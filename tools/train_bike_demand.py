#!/usr/bin/env python3
"""Frozen UCI bike-demand training and native replay; see docs/bike-demand-study-v1.md."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import stealth_cpu as native
import bike_demand as data
from build_identity import extension_identity
from torch_forecaster import Config, Forecaster
from training_state import (canonical,digest,bounded,strict_json,environment,make_optimizer,
                            train_epoch,save_checkpoint,load_checkpoint)
from train_ettm1 import ROOT,clean_commit,write,file_hash,predictions
# These helpers are domain-independent. Leave the frozen BTC study unchanged.
from train_btc_returns import tensors,native_predictions,expected_names

PROTOCOL=ROOT/'benchmarks/bike/demand_v1.json'


def read_recipe():
    p=strict_json(bounded(PROTOCOL,65536))
    if (p['schema_version']!=1 or p['experiment_id']!='bike_hourly_demand_v1'
            or p['seeds']!=[7,19,43] or p['epochs']!=8 or p['features']!=list(data.FEATURES)
            or p['model']!={'features':5,'width':64,'heads':4,'hidden':256,'context':48,
                            'horizon':1,'target':0,'epsilon':1e-5,'output':'correction'}
            or p['cpu_threads']!=1 or p['deterministic_algorithms'] is not True):
        raise ValueError('Changed frozen task; create a separately registered version')
    return p


def load_frame(path,p):
    frame=data.parse_csv(bounded(path,data.MAX_BYTES),p['data']['csv_sha256'])
    if len(frame.hours)!=p['data']['rows'] or frame.hours[0]!=0 or frame.hours[-1]!=data.MAX_HOURS-1:
        raise ValueError('Unexpected source coverage')
    return frame


def parity(model,samples,p):
    x,_=tensors(samples);expected=predictions(model,x,p['evaluation_batch_size'])
    actual=native_predictions(model.export_bytes(),samples,p['evaluation_batch_size'])
    absolute=p['parity_absolute_tolerance'];relative=p['parity_relative_tolerance']
    difference=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    if not np.isfinite(actual).all() or not (difference<=absolute+relative*np.abs(expected)).all():
        raise ValueError('Native latest prediction parity failed')
    indices=np.unique(np.linspace(0,len(x)-1,min(len(x),p['full_trace_parity_windows']),dtype=int))
    raw=np.ascontiguousarray(samples.x[indices])
    with torch.inference_mode():full=model(torch.from_numpy(raw)).numpy()
    maximum=float(difference.max())
    for kernel in ('reference','packed'):
        trace=native.ForecastModel.from_bytes(model.export_bytes(),kernel=kernel).session().predict_batch(raw)
        delta=np.abs(trace.astype(np.float64)-full.astype(np.float64))
        if not np.isfinite(trace).all() or not (delta<=absolute+relative*np.abs(full)).all():
            raise ValueError('Native full trace parity failed')
        maximum=max(maximum,float(delta.max()))
    return actual,dict(latest_windows=len(actual),full_trace_windows=len(indices),full_trace_kernels=['reference','packed'],
                       max_absolute_error_rentals=maximum,absolute_tolerance=absolute,relative_tolerance=relative)


def baseline_values(record,samples):
    return dict(last_hour=samples.x[:,-1,0].astype(np.float64),
                same_hour_yesterday=samples.x[:,samples.x.shape[1]-24,0].astype(np.float64),
                training_weekday_hour_mean=np.asarray(record['calendar_means'])[data.calendar_index(samples.target_hours)],
                ridge=data.ridge_predict(record['ridge'],samples.x))


def fit_baselines(train,val,p):
    candidates=[];models=[]
    for alpha in p['ridge_alphas']:
        model=data.fit_ridge(train.x,train.actual,alpha)
        candidates.append(dict(alpha=alpha,validation=data.scores(data.ridge_predict(model,val.x),val.actual)));models.append(model)
    best=min(range(len(candidates)),key=lambda i:candidates[i]['validation']['mae'])
    groups=data.calendar_index(train.target_hours);counts=np.bincount(groups,minlength=168)
    sums=np.bincount(groups,weights=train.actual,minlength=168)
    means=np.full(168,float(train.actual.mean()));np.divide(sums,counts,out=means,where=counts>0)
    record=dict(calendar_means=means.tolist(),calendar_counts=counts.tolist(),ridge=models[best],ridge_candidates=candidates)
    record['validation']={name:data.scores(values,val.actual) for name,values in baseline_values(record,val).items()}
    return record


def ledger(samples,model_predictions,baselines):
    lines=[]
    for i,h in enumerate(samples.target_hours):
        row=dict(information_end_source_local=data.label(samples.input_end_hours[i]),target_start_source_local=data.label(h),
                 target_end_source_local=data.label(h+1),actual_rentals=float(samples.actual[i]),
                 native_rentals={str(seed):float(values[i]) for seed,values in model_predictions.items()},
                 baseline_rentals={name:float(values[i]) for name,values in baselines.items()})
        lines.append(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    return ''.join(lines).encode()


def run_study(output,frame,p,binding):
    """Configurable only for synthetic tests; the real CLI uses the frozen recipe."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    try:
        c=Config(**p['model']);ends={k:data.hour(v) for k,v in p['split_ends'].items()}
        if not 0<ends['train']<ends['validation']<ends['test'] or ends['test']!=int(frame.hours[-1])+1:
            raise ValueError('Invalid study split coverage')
        if c.features!=5 or c.context<24 or c.horizon!=1 or c.target!=0 or c.output!='correction':
            raise ValueError('Unsupported bike model contract')
        if torch.get_num_threads()!=1 or not torch.are_deterministic_algorithms_enabled():
            raise ValueError('Study requires deterministic single-thread CPU Torch')
        write(output/'protocol.json',p)
        train=data.split(frame,c.context,0,ends['train']);val=data.split(frame,c.context,ends['train'],ends['validation'])
        mean,scale=data.fit_scaler(frame,train)
        scaler=dict(features=list(data.FEATURES),mean=mean.tolist(),scale=scale.tolist(),
                    last_fitted_input_hour=int(frame.hours[train.origins[-1]]))
        bound={**binding,'scaler_sha256':digest(canonical(scaler))};write(output/'scaler.json',scaler)
        write(output/'started.json',dict(binding=bound,environment=environment(),native=extension_identity(),
                                         training_samples=len(train.y),validation_samples=len(val.y)))
        base=fit_baselines(train,val,p);write(output/'baselines.json',base)
        tx,ty=tensors(train);vx,_=tensors(val);selected={};models={}
        for seed in p['seeds']:
            root=output/f'seed-{seed}';root.mkdir();model=Forecaster(c,mean,scale,seed=seed)
            optimizer=make_optimizer(model,p['optimizer']);rng=torch.Generator().manual_seed(seed+1000)
            history=[];best_epoch=0;best_mae=float('inf')
            for epoch in range(p['epochs']+1):
                started=time.perf_counter();loss=None
                if epoch:
                    loss=train_epoch(model,optimizer,tx,ty,rng,batch_size=p['batch_size'],
                                     accumulate=p['accumulate'],clip_norm=p['clip_norm'])
                validation=data.scores(predictions(model,vx,p['evaluation_batch_size']),val.actual)
                if validation['mae']<best_mae:best_epoch=epoch;best_mae=validation['mae']
                history.append(dict(epoch=epoch,training_standardized_mse=loss,validation=validation))
                save_checkpoint(root/f'epoch-{epoch:03d}',model,optimizer,rng,
                                dict(epoch=epoch,best_epoch=best_epoch,history=history),{**bound,'seed':seed})
                print(json.dumps(dict(seed=seed,epoch=epoch,validation_mae_rentals=validation['mae'],
                                      seconds=round(time.perf_counter()-started,3))),flush=True)
            chosen,_,_,_=load_checkpoint(root/f'epoch-{best_epoch:03d}',{**bound,'seed':seed})
            _,detail=parity(chosen,val,p);blob=chosen.export_bytes();(root/'model.stlf').write_bytes(blob)
            write(root/'history.json',history)
            selected[str(seed)]=dict(selected_epoch=best_epoch,model_sha256=digest(blob),validation_history=history,
                                      validation_parity=detail,initial_persistence_selected=best_epoch==0)
            models[str(seed)]=chosen
        write(output/'selection.json',dict(seeds=selected,ridge_alpha=base['ridge']['alpha'],evaluation_started=False))
        # The test arrays and metrics are created only after every model is fixed.
        test=data.split(frame,c.context,ends['validation'],ends['test']);bases=baseline_values(base,test)
        model_predictions={};results={}
        for seed,model in models.items():
            actual,detail=parity(model,test,p);model_predictions[seed]=actual
            results[seed]={**selected[seed],'test':data.scores(actual,test.actual),'test_parity':detail,
                           'negative_predictions':int((actual<0).sum())}
        baseline_scores={name:{**data.scores(values,test.actual),'negative_predictions':int((values<0).sum())}
                         for name,values in bases.items()}
        interval=data.loss_interval(model_predictions[str(p['uncertainty']['primary_seed'])],bases['last_hour'],test.actual,
                                    test.target_hours,p['uncertainty'])
        gate=all(r['test']['mae']<=.9*baseline_scores['last_hour']['mae'] for r in results.values()) and interval['upper']<0
        result=dict(schema_version=1,experiment_id=p['experiment_id'],binding=bound,models=results,baselines=baseline_scores,
                    primary_seed_last_hour_loss_difference=interval,demonstration_gate_passed=bool(gate),
                    train_samples=len(train.y),validation_samples=len(val.y),test_samples=len(test.y),
                    excluded_hours=dict(train=train.excluded_hours,validation=val.excluded_hours,test=test.excluded_hours),
                    missing_source_hours=int(frame.hours[-1]-frame.hours[0]+1-len(frame.hours)),
                    source_calendar_timezone_verified=False,forecast_clipping_applied=False)
        (output/'predictions.jsonl').write_bytes(ledger(test,model_predictions,bases));write(output/'scores.json',result)
        write(output/'complete.json',dict(schema_version=1,status='complete',binding=bound,
                                          files={name:file_hash(output/name) for name in expected_names(p)}))
        return result
    except Exception as exc:
        write(output/'failure.json',dict(error_type=type(exc).__name__));raise


def verify_run(path):
    path=Path(path);sealed=strict_json(bounded(path/'complete.json',1024*1024));p=strict_json(bounded(path/'protocol.json',65536))
    if (sealed.get('schema_version')!=1 or sealed.get('status')!='complete'
            or set(sealed.get('files',{}))!=set(expected_names(p))):raise ValueError('Invalid complete study seal')
    for name,sha in sealed['files'].items():
        if file_hash(path/name)!=sha:raise ValueError('Study artifact digest mismatch')
    selection=strict_json(bounded(path/'selection.json',1024*1024))
    for seed in p['seeds']:
        s=selection['seeds'][str(seed)];history=strict_json(bounded(path/f'seed-{seed}/history.json',1024*1024))
        if (len(history)!=p['epochs']+1 or [r['epoch'] for r in history]!=list(range(p['epochs']+1))
                or s['selected_epoch']!=min(history,key=lambda r:r['validation']['mae'])['epoch']
                or s['model_sha256']!=file_hash(path/f'seed-{seed}/model.stlf')):
            raise ValueError('Invalid validation selection')
    return sealed


def replay(path,frame):
    path=Path(path);verify_run(path);p=strict_json(bounded(path/'protocol.json',65536))
    test=data.split(frame,p['model']['context'],data.hour(p['split_ends']['validation']),data.hour(p['split_ends']['test']))
    base=strict_json(bounded(path/'baselines.json',1024*1024))
    preds={str(seed):native_predictions(bounded(path/f'seed-{seed}/model.stlf',16*1024*1024),test,p['evaluation_batch_size'])
           for seed in p['seeds']}
    return ledger(test,preds,baseline_values(base,test))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('train','replay'))
    parser.add_argument('--csv',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run',type=Path);args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'local'):raise ValueError('Use ignored local/ for outputs')
    if args.output.exists():raise FileExistsError('Output already exists')
    p=read_recipe();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    commit=clean_commit();extension_identity();frame=load_frame(args.csv,p)
    if args.command=='train':
        if args.run is not None:parser.error('Training does not accept --run')
        r=run_study(args.output,frame,p,dict(protocol_sha256=file_hash(PROTOCOL),data_sha256=p['data']['csv_sha256'],code_commit=commit))
        print(json.dumps(dict(status='complete',test_samples=r['test_samples'],gate=r['demonstration_gate_passed'])))
    else:
        if args.run is None:parser.error('Replay requires --run')
        seal=verify_run(args.run)
        if seal['binding']['data_sha256']!=p['data']['csv_sha256'] or seal['binding']['protocol_sha256']!=file_hash(PROTOCOL):
            raise ValueError('Replay data or protocol mismatch')
        raw=replay(args.run,frame)
        if raw!=bounded(args.run/'predictions.jsonl',64*1024*1024):raise ValueError('Native replay differs')
        args.output.mkdir(parents=True,exist_ok=False);(args.output/'predictions.jsonl').write_bytes(raw)
        write(args.output/'complete.json',dict(status='complete',byte_identical=True,ledger_sha256=digest(raw),
                                              code_commit=commit,native=extension_identity()))
        print(json.dumps(dict(status='complete',byte_identical=True)))


if __name__=='__main__':main()
