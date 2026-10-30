#!/usr/bin/env python3
"""Frozen exploratory BTC return training, export, evaluation and native replay."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
import stealth_cpu as native
import btc_returns as data
from hourly_bars import BarContract, parse_bars, timestamp_ns, MAX_BYTES
from build_identity import extension_identity
from torch_forecaster import Config, Forecaster
from training_state import (canonical,digest,bounded,strict_json,environment,make_optimizer,
                            train_epoch,save_checkpoint,load_checkpoint)
from train_ettm1 import ROOT,clean_commit,write,file_hash,predictions

PROTOCOL=ROOT/'benchmarks/btc/return_v1.json'


def read_recipe():
    p=strict_json(bounded(PROTOCOL,65536))
    if (p['schema_version']!=1 or p['experiment_id']!='btc_hourly_return_v1'
            or p['seeds']!=[7,19,43] or p['epochs']!=3 or p['features']!=list(data.FEATURES)
            or p['model']!={'features':5,'width':64,'heads':4,'hidden':256,'context':96,
                            'horizon':1,'target':0,'epsilon':1e-5,'output':'level'}
            or p['cpu_threads']!=1 or p['deterministic_algorithms'] is not True
            or p['data']['target_symbol']!='BTCUSDT' or p['scientific_evidence_ceiling']!='E0'):
        raise ValueError('Changed frozen task; create a separately registered version')
    return p


def load_frame(path,p):
    c=p['data']
    bars=parse_bars(bounded(path,MAX_BYTES),BarContract(c['sha256'],tuple(c['symbols']),c['start'],c['end']))
    return data.features(tuple(bar for bar in bars if bar.symbol=='BTCUSDT'))


def tensors(samples):
    # Separate mutable Torch storage; NumPy sample windows are read-only views.
    return torch.tensor(samples.x,dtype=torch.float32),torch.tensor(samples.y,dtype=torch.float32)


def native_predictions(blob,samples,batch_size):
    session=native.ForecastModel.from_bytes(blob,kernel='packed').session()
    output=[]
    for i in range(0,len(samples.y),batch_size):
        output.append(session.predict_last_batch(np.ascontiguousarray(samples.x[i:i+batch_size])))
    return np.concatenate(output)


def parity(model,samples,p):
    x,_=tensors(samples)
    expected=predictions(model,x,p['evaluation_batch_size'])
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
    return actual,dict(latest_windows=len(actual),full_trace_windows=len(indices),
                       full_trace_kernels=['reference','packed'],max_absolute_error_bps=maximum,
                       absolute_tolerance=absolute,relative_tolerance=relative)


def baseline_values(record,samples):
    return dict(zero=np.zeros(len(samples.y)),training_label_mean=np.full(len(samples.y),record['training_label_mean']),
                last_hour_return=samples.x[:,-1,0].astype(np.float64),ridge=data.ridge_predict(record['ridge'],samples.x))


def fit_baselines(train,val,p):
    candidates=[];models=[]
    for alpha in p['ridge_alphas']:
        model=data.fit_ridge(train.x,train.actual,alpha)
        candidates.append(dict(alpha=alpha,validation=data.scores(data.ridge_predict(model,val.x),val.actual)))
        models.append(model)
    best=min(range(len(candidates)),key=lambda i:candidates[i]['validation']['mse'])
    record=dict(training_label_mean=float(train.actual.mean()),ridge=models[best],ridge_candidates=candidates)
    record['validation']={name:data.scores(values,val.actual) for name,values in baseline_values(record,val).items()}
    return record


def expected_names(p):
    files=['started.json','protocol.json','scaler.json','baselines.json','selection.json','scores.json','predictions.jsonl']
    for seed in p['seeds']:
        files.extend([f'seed-{seed}/model.stlf',f'seed-{seed}/history.json'])
        files.extend(f'seed-{seed}/epoch-{epoch:03d}/{name}' for epoch in range(p['epochs']+1)
                     for name in ('manifest.json','arrays.npz'))
    return files


def ledger(samples,model_predictions,baselines):
    lines=[]
    for i in range(len(samples.y)):
        row=dict(information_end_ns=int(samples.input_end_ns[i]),
                 assumed_decision_at_ns=int(samples.input_end_ns[i])+1,
                 target_start_ns=int(samples.input_end_ns[i]),target_end_ns=int(samples.label_end_ns[i]),
                 actual_return_bps=float(samples.actual[i]),
                 native_return_bps={str(seed):float(values[i]) for seed,values in model_predictions.items()},
                 baseline_return_bps={name:float(values[i]) for name,values in baselines.items()})
        lines.append(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    return ''.join(lines).encode()


def run_study(output,frame,p,binding):
    """Internal configurable path for tiny tests; the real CLI uses the frozen recipe."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    try:
        c=Config(**p['model']);ends={k:timestamp_ns(v) for k,v in p['split_ends'].items()}
        if not 0<ends['train']<ends['validation']<ends['test'] or ends['test']!=int(frame.end_ns[-1]):
            raise ValueError('Invalid study split coverage')
        if c.features!=len(data.FEATURES) or c.horizon!=1 or c.target!=0 or c.output!='level':
            raise ValueError('Unsupported BTC model contract')
        if torch.get_num_threads()!=1 or not torch.are_deterministic_algorithms_enabled():
            raise ValueError('Study requires deterministic single-thread CPU Torch')
        write(output/'protocol.json',p)
        train=data.split(frame,c.context,0,ends['train'])
        val=data.split(frame,c.context,ends['train'],ends['validation'])
        mean,scale=data.fit_scaler(frame,train)
        scaler=dict(features=list(data.FEATURES),mean=mean.tolist(),scale=scale.tolist(),
                    last_fitted_input_end_ns=int(train.input_end_ns[-1]))
        bound={**binding,'scaler_sha256':digest(canonical(scaler))}
        write(output/'scaler.json',scaler)
        write(output/'started.json',dict(binding=bound,environment=environment(),native=extension_identity(),
                                         training_samples=len(train.y),validation_samples=len(val.y)))
        baseline=fit_baselines(train,val,p);write(output/'baselines.json',baseline)
        tx,ty=tensors(train);vx,_=tensors(val);selected={};models={}
        for seed in p['seeds']:
            root=output/f'seed-{seed}';root.mkdir()
            model=Forecaster(c,mean,scale,seed=seed)
            with torch.no_grad():
                model.head_weight.zero_()
                model.head_bias.fill_((float(train.actual.mean())-float(mean[0]))/float(scale[0]))
            optimizer=make_optimizer(model,p['optimizer']);rng=torch.Generator().manual_seed(seed+1000)
            history=[];best_epoch=0;best_mse=float('inf')
            for epoch in range(p['epochs']+1):
                started=time.perf_counter();loss=None
                if epoch:
                    loss=train_epoch(model,optimizer,tx,ty,rng,batch_size=p['batch_size'],
                                     accumulate=p['accumulate'],clip_norm=p['clip_norm'])
                validation=data.scores(predictions(model,vx,p['evaluation_batch_size']),val.actual)
                if validation['mse']<best_mse:best_epoch=epoch;best_mse=validation['mse']
                history.append(dict(epoch=epoch,training_standardized_mse=loss,validation=validation))
                save_checkpoint(root/f'epoch-{epoch:03d}',model,optimizer,rng,
                                dict(epoch=epoch,best_epoch=best_epoch,history=history),{**bound,'seed':seed})
                print(json.dumps(dict(seed=seed,epoch=epoch,validation_mse_bps2=validation['mse'],
                                      seconds=round(time.perf_counter()-started,3))),flush=True)
            chosen,_,_,_=load_checkpoint(root/f'epoch-{best_epoch:03d}',{**bound,'seed':seed})
            _,validation_parity=parity(chosen,val,p)
            blob=chosen.export_bytes();(root/'model.stlf').write_bytes(blob)
            write(root/'history.json',history)
            selected[str(seed)]=dict(selected_epoch=best_epoch,model_sha256=digest(blob),
                                      validation_history=history,validation_parity=validation_parity,
                                      initial_constant_selected=best_epoch==0)
            models[str(seed)]=chosen
        # All seeds and the baseline are fixed before creating/scoring test samples.
        write(output/'selection.json',dict(seeds=selected,ridge_alpha=baseline['ridge']['alpha'],
                                          evaluation_started=False))
        test=data.split(frame,c.context,ends['validation'],ends['test'])
        bases=baseline_values(baseline,test);model_predictions={};results={}
        for seed,model in models.items():
            actual,detail=parity(model,test,p);model_predictions[seed]=actual
            results[seed]={**selected[seed],'test':data.scores(actual,test.actual),'test_parity':detail}
        baseline_scores={name:data.scores(values,test.actual) for name,values in bases.items()}
        interval=data.loss_interval(model_predictions[str(p['uncertainty']['primary_seed'])],bases['zero'],test.actual,p['uncertainty'])
        gate=all(r['test']['mse']<min(baseline_scores['zero']['mse'],baseline_scores['ridge']['mse']) for r in results.values()) and interval['upper']<0
        result=dict(schema_version=1,experiment_id=p['experiment_id'],hypothesis_id=p['hypothesis_id'],
                    scientific_evidence_ceiling='E0',binding=bound,models=results,baselines=baseline_scores,
                    primary_seed_zero_loss_difference=interval,descriptive_gate_passed=bool(gate),
                    historical_availability_verified=False,trading_pnl_computed=False,
                    train_samples=len(train.y),validation_samples=len(val.y),test_samples=len(test.y))
        (output/'predictions.jsonl').write_bytes(ledger(test,model_predictions,bases))
        write(output/'scores.json',result)
        write(output/'complete.json',dict(schema_version=1,status='complete',binding=bound,
                                          files={name:file_hash(output/name) for name in expected_names(p)}))
        return result
    except Exception as exc:
        write(output/'failure.json',dict(error_type=type(exc).__name__))
        raise


def verify_run(path):
    path=Path(path);sealed=strict_json(bounded(path/'complete.json',1024*1024))
    p=strict_json(bounded(path/'protocol.json',65536))
    if (sealed.get('schema_version')!=1 or sealed.get('status')!='complete'
            or set(sealed.get('files',{}))!=set(expected_names(p))):raise ValueError('Invalid complete study seal')
    for name,sha in sealed['files'].items():
        if file_hash(path/name)!=sha:raise ValueError('Study artifact digest mismatch')
    selection=strict_json(bounded(path/'selection.json',1024*1024))
    for seed in p['seeds']:
        s=selection['seeds'][str(seed)]
        history=strict_json(bounded(path/f'seed-{seed}/history.json',1024*1024))
        if (len(history)!=p['epochs']+1 or [r['epoch'] for r in history]!=list(range(p['epochs']+1))
                or s['selected_epoch']!=min(history,key=lambda r:r['validation']['mse'])['epoch']
                or s['model_sha256']!=file_hash(path/f'seed-{seed}/model.stlf')):
            raise ValueError('Invalid validation selection')
    return sealed


def replay(path,frame):
    path=Path(path);verify_run(path)
    p=strict_json(bounded(path/'protocol.json',65536))
    test=data.split(frame,p['model']['context'],timestamp_ns(p['split_ends']['validation']),timestamp_ns(p['split_ends']['test']))
    base=strict_json(bounded(path/'baselines.json',1024*1024))
    preds={str(seed):native_predictions(bounded(path/f'seed-{seed}/model.stlf',16*1024*1024),test,p['evaluation_batch_size'])
           for seed in p['seeds']}
    return ledger(test,preds,baseline_values(base,test))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('train','replay'))
    parser.add_argument('--csv',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'local'):raise ValueError('Use ignored local/ for outputs')
    if args.output.exists():raise FileExistsError('Output already exists')
    p=read_recipe();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    commit=clean_commit();extension_identity();frame=load_frame(args.csv,p)
    if args.command=='train':
        if args.run is not None:parser.error('Training does not accept --run')
        result=run_study(args.output,frame,p,dict(protocol_sha256=file_hash(PROTOCOL),
                                                data_sha256=p['data']['sha256'],code_commit=commit))
        print(json.dumps(dict(status='complete',test_samples=result['test_samples'],gate=result['descriptive_gate_passed'])))
    else:
        if args.run is None:parser.error('Replay requires --run')
        sealed=verify_run(args.run)
        if (sealed['binding']['data_sha256']!=p['data']['sha256']
                or sealed['binding']['protocol_sha256']!=file_hash(PROTOCOL)):
            raise ValueError('Replay data or protocol mismatch')
        raw=replay(args.run,frame)
        if raw!=bounded(args.run/'predictions.jsonl',64*1024*1024):raise ValueError('Native replay differs')
        args.output.mkdir(parents=True,exist_ok=False);(args.output/'predictions.jsonl').write_bytes(raw)
        write(args.output/'complete.json',dict(status='complete',byte_identical=True,
                                              ledger_sha256=digest(raw),code_commit=commit,native=extension_identity()))
        print(json.dumps(dict(status='complete',byte_identical=True)))


if __name__=='__main__':main()
