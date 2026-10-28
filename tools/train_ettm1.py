#!/usr/bin/env python3
"""Frozen CPU learning experiment: train/validate first, then sealed test scoring."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import torch
import stealth_cpu as native
import prepare_public_benchmark as data
from torch_forecaster import Config, Forecaster
from training_state import (bounded,canonical,digest,environment,load_checkpoint,make_optimizer,
                            metrics,save_checkpoint,strict_json,train_epoch)

ROOT=Path(__file__).resolve().parents[1]


def read_recipe():
    task=strict_json((ROOT/'benchmarks/ettm1/one_hour.json').read_bytes())
    protocol=strict_json((ROOT/'benchmarks/ettm1/training_v1.json').read_bytes())
    m=task['model']
    if (task['task_id']!=protocol['task_id'] or task['target_mode']!='point' or m['blocks']!=1
            or m['dropout']!=0 or m['activation']!='exact_gelu' or not m['final_normalization']
            or m['positions']!='fixed_sinusoidal_window_relative'
            or m['output']!='linear_scalar_then_inverse_target_scaling'
            or task['input_columns']!=list(data.COLUMNS[1:]) or task['interval_seconds']!=900):
        raise ValueError('Unsupported task contract')
    c=Config(len(task['input_columns']),m['width'],m['heads'],m['hidden'],task['context'],
             task['horizon_steps'],task['input_columns'].index(task['target_column']),m['epsilon'])
    if (protocol['schema_version']!=1 or protocol['seeds']!=[7,19,43] or protocol['epochs']!=6
            or protocol['cpu_threads']!=1 or protocol['deterministic_algorithms'] is not True):
        raise ValueError('Unsupported experiment recipe; create a new version for changes')
    return protocol,task,c


def split_tensors(rows,ends,split,c):
    if rows.ndim!=2 or rows.shape[1]!=c.features or rows.dtype!=torch.float32 or rows.device.type!='cpu':
        raise ValueError('Invalid observation matrix')
    descriptors=list(data.windows(len(rows),ends,split,c.context,c.horizon,'point'))
    if not descriptors: raise ValueError('Empty split')
    first,last=descriptors[0],descriptors[-1]
    # Views share the original observations. Only selected minibatches are copied.
    x=rows[first.input_start:last.input_end].unfold(0,c.context,1).transpose(1,2)
    y=rows[first.target_start:last.target_end,c.target]
    if len(x)!=len(descriptors) or len(y)!=len(x): raise ValueError('Noncontiguous split')
    return x,y


def baseline_predictions(x,c,period=96):
    index=c.context+c.horizon-1-period
    if x.ndim!=3 or x.shape[1:]!=(c.context,c.features) or not 0<=index<c.context:
        raise ValueError('Seasonal target must be a known input observation')
    return x[:,-1,c.target].numpy().copy(),x[:,index,c.target].numpy().copy()


def predictions(model,x,batch_size):
    model.eval(); outputs=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):
            outputs.append(model(x[start:start+batch_size])[:,-1].numpy().copy())
    return np.concatenate(outputs)


def parity(model,x,protocol):
    indices=np.unique(np.linspace(0,len(x)-1,min(protocol['native_parity_windows_per_split'],len(x)),dtype=np.int64))
    sample=np.ascontiguousarray(x[torch.from_numpy(indices)].numpy())
    with torch.inference_mode(): expected=model(torch.from_numpy(sample)).numpy()
    maximum=0.
    for kernel in ('reference','packed'):
        actual=native.ForecastModel.from_bytes(model.export_bytes(),kernel=kernel).session().predict_batch(sample)
        difference=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
        allowance=protocol['parity_absolute_tolerance']+protocol['parity_relative_tolerance']*np.abs(expected)
        if not np.isfinite(actual).all() or not np.all(difference<=allowance):
            raise ValueError(f'Trained native parity failed: {kernel}')
        maximum=max(maximum,float(difference.max()))
    return dict(windows=len(indices),origins_per_window=x.shape[1],kernels=['reference','packed'],
                max_absolute_error=maximum,absolute_tolerance=protocol['parity_absolute_tolerance'],
                relative_tolerance=protocol['parity_relative_tolerance'])


def write(path,value):
    with Path(path).open('xb') as f: f.write(canonical(value))


def file_hash(path): return digest(bounded(path,64*1024*1024))


def expected_files(epochs):
    return ['result.json','export.json','model.stlf']+[
        f'epoch-{epoch:03d}/{name}' for epoch in range(1,epochs+1) for name in ('manifest.json','arrays.npz')]


def verify_run(path,protocol,binding,seed):
    path=Path(path); complete=strict_json(bounded(path/'complete.json',1024*1024))
    if (complete.get('status')!='complete' or complete.get('seed')!=seed or complete.get('binding')!={**binding,'seed':seed}
            or complete.get('environment')!=environment()
            or set(complete.get('files',{}))!=set(expected_files(protocol['epochs']))):
        raise ValueError('Run is incomplete or belongs to another experiment/environment')
    for name,sha in complete['files'].items():
        if file_hash(path/name)!=sha: raise ValueError('Run artifact hash mismatch')
    result=strict_json(bounded(path/'result.json',1024*1024))
    history=result['history']
    if len(history)!=protocol['epochs'] or [r['epoch'] for r in history]!=list(range(1,protocol['epochs']+1)):
        raise ValueError('Incomplete learning history')
    best=min(history,key=lambda r:r['validation']['mae'])['epoch']
    if best!=result['best_epoch']: raise ValueError('Invalid validation selection')
    return result


def run_seed(output,rows,ends,c,mean,scale,protocol,task,binding,seed,resume=None):
    if seed not in protocol['seeds']: raise ValueError('Seed outside frozen recipe')
    output=Path(output)
    if output.exists(): raise FileExistsError('Run destination already exists')
    bound={**binding,'seed':seed}
    train_x,train_y=split_tensors(rows,ends,'train',c)
    val_x,val_y=split_tensors(rows,ends,'validation',c)
    start=0; history=[]; best_epoch=0
    if resume is None:
        model=Forecaster(c,mean,scale,seed=seed)
        optimizer=make_optimizer(model,protocol['optimizer']); rng=torch.Generator().manual_seed(seed+1000)
    else:
        resume=Path(resume)
        model,optimizer,rng,record=load_checkpoint(resume,bound)
        start=record['epoch']; history=record['history']; best_epoch=record['best_epoch']
        if (model.config!=c or not 1<=start<protocol['epochs'] or len(history)!=start
                or [r['epoch'] for r in history]!=list(range(1,start+1))
                or best_epoch!=min(history,key=lambda r:r['validation']['mae'])['epoch']):
            raise ValueError('Invalid resume position')
    output.mkdir(parents=True,exist_ok=False)
    write(output/'started.json',dict(binding=bound,environment=environment(),resume_epoch=start))
    try:
        if resume is not None:
            for epoch in range(1,start+1):
                prior=resume.parent/f'epoch-{epoch:03d}'
                _,_,_,record=load_checkpoint(prior,bound)
                if record['epoch']!=epoch or record['history']!=history[:epoch]:
                    raise ValueError('Resume history mismatch')
                shutil.copytree(prior,output/prior.name)
        best_mae=float('inf') if not history else min(r['validation']['mae'] for r in history)
        for epoch in range(start+1,protocol['epochs']+1):
            clock=time.perf_counter()
            loss=train_epoch(model,optimizer,train_x,train_y,rng,batch_size=protocol['batch_size'],
                             accumulate=protocol['accumulate'],clip_norm=protocol['clip_norm'])
            validation=metrics(predictions(model,val_x,protocol['evaluation_batch_size']),val_y.numpy())
            history.append(dict(epoch=epoch,training_standardized_mse=loss,validation=validation))
            if validation['mae']<best_mae: best_mae=validation['mae']; best_epoch=epoch
            record=dict(epoch=epoch,history=history,best_epoch=best_epoch)
            save_checkpoint(output/f'epoch-{epoch:03d}',model,optimizer,rng,record,bound)
            print(json.dumps(dict(seed=seed,epoch=epoch,training_mse=loss,validation_mae=validation['mae'],
                                  seconds=round(time.perf_counter()-clock,2))),flush=True)
        best,_,_,_=load_checkpoint(output/f'epoch-{best_epoch:03d}',bound)
        result=dict(seed=seed,best_epoch=best_epoch,history=history,
                    parity={'train':parity(best,train_x,protocol),'validation':parity(best,val_x,protocol)})
        blob=best.export_bytes()
        with (output/'model.stlf').open('xb') as f: f.write(blob)
        export=dict(schema_version=1,model_sha256=digest(blob),binding=bound,environment=environment(),
                    config=asdict(c),task_id=task['task_id'],input_columns=task['input_columns'],
                    interval_seconds=task['interval_seconds'],target_column=task['target_column'],
                    scaler_fp32=dict(mean=best.mean.tolist(),scale=best.scale.tolist()),
                    selected_epoch=best_epoch,selection='validation_mae_earliest_minimum',
                    checkpoint_sha256=file_hash(output/f'epoch-{best_epoch:03d}'/'manifest.json'))
        write(output/'export.json',export); write(output/'result.json',result)
        files={name:file_hash(output/name) for name in expected_files(protocol['epochs'])}
        write(output/'complete.json',dict(status='complete',seed=seed,binding=bound,environment=environment(),files=files))
        return result
    except Exception as exc:
        write(output/'failure.json',dict(error_type=type(exc).__name__,completed_epochs=len(history)))
        raise


def evaluate_runs(output,runs,rows,ends,c,protocol,task,binding,period=96):
    # Validate every seed and artifact BEFORE constructing or scoring test windows.
    if len(runs)!=len(protocol['seeds']): raise ValueError('All preregistered seeds required before test scoring')
    sealed=[]; observed=[]
    for path in runs:
        complete=strict_json(bounded(Path(path)/'complete.json',1024*1024)); seed=complete['seed']
        observed.append(seed); result=verify_run(path,protocol,binding,seed)
        model,_,_,_=load_checkpoint(Path(path)/f"epoch-{result['best_epoch']:03d}",{**binding,'seed':seed})
        if model.export_bytes()!=bounded(Path(path)/'model.stlf',16*1024*1024):
            raise ValueError('Export differs from selected checkpoint')
        sealed.append((path,seed,result,model))
    if sorted(observed)!=sorted(protocol['seeds']): raise ValueError('Missing or duplicate seed')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    write(output/'started.json',dict(binding=binding,sealed_runs={str(seed):file_hash(Path(path)/'complete.json') for path,seed,_,_ in sealed}))
    try:
        x,y=split_tensors(rows,ends,'test',c)
        last,seasonal=baseline_predictions(x,c,period)
        result=dict(schema_version=1,experiment_id=protocol['experiment_id'],binding=binding,environment=environment(),
                    baselines=dict(last_observation=metrics(last,y.numpy()),seasonal=metrics(seasonal,y.numpy())),models={})
        for path,seed,selected,model in sealed:
            score=metrics(predictions(model,x,protocol['evaluation_batch_size']),y.numpy())
            result['models'][str(seed)]=dict(best_epoch=selected['best_epoch'],test=score,
                                           parity=parity(model,x,protocol),model_sha256=file_hash(Path(path)/'model.stlf'))
        write(output/'scores.json',result)
        write(output/'complete.json',dict(status='complete',scores_sha256=file_hash(output/'scores.json')))
        return result
    except Exception as exc:
        write(output/'failure.json',dict(error_type=type(exc).__name__))
        raise


def clean_commit():
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    if dirty.strip(): raise ValueError('Commit the frozen code/protocol before running a real experiment')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','evaluate'])
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--runs',nargs='+',type=Path)
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'local'): raise ValueError('Keep experiment outputs under ignored local/')
    protocol,task,c=read_recipe()
    torch.set_num_threads(protocol['cpu_threads']); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    commit=clean_commit()
    data.check_snapshot(args.raw,args.prepared)
    frame=data.verified_frame(args.raw)
    scaler=strict_json(bounded(args.prepared/'scaler.json',65536))
    binding=dict(protocol_sha256=file_hash(ROOT/'benchmarks/ettm1/training_v1.json'),
                 task_sha256=file_hash(ROOT/'benchmarks/ettm1/one_hour.json'),
                 data_sha256=file_hash(args.raw/'ETTm1.csv'),scaler_sha256=file_hash(args.prepared/'scaler.json'),code_commit=commit)
    rows=torch.tensor(frame.values,dtype=torch.float32)
    if args.command=='train':
        if args.seed is None or args.runs is not None: parser.error('Training requires --seed and no --runs')
        result=run_seed(args.output,rows,data.SOURCE['split_ends'],c,scaler['mean'],scaler['scale'],protocol,task,binding,args.seed,args.resume)
        print(json.dumps(dict(status='complete',seed=args.seed,best_epoch=result['best_epoch'])))
    else:
        if args.seed is not None or args.resume is not None or not args.runs: parser.error('Evaluation requires --runs only')
        print(json.dumps(evaluate_runs(args.output,args.runs,rows,data.SOURCE['split_ends'],c,protocol,task,binding),indent=2))


if __name__=='__main__': main()
