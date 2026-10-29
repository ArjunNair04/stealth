#!/usr/bin/env python3
"""Retrospective v1 diagnosis; validation-selected linear fits; reserved tail unused."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import numpy as np
import torch
import prepare_public_benchmark as data
from forecast_diagnostics import (window_view, design_matrix, fit_linear_families,
                                  select_candidate, change_thresholds, describe)
from training_state import canonical, strict_json, bounded, digest, environment, load_checkpoint
from train_ettm1 import clean_commit, read_recipe, verify_run, file_hash

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    with Path(path).open('xb') as f: f.write(canonical(value))


def write_arrays(path,arrays):
    with Path(path).open('xb') as f: np.savez(f,**arrays)


def run_analysis(output,rows,timestamps,scaler,protocol,predictors,binding,original_scores=None):
    output=Path(output)
    if output.exists(): raise FileExistsError('Diagnostic destination exists')
    end=protocol['maximum_observation_end_exclusive']; ends=protocol['split_ends']
    if type(end) is not int or end!=ends['test'] or not 1<=end<=len(rows) or len(timestamps)!=len(rows):
        raise ValueError('Invalid analysis extent')
    # No features, targets, statistics or predictions are constructed past this boundary.
    rows=rows[:end]; timestamps=timestamps[:end]
    c,h,target=protocol['context'],protocol['horizon_steps'],protocol['target_index']
    mean=np.asarray(scaler['mean'],dtype=np.float32).astype(np.float64)
    scale=np.asarray(scaler['scale'],dtype=np.float32).astype(np.float64)
    train_x,train_y,_=window_view(rows,ends,'train',c,h,target)
    val_x,val_y,_=window_view(rows,ends,'validation',c,h,target)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'started.json',dict(binding=binding,experiment_id=protocol['experiment_id']))
    try:
        records,models=fit_linear_families(train_x,train_y,val_x,val_y,mean,scale,
                                          protocol['feature_sets'],protocol['target_modes'],protocol['ridge_penalties'],target)
        choices={name:select_candidate([r for r in records if r['family']==name]) for name in models}
        # Seal all validation selections before constructing test windows or scoring them.
        write(output/'selection.json',dict(candidates=records,selected=choices))
        linear_arrays={}
        for name,model in models.items():
            linear_arrays[name+'_coefficients']=model.coefficients
            linear_arrays[name+'_intercept']=np.array(model.intercept)
        write_arrays(output/'linear_models.npz',linear_arrays)
        thresholds=change_thresholds(train_y,train_x[:,-1,target],protocol['movement_quantiles'])
        result=dict(experiment_id=protocol['experiment_id'],binding=binding,
                    inference_status='retrospective_exploratory; old test already inspected',
                    reserved_rows_used=False,maximum_observation_end_exclusive=end,
                    movement_thresholds=thresholds,candidates=records,selected=choices,splits={})
        arrays={}
        for split in ('train','validation','test'):
            x,y,indices=window_view(rows,ends,split,c,h,target)
            last=x[:,-1,target].astype(np.float64)
            seasonal_index=c+h-1-protocol['seasonal_period_steps']
            if not 0<=seasonal_index<c: raise ValueError('Seasonal observation is outside history')
            outputs={'persistence':last,'seasonal':x[:,seasonal_index,target].astype(np.float64)}
            for name,model in models.items():
                choice=choices[name]
                design=design_matrix(x,choice['features'],mean,scale)
                outputs['linear_'+name]=model.predict(design)+(last if choice['target_mode']=='change' else 0.)
                del design
            for name,predict in predictors.items(): outputs[name]=predict(x)
            labels=[timestamps[i][:7] for i in indices]
            result['splits'][split]={name:describe(prediction,y,last,thresholds,labels) for name,prediction in outputs.items()}
            arrays[split+'_target_indices']=indices; arrays[split+'_target']=y
            arrays.update({split+'__'+name:prediction for name,prediction in outputs.items()})
            print(json.dumps(dict(split=split,count=len(y),methods=len(outputs))),flush=True)
        if original_scores is not None:
            differences={}
            for name,original in original_scores.items():
                actual=result['splits']['test'][name]['overall']
                differences[name]={key:abs(actual[key]-original[key]) for key in ('mae','rmse')}
                if actual['count']!=original['count'] or any(v>protocol['original_score_replay_absolute_tolerance'] for v in differences[name].values()):
                    raise ValueError('Original score replay mismatch')
            result['original_score_replay_absolute_differences']=differences
        write_arrays(output/'predictions.npz',arrays)
        write(output/'results.json',result)
        files=('selection.json','linear_models.npz','predictions.npz','results.json')
        write(output/'complete.json',dict(status='complete',files={name:file_hash(output/name) for name in files}))
        return result
    except Exception as exc:
        write(output/'failure.json',dict(error_type=type(exc).__name__))
        raise


def predictor(model,batch_size=128):
    def predict(x):
        outputs=[]; model.eval()
        with torch.inference_mode():
            for start in range(0,len(x),batch_size):
                # NumPy window views are read-only; hand Torch an owned contiguous batch.
                batch=torch.from_numpy(np.array(x[start:start+batch_size],dtype=np.float32,copy=True,order='C'))
                outputs.append(model(batch)[:,-1].numpy().copy())
        return np.concatenate(outputs)
    return predict


def verified_predictors(prior,expected,recipe):
    predictors={}; hashes={}
    bound=expected['binding']
    for seed in recipe['seeds']:
        path=Path(prior)/f'seed{seed}'
        sha=file_hash(path/'complete.json')
        if sha!=expected['run_manifest_sha256'][str(seed)]: raise ValueError('Changed original run manifest')
        result=verify_run(path,recipe,bound,seed)
        model,_,_,_=load_checkpoint(path/f"epoch-{result['best_epoch']:03d}",{**bound,'seed':seed})
        blob=bounded(path/'model.stlf',16*1024*1024)
        if model.export_bytes()!=blob or digest(blob)!=expected['models'][str(seed)]['model_sha256']:
            raise ValueError('Changed selected original model')
        predictors[f'transformer_{seed}']=predictor(model); hashes[str(seed)]=sha
    return predictors,hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--prior-runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'local'): raise ValueError('Outputs belong under ignored local/')
    for name in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):
        if os.environ.get(name)!='1': raise ValueError(f'Set {name}=1 before starting the process')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    commit=clean_commit()
    protocol_path=ROOT/'benchmarks/ettm1/diagnostics_v1.json'
    protocol=strict_json(protocol_path.read_bytes())
    recipe,task,c=read_recipe()
    if (protocol['schema_version']!=1 or protocol['task_id']!=task['task_id'] or protocol['context']!=c.context
            or protocol['horizon_steps']!=c.horizon or protocol['target_index']!=c.target
            or protocol['split_ends']!=data.SOURCE['split_ends']
            or protocol['maximum_observation_end_exclusive']!=data.SOURCE['split_ends']['test']):
        raise ValueError('Diagnostic protocol changed task or source boundaries')
    previous_path=ROOT/'docs/evidence/training-v1/results.json'
    previous=strict_json(previous_path.read_bytes())
    data.check_snapshot(args.raw,args.prepared)
    if (file_hash(args.raw/'ETTm1.csv')!=previous['binding']['data_sha256']
            or file_hash(args.prepared/'scaler.json')!=previous['binding']['scaler_sha256']
            or file_hash(ROOT/'benchmarks/ettm1/one_hour.json')!=previous['binding']['task_sha256']):
        raise ValueError('Original data/task/scaler changed')
    models,run_hashes=verified_predictors(args.prior_runs,previous,recipe)
    frame=data.verified_frame(args.raw)
    stop=protocol['maximum_observation_end_exclusive']
    rows=np.asarray(frame.values[:stop],dtype=np.float32)
    scaler=strict_json(bounded(args.prepared/'scaler.json',65536))
    build=io.StringIO()
    with contextlib.redirect_stdout(build): np.__config__.show()
    binding=dict(code_commit=commit,protocol_sha256=file_hash(protocol_path),
                 original_evidence_sha256=file_hash(previous_path),original_binding=previous['binding'],
                 original_run_hashes=run_hashes,environment=environment(),
                 numpy_build_sha256=digest(build.getvalue().encode()))
    original_scores={f'transformer_{seed}':previous['models'][str(seed)]['test'] for seed in recipe['seeds']}
    original_scores.update(persistence=previous['baselines']['last_observation'],seasonal=previous['baselines']['seasonal'])
    run_analysis(args.output,rows,frame.timestamps[:stop],scaler,protocol,models,binding,original_scores)
    print('Retrospective diagnosis complete; reserved tail remains unused',flush=True)


if __name__=='__main__': main()
