#!/usr/bin/env python3
"""Frozen correction-head comparison; prior test exploratory, reserved tail unused."""
import argparse
from dataclasses import replace
from pathlib import Path
import copy
import json
import numpy as np
import torch
import prepare_public_benchmark as data
from training_state import strict_json,bounded
from train_ettm1 import (read_recipe as original_recipe,run_seed,evaluate_runs,
                         clean_commit,file_hash)
ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'benchmarks/ettm1/one_hour_correction_v1.json'
PROTOCOL=ROOT/'benchmarks/ettm1/training_correction_v1.json'


def validate_recipe(protocol,task):
    original,base,c=original_recipe();expected=copy.deepcopy(base)
    expected['task_id']='ettm1_one_hour_correction_v1'
    expected['model']['output']='latest_target_plus_scaled_linear_correction'
    expected['model']['head_initialization']='zeros'
    expected['user_choices']['output']='Start with the latest temperature and learn a correction'
    descriptions={'experiment_id','task_id','purpose','initialization','test_policy','limitations'}
    extras={'maximum_observation_end_exclusive','output_mode','comparison','initial_candidate'}
    if (task!=expected or set(protocol)!=set(original)|extras
            or any(protocol[k]!=v for k,v in original.items() if k not in descriptions)
            or protocol['experiment_id']!='ettm1_one_hour_correction_v1'
            or protocol['task_id']!=expected['task_id'] or protocol['output_mode']!='correction'
            or protocol['maximum_observation_end_exclusive']!=data.SOURCE['split_ends']['test']):
        raise ValueError('Unsupported correction recipe; preserve the frozen comparison')
    return replace(c,output='correction')


def read_recipe():
    protocol=strict_json(PROTOCOL.read_bytes());task=strict_json(TASK.read_bytes())
    return protocol,task,validate_recipe(protocol,task)


def experiment_rows(values):
    stop=data.SOURCE['split_ends']['test']
    if len(values)<stop: raise ValueError('Insufficient frozen data extent')
    selected=np.asarray(values[:stop],dtype=np.float32)
    if selected.shape!=(stop,7) or not np.isfinite(selected).all(): raise ValueError('Invalid selected observations')
    return torch.from_numpy(selected.copy())


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
    if not args.output.resolve().is_relative_to(ROOT/'local'): raise ValueError('Outputs belong under ignored local/')
    protocol,task,c=read_recipe()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    commit=clean_commit();data.check_snapshot(args.raw,args.prepared)
    original=strict_json((ROOT/'docs/evidence/training-v1/results.json').read_bytes())
    if (file_hash(args.raw/'ETTm1.csv')!=original['binding']['data_sha256']
            or file_hash(args.prepared/'scaler.json')!=original['binding']['scaler_sha256']):
        raise ValueError('Matched comparison requires the original data and scaler')
    frame=data.verified_frame(args.raw);rows=experiment_rows(frame.values)
    scaler=strict_json(bounded(args.prepared/'scaler.json',65536))
    binding=dict(protocol_sha256=file_hash(PROTOCOL),task_sha256=file_hash(TASK),
                 data_sha256=file_hash(args.raw/'ETTm1.csv'),scaler_sha256=file_hash(args.prepared/'scaler.json'),
                 code_commit=commit)
    if args.command=='train':
        if args.seed is None or args.runs is not None: parser.error('Training requires --seed and no --runs')
        result=run_seed(args.output,rows,data.SOURCE['split_ends'],c,scaler['mean'],scaler['scale'],
                        protocol,task,binding,args.seed,args.resume)
        print(json.dumps(dict(status='complete',seed=args.seed,best_epoch=result['best_epoch'])))
    else:
        if args.seed is not None or args.resume is not None or not args.runs: parser.error('Evaluation requires --runs only')
        print(json.dumps(evaluate_runs(args.output,args.runs,rows,data.SOURCE['split_ends'],c,
                                       protocol,task,binding),indent=2))

if __name__=='__main__':main()
