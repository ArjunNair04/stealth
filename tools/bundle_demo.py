#!/usr/bin/env python3
"""Small CPU ETTm1 train/export/replay demonstration, with immutable local outputs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
import prepare_public_benchmark as data
from build_identity import extension_identity
from feature_inference import FeatureReplay
from model_bundle import create_bundle, load_bundle
from signal_timeline import SignalSpec, Observation, SignalTimeline
from torch_forecaster import Config
from training_state import canonical, digest, bounded, strict_json, environment, load_checkpoint, metrics
from train_ettm1 import (ROOT, clean_commit, file_hash, write, run_seed, split_tensors,
                        baseline_predictions, predictions, expected_files)

PROTOCOL = ROOT/'benchmarks/ettm1/bundle_demo_v1.json'
TASK = ROOT/'benchmarks/ettm1/one_hour_correction_v1.json'


def read_recipe():
    protocol, task = (strict_json(bounded(p,65536)) for p in (PROTOCOL,TASK))
    if (protocol['experiment_id'] != 'ettm1_bundle_demo_v1' or protocol['seeds'] != [7] or
            protocol['epochs'] != 2 or protocol['split_ends'] != dict(train=2048,validation=2560,test=3072) or
            protocol['maximum_observation_end_exclusive'] != 3072 or protocol['cpu_threads'] != 1 or
            protocol['deterministic_algorithms'] is not True or protocol['clock_domain'] != 'synthetic'):
        raise ValueError('Demo recipe changed; introduce a new version')
    m = task['model']
    if (task['input_columns'] != list(data.COLUMNS[1:]) or task['context'] != 96 or
            task['horizon_steps'] != 4 or task['target_column'] != 'OT' or task['interval_seconds'] != 900 or
            (m['blocks'],m['width'],m['heads'],m['hidden'],m['dropout']) != (1,64,4,256,0) or
            m['output'] != 'latest_target_plus_scaled_linear_correction' or
            m['positions'] != 'fixed_sinusoidal_window_relative' or m['activation'] != 'exact_gelu' or
            m['final_normalization'] is not True or m['head_initialization'] != 'zeros'):
        raise ValueError('Unsupported demo task')
    c = Config(7,64,4,256,96,4,6,m['epsilon'],output='correction')
    return protocol, task, c


def demo_specs(columns, interval_ns):
    # Invented instantaneous delivery, with cutoff one tick later. A missing
    # observation becomes stale on the next grid step; no hidden filling.
    if type(interval_ns) is not int or interval_ns < 2:
        raise ValueError('Demo interval must exceed its one-tick receipt offset')
    return tuple(SignalSpec(name,'ettm1_synthetic_clock','ETTm1',name,1,1) for name in columns)


def replay_decisions(bundle, values, first_input, end_input):
    """Replay [first_input,end_input); values after end_input are never observed."""
    if (bundle.clock_domain != 'synthetic' or bundle.grid_origin_ns != 1 or
            type(first_input) is not int or type(end_input) is not int or
            not 0 <= first_input < end_input <= len(values)):
        raise ValueError('Invalid synthetic replay range or clock')
    records = (Observation(s.source,s.entity,s.field,str(i),0,
                           i*bundle.interval_ns,i*bundle.interval_ns,i*bundle.interval_ns,float(values[i][j]))
               for i in range(first_input,end_input) for j,s in enumerate(bundle.specs))
    timeline = SignalTimeline(bundle.specs,records)
    replay = FeatureReplay(bundle)
    return [replay.advance(timeline,i*bundle.interval_ns+1) for i in range(first_input,end_input)]


def ledger_bytes(decisions):
    return b''.join((json.dumps(asdict(r),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
                    for r in decisions)


def score_replay(bundle, frame, rows, ends, c, model, protocol):
    descriptors = list(data.windows(len(rows),ends,'test',c.context,c.horizon,'point'))
    decisions = replay_decisions(bundle,rows.numpy(),descriptors[0].input_start,descriptors[-1].input_end)
    forecasts = [r for r in decisions if r.state == 'forecast']
    if len(forecasts) != len(descriptors) or any(r.state == 'unavailable' for r in decisions):
        raise ValueError('Incomplete demo replay')
    for r,w in zip(forecasts,descriptors):
        if (r.decision_at_ns != (w.input_end-1)*bundle.interval_ns+1 or
                r.target_at_ns != w.target_start*bundle.interval_ns+1):
            raise ValueError('Replay target alignment mismatch')
    x,y = split_tensors(rows,ends,'test',c)
    actual = np.array([r.value for r in forecasts],dtype=np.float32)
    direct = bundle.model.session().predict_last_batch(np.ascontiguousarray(x.numpy()))
    if not np.array_equal(actual,direct):
        raise ValueError('Feature bridge differs from direct native inference')
    expected = predictions(model,x,protocol['evaluation_batch_size'])
    difference = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    if not np.all(difference <= protocol['parity_absolute_tolerance'] + protocol['parity_relative_tolerance']*np.abs(expected)):
        raise ValueError('Demo Torch/native parity failed')
    last, seasonal = baseline_predictions(x,c,period=c.context)
    scores = {name:metrics(p,y.numpy()) for name,p in
              (('native',actual),('torch',expected),('last_observation',last),('previous_day',seasonal))}
    examples = [dict(origin_row=w.input_end-1,target_row=w.target_start,
                     source_origin_time=frame.timestamps[w.input_end-1],
                     source_target_time=frame.timestamps[w.target_start],
                     prediction=float(p),actual=float(target),last_observation=float(base))
                for w,p,target,base in list(zip(descriptors,actual,y.numpy(),last))[:5]]
    details = dict(forecasts=len(forecasts),warming_up=len(decisions)-len(forecasts),
                   max_torch_native_absolute_error=float(difference.max()),
                   direct_native_exact=True,clock_domain='synthetic')
    return scores,details,examples,ledger_bytes(decisions)


def artifact_names(epochs):
    return (['started.json','protocol.json','task.json','scaler.json','scores.json','decisions.jsonl',
             'bundle/manifest.json','bundle/model.stlf','training/started.json','training/complete.json'] +
            ['training/'+p for p in expected_files(epochs)])


def run_demo(output,frame,protocol,task,c,binding):
    """Internal configurable entry point for tiny offline tests; CLI uses the frozen recipe."""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    try:
        ends = protocol['split_ends']
        data.split_ranges(len(frame.values),ends)
        # Never use the old full-training scaler: this demo has a smaller training prefix.
        scaler = data.fit_scaler(frame.values,ends['train'])
        scaler_record = dict(training_rows=ends['train'],columns=task['input_columns'],
                             mean=scaler.mean,scale=scaler.scale,variance_ddof=0)
        bound = {**binding,'scaler_sha256':digest(canonical(scaler_record))}
        write(output/'started.json',dict(binding=bound,environment=environment(),native=extension_identity()))
        write(output/'protocol.json',protocol); write(output/'task.json',task)
        write(output/'scaler.json',scaler_record)
        rows = torch.tensor(frame.values[:ends['test']],dtype=torch.float32)
        selected = run_seed(output/'training',rows,ends,c,scaler.mean,scaler.scale,
                            protocol,task,bound,protocol['seeds'][0])
        training_seconds = time.perf_counter()-started
        model,_,_,_ = load_checkpoint(output/'training'/f"epoch-{selected['best_epoch']:03d}",
                                      {**bound,'seed':protocol['seeds'][0]})
        interval = task['interval_seconds']*1_000_000_000
        ident = create_bundle(output/'bundle',model.export_bytes(),demo_specs(task['input_columns'],interval),
                              interval_ns=interval,grid_origin_ns=1,clock_domain='synthetic')
        bundle = load_bundle(output/'bundle',expected_id=ident)
        scores,details,examples,ledger = score_replay(bundle,frame,rows,ends,c,model,protocol)
        with (output/'decisions.jsonl').open('xb') as target: target.write(ledger)
        result = dict(schema_version=1,purpose='engineering_demonstration',bundle_id=ident,
                       selected_epoch=selected['best_epoch'],scores=scores,replay=details,examples=examples,
                       training_and_setup_seconds=training_seconds,total_seconds=time.perf_counter()-started,
                       timing_is_benchmark=False,binding=bound)
        write(output/'scores.json',result)
        files = {name:file_hash(output/name) for name in artifact_names(protocol['epochs'])}
        write(output/'complete.json',dict(schema_version=1,status='complete',bundle_id=ident,binding=bound,files=files))
        return result
    except Exception as exc:
        write(output/'failure.json',dict(error_type=type(exc).__name__))
        raise


def verify_demo(path):
    path = Path(path)
    complete = strict_json(bounded(path/'complete.json',1024*1024))
    protocol = strict_json(bounded(path/'protocol.json',65536))
    if (complete.get('schema_version') != 1 or complete.get('status') != 'complete' or
            protocol.get('epochs') != 2 or set(complete.get('files',{})) != set(artifact_names(2))):
        raise ValueError('Incomplete demo seal')
    for name,sha in complete['files'].items():
        if file_hash(path/name) != sha:
            raise ValueError('Demo artifact digest mismatch')
    load_bundle(path/'bundle',expected_id=complete['bundle_id'])
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('train','replay'))
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run',type=Path)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'local'):
        raise ValueError('Keep generated outputs under ignored local/')
    protocol,task,c = read_recipe()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    commit = clean_commit()
    frame = data.verified_frame(args.raw)
    if args.command == 'train':
        if args.run is not None: parser.error('Training does not accept --run')
        binding = dict(protocol_sha256=file_hash(PROTOCOL),task_sha256=file_hash(TASK),
                        data_sha256=file_hash(args.raw/'ETTm1.csv'),code_commit=commit)
        print(json.dumps(run_demo(args.output,frame,protocol,task,c,binding),indent=2))
    else:
        if args.run is None: parser.error('Replay requires --run')
        sealed = verify_demo(args.run)
        if (sealed['binding']['data_sha256'] != file_hash(args.raw/'ETTm1.csv') or
                sealed['binding']['protocol_sha256'] != file_hash(PROTOCOL) or
                sealed['binding']['task_sha256'] != file_hash(TASK)):
            raise ValueError('Replay source/task/protocol mismatch')
        bundle = load_bundle(args.run/'bundle',expected_id=sealed['bundle_id'])
        rows = np.asarray(frame.values[:protocol['split_ends']['test']],dtype=np.float32)
        windows = list(data.windows(len(rows),protocol['split_ends'],'test',c.context,c.horizon,'point'))
        raw = ledger_bytes(replay_decisions(bundle,rows,windows[0].input_start,windows[-1].input_end))
        if raw != bounded(args.run/'decisions.jsonl',64*1024*1024):
            raise ValueError('Fresh-process replay differs from saved decisions')
        args.output.mkdir(parents=True,exist_ok=False)
        with (args.output/'decisions.jsonl').open('xb') as target: target.write(raw)
        result = dict(status='complete',byte_identical=True,bundle_id=bundle.bundle_id,
                       ledger_sha256=digest(raw),code_commit=commit,native=extension_identity())
        write(args.output/'complete.json',result)
        print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
