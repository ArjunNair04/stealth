#!/usr/bin/env python3
"""One fresh process, one engine, one saved model and one batch size."""
import argparse
import gc
import json
from pathlib import Path
import struct
import time
import numpy as np
from runtime_common import bounded,canonical,digest,parity,peak_rss_bytes,read_input


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',choices=['native_python','torch_reference','torch_standard'],required=True)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--expected',type=Path,required=True);parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--samples',type=int,default=1000);parser.add_argument('--warmup',type=int,default=64)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--output',choices=['all','last'],default='all')
    args=parser.parse_args()
    if not 1<=args.samples<=1000000 or not 1<=args.warmup<=100000 or not 1<=args.batch<=1024:
        raise ValueError('Invalid bounded positive count')
    if args.prepare and args.output!='all': raise ValueError('Prepare the full reference trace')
    if args.engine=='native_python':
        import stealth_cpu as native
        from build_identity import extension_identity
        versions=dict(numpy=np.__version__,**extension_identity())
    else:
        import torch
        from runtime_torch import load_export
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        # inference_mode covers setup, parity, warmup and measured calls consistently.
        inference_guard=torch.inference_mode();inference_guard.__enter__()
        versions=dict(numpy=np.__version__,torch=torch.__version__,torch_build_sha256=digest(torch.__config__.show().encode()),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),deterministic=torch.are_deterministic_algorithms_enabled(),inference_mode=torch.is_inference_mode_enabled())
    baseline_rss=peak_rss_bytes();start=time.perf_counter_ns()
    windows,tokens,features,values=read_input(bounded(args.input))
    if windows%args.batch: raise ValueError('Batch must divide fixed window count')
    x=np.array(values,np.float32).reshape(windows,tokens,features);del values
    if args.engine=='native_python':
        model=native.ForecastModel.from_bytes(bounded(args.model));session=model.session()
        groups=[x[i:i+args.batch] for i in range(0,windows,args.batch)]
        if args.batch==1:
            groups=[a[0] for a in groups]
            def call(i): return session.predict_last(groups[i]) if args.output=='last' else session.predict(groups[i])
        else:
            def call(i): return session.predict_last_batch(groups[i]) if args.output=='last' else session.predict_batch(groups[i])
        owned=dict(parameter_bytes=model.parameter_bytes,workspace_bytes=session.workspace_bytes)
    else:
        model=load_export(bounded(args.model),standard=args.engine=='torch_standard')
        groups=[torch.from_numpy(x[i:i+args.batch]) for i in range(0,windows,args.batch)]
        def call(i):
            if args.engine=='torch_standard': return model(groups[i],latest_only=args.output=='last')
            y=model(groups[i]);return y[:,-1:] if args.output=='last' else y
        owned=dict(parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),buffer_bytes=sum(p.numel()*p.element_size() for p in model.buffers()))
    setup_ns=time.perf_counter_ns()-start;setup_rss=peak_rss_bytes()
    def array(y):
        if isinstance(y,np.ndarray): return y
        if isinstance(y,float): return np.asarray(y,dtype=np.float32)
        return y.numpy()
    start=time.perf_counter_ns();first=call(0);first_call_ns=time.perf_counter_ns()-start
    del first
    if args.engine=='native_python':
        owned['workspace_bytes_before_first_call']=owned['workspace_bytes']
        owned['workspace_bytes']=session.workspace_bytes
    if args.prepare:
        if args.engine!='torch_reference': raise ValueError('Expected outputs must use reference')
        expected=np.concatenate([array(call(i)).reshape(-1) for i in range(len(groups))]).astype('<f4')
        if not np.isfinite(expected).all(): raise ValueError('Nonfinite expected outputs')
        with args.expected.open('xb') as f: f.write(expected.tobytes())
        print(json.dumps(dict(versions=versions,model_sha256=digest(bounded(args.model)),expected_sha256=digest(expected.tobytes()))))
        return
    expected_blob=bounded(args.expected)
    if len(expected_blob)!=windows*tokens*4: raise ValueError('Expected output length mismatch')
    expected=np.frombuffer(expected_blob,dtype='<f4').reshape(windows,tokens)
    if args.output=='last': expected=expected[:,-1:]
    maximum=0.
    for i in range(len(groups)):
        maximum=max(maximum,parity(expected[i*args.batch:(i+1)*args.batch].reshape(-1),array(call(i)).reshape(-1),5e-5,2e-4))
    for i in range(args.warmup): call(i%len(groups))
    gc.collect();gc.disable()
    times=[0]*args.samples;clock_times=[0]*1000;checksum=0.
    for i in range(len(clock_times)):
        start=time.perf_counter_ns();clock_times[i]=time.perf_counter_ns()-start
    for i in range(args.samples):
        start=time.perf_counter_ns();y=call(i%len(groups));times[i]=time.perf_counter_ns()-start
        a=array(y).reshape(-1);checksum+=float(a[i%a.size]);del a,y
    gc.enable()
    if not np.isfinite(checksum): raise ValueError('Nonfinite checksum')
    print(json.dumps(dict(engine=args.engine,output_mode=args.output,batch=args.batch,setup_ns=setup_ns,first_call_ns=first_call_ns,
        max_abs_difference=maximum,checksum=checksum,**owned,versions=versions,
        peak_rss_bytes=dict(baseline=baseline_rss,setup=setup_rss,final=peak_rss_bytes()),
        times_ns=times,clock_times_ns=clock_times),allow_nan=False))

if __name__=='__main__':main()
