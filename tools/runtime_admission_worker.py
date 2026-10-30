#!/usr/bin/env python3
"""Internal fresh-process admission worker; use admit_runtime.py as supervisor."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from runtime_adapters import TorchAdapter,OnnxAdapter
from runtime_common import bounded,digest,parity,read_input,peak_rss_bytes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',choices=('torch_eager','torch_compiled_inductor','onnxruntime_cpu'),required=True)
    p.add_argument('--cell',type=Path,required=True);p.add_argument('--batch',type=int,required=True)
    a=p.parse_args()
    if not 1<=a.batch<=1024:raise ValueError('Invalid batch')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    blobs={name:bounded(a.cell/name) for name in ('model.stlf','input.bin','expected.bin')}
    windows,tokens,features,values=read_input(blobs['input.bin'])
    if windows%a.batch or len(blobs['expected.bin'])!=windows*tokens*4:
        raise ValueError('Batch or reference trace length mismatch')
    x=np.asarray(values,dtype=np.float32).reshape(windows,tokens,features)
    expected=np.frombuffer(blobs['expected.bin'],dtype='<f4').reshape(windows,tokens)[:,-1:]
    if not np.isfinite(expected).all():raise ValueError('Nonfinite reference outputs')
    baseline_rss=peak_rss_bytes();start=time.perf_counter_ns()
    with torch.inference_mode():
        # Exporter progress is diagnostic text, never the machine-readable result.
        with contextlib.redirect_stdout(sys.stderr):
            if a.engine=='onnxruntime_cpu':
                adapter=OnnxAdapter(blobs['model.stlf'],batch=a.batch,tokens=tokens,artifacts=a.cell/'onnx')
            else:
                adapter=TorchAdapter(blobs['model.stlf'],batch=a.batch,tokens=tokens,
                                     compiled=a.engine=='torch_compiled_inductor')
        groups=[adapter.prepare(x[i:i+a.batch]) for i in range(0,windows,a.batch)]
        setup_ns=time.perf_counter_ns()-start
        start=time.perf_counter_ns();first=adapter.run(groups[0]);first_call_ns=time.perf_counter_ns()-start
        maximum=0.
        for i,group in enumerate(groups):
            result=first if i==0 else adapter.run(group)
            if isinstance(result,torch.Tensor):result=result.numpy()
            maximum=max(maximum,parity(expected[i*a.batch:(i+1)*a.batch].reshape(-1),result.reshape(-1),5e-5,2e-4))
    if a.engine=='torch_compiled_inductor':
        from torch._inductor import metrics
        generated=int(metrics.generated_kernel_count)
        if generated<=0:
            raise RuntimeError('No generated Inductor kernels; compilation admission cannot be established')
        adapter.metadata['generated_kernel_count']=generated
    result=dict(status='admitted',engine=a.engine,batch=a.batch,windows_checked=windows,
                 max_abs_difference=maximum,absolute_tolerance=5e-5,relative_tolerance=2e-4,
                 inputs_sha256={name:digest(blob) for name,blob in blobs.items()},adapter=adapter.metadata,
                 setup_ns=setup_ns,first_call_ns=first_call_ns,timing_is_benchmark=False,
                 peak_rss_bytes=dict(baseline=baseline_rss,final=peak_rss_bytes()),
                 environment=dict(numpy=np.__version__,torch=str(torch.__version__),
                                  torch_build_sha256=digest(torch.__config__.show().encode()),
                                  intra_op=torch.get_num_threads(),inter_op=torch.get_num_interop_threads(),
                                  deterministic=torch.are_deterministic_algorithms_enabled()))
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
