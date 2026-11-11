#!/usr/bin/env python3
"""One fixed-shape Torch/ONNX timing cell; launched by benchmark_frontier.py."""
import argparse
import contextlib
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from runtime_adapters import TorchAdapter, OnnxAdapter
from runtime_common import bounded, digest, parity, peak_rss_bytes, read_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('torch_eager', 'torch_compiled_inductor', 'onnxruntime_cpu'), required=True)
    for name in ('model', 'input', 'expected', 'artifacts'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--warmup', type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.batch <= 1024 or not 1 <= args.samples <= 1000000 or not 1 <= args.warmup <= 100000:
        raise ValueError('Invalid bounded count')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    blobs = {name: bounded(getattr(args, name)) for name in ('model', 'input', 'expected')}
    windows, tokens, features, values = read_input(blobs['input'])
    if windows % args.batch or len(blobs['expected']) != windows * tokens * 4:
        raise ValueError('Batch or reference trace length mismatch')
    x = np.asarray(values, dtype=np.float32).reshape(windows, tokens, features)
    expected = np.frombuffer(blobs['expected'], dtype='<f4').reshape(windows, tokens)[:, -1:]
    if not np.isfinite(expected).all():
        raise ValueError('Nonfinite reference')
    baseline_rss = peak_rss_bytes()
    with torch.inference_mode():
        start = time.perf_counter_ns()
        with contextlib.redirect_stdout(sys.stderr):
            if args.engine == 'onnxruntime_cpu':
                adapter = OnnxAdapter(blobs['model'], batch=args.batch, tokens=tokens, artifacts=args.artifacts / 'onnx')
            else:
                adapter = TorchAdapter(blobs['model'], batch=args.batch, tokens=tokens,
                                       compiled=args.engine == 'torch_compiled_inductor')
        groups = [adapter.prepare(x[i:i + args.batch]) for i in range(0, windows, args.batch)]
        setup_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        first = adapter.run(groups[0])
        first_call_ns = time.perf_counter_ns() - start
        def array(result):
            return result.numpy() if isinstance(result, torch.Tensor) else result
        maximum = 0.
        for i, group in enumerate(groups):
            result = first if i == 0 else adapter.run(group)
            maximum = max(maximum, parity(expected[i * args.batch:(i + 1) * args.batch].reshape(-1),
                                           array(result).reshape(-1), 5e-5, 2e-4))
        if args.engine == 'torch_compiled_inductor':
            from torch._inductor import metrics
            generated = int(metrics.generated_kernel_count)
            if generated <= 0:
                raise RuntimeError('No generated Inductor kernels')
            adapter.metadata['generated_kernel_count'] = generated
        for i in range(args.warmup):
            adapter.run(groups[i % len(groups)])
        gc.collect()
        times = [0] * args.samples
        clock_times = [0] * 1000
        checksum = 0.
        gc.disable()
        try:
            for i in range(len(clock_times)):
                start = time.perf_counter_ns()
                clock_times[i] = time.perf_counter_ns() - start
            for i in range(args.samples):
                group = groups[i % len(groups)]
                start = time.perf_counter_ns()
                result = adapter.run(group)
                times[i] = time.perf_counter_ns() - start
                values = array(result).reshape(-1)
                checksum += float(values[i % values.size])
                del result, values
        finally:
            gc.enable()
    if not np.isfinite(checksum):
        raise ValueError('Nonfinite checksum')
    print(json.dumps(dict(engine=args.engine, batch=args.batch, output_mode='last',
                          windows_checked=windows, inputs_sha256={k: digest(v) for k, v in blobs.items()},
                          setup_ns=setup_ns, first_call_ns=first_call_ns, max_abs_difference=maximum,
                          absolute_tolerance=5e-5, relative_tolerance=2e-4,
                          times_ns=times, clock_times_ns=clock_times, checksum=checksum,
                          adapter=adapter.metadata,
                          peak_rss_bytes=dict(baseline=baseline_rss, final=peak_rss_bytes())), allow_nan=False))


if __name__ == '__main__':
    main()
