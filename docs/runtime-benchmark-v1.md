# Saved-model CPU runtime comparison

A forecast is the answer; a runtime is the machinery that calculates it. This
experiment keeps the saved answers' recipe fixed and changes only the machinery.
It measures all three validation-selected correction models, without retraining,
changing their structure, selecting a seed from test accuracy, or reading the
reserved source tail as numeric features.

## Frozen scope

The [protocol](../benchmarks/ettm1/runtime_v1.json) fixes 32 evenly spaced
validation input windows, 96 rows by seven features, seeds 7/19/43, three trials,
64 warm-up calls, 1,000 measured batch-one calls and 200 batch-eight calls.
Every call returns all 96 per-origin predictions. All inputs are contiguous FP32,
prepared before measurement. Each engine/model/batch/trial runs in a fresh process,
serially; engine order rotates by trial, seed and workload. No timing outliers are
removed. This is a finite local experiment, not a latency deadline guarantee.

Four engines participate:

- `native_core`: Release C++ packed kernels, caller-owned input/output, reusable workspace.
- `native_python`: the installed NumPy wrapper, including input snapshot, Python
  boundary, GIL release/reacquisition and output allocation. Batch eight evaluates
  eight windows sequentially at the original study revision. Current builds use
  [native batching](native-batching-v1.md). The C++ `stealth_runtime_bench` executable
  retains its serial diagnostic path via `--batch-execution serial`.
- `torch_reference`: the training reference with explicit double reductions.
- `torch_standard`: the same weights and architecture using ordinary FP32
  `layer_norm`, exact `gelu` and causal `scaled_dot_product_attention` kernels.
  Scaling and output reconstruction retain the original double arithmetic.

The last implementation uses PyTorch's documented
[causal attention API](https://docs.pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html),
with dropout explicitly zero. It is an eager CPU baseline, not compiled Torch,
ONNX Runtime or the strongest available deployment. No GPU backend is used.
Torch runs under `inference_mode`, with intra-op and inter-op threads both one,
deterministic algorithms enabled and common BLAS thread environment variables
set to one. Python cyclic garbage collection is disabled during timings only;
reference-count destruction still occurs. Torch tensors and NumPy views are
prepared before timing. Outputs are consumed outside the timed interval; result
allocation remains inside each framework call. C++ outputs are preallocated.

## Correctness and statistics

Every origin of all 32 windows must match the strict reference before that cell's
warm timings are accepted. The existing trained-export tolerance is unchanged:
`abs(actual-reference) <= 5e-5 + 2e-4*abs(reference)`. This allows ordinary FP32
rounding, not a forecast objective change. Native kernels check intermediate
arithmetic as well; the Torch paths check inputs and final outputs. Equal results
on these finite windows do not imply identical failure handling for every input. A parity failure aborts the experiment;
no slower or failing engine is silently removed. Input, model and expected-output
artifacts are bounded and hashed. Saved model hashes must match the published
correction results. Model decoding is pickle-free and checked against an exact
re-export. Synthetic independent equations and failure tests run without ETTm1.

Monotonic clocks measure one synchronous call at a time. Report nearest-rank p50,
p95, p99, mean, min and max for every trial, plus timer overhead without subtracting
it. A checksum consumes outputs. Trial ranges describe observed variability;
overlapping calls/windows do not justify an IID confidence claim. Batch-one latency
is primary. Batch-eight throughput is separate: the native implementation is
sequential while Torch can use batched kernels. It is a public API workload
comparison, not a claim that their scheduling or allocation is identical.

A separate first call follows setup and precedes parity/warm-up. It includes lazy
initialization but excludes interpreter startup/imports, model loading and input
preparation. Setup timing is diagnostic and has engine-specific boundaries. These
numbers are not a cold-disk/cache or full process startup measurement.

## Memory and provenance

Fresh processes report lifetime peak RSS after imports (C++ after process entry),
after setup and after timing. macOS reports bytes; Linux's KiB are converted to
bytes. Unsupported systems report null. A high-water mark is not current memory,
and subtracting two peaks does not isolate allocations. Torch/NumPy/interpreter
footprint is part of the process total. Native parameter, position and per-session
workspace buffers are reported separately from input/output, allocator bookkeeping
and executable code. Torch parameter counts exclude registered buffers, which
are listed separately. Compare scope before comparing numbers.

The recorder requires clean Git, rebuilds Release, validates source/model hashes,
and refuses existing output directories. It records the commit, protocol, source,
input, model, native binary and loaded extension hashes, compiler flags, CPU,
Python/NumPy/Torch versions and threading. Raw timings, data-derived inputs and
expected outputs stay in ignored `local/`. Public evidence contains aggregate
records and hashes only. No test outcomes or reserved-tail forecasts are computed.

## Run a new experiment

Use the already provisioned training environment, with the native wheel rebuilt
from the intended source. No dependencies are installed by the recorder.

```sh
cmake --preset release -DSTEALTH_BUILD_BENCHMARKS=ON
cmake --build --preset release
python -m unittest discover -s tests -p test_runtime_benchmark.py -v
python -m unittest discover -s tests/training -p test_runtime_torch.py -v
python tools/benchmark_runtime.py --raw local/datasets/ettm1-v1 \
  --runs local/ettm1-correction-v1 --output local/runtime-v1-new
```

Use a new output name. An exception leaves the partial run for inspection; it does
not overwrite prior evidence. Hardware-specific performance has no CI pass/fail
threshold. CI checks arithmetic and runner contracts on the platform matrix.

Current-host Torch 2.2.2 is a compatibility comparison. A current supported
Linux/ARM environment, compiled Torch and an independent deployment runtime remain
necessary before describing a performance frontier. CPU pinning, frequency
control, cold caches, contention, sustained streams, network input and order
execution remain outside this experiment. QTS integration stays a separate task.
