# Compiled Torch and ONNX CPU admission

We can now give the same saved model to three comparison engines: ordinary
PyTorch, PyTorch's Inductor compiler, and ONNX Runtime's CPU engine. Before a speed
comparison, each engine must produce the expected latest forecast for every
supplied input window. This stage establishes that comparison machinery.

## What is implemented

`runtime_torch.StandardForecaster.tensor_forward` contains the existing eager
baseline's tensor equations. Its public `forward` keeps the original checks.
`runtime_adapters.LatestGraph` selects the latest-query path from those shared
equations, avoiding separate compiler/export implementations of the model.
Weights, scaling, sinusoidal positions, layer normalization, exact GELU and the
correction head keep their existing meanings. Native C++ is unchanged.

`TorchAdapter` has eager and Inductor modes. Both accept fixed batch/context shapes
and require inference mode. Compilation requests `backend='inductor'`,
`fullgraph=True`, `dynamic=False`, `mode='default'`. Shape and finite-value checks
surround the pure graph and remain inside each public `run` call. Suppressed
compiler errors and disabled compilation are rejected. The supervised compiled
worker additionally requires a positive generated-kernel count; unknown compiler
instrumentation must fail admission instead of being called a successful compile.

`OnnxAdapter` uses the modern Dynamo exporter, ONNX opset 18, fixed shapes, static
weights and a single model file. It validates the exported graph, then creates an
ONNX Runtime session with only CPUExecutionProvider, sequential execution, full
graph optimization, one intra-op/inter-op thread and spinning disabled. Provider
fallback is disabled. Both exported and optimized graph files are hashed. The
optimized graph may contain host-specific transformations: retain it as evidence
for that host, not as a portable preoptimized deployment artifact.

Inputs are aligned contiguous FP32 arrays of the declared shape. Torch preparation
creates resident tensor views before a runtime call. ONNX preparation retains the
resident array. The caller owns these buffers and must not mutate them concurrently.
Every call checks input and output shape/type/finiteness. These are benchmark
adapters, not a general serving API or replacement for the feature bundle.

## Bounded admission

`admit_runtime.py` creates a fresh output directory, captures bounded immutable
copies of the model, input bundle and full reference trace, and launches one
worker in a new POSIX process group. The complete worker gets at most 180 seconds,
including imports, export/compile and parity. This is stricter than the future
measurement protocol's compile-only allowance. A timeout kills the group,
including compiler descendants. Logs and failure records remain; no success seal
is written. A successful cell checks every window, with the existing tolerance
`5e-5 + 2e-4 * abs(reference)`, and seals its artifacts by hash.

The supervisor fixes thread settings, removes inherited compiler-disable/error-
suppression switches, and uses a fresh compilation-cache directory with caches
explicitly disabled. The worker records versions, source/input hashes, graph
hashes, memory high-water marks, setup and first-call costs. These single-call
costs are diagnostics, not latency benchmark results. No p50/p99 claim is emitted.
The supervisor currently targets Linux/macOS POSIX process groups; native Stealth
retains its existing cross-platform support.

On an explicitly prepared Linux environment, using previously prepared matching
reference/input artifacts:

```sh
local/frontier-env/bin/python tools/admit_runtime.py \
  --engine torch_compiled_inductor --batch 1 \
  --model local/ettm1-correction-v1/seed7/model.stlf \
  --input local/runtime-latest-v1/input.bin \
  --expected local/runtime-latest-v1/expected-7.bin \
  --output local/frontier-admission/seed7-batch1-compiled
```

Run separate cells for `torch_eager` and `onnxruntime_cpu`, both batch sizes 1/8,
and all three saved seeds. Output paths must be new. Files remain local, including
ONNX graphs and any compiler cache. For the frozen study, independently verify
all model/input/data hashes against `benchmarks/ettm1/frontier_v1.json`; successful
admission of arbitrary input files is not admission of that whole study. The
historical runtime workers and historical timing artifacts remain unchanged.

## Dependencies and tests

`requirements-frontier.txt` declares optional top-level pins for Linux/Python 3.12:
Torch 2.15.0+cpu from its official CPU index, ONNX 1.23.2, ONNX Runtime 1.31.1 and
ONNX Script 0.7.2, plus the existing NumPy/build/reference dependencies. Install
only in the explicit optional environment. CI prints all resolved package
versions and runs `pip check`. This is not the quiet measurement host's complete
wheel-hash lock, which remains required before acceptance timings.

The separate Linux CI job executes real Inductor and ONNX conversions against the
independent arithmetic fixture in level/correction modes, plus the deployed
96×7 input shape at batch sizes 1 and 8. Nonzero heads and changed inputs prevent
an identity/constant graph from passing accidentally. Supervised engine tests
also exercise the actual subprocess/export path. Other tests cover shape and
finite checks, fixed-graph capture, compiler failure/disablement, a mismatch only
in the last window, incomplete reference files, immutable destinations and
process-group timeouts. Tests use fixtures and never download market data.

The legacy local environment can check shared equations and eager admission
without installing anything. Its real compiler/export tests are explicitly
skipped; only the optional Linux job supplies that evidence.

## Recorded evidence

The [portable verification record](evidence/runtime-adapters-v1/verification.json)
binds the initial passing real-engine Linux CI job and six local saved-model
admission cells. Each of the three saved trained models passed all 32 frozen
validation windows at batches 1 and 8 under the local eager Torch 2.2.2 adapter.
Maximum absolute difference from the saved independent reference was
`0.0000019073486328125`. The copied input/model/reference hashes matched the frozen
frontier inputs and previously committed reference preparation hashes.

The local full reference/training suite has 80 tests: 78 pass and two optional
real-engine tests are reserved for Linux CI. All 66 offline tests and 24 wrapper
tests pass. The compiler-disablement regression failed before its guard was added;
malformed final-window references and actual process-group timeouts are retained
as deliberate failure tests. The final Linux supervised-path checks are tracked
on [PR #21](https://github.com/ArjunNair04/stealth/pull/21).

## Remaining performance gate

The adapter milestone does not complete the 90-cell frontier comparison. The
quiet Linux/Beaker CPU allocation, host and wheel lock, fixed saved-model admission,
full multi-engine timed orchestration, aggregate gate calculation and one frozen
acceptance run remain open. The old local speed result still compares with Torch
2.2.2 only. No kernel tuning, retraining or predictive-quality claim follows from
compiler/export correctness alone.

Primary API references checked for this implementation: [Torch ONNX exporter](https://docs.pytorch.org/docs/2.15/onnx.html)
(the modern exporter and fixed-shape controls), [Torch compiler](https://docs.pytorch.org/docs/2.15/generated/torch.compile.html),
and [ONNX Runtime threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
