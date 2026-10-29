# What comparable CPU performance must mean

Status: comparison requirements, not completed frontier evidence. The saved-model
reports measure one Intel Mac against Torch 2.2.2. They establish repeatable local
improvements and remaining gaps. They cannot establish competitiveness with the
best available deployments. Forecast accuracy is a separate question: faster
execution of the same weights leaves the earlier persistence comparison unchanged.

The [latest-only experiment](latest-results-v1.md) now beats the local eager
baseline for one requested forecast, while losing for batches of eight. Both
engines skip unused outputs. That useful workload-specific result still leaves
current compiled and independent runtime comparisons outstanding.

Think of this as a race between calculators. Each calculator must receive the
same numbers, return the same requested answers, and use the same number of
workers. We must also count preparation time separately from repeated calls.

## Next comparison environment

Use a supported Linux x86_64 or ARM CPU machine with a dedicated measurement
window. Pin its CPU/ISA, operating system, compiler, Python and package versions,
thread controls, power mode and affinity policy before collecting results.
Keep all engines on the same machine. Do not compare this Mac's milliseconds to
another machine's numbers and attribute the difference to software.

At the 29 October 2026 review, ONNX Runtime 1.31.1 requires Python 3.11+ and its
published macOS wheels are ARM64; the current local benchmark is Python 3.10 on
Intel macOS. Its Linux x86_64/ARM64 wheels provide a practical supported route.
No current ONNX Runtime benchmark has run here, and no dependency was silently
installed. [Pinned release metadata](https://pypi.org/project/onnxruntime/1.31.1/)
records the package/platform constraint. A source build would be a separate,
explicitly recorded environment, not a drop-in wheel replacement.

## Engines and matched work

Compare the portable C++ core and its Python interface with a supported eager
PyTorch release, compiled PyTorch using Inductor, and ONNX Runtime's CPU provider.
Freeze exact releases and build hashes for that machine. PyTorch's CPU compilation
path optimizes graphs and generates compiled kernels, so eager execution is only
one relevant baseline. [PyTorch CPU compiler guide](https://docs.pytorch.org/tutorials/intermediate/inductor_debug_cpu.html)
explains generated code, profiling and correctness checks; it does not predict
a speedup for this model.

Start with the existing 96-by-7, one-block workload returning the latest forecast,
all three saved models and identical validation windows. Keep the full trace for
diagnostic parity and causality checks. Preserve preprocessing, causal mask,
epsilon, exact-GELU definition, positions and correction head. Freeze the
converted graph hashes and verify every output against the reference before any
timing. Report dtype and numerical differences. Quantization or approximate
functions need a separate experiment; they cannot silently replace this baseline.
Every timed engine must implement the same latest-only request. The bounded
[Linux specification](frontier-linux.md) records the next gate; it remains pending
a host lock and compiled/ONNX adapter validation.

The first comparison uses one intra-op and one inter-op thread. ONNX Runtime's
default intra-op setting can use multiple physical cores; explicitly set one
rather than treating the default as comparable. Record graph optimization level,
execution mode, thread-pool/spinning policy and execution providers. Report export
and session/graph preparation costs separately. [ONNX Runtime thread controls](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
describe these settings. Compile failures and unsupported operators are recorded
outcomes, not a reason to omit an inconvenient engine without explanation.

## Acceptance target

For a named machine and workload, the working engineering target is native p50
and p99 within 20% of the fastest tested, parity-passing FP32 baseline, with memory
and startup costs reported. This is a target, not a result or universal guarantee.
Report batch one and batch eight independently; neither a batch-eight throughput
win nor smaller RAM establishes a single-request latency win. An engine that
misses the target remains visible. Broader shape coverage needs fresh runs rather
than extrapolating this tiny transformer to other architectures.

Use fresh processes, alternating engine order, repeated paired trials, fixed
warm-up and retained raw timings. Keep tuning runs separate from acceptance runs
and retain failed candidates. Report p50/p95/p99/max, mean throughput, first call,
setup/compile time, clock cost, owned buffers and process peak RSS with their
scopes. No subtraction of convenient overheads or undocumented thread advantage.
The precise run config, model/input hashes, sample counts and acceptance rules
must be committed before timing starts on the chosen environment.

CI continues to verify portability and arithmetic. Shared CI machines do not
supply stable latency evidence. A later deployment gate additionally needs a
specified arrival rate, queue/backpressure behavior, concurrency, sustained load,
restarts and recovery. Trading usefulness needs separate point-in-time data,
forecast validation, costs and risk controls. Those requirements are still open.

## Native implementation direction

The row APIs expose a matrix workload. Complete the current-runtime comparison
before choosing further native optimization. If it identifies a worthwhile gap,
profile that workload and compare further native tiling/fusion with optional
mature CPU matrix and vector-math backends. Keep the
portable fallback and numerical checks. Measure the integration, packing/setup,
code size and memory costs; a microkernel win alone is insufficient. These are
established engineering techniques, not a research-novelty claim.
