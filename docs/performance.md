# How the first latency improvements work

For subsequent full-sequence attention/block measurements, see [attention v1](attention-v1.md).

Think of inference as cooking the same recipe repeatedly. We prepare the
workspace and put the ingredients in a useful order before the first serving.
Each later serving follows a short, predictable path. Here, the ingredients are
numbers and the recipe is a small feed-forward layer.

## Prepare memory once

Model construction copies, validates and arranges the weights. Workspace
construction reserves intermediate storage. Forward calls reuse those buffers.
This separates preparation from repeated inference and removes explicit allocator
calls from the measured path. The allocation test checks that property on the
implemented forward and selected error paths.

The 64-wide, 256-hidden feed-forward example owns 132,864 parameter bytes and
needs a 3,584-byte workspace per caller, excluding caller input/output and small
object bookkeeping. There is one parameter layout per model. Choosing the packed
layout does not retain a second scalar copy.

Owned buffers begin at 64-byte boundaries. Individual workspace slices need not
all be aligned to 64 bytes. The portable implementation does not assume that
alignment for caller spans or use machine-specific intrinsics.

## Put independent work side by side

A dense projection computes several output sums. The simple reference finishes
one sum, then starts the next. The packed implementation places weights for eight
outputs next to one another and updates eight independent sums for each input.
It reuses that input value while walking contiguous weights.

The compiler can combine those independent calculations into vector instructions:
one instruction handles several numbers. LLVM describes this mechanism in its
[vectorization documentation](https://llvm.org/docs/Vectorizers.html). In the
measured AppleClang build, inspecting `Dense::forward` found `mulps` and `addps`
in the packed loop: four-lane FP32 vector operations, with two vectors covering
eight outputs. This is observed code generation on this build, not a promise
about every compiler or CPU.

Each output still sums inputs in the same order. Fused multiply-add contraction
is disabled; NaN/infinity checks stay enabled. Fast-math and finite-math-only
are rejected by the GCC/Clang compile guard because they can remove those checks.
See [Clang's floating-point rules](https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior).

The final output tile is padded to eight lanes. This costs space for small or
irregular output widths. The 3-input, 4-output case uses 112 parameter bytes packed
versus 64 bytes scalar. Shapes whose output count is divisible by eight have no
weight-padding overhead. The caller can explicitly select either kernel; there
is no automatic shape-specific dispatch yet.

## Measurements

Three preserved trials, 10,000 measured calls per kernel per trial, following
1,000 warm-up calls. Hardware: Intel Core i5-10600, x86_64; macOS 14.8.7;
AppleClang 16.0.0.16000026; C++20 Release `-O3`, `-ffp-contract=off`; one inference
thread. The scalar and packed calls alternate order, use the same finite inputs
and include validation and output publication. Code commit:
`cb8bb3108b4dcd9d5a6ccdfb32115f11787f76db`.

Ranges below are the minimum and maximum of the three trials' reported quantiles,
not confidence intervals. Times are microseconds; lower is faster. p50 is the
median call, and p99 is the point below which 99% of measured calls fell.

| Workload, input → output | Scalar p50 | Packed p50 | Packed p99 | p50 speedup |
| --- | ---: | ---: | ---: | ---: |
| Dense 3 → 4 | 0.054–0.055 | 0.058–0.059 | 0.059–0.105 | 0.93× |
| Dense 64 → 64 | 2.381–2.443 | 0.547–0.554 | 0.571–0.863 | 4.31–4.41× |
| Dense 64 → 256 | 9.180–9.400 | 2.053–2.142 | 2.934–3.538 | 4.39–4.47× |
| Dense 256 → 64 | 13.013–13.603 | 2.049–2.140 | 2.133–3.270 | 6.33–6.36× |
| Feed-forward width 64, hidden 256 | 27.697–29.437 | 10.169–10.768 | 19.090–26.137 | 2.72–2.75× |

The tiny dense case was slightly slower packed. Its timing is also close to the
measured clock overhead: p50 34–35 ns, which was reported and not subtracted.
Larger tested projections show a clear local improvement. The complete
feed-forward block improves less because normalization, activation, validation
and copying still take time.

Separate component probes measured p50 ranges of 0.383–0.386 µs for normalization
of 64 values, 2.062–2.066 µs for GELU of 256 values, 1.580–1.593 µs for softmax of
128 values, and 0.180–0.181 µs for residual addition of 64 values. These probes use
their own fixed inputs and call boundaries. They are not a trace of the composed
block and their timings should not be added to reconstruct its latency.

All scalar/packed benchmark outputs matched exactly on these fixed inputs. The
larger test suite uses the declared numerical tolerances. Raw compact results,
source and binary hashes, the sanitized compile command and protocol are in
[evidence/native-v1](evidence/native-v1/manifest.json).

These are warmed desktop microbenchmarks. There was no CPU affinity, frequency
control, cold-cache experiment, sustained-load soak, concurrent stream load,
network traffic or order execution. No comparison against ONNX Runtime, oneDNN,
BLAS or another mature runtime has been made. The numbers do not establish the
performance frontier or a hard deadline guarantee.

## Reproduce a new run

The standard-library recording tool requires a clean Git checkout, configures and
builds Release, then saves trials to a new directory. Existing runs are refused.
Use a new name each time:

```sh
python3 tools/record_benchmarks.py --output local/my-new-run --samples 10000 --trials 3
```

For a direct native run without Python:

```sh
cmake --preset release -DSTEALTH_BUILD_BENCHMARKS=ON
cmake --build --preset release
./build/release/stealth_bench --samples 10000
```

On Windows, use `stealth_bench.exe`. CI compiles and smoke-runs the benchmark but
has no wall-clock pass/fail threshold: shared runner timings are not stable enough
for a meaningful latency budget.

## How this relates to later streaming work

Preallocation, contiguous storage, immutable parameters and bounded operations
are useful foundations for latency-sensitive systems. They are established
techniques; this milestone makes no research-novelty claim.

After causal attention and ordinary forward correctness, the next performance
steps are bounded incremental state, small-shape dispatch, fair established-runtime
comparisons, and profiling under realistic stream arrivals. Optional AVX/NEON
backends need the same parity suite and a portable fallback. Approximate activation
or reduced precision needs a separate numerical contract and forecast-quality
validation before adoption.

If a measured ingestion pipeline later needs a bounded queue with one producer
and one consumer, its capacity, wraparound, backpressure, memory ordering and
shutdown behavior must be tested separately before integration. CPU pinning,
NUMA placement and network/kernel changes belong to a specified deployment
machine and workload. None is assumed by this portable numeric core.

## Deferred optimization notebook

The user wants the advanced optimization work after the complete model is correct.
Retain a portable path and an independent reference. Record a measured bottleneck,
the proposed change, numerical difference, memory cost and median/tail timing
before accepting any specialization. Current candidates are:

- Fixed-shape/template kernels and small-shape dispatch after benchmark dimensions
  are agreed; check code size and tiny workloads as well as larger ones.
- Explicit AVX/NEON, fusion and packing choices after compiler output and profiler
  evidence; do not assume a branch-free rewrite is faster.
- Bounded KV caching after reset/capacity/position semantics and prefix parity.
- Prefetching and profile-guided optimization only for a stable measured workload.
- Affinity, queue design and deployment-specific scheduling when real stream
  arrival rates and latency budgets exist.

ETTm1 downloading, parsing and scaler fitting are cold preparation tasks. Keep
these outside timed native inference. The window iterator stores indices and
creates one descriptor at a time; it avoids copying every overlapping window
into one large array. This is a memory design choice, not a measured latency win.


The first complete forecaster exposes per-origin outputs for causal diagnostics.
A future final-only readout can avoid normalization/head work for unused earlier
origins while retaining the trace path as a reference. Measure that change after
trained parity; it does not require changing the agreed forecasting objective.

## Python and raw-window follow-ups

The first wrapper copies input explicitly; batch calls snapshot their bounded
input and evaluate sequences sequentially. Primitive convenience calls allocate
workspaces and retain the GIL. Measure these interface costs separately from
native arithmetic. Later candidates include caller-owned output/workspace APIs
and fused batches, only if measured workloads justify them.

Raw-window streams currently recalculate each accepted prefix/window. Position
markers restart at zero. A future cache must prove equivalence for its stated
semantics; simply retaining prior keys/values is not an equivalent optimization.
No wrapper, full-stream or training speed claim is recorded in this milestone.

## Full-model runtime evidence and next experiment

The [saved correction-model comparison](runtime-results-v1.md) is complete. Across
all three saved models and three trials, C++ core batch-one p50 is 1.909–1.974 ms;
ordinary eager Torch is 0.487–0.506 ms. Native process memory is much smaller.
All engines pass fixed-window parity before measurement. The Python wrapper has
similar median latency to direct C++. Subsequent native sampling puts attention
at about 39% of stacks. The [first measured attention change](attention-results-v1.md)
computes eight independent FP64 value sums together, preserving their order.
It reduces paired full-model batch-one p50 by 10.36% with byte-identical native
predictions and no additional owned heap workspace. Two individual p99 pairs
worsen despite an 8.23% median improvement, and all tails are retained.

In the fresh four-engine comparison, C++ core is 1.714 ms and ordinary Torch
0.491 ms at the median of nine cell medians. Torch remains faster. The new loop
is portable C++20, with explicit scalar reference/tails; this compiler generates
double-precision vector instructions. Profile the improved model before selecting
the next change.

Use the frozen benchmark to test one measured change at a time, keeping weights,
inputs, full-output workload, threads and tolerances fixed. Compare matrix,
attention and normalization costs before choosing fusion, an optional optimized
matrix library, fixed-shape specialization or improved batching. Keep the portable
path and independent reference. Do not accept a local microkernel win without
checking full-model latency, tail behavior and memory.

The parity-focused Torch reference deliberately uses double reductions; the
experiment also includes ordinary FP32 layer normalization, exact GELU and causal
SDPA. Both use Torch 2.2.2 on the local Intel Mac. This is a compatibility result.
A current supported Linux/ARM environment, compiled Torch and an independent
inference runtime remain necessary to map the performance frontier. Shared CI
runners establish portability and numerical correctness, not stable latency.

Do not change the model while comparing runtimes. If the initial learner fails to
beat persistence, discuss its learning objective, validation behaviour and simple
model baselines before introducing a second transformer block. Runtime speed and
forecast quality are separate questions. QTS adapters remain a later stage with
explicit data clocks, missing observations and feature-order checks.


## Bounded matrix rows

The next accepted change adds `Dense::forward_rows` and bounded feed-forward row
workspaces. The transformer normalizes rows independently, then processes up to
four together through QKV and post-attention projections/MLP. Each output keeps
its arithmetic order and the complete output is staged before publication.
The same saved-model experiment records another 12.98% paired batch-one p50
reduction and 12.80% for eight windows, with 17.25 KiB extra workspace and no hot
allocations. A scalarized first candidate was rejected despite passing numerical
tests; exposing independent accumulator vectors fixed its compiled loop.
[All trials, rejected results and memory scopes](dense-rows-results-v1.md) are
retained. This is a within-window matrix tile; cross-window batching is still
sequential. [Frontier requirements](runtime-frontier.md) define the supported
current-runtime comparison and the remaining performance target.

## Request only the latest forecast

The accepted one-block schedule retains all input embeddings and QKV projections,
then evaluates only the last attention output, MLP and readout. The fixed 108-cell
experiment applies the same schedule to Torch and keeps full-trace comparisons
separate. Native full-to-last paired p50 improves 87.62%; native latest-only p50
is about 0.181 ms versus 0.287 ms for ordinary eager Torch 2.2.2 on the local Intel
Mac. Native loses for batch eight because windows are still processed sequentially.
The full trace and all 9,216 latest-prefix answers are bitwise unchanged; there is
no extra workspace and no native hot allocation. [Evidence and limitations](latest-results-v1.md)
include all timings, tails and the different policy for skipped unused overflow.

Profile this new execution schedule before another kernel change. Full-trace
component shares cannot be reused as latest-only bottleneck evidence. Candidate
work includes all-row QKV/input projections and effective batching; changing
approximations or backends requires its own parity and performance gates. A second
block needs earlier first-block states and changes what work can be skipped.
