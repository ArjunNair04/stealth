# Native batching with fixed arithmetic, 31 October 2026

Native batching lowers paired batch-eight p50 by 11.84% in the matched Linux
comparison. It processes independent forecast windows together. Each window keeps
its own history and restarts its position indices at zero. It changes execution,
not the model, trained weights, forecast objective or numerical tolerances.

The preceding [Beaker compatibility run](beaker-linux-v1.md) exposed the problem:
eight native forecasts cost almost eight single calls, while eager Torch grouped
the work. The new comparison runs the old native binary, the candidate, its Python
wrapper and the same eager Torch 2.6 implementation on one allocated CPU host.

## What changed

`ForecastBatchWorkspace` preallocates scratch, and `ForecastModel::forward_batch`
accepts `[batch,tokens,features]` inputs. It processes at most four windows at a
time, with 32-row matrix and normalization tiles. It groups embedding, QKV,
projection, MLP and readout work; attention remains separate for each window.
The full trace and latest-only output modes use the same contract.

The new `layer_norm_rows` interleaves four independent FP64 reductions. Each
individual row sums its channels in the original order. Scale/shift and buffer
checks are shared across the call. Dense accumulation, exact GELU, FP32 weights,
FP64 attention and finite checks are unchanged. FMA contraction and fast-math
remain disabled. No BLAS, platform SDK, thread pool or native dependency was added.

All outputs are staged until the entire native batch succeeds. A failure in the
last window leaves caller output unchanged. Native forward calls allocate no heap
memory, including tested error paths. The [API contract](forecast-contract.md#independent-window-batches)
records capacity, aliasing, ownership and failure behavior.

The Python session calls the same core and caches batch scratch lazily. It retains
its busy guard, owned outputs and input snapshot before releasing the GIL. Existing
Python batches above 1024 windows are split into bounded native calls. Single-window
Python calls and batch-one benchmark calls retain the original execution path.
The benchmark now reports batch scratch after its first allocation and retains
an explicit `--batch-execution serial` diagnostic option.

## Investigation and rejected work

A four-second local sample attributed 49.52% of sampled stacks to dense work and
20.25% to normalization. After batching, normalization represented 12.80% and
dense work 51.37%. These are sampled proportions, not before/after latency
measurements or a causal decomposition of saved time. The attribution parser now
recognizes batched normalization without counting nested scalar calls twice.

The first grouped execution probe improved batch-eight local timings modestly.
An eight-row AVX tile and altered weight traversal did not improve that result,
so the specialization was removed. The existing four-row AVX kernel remains
unchanged. The retained normalization change passed exact arithmetic tests;
noisy development probes are preserved separately from Linux acceptance evidence.

The first complete Linux candidate failed the frozen performance gate: batch-eight
paired p50 regressed 9.79% and p99 regressed 11.31%, despite passing correctness.
Four scheduling and reduction variants also failed. Their raw observations remain
in the public evidence; none is part of the final implementation.

Component timings showed that grouped matrix and normalization operations were
faster. Instrumenting the complete forecast then isolated input preparation:
about 26 µs per window in serial execution versus 71 µs in the first batched path.
The transformer block itself improved from about 178 to 156 µs. These stage
measurements include instrumentation overhead and diagnose the failed candidate;
they are not the final acceptance timings.

GCC 11 assembly exposed a 64-bit integer division inside the position-addition
channel loop. The expression calculated the same row position once per channel.
The correction computes the initial position once per tile, then increments and
wraps it once per row. Position resets and arithmetic values stay unchanged.
The added 31-token case exercises a window boundary inside a 32-row tile.

GCC's corrected object places the divide before the channel loop: the old loop
jumps from `0x1b72` to `0x1b0a`, passing the divide at `0x1b29`; the new loop jumps
from `0x1b5a` to `0x1b13`, after the divide at `0x1af6`. The
[before](evidence/native-batching-v1/diagnostics/before-position-loop.asm) and
[after](evidence/native-batching-v1/diagnostics/after-position-loop.asm) excerpts
include the checked arithmetic and loop branches.

## Correctness evidence

Local validation passed 73 tests each in Debug, Release and forced-portable
configurations, and 72 address/undefined-behavior sanitizer tests. The relocated
native consumer and an isolated source-distribution-to-wheel installation passed.
There are 36 wrapper tests, 86 offline Python tests, and 114 passing
training/integration tests with three optional Linux-specific checks skipped.

The new tests cover independent reference fixtures, generated regular/irregular
widths, single-token windows, row/tile tails, both output modes, capacity limits,
workspace reuse and moved state, nonfinite inputs, aliasing, late arithmetic
overflow, allocation failure and zero native hot allocations. Three deliberate
faults were caught: cross-window queries, partial batch publication and an
incorrect normalization mean. The original code was restored and retested.

On the local saved-model check, all three models' full and latest outputs are
bitwise equal between serial and grouped native execution at batch sizes
1, 2, 4, 8, 16 and 32. Every saved output also passes the unchanged independent
reference tolerance, `5e-5 + 2e-4 * abs(reference)`.

One initial training test launch accidentally used the packaging environment's
NumPy 2 with local Torch 2.2.2. Its failure log is retained. Repeating in the
existing Torch/NumPy 1.26 environment passes; no native, model or tolerance change
was required.

The first corrected Linux run passed all native, sanitizer, offline and wrapper
gates, then stopped before timing because a process-group test assumed Python
would start within 200 ms. The repaired fixture waits for a real descendant's
readiness signal and requires its pipe to close after termination. A forced
400 ms startup passes, while deliberately killing only the parent is still caught.
The production supervisor is unchanged. The final run reuses the exact validated
native binaries and twelve successful build/test records by hash, verifies that
only this test source changed, and runs fresh Python/training gates before timing.
Two continuations stopped on incomplete executable placement: the sizing test
needed the benchmark at its conventional path, and the model-inspection test
needed the companion CLI. An exhaustive test-path inventory and hash-verified
copies of both unchanged executables repaired that layout. All three stopped
validation runs remain recorded; none collected acceptance timings.

## Linux comparison

The accepted Linux comparison tests native source commit
`2b19980f0f556d81bb6c4c015830a202a2846450`. Test-only commit
`c5f4374ff1c1d17d13e3f02a9b108edb2caffd0e` repairs the integration fixture.
The rejected first run used
`67e6399e9a8a041044969a39062ebd8d092de5b1`. The baseline binary comes from the preceding admitted
source `a65ba9f229e360cfea92b70438c98c89e19a5416`, bound by SHA-256.

The primary grid has three saved seeds, three trials, five batch sizes
(1, 2, 4, 8, 16) and four engines: 180 fresh serial workers. It retains the same
32 validation windows and 64 warm-up calls. Each cell validates predictions before
timing. Engine order rotates, all observations are retained, and no build, test
or profiler runs concurrently with measurement. Each reported p50/p99 is the
median of nine worker quantiles, not a pooled percentile or confidence interval.

The final run passes all frozen performance gates. Median paired batch-eight
p50 falls 11.84%; every saved seed improves. The median paired batch-one
p50 change is -0.27%. Paired p99 changes
are -5.88% for one window and
-12.20% for eight windows (negative is better).

Measured on Intel(R) Xeon(R) CPU E3-1240 V2 @ 3.40GHz, with one timed CPU thread, GCC 11.2.1 and
eager Torch 2.6.0. The earlier rejected run and the inherited native validation
used separate scheduled hosts; each timing comparison reruns all engines on its
own host. No cross-host subtraction is used.

Times below are milliseconds per call, p50 / p99, each the median of nine
worker percentiles. A batch-eight call returns eight forecasts.

| Batch | Old C++ | New C++ | New Python wrapper | Eager Torch 2.6 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.227 / 0.315 | 0.226 / 0.244 | 0.230 / 0.266 | 0.721 / 0.761 |
| 2 | 0.451 / 0.553 | 0.405 / 0.534 | 0.412 / 0.476 | 0.880 / 0.929 |
| 4 | 0.900 / 1.069 | 0.793 / 0.823 | 0.799 / 0.840 | 1.154 / 1.215 |
| 8 | 1.797 / 1.851 | 1.584 / 1.641 | 1.595 / 1.646 | 1.708 / 1.780 |
| 16 | 3.594 / 3.676 | 3.170 / 3.218 | 3.183 / 3.261 | 2.788 / 2.913 |

At batch eight the native core is 7.2% faster than this eager Torch baseline
by the table's median p50 (1.584 versus 1.708 ms). At batch sixteen, Torch
is still faster: 2.788 versus 3.170 ms.
This result concerns the fixed workload and precision policies, not all model
sizes or current compiled runtimes. The Python wrapper includes snapshot and
output allocation costs. It is reported separately from the native core.

The verifier reconciles 426 sealed files, 180 primary worker cells and
75,600 measured calls. All predictions pass the unchanged independent
reference tolerance; maximum latest-output difference is
1.90734863e-06. Native before/after checksums match in every
primary and full-output comparison. Full-output timings remain secondary;
all 18 full-output worker observations, including their tails, are retained.

## Memory and reproduction

For this configuration, native batch-eight scratch is 922,752 bytes versus
222,240 bytes for the old single-window workspace. The Python session retains
both its original 224,928 bytes of workspace/input storage and the lazily allocated
batch scratch: 1,147,680 bytes after a fresh session's first batch-eight call. Native parameters
(204,604 bytes) and position values (24,576 bytes) are unchanged. Caller arrays,
Python snapshots and object bookkeeping are additional. Scratch covers at most
four active windows; output staging scales with the admitted batch capacity.

Raw process peak-RSS readings are retained but inherit the parent's high-water
mark on this platform. They do not establish an isolated memory advantage.

The [evidence directory](evidence/native-batching-v1/) retains the accepted and
rejected runs, all worker observations, source/input/binary hashes, diagnostic
variants and verification scripts. Run:

```sh
python3 docs/evidence/native-batching-v1/verify_published.py
```

This standard-library check verifies verbatim worker payloads against their
original seals, recalculates percentiles and paired acceptance, and preserves
the rejected run as rejected. It does not rerun inference. The original complete
trees and logs remain in the ignored local evidence directory; their hashes are
retained publicly. Source commits, protocol and model/input hashes identify the
measurements without publishing private cluster accounts or hostnames.

## Limits and next work

This is a warm inference comparison for the existing 96-step, seven-feature,
64-wide, four-head, 256-hidden, one-block ETTm1 workload. Cold construction and
first-call costs are recorded separately. A shared compute node and CPU affinity
do not guarantee exclusive execution or fixed frequency. Tail observations are
not hard execution deadlines.

Current compiled Torch/ONNX comparison, other model families, sustained serving
load, ARM-specific acceleration and financial usefulness remain separate gates.
The earlier large-bike-model numerical discrepancy is unchanged. Batching needs
several inputs already available; waiting for a batch to fill can add response
latency to a streaming system.
