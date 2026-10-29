# Latest-only forecast results, 29 October 2026

The native engine now calculates just the requested last forecast in a window.
Its batch-one p50 falls from 1.455 ms for the full diagnostic trace to
0.181 ms for the last answer. The median paired reduction is
87.62%; the frozen work-reduction gate passes.
All 9,216 saved native prediction origins still match exactly through prefix calls.
The default full trace is also byte-identical to the previous implementation.

Imagine needing only the final answer on a worksheet. We still read every earlier
observation that influences it, but stop calculating earlier answers that nobody
requested. The same pruning is applied to Torch. This changes the execution
schedule of the existing one-block model, without changing its weights or target.
See [the execution contract](latest-forecast-v1.md).

## Matched work and results

The frozen [protocol](../benchmarks/ettm1/latest_v1.json) covers three saved models,
three trials, batch one/eight, three engines and both output modes: 108 complete
cells, 64,800 timed calls and 140,400 window evaluations. Each cell uses a fresh
process; engine/mode order rotates, inference uses one thread, and every requested
output passes reference parity before warm-up and timing. Warm-up is 64 calls;
measurement is 1,000 calls for batch one and 200 for batch eight.

Each row below combines nine cells. Centres are medians of cell p50 values;
ranges retain every cell and are not confidence intervals. p99 is a measured
percentile rather than a latency deadline. `all` returns 96 predictions per window;
`last` returns one. Compare engines within the same mode.

One window per call:

| Engine / requested outputs | Median cell p50 (ms) | Cell p50 range (ms) | Cell p99 range (ms) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| C++ core / all | 1.455 | 1.429–1.500 | 1.759–1.941 | 1.469–1.508 |
| C++ core / last | 0.181 | 0.177–0.185 | 0.237–0.297 | 1.453–1.516 |
| C++ through Python / all | 1.449 | 1.427–1.488 | 1.748–2.053 | 26.770–26.965 |
| C++ through Python / last | 0.182 | 0.176–0.185 | 0.217–0.314 | 26.766–26.934 |
| Torch ordinary eager / all | 0.472 | 0.454–0.480 | 0.622–0.720 | 157.223–158.484 |
| Torch ordinary eager / last | 0.287 | 0.284–0.294 | 0.385–0.477 | 156.629–157.730 |

Eight windows per call:

| Engine / requested outputs | Median cell p50 (ms) | Cell p50 range (ms) | Cell p99 range (ms) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| C++ core / all | 11.605 | 11.405–11.873 | 12.752–13.380 | 1.461–1.504 |
| C++ core / last | 1.452 | 1.421–1.478 | 1.648–2.080 | 1.461–1.500 |
| C++ through Python / all | 11.576 | 11.420–11.837 | 12.810–13.405 | 26.926–27.082 |
| C++ through Python / last | 1.440 | 1.388–1.468 | 1.594–1.970 | 26.797–27.055 |
| Torch ordinary eager / all | 1.921 | 1.861–1.957 | 2.322–2.801 | 158.922–160.250 |
| Torch ordinary eager / last | 0.566 | 0.551–0.594 | 0.707–0.933 | 156.504–157.852 |

Native full-to-last median paired p50 reductions are
87.62% for one window and
87.40% for eight. Batch-one seed medians are
7: 87.62%, 19: 87.36%, 43: 87.69%.
Its median paired batch-one p99 reduction is 85.56%,
with individual reductions ranging 84.566–87.156%.
Torch also benefits: its full-to-last median paired batch-one p50 reduction is
38.89%.

For the fair latest-only race, native/Torch median paired p50 ratios are
0.627 for batch one and
2.588 for batch eight (below one means native is faster).
Median paired p99 ratios are 0.616 and
2.277, respectively. These are comparisons
with local eager Torch 2.2.2, not current compiled Torch or ONNX Runtime. Native
batching still loops over windows; Torch executes a batched graph.

## Correctness, API and scope

Native `SequenceOutput::last` is explicit on attention, block and forecaster calls;
the default is `all`. Python adds `session.predict_last(x)` for `[T,F]` input,
returning a Python float, and `session.predict_last_batch(x)` for `[B,T,F]`, returning
an owned FP32 array of shape `[B]`. Both retain finite-input, shape, input-copy,
busy-session and failure checks. Existing streams retain full-trace execution.

For this one-block architecture, latest mode computes all embeddings and QKV
projections, then only the final attention output, MLP and readout. It checks all
raw inputs and the complete QKV arrays. An overflow in a skipped earlier MLP row
is outside this mode's calculation graph; latest may succeed where the full
trace fails. Required-calculation errors still fail without publishing output.
A second block needs earlier first-block outputs, so this schedule must not be
copied blindly into a deeper model.

Independent fixture expectations, every-prefix equality, both native kernel
choices, irregular dimensions, invalid modes/shapes, aliasing, moved state,
nonfinite input, required and unused overflow, mixed-mode reuse and zero native
hot allocations are covered. Deliberately using the wrong query failed the native
regression; an incorrect single-query causal mask failed the Torch regression.
The correct sources were restored, rebuilt and retested before measurement.

All 9,216 saved full-trace values are unchanged and every latest-prefix answer
matches them bit for bit; all 96 batched final values also match. Across the
runtime cells, the maximum absolute reference difference is
1.9073486328e-06 temperature units, within the
unchanged 5e-5 absolute / 2e-4 relative tolerance. Latest cells validate only the
requested final outputs; the separate prefix audit checks all saved origins.

Local validation: 59 Debug groups, 59 Release groups, 58 sanitizer groups,
47 Torch tests, 23 wrapper tests, 11 runtime contracts in ordinary and sanitized
builds, and four profile/gate tests pass. Debug and Release include relocated
native consumers. The recorded measured-code CI passes native Linux GCC/Clang,
macOS Intel/ARM and Windows MSVC, plus Torch 2.2.2/2.13.0 CPU reference jobs.

Owned native workspace remains 222,240 bytes; the Python session additionally
owns a 2,688-byte input copy. Parameters remain 204,604 bytes and positional
storage 24,576 bytes. Latest mode saves arithmetic, not session workspace.
Process RSS above includes runtime imports, setup and timing; it is a high-water
mark, not attributable model memory. Python batch calls still allocate their
input snapshot and output. Setup, first-call, clock cost and complete cell
statistics are retained in the JSON evidence; first call is after model loading,
not a claim about process or cold-disk startup.

Measured code: `6197147144cfbd90bf07a322bb44c7a702be9ba8`. Intel Core i5-10600,
x86_64 macOS 14.8.7, AppleClang 16, Release `-O3`, native `-ffp-contract=off`,
Python 3.10.6, NumPy 1.26.4, Torch 2.2.2. Scheduling, caches and CPU frequency
are uncontrolled. No timing is a hard deadline or universal platform claim.

The [complete compact results](evidence/latest-v1/runtime.json) and
[verification record](evidence/latest-v1/verification.json) bind code, protocol,
models, inputs, installed extension, binary, packages and local raw-artifact hashes.
The audit recomputes every timing summary and consumed-output checksum and checks
all 108 unique cells. Raw inputs, predictions, weights and timing traces stay local.
No QTS data or reserved source-tail observations were used.

## Remaining work

The [current-runtime comparison](runtime-frontier.md) still needs a supported
measurement host, current eager/compiled Torch and an independent CPU runtime.
Profile latest-only execution before the next kernel change: all-row QKV and
input projection remain, and further specialization needs its own numerical and
performance gates. Integrating the latest path into streams needs an explicit
failure-policy decision. Sustained load, concurrency, queueing and recovery are
still unmeasured. Prediction accuracy is unchanged: the saved correction models
still miss the persistence baseline, so this is execution progress, not evidence
of better forecasts or trading readiness.
