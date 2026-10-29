# Saved-model runtime results, 29 October 2026

Ordinary eager PyTorch is faster than the current portable C++ engine on this
machine and model. C++ uses substantially less OS-reported process memory. The
Python wrapper has roughly the same typical latency as direct C++; removing the
Python call alone does not close the observed gap.

Think of these as different calculators doing the same sums. A calculator written
in C++ does not automatically win: Torch also calls compiled numerical kernels.
We now have a repeatable measurement against which to test improvements.

## Single-window latency

Each call reads one 96-row window and returns all 96 per-origin predictions. The
last is the agreed one-hour temperature forecast. Numbers below combine nine
separate cells per engine: three saved models times three trials. The central
column is the median of those nine medians; ranges retain every cell, including
slow calls. Lower latency is better. RSS means the operating system's lifetime
peak resident-memory counter, not model-weight size.

| Engine | Median of trial p50s (ms) | Trial p50 range (ms) | Trial p99 range (ms) | Peak RSS range (MiB) |
| --- | ---: | ---: | ---: | ---: |
| C++ core | 1.942 | 1.909–1.974 | 2.314–3.022 | 1.457–1.500 |
| C++ through Python | 1.932 | 1.887–1.973 | 2.338–6.376 | 26.820–27.047 |
| Torch strict reference | 1.274 | 1.249–1.306 | 1.690–2.406 | 157.668–158.109 |
| Torch ordinary eager | 0.490 | 0.487–0.506 | 0.731–1.268 | 157.676–158.258 |

Paired by model/trial, Torch ordinary eager is 3.84–4.05 times faster than C++ core
at p50 (median ratio 3.92). The strict double-reduction reference is also faster
than this C++ implementation, but ordinary Torch is the more useful baseline.
All engines agree within the existing tolerance; the largest absolute difference
is 0.00000190735 temperature units across the checked windows. No forecasting
quality changed or new targets were scored.

The Python wrapper's p50 ranges overlap the direct core. This does not prove zero
overhead: the processes/builds are separate and desktop scheduling varies. Its
worst trial p99 is 6.376 ms, versus 3.022 ms for the direct core. These measured
tails are retained, and neither is a hard latency guarantee.

## Eight windows per call

| Engine | Median of trial p50s (ms) | Trial p50 range (ms) | Trial p99 range (ms) | Peak RSS range (MiB) |
| --- | ---: | ---: | ---: | ---: |
| C++ core | 15.823 | 15.676–15.985 | 16.964–19.671 | 1.449–1.500 |
| C++ through Python | 15.876 | 15.263–16.070 | 16.151–41.042 | 27.004–27.195 |
| Torch strict reference | 8.062 | 7.855–8.427 | 9.681–12.247 | 168.223–168.629 |
| Torch ordinary eager | 2.014 | 1.984–2.154 | 2.488–3.178 | 159.078–159.539 |

Native batching currently loops over windows; Torch uses batched tensor kernels.
The median paired p50 ratio is 7.81 in Torch's favour, with a 7.28–8.01 range.
Median per-trial throughput, computed from mean call time, is approximately 497
windows/second for C++ core and 3,831 for ordinary Torch. The improvement from
batching is useful for many instruments, but waiting for a batch can itself add
latency in a live system; that waiting time is outside this experiment.

## Memory and initialization

C++ owns 204,604 parameter bytes, 24,576 position bytes and 204,576 workspace bytes
per session. That is 433,756 bytes in these buffers before caller input/output,
object bookkeeping, executable code and allocator overhead. The Python session
also reserves a 2,688-byte input copy; batch calls have additional snapshots and
allocated outputs. Torch's 202,756 trainable parameter bytes exclude its scaler
and position buffers. These scopes must not be confused with whole-process RSS.

The process-memory comparison includes interpreter and library overhead. It is
useful when evaluating a small native executable, but does not establish the
incremental RAM cost of adding a model to an already running Torch service.
RSS accounting is OS-specific and is not a memory reservation guarantee.

For batch one, first calls after setup span 1.899–2.046 ms (C++ core),
2.040–2.276 ms (Python wrapper), 2.400–3.104 ms (strict Torch), and 1.614–1.950 ms
(ordinary Torch). Model loading and interpreter startup are excluded. Native clock
p50 is 34–35 ns; Python clock p50 is 91–102 ns. Clock cost is not subtracted.

## Reproducibility and verification

Frozen measurement code: `e2bdec4b23a93e1f234c22d6033827514f0df001`.
Intel Core i5-10600, x86_64, Darwin 23.6.0; AppleClang 16.0.0.16000026,
Release `-O3`, native core `-ffp-contract=off`, one inference thread. Python 3.10.6,
NumPy 1.26.4 and Torch 2.2.2. The local Torch wheel is a compatibility baseline,
not a current-platform performance frontier.

The [protocol](runtime-benchmark-v1.md) fixes 32 validation windows, all three
saved correction models, 64 warm-up calls, 1,000 measured batch-one calls and
200 batch-eight calls per cell. All 72 cells completed: 43,200 measured calls
covering 93,600 windows. Engines run serially in fresh processes with rotated
order. Every output in every fixed window passes parity before its timing cell.
No timing threshold is used in CI. No CPU affinity, frequency/cache control,
contention test or sustained live-stream measurement was performed.

Debug and Release each pass 52 native groups; sanitizers pass 51. The nine new
runner/statistics contracts pass in Release, Debug and under sanitizers. The full
Torch suite passes 46 tests, the wrapper 22 and public-data preparation 23. The
frozen code passes all five native platform jobs and both Torch-version jobs.
[Verification](evidence/runtime-v1/verification.json) records those checks.

The first run stopped when a nonzero NumPy parity difference could not be encoded
as JSON. Its partial evidence remains local. A failing regression was added,
fixed without changing tolerances or workloads, and the entire experiment rerun
in a new directory. No interrupted-run timing enters these tables.

[Full aggregate records](evidence/runtime-v1/results.json) include all models,
trials, setup/first-call timings, percentiles, memory counters and hashes. Raw
sample timings, data-derived inputs, expected outputs and model weights stay
local. An independent audit recomputed every timing statistic, checked all file
hashes and the complete 72-cell matrix, and checked the consumed-output sums.
The reserved source tail and QTS data remain unused.

## Next work

Profile the native computation before choosing an optimization. The measurements
point to computation rather than the Python boundary, but do not identify which
kernel is responsible. Compare matrix kernels, attention and normalization time;
then try one change at a time against these fixed weights and parity gates.
Potential candidates are optimized CPU matrix libraries, fusion and batch-aware
kernels. Preserve the portable implementation as the reference/fallback.

A current Linux/ARM runtime comparison, compiled Torch and an independent
inference engine remain outstanding. Streaming timestamps, reset/recovery,
concurrency, load tests and QTS integration are separate work. Runtime progress
does not repair the earlier accuracy result: persistence still wins on the
previously inspected test period's primary MAE.
