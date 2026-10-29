# Four-row execution results, 29 October 2026

Reusing packed weights across four observations reduces full-model batch-one p50
by 12.98% against the accepted attention engine in fresh paired trials.
All 9,216 checked trained-model outputs remain byte-identical. Each prediction
keeps its arithmetic order. The extra owned workspace is 17.25 KiB per full-model
session, allocated before inference; parameters and positions are unchanged.
Ordinary Torch remains faster in the fresh comparison below.

Imagine four people using the same multiplication table. We now read a weight
once and update their four separate answers together. This shares the work of
fetching weights without mixing observations. The [contract](dense-rows-v1.md)
describes the checked native row APIs and bounded transformer integration.

## Paired experiment

The baseline includes the previous attention optimization. Both binaries use
identical saved weights and validation inputs, one inference thread, three trials
and alternating process order. Each table combines nine cells per implementation:
three models times three trials. Centres are medians of cell medians; ranges
retain all cells and are not confidence intervals. p99 is a measured percentile,
not a deadline. Each call returns all 96 per-origin predictions.

One window per call:

| Implementation | Median cell p50 (ms) | Cell p50 range (ms) | Cell p99 range (ms) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Previous C++ engine | 1.745 | 1.643–1.790 | 2.001–3.044 | 1.473–1.531 |
| Four-row C++ engine | 1.470 | 1.431–1.560 | 1.800–2.970 | 1.469–1.520 |

Median paired p50 reduction: 12.98%, range
11.798–16.050%. Seed medians:
7: 14.04%, 19: 12.85%, 43: 12.33%.
Median paired p99 change: -8.09%; 1 of nine pairs have
higher p99, and the largest increase is 8.26%. The frozen performance
gate passes; all individual tails remain reported.

Eight windows per call:

| Implementation | Median cell p50 (ms) | Cell p50 range (ms) | Cell p99 range (ms) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Previous C++ engine | 13.856 | 13.125–14.343 | 14.585–23.949 | 1.453–1.500 |
| Four-row C++ engine | 11.989 | 11.445–12.652 | 12.850–20.971 | 1.461–1.508 |

Median paired batch-eight p50 reduction: 12.80%.
Four-row tiling is within each window. Cross-window native batching still loops
sequentially; it is not a fused multi-instrument matrix batch.

## Fresh runtime comparison

The unchanged four-engine protocol was rerun after acceptance: 72 complete cells,
43,200 timed calls, parity on every fixed output before each cell. It uses the
same native binary, inputs and weights as the accepted paired experiment. The
Python wrapper was rebuilt and installed before testing.

One window per call:

| Implementation | Median cell p50 (ms) | Cell p50 range (ms) | Cell p99 range (ms) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| C++ core | 1.529 | 1.488–1.562 | 1.958–2.800 | 1.484–1.527 |
| C++ through Python | 1.529 | 1.500–1.562 | 1.860–3.126 | 26.785–27.125 |
| Torch strict reference | 1.286 | 1.259–1.343 | 2.347–2.537 | 158.031–158.371 |
| Torch ordinary eager | 0.502 | 0.482–0.520 | 0.685–1.167 | 157.562–159.238 |

Eight windows per call:

| Implementation | Median cell p50 (ms) | Cell p50 range (ms) | Cell p99 range (ms) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| C++ core | 12.272 | 11.905–12.616 | 17.835–20.722 | 1.477–1.512 |
| C++ through Python | 12.251 | 11.979–12.558 | 13.218–21.094 | 27.129–27.449 |
| Torch strict reference | 8.370 | 8.017–9.378 | 12.879–16.476 | 168.355–169.320 |
| Torch ordinary eager | 2.109 | 2.010–2.557 | 2.641–4.426 | 159.121–159.582 |

The median paired p50 ratio still favours ordinary Torch by
3.09 times for one window and 5.86 times for eight.
The maximum absolute difference against the Torch reference is
1.9073486328e-06 temperature units, within the unchanged tolerance.
This is progress against our baseline, not evidence of leading-runtime parity.

## Rejected first implementation

The first nested-array loop passed every numerical check but was
79.38% slower at paired batch-one p50 and failed the performance gate.
Its 36 cells are retained in [the rejected result](evidence/dense-rows-v1/rejected.json).
Compiler inspection found 35 scalar versus six vector multiply instructions in
the complete function. Exposing four named accumulator vectors reduced that to
three scalar and 14 vector multiplies. These are static instruction counts on
this build, not execution frequencies or promises about another compiler.
The revised code kept the same four-row contract and numerical order. It was
retested and measured in a new immutable run, without replacing the failed data.

## Verification and memory

Accepted paired code: `91561ea77f1c4adf72f7daf3fac563c296fbde7b`.
Four-engine code: `91561ea77f1c4adf72f7daf3fac563c296fbde7b`.
Rejected code: `c94308b16dcf072bb7cfae5df06bb39a4c726cd8`.
Intel Core i5-10600, x86_64, macOS 14.8.7, AppleClang 16.0.0.16000026,
Release `-O3`, native `-ffp-contract=off`; Python 3.10.6, NumPy 1.26.4,
Torch 2.2.2. No new dependencies or architecture-specific intrinsics were added.

Debug and Release pass 56 native groups each; sanitizers pass 55. Torch passes
46 tests, the installed wrapper 22, runtime contracts nine and profile/gate tests
four. The final wrong-row mutation fails; correct code is restored before the
complete checks. Tests cover independent answers, scalar/packed bitwise equality,
row and column remainders, isolation, invalid shapes, overflow, overlapping spans,
moved state, late failure preservation and zero hot allocations. The new profiler
recognizes row kernels without double counting wrappers; all six previous
profile attributions remain unchanged. [Verification](evidence/dense-rows-v1/verification.json)
binds packages, test logs and cross-platform CI to exact source commits.

Native owned workspace grows from 204,576 to 222,240 bytes. Parameters stay at
204,604 bytes and positions at 24,576, for 451,420 owned buffer bytes in total,
excluding caller input/output, allocator overhead and object bookkeeping. The
Python session also owns a 2,688-byte input copy. Single-row feed-forward buffer
sizes stay unchanged, although workspace object layouts add capacity metadata;
rebuild native consumers and Python wheels. No model-file migration is needed.
RSS includes the executable/interpreter and libraries. It does not measure the
incremental cost of a model inside an existing service.

Every artifact hash, unique timing cell, latency/clock statistic and consumed-output
checksum was audited, including the rejected run. Aggregate reports regenerate
byte-identically. All accepted [paired records](evidence/dense-rows-v1/paired.json)
and [four-engine records](evidence/dense-rows-v1/runtime.json) remain available.
Raw traces, datasets, weights and prediction arrays stay local. No reserved-tail
rows, QTS data, new targets or forecast-quality conclusions were used.

Measurements are warm desktop runs with no affinity/frequency/cache control,
sustained or concurrent streams, full cold start, network or order execution.
Other shapes/platforms have correctness coverage, not a measured speed guarantee.
A current-platform, compiled and independent-runtime comparison is still required;
see [the performance frontier requirements](runtime-frontier.md).
