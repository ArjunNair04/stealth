# Attention optimization results, 29 October 2026

Computing eight independent attention sums side by side reduced full-model
batch-one p50 latency by 10.36% in the paired experiment. Every saved
model improved across its trials. All 9,216 checked trained-model outputs are
byte-identical to the old native Python package. Ordinary PyTorch remains faster
in the fresh four-engine comparison; this is one measured improvement on one
Intel Mac, not a claim about the performance frontier.

Imagine adding eight columns of numbers. We now update all eight totals while
looking at each row, instead of walking down the table eight separate times.
Each column still adds its numbers in the same order. The compiler generated
double-precision vector instructions for this loop on the measured machine.
The [design and numerical contract](attention-tiling-v1.md) explains the change.

## Paired baseline and candidate

The exact preserved baseline executable and rebuilt candidate alternate order,
with the same 32 windows, three saved models, three trials and one inference
thread. Sampling is disabled during these runs. These fresh pairs measure the
change; comparisons with earlier morning timings are not used to accept it.
Each cell's p50 is the median call; p99 covers 99% of its measured calls. Table
centres are medians of nine cells, and ranges include every cell, not confidence
intervals. Lower latency is better.

One window per call, returning all 96 predictions:

| Implementation | Median of cell p50s (ms) | Cell p50 range (ms) | Cell p99 range (ms) | Peak RSS range (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Preserved C++ baseline | 1.900 | 1.842–1.989 | 2.296–2.639 | 1.457–1.480 |
| Tiled C++ candidate | 1.688 | 1.652–1.782 | 2.057–2.470 | 1.453–1.496 |

The median of paired percentage reductions is 10.36%, with a
10.043–12.667% range. Seed medians are
7: 10.36%, 19: 10.26%, 43: 10.41%.
This passes the frozen 5% improvement gate. Median paired p99 improves by
8.23%; 2 of nine individual p99s worsen, with a largest
increase of 6.98%. All tails remain in the aggregate evidence. The gate concerns
the median paired change, not a guarantee for every call or trial.

Eight windows per call:

| Implementation | Median of cell p50s (ms) | Cell p50 range (ms) | Cell p99 range (ms) | Peak RSS range (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Preserved C++ baseline | 15.584 | 14.621–16.057 | 16.161–17.500 | 1.449–1.473 |
| Tiled C++ candidate | 13.983 | 13.143–14.364 | 14.564–16.920 | 1.453–1.488 |

Median paired batch-eight p50 reduction is 10.32%.
Native batching still executes windows sequentially.

## Fresh comparison with Torch

After acceptance, the unchanged four-engine protocol was repeated in a separate
run. There are 72 cells and 43,200 timed calls, with output parity checked before
each cell. The candidate binary, saved weights and input hashes match the paired
experiment. Python uses the rebuilt native wheel.

One window per call:

| Implementation | Median of cell p50s (ms) | Cell p50 range (ms) | Cell p99 range (ms) | Peak RSS range (MiB) |
| --- | ---: | ---: | ---: | ---: |
| C++ core | 1.714 | 1.633–1.777 | 2.028–2.379 | 1.469–1.500 |
| C++ through Python | 1.748 | 1.673–1.806 | 2.107–2.375 | 26.691–26.883 |
| Torch strict reference | 1.224 | 1.177–1.294 | 1.536–1.861 | 157.152–157.930 |
| Torch ordinary eager | 0.491 | 0.459–0.502 | 0.619–0.727 | 157.137–158.617 |

Eight windows per call:

| Implementation | Median of cell p50s (ms) | Cell p50 range (ms) | Cell p99 range (ms) | Peak RSS range (MiB) |
| --- | ---: | ---: | ---: | ---: |
| C++ core | 13.536 | 13.111–14.292 | 14.525–15.776 | 1.453–1.480 |
| C++ through Python | 13.667 | 13.400–14.552 | 14.997–18.643 | 26.855–27.086 |
| Torch strict reference | 8.542 | 7.112–8.923 | 8.359–12.206 | 168.129–168.594 |
| Torch ordinary eager | 1.922 | 1.877–2.087 | 2.218–2.577 | 158.941–159.957 |

Ordinary eager Torch's median paired p50 advantage is 3.53 times for
one window and 6.96 times for eight. Torch's batched tensor kernels
remain a substantial advantage over the native sequential batch loop. Both tables
retain all trial tails. The largest absolute checked difference against the Torch
reference remains 1.9073486328e-06 temperature units.

RSS is the operating system's process lifetime peak resident-memory counter,
including interpreter and library overhead. It does not measure incremental model
cost in an existing service. Native owned buffers remain 204,604 parameter bytes,
24,576 position bytes and 204,576 workspace bytes. The tile adds 64 bytes of
bounded temporary accumulator storage, plus a selector in the workspace object;
it adds no heap workspace or hot-path allocations. Python also owns its input
copy and allocates returned arrays. These are distinct memory scopes.

## Verification and reproducibility

Paired measurement commit: `9f67f1719033a48c808de57af39540ffe2b0e968`.
Four-engine commit: `7c44028ccf6621a98950d602f413b6851cba6d6e`. The intervening change only
corrects test fill literals from integer zero to float zero so strict Windows
compilation succeeds; the native binary hash is identical in both experiments.
Intel Core i5-10600, x86_64, macOS 14.8.7, AppleClang 16.0.0.16000026,
Release `-O3` with native `-ffp-contract=off`. One inference thread; Python 3.10.6,
NumPy 1.26.4, Torch 2.2.2. Current Torch versions in CI test correctness, not speed.

Debug and Release each pass 53 native groups; sanitizers pass 52. Torch passes
46 tests, the installed wrapper 22, runtime contracts nine, and profile/gate
contracts three. Both kernels have zero-allocation coverage. Tile boundaries,
multiple heads, maximum dimensions, causality, invalid input and unchanged output
on failure are tested. A deliberate wrong-channel mutation failed before the
correct implementation was restored. No tolerance was relaxed. Cross-platform
correctness and package results are bound to exact commits in
[verification evidence](evidence/attention-tiling-v1/verification.json).

The audit verifies every local file hash, all 108 unique timing cells, every
latency and clock statistic from raw samples, consumed-output checksums, profile
counts and matching native binaries/weights/inputs. Aggregate reports regenerate
byte-identically. [Paired records](evidence/attention-tiling-v1/paired.json) and
[four-engine records](evidence/attention-tiling-v1/runtime.json) retain every
seed, trial, percentile, first call, setup timing, memory scope and binding hash.
Raw profiler traces, data, weights and prediction arrays remain local.

There is no affinity, frequency or cache control, sustained load, concurrent
stream, network traffic, full cold-start measurement or hard deadline guarantee.
The reserved source tail and QTS data were not used. No training or forecasting
objective changed, and the earlier accuracy result still favours persistence.

Next, profile the improved engine before choosing another kernel change. A
current-platform comparison, compiled Torch, an independent inference runtime and
batch-aware native computation remain open work, followed by separately scoped
streaming and QTS integration gates.
