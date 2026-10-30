# Beaker Linux: correctness passes, single forecasts win, batching loses

Stealth runs on a UCL Beaker CPU compute node without changing application code.
All native, Python and training/reference gates pass after preparing an isolated
compiler environment. For the saved ETTm1 workload, native single forecasts take
206.972 µs versus 502.962 µs in eager PyTorch 2.6: a 2.43× speed ratio. PyTorch
wins at batch eight, taking 928.392 µs versus 1,656.781 µs native.

This is a measured Linux compatibility result on 30 October 2026. It is not the
pending comparison with current compiled Torch and ONNX Runtime. No VM was
needed for these tests. Model training, weights, numerical tolerances and native
kernels are unchanged; no financial data or reserved ETTm1 tail was accessed.

## What passed

| Gate | Result |
| --- | --- |
| Debug native | 67/67 |
| Release native | 67/67 |
| Forced portable implementation, AVX disabled | 67/67 |
| Address and undefined-behavior sanitizers | 66/66 |
| Offline Python contracts | 84/84 |
| Installed native Python interface | 34/34 |
| Torch/training/integration | 114 passed; 3 optional runtime tests skipped |

Native installation and relocated-consumer checks pass in Debug, Release and
portable configurations. The sanitizer configuration intentionally omits that
packaging test, as in the existing test matrix. The optional skips cover the
separate compiler/export dependency environment; they are not Linux passes.

Every measured worker checks all 32 supplied latest forecasts before warmup.
The maximum absolute difference from the saved reference is
`1.9073486328125e-6`, within the unchanged per-value bound
`5e-5 + 2e-4 * abs(reference)`. This does not resolve or supersede the separate
[792k bike-family rejection](bike-size-results-v1.md).

## Same machine, same saved models

Hardware is an Intel Xeon E3-1240 v3 at a nominal 3.40 GHz. The node runs CentOS 7,
Linux 3.10 and glibc 2.17. The build uses GCC 11.2.1, CMake 3.26.4 and Ninja 1.10.0;
this tests that installed Ninja version without changing the general setup
recommendation. Python is 3.11.0, NumPy 2.2.6 and pybind11 3.0.1. The retained Torch
wheel is `2.6.0+cu124`, executed entirely on CPU with one intra-op and inter-op
thread, deterministic algorithms and inference mode.

The original source is commit `a65ba9f229e360cfea92b70438c98c89e19a5416`.
Release core flags include `-O3 -DNDEBUG -std=c++20 -ffp-contract=off`; no fast-math,
global native-CPU flags, quantization or model change was introduced. The native
source fingerprint matches the previously tested Mac implementation.

The three saved ETTm1 correction models use 96 observations, seven features,
width 64, four heads and one block, with a 256-wide MLP. Inputs are the same 32
validation windows used in the earlier latest-only study. Source, models,
reference answers, inputs, compiler, sanitizer libraries and deployed scripts
are hash-bound. Every original source-file byte was checked against the Git
archive again before the final validation run.

The scheduler reserved one physical core; timed workers were pinned to one of
its hardware threads. Threads are fixed to one. The node was shared, frequency
was not locked, and no exclusive-machine or sustained-load guarantee is claimed.
Builds and tests finished before timing. Each cell runs in a fresh serial process,
with engine order rotated across trials, seeds and batch sizes.

## All timing results

There are three seeds × three trials × two batch sizes × four engines: 72 cells
and 43,200 timed calls. Each worker has 64 warmups, then 1,000 batch-one or 200
batch-eight calls. Inputs and weights are resident; model loading and first-call
costs are recorded separately. The requested output is the latest forecast.
Finite-value checks remain inside each normal inference call.

p50 is the middle measured call; p99 is the per-worker 99th percentile. Each table
entry is the median of nine worker quantiles, not a pooled percentile or a
confidence interval. Batch-eight latency is for the complete group of eight.

| Engine | Batch-one p50 | Batch-one p99 | Batch-eight p50 | Batch-eight p99 |
| --- | ---: | ---: | ---: | ---: |
| C++ with guarded AVX | 206.972 µs | 212.387 µs | 1,656.781 µs | 1,707.831 µs |
| C++, AVX disabled | 504.698 µs | 536.181 µs | 4,033.825 µs | 4,126.473 µs |
| Python calling the native core | 205.786 µs | 212.753 µs | 1,638.584 µs | 1,655.752 µs |
| Eager PyTorch 2.6 | 502.962 µs | 523.678 µs | 928.392 µs | 971.009 µs |

The native batch-one p50 range across all nine cells is 206.510–207.992 µs;
Torch's is 494.448–512.396 µs. For batch eight, ranges are
1,653.700–1,662.159 µs and 920.114–940.451 µs respectively. All raw calls and
worker quantiles are retained in the [evidence](evidence/beaker-linux-v1/manifest.json).

The AVX build reduces single-forecast median latency by 58.99% relative to the
forced portable build on this host. This compares two existing implementations,
not a new optimization made during this test. Native batching currently loops
over windows; Torch processes them together. That implementation difference is
a plausible explanation for the batch-eight result, not a measured profile of
its cause. Torch is 1.78× faster than native for that group.

The Python and C++ call paths have similar observed timings. Their small
reversal does not measure negative Python overhead; subtracting independently
compiled call paths does not isolate wrapper cost. The older Mac timings use
different hardware and compilers, and the 53 µs bike demonstration uses a shorter
48-step, five-feature input. Neither establishes a Linux penalty or speedup.

Native weights, positions and reusable workspace total 451,420 bytes, excluding
caller inputs, object/allocator bookkeeping, executable code and runtime memory.
Raw process high-water RSS is retained only as a diagnostic: fork/exec history
can affect that counter, so it is not an isolated inference-memory comparison.

## Environment failures are preserved

The first job stopped before building because a compiler module visible on the
login machine was absent from the compute node. The second correctly rejected
that node's GCC 4.8.5. The third used an isolated copy of the installed GCC 11 and
passed native/Python checks, but lacked the sanitizer link libraries. Its gate
correctly prevented timing.

The repair copied no credentials and changed no cluster system package or
existing Python environment. Exact GCC 11.2.1-9 sanitizer package payloads from
[Oracle's official repository](https://yum.oracle.com/repo/OracleLinux/OL7/SoftwareCollections/x86_64/)
were extracted into the project toolchain directory; versions, URLs and hashes
are [recorded](evidence/beaker-linux-v1/sanitizer-packages.json). Only copied
linker paths were relocated. The final job rechecked all source bytes and
reran every gate before the sole complete timed run. Earlier failures remain
[recorded](evidence/beaker-linux-v1/environment-failures.json); no timing was
rerun because of an unfavorable outcome.

All 169 completion-listed files were mirrored and independently verified.
The local verifier reconstructed the complete 72-cell grid, every raw p50/p95/p99
and all aggregate rows. The scheduler is empty and the account coordination lease
is released. Connection details, machine dumps, package inventory, saved model
files and full logs remain under ignored `local/beaker-linux-v1/`; the public
record contains compact metrics, hashes, the frozen protocol and archived driver.
The archived driver documents this deployment and is not a general cluster installer.

## Next performance gate

Keep single-forecast inference as the measured strength and record batched
execution as an open weakness. Before choosing another optimization, complete
the [current-runtime comparison](frontier-linux.md), including compiled Torch
and ONNX CPU, on one admitted environment. Beaker has Singularity 3.8.5 installed;
a newer container is a candidate, but no container or current-runtime environment
was qualified in this run. The old shared host does not establish a latency
frontier, predictable trading deadlines or production readiness.
