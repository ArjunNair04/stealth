# Compare the CPU runtimes

We want to answer a practical question: is the custom C++ engine worth keeping
when the same model can run through current PyTorch and ONNX Runtime?

The comparison runner is implemented. The controlled Linux run has not happened.
Local tests exercise the native and eager PyTorch workers; the optional Linux CI
job exercises real compilation and ONNX export. CI timings are test diagnostics,
not performance results.

## What is held fixed

The [original protocol](../benchmarks/ettm1/frontier_v1.json) is unchanged: three
saved models, 32 input windows, batches of one and eight, and three trials. Five
engines give 90 cells and 54,000 timed calls:

- Native C++ directly.
- Native C++ through Python.
- Eager PyTorch.
- PyTorch compiled with Inductor.
- ONNX Runtime on CPU.

Every engine checks every supplied window against the saved independent reference
before warming up or timing. The tolerance remains `5e-5 + 2e-4 * abs(reference)`.
There is no training, data download, kernel tuning or use of the reserved test tail.
Each cell gets a fresh process; execution order rotates. Inputs are resident before
timing, while normal API work and finite-value checks remain inside each call.

For each batch size, the report takes the median of nine cell medians and the
median of nine cell p99 values. It compares native C++ with the fastest baseline
at each percentile. Both ratios must be at most 1.20 to meet the working target.
The Python wrapper is reported separately. Missing or failed cells leave the
comparison incomplete and produce no pass/fail speed verdict.

## What the scripts do

| Script | Purpose |
| --- | --- |
| `tools/benchmark_frontier.py` | Run admission, create the host lock, schedule the fixed comparison and verify its files/results |
| `tools/frontier_worker.py` | Check and time one eager, compiled or ONNX cell |
| `tools/frontier_host.py` | Check the Linux environment, build identities, affinity, packages and archived wheel hashes |
| `tools/runtime_worker.py` | Existing Python wrapper worker |
| `stealth_runtime_bench` | Existing native benchmark executable |

A failed worker leaves its stdout, stderr, process status and record on disk; other
competitors can still finish. Compiler child processes are stopped after success,
failure or timeout, so they cannot keep working during the next cell. Each worker has a 180-second wall limit, including
imports, compilation, accuracy checks and timing. This is stricter than the
protocol's 180-second compilation allowance. The measurement matrix has a
45-minute limit. Remaining cells are recorded as not run if it expires.

The runner never retries a failed or slow result. It creates one acceptance claim
beside the admission directory, so requesting another output directory does not
silently start a second attempt. A demonstrated environment or correctness defect
needs a separately documented decision before another admission/run.

## Prepare the Linux environment

Use the [Linux setup specification](frontier-linux.md) for the pinned package set.
Keep Stealth in its own checkout and environment. A scheduler allocation must
provide a quiet CPU window; record that allocation when creating the lock. The
runner checks affinity, but cannot prove that another user or the hypervisor is
not competing for the same physical core.

Archive the exact wheels used to install **every** package in the environment,
including pip, build helpers and the locally built Stealth wheel. The lock reads
their package metadata and hashes the complete files. Missing, duplicate or
wrong-version wheels fail the check. These scripts install nothing.

Build the release native worker and install the matching wheel. Run the normal
native, Python and reference suites. Confirm that `tools/build_identity.py
--check-extension` passes. Prepare the saved models and reference files using the
existing experiment artifacts; the runner checks their frozen hashes.

Commit the runner and protocol before admission. The following commands run inside
the prepared environment, pinned to a CPU assigned by the scheduler. Replace
`<allocated-cpu>` with that CPU number; it is not necessarily CPU 0.

```sh
taskset -c <allocated-cpu> local/frontier-env/bin/python tools/benchmark_frontier.py admit \
  --runs local/ettm1-correction-v1 \
  --references local/runtime-latest-v1 \
  --output local/frontier-admission-v1
```

Admission checks 30 model/batch/engine combinations. Its one-call timing diagnostics
are not the acceptance sample. All 30 must pass before a host lock can be written.

```sh
taskset -c <allocated-cpu> local/frontier-env/bin/python tools/benchmark_frontier.py lock \
  --admission local/frontier-admission-v1 \
  --wheels local/frontier-wheelhouse \
  --allocation 'Describe the reserved CPU window and any known sharing' \
  --output docs/evidence/frontier-host-v1.json
```

Inspect the lock before committing it, particularly compiler commands and inherited
environment settings that could contain private paths. The lock records the CPU,
OS, Python, compiler flags, package and binary hashes, source identity, affinity,
power policy, thread settings and admitted ONNX graph hashes. No credentials or
private cluster identifiers belong in a public report. Commit the reviewed lock;
changing the environment afterwards requires admission again.

## Run once and verify

```sh
taskset -c <allocated-cpu> local/frontier-env/bin/python tools/benchmark_frontier.py run \
  --admission local/frontier-admission-v1 \
  --host-lock docs/evidence/frontier-host-v1.json \
  --output local/frontier-measurement-v1

local/frontier-env/bin/python tools/benchmark_frontier.py verify \
  local/frontier-measurement-v1
```

The run checks the host against the committed lock before and after measurement.
Each ONNX export must match its admitted graph hashes. Outputs are new directories;
existing results are never overwritten. The final receipt hashes the inputs, raw
records, logs and exported graphs. Compiler caches are explicitly excluded from
that receipt. Verification checks the files and recalculates the aggregate from
raw timing samples.

Report every trial, setup/first-call cost, memory high-water mark, wrapper cost and
worst call alongside the median/p99 comparison. Percentiles are observations, not
hard latency deadlines. If a standard runtime wins, the next decision is whether
to use it behind the existing feature and replay interfaces.

## Verification record

The [validation record](evidence/frontier-runner-v1/verification.json) binds the
implementation revision, local test logs and 18 saved-model checks. All 100 offline
tests and 36 installed-wrapper tests pass; the reference/training suite passes
118 of 123 tests, with five optional Linux checks skipped locally. Real Inductor,
ONNX, graph binding and host-probe tests pass in the linked Linux CI jobs.
The saved-model checks use native C++, its Python wrapper and local eager Torch
2.2.2. Their maximum absolute reference difference is `1.9073486328125e-6`.

A regression test reproduced compiler children surviving both successful and
failed worker exits. The supervisor now stops the whole process group on every
outcome. The record retains that failure, the temporary duplicate-cleanup test
error on macOS, and the stale-wheel rejection. None of these correctness runs
is the controlled latency comparison.
