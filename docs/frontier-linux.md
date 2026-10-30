# Prepare the next CPU comparison

Status: environment and experiment specification, with [implemented adapters and
bounded correctness admission](runtime-adapters-v1.md). Real Inductor/ONNX fixture
and deployed-shape parity has passed Linux CI. No current-runtime acceptance
timings are claimed. A quiet machine, full host lock and timed orchestration
are still required. The current Intel Mac benchmark
remains a separate Torch 2.2.2 compatibility result.

The default preparation target is Linux x86_64, Python 3.12, with a quiet CPU
measurement window. Apple silicon can be a separately recorded host choice.
Every competing engine must run on the same host; comparisons between machines
cannot establish a software advantage. Shared CI verifies behavior and packaging,
not the final latency claim.

## Candidate package set and setup

As checked on 29 October 2026, PyTorch publishes 2.15.0 and its official CPU index
contains a CPython 3.12 Linux x86_64 wheel. ONNX Runtime publishes 1.31.1 (Python
3.11+); ONNX publishes 1.23.2. These are candidate pins awaiting a complete host
lock and compatibility checks, not packages installed or validated on this Mac.
Primary sources: [Torch release](https://pypi.org/project/torch/2.15.0/),
[official CPU wheels](https://download.pytorch.org/whl/cpu/torch/),
[ONNX Runtime release](https://pypi.org/project/onnxruntime/1.31.1/),
[ONNX release](https://pypi.org/project/onnx/1.23.2/).

Use a permanent checkout on the selected host. Install the compiler, CMake, Ninja
and Python venv prerequisites through the host's normal administration process.
The explicit opt-in Python setup below is prepared for that host; it has not been
executed here and is not yet a complete reproducible lock:

```sh
python3.12 -m venv local/frontier-env
local/frontier-env/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0+cpu'
local/frontier-env/bin/python -m pip install -r requirements-frontier.txt
local/frontier-env/bin/python -m pip install --no-build-isolation --no-deps .
local/frontier-env/bin/python -m pip check
local/frontier-env/bin/python tools/build_identity.py --check-extension
local/frontier-env/bin/python -m pip freeze --all > local/frontier-environment.txt
cmake --preset release -DSTEALTH_BUILD_BENCHMARKS=ON
cmake --build --preset release
ctest --preset release
local/frontier-env/bin/python -m unittest discover -s tests/python -v
local/frontier-env/bin/python -m unittest discover -s tests/training -v
```

Resolve and record any exporter/compiler helper dependencies explicitly during
admission. Archive all resolved wheel hashes and versions, CPU/ISA, OS/kernel,
Python patch, compiler and flags, build identities, CPU affinity, power policy,
thread environment and graph settings. Commit that host lock before collecting
acceptance timings. A package conflict is a recorded environment outcome; do not
silently fall back to a different version or install unrelated dependencies.
Saved models and the fixed input bundle remain local. Transfer only those needed
artifacts after checking their hashes against the protocol; no QTS data is needed.

## One question and fixed acceptance rules

Can the existing native engine approach current optimized runtimes for the
requested latest forecast without changing its answer?

The [machine-readable specification](../benchmarks/ettm1/frontier_v1.json) fixes
three saved models, 32 validation windows and three trials. Latest-only batch one
is primary; batch eight is secondary. Five engines produce 90 cells and 54,000
timed calls. Keep the same scaler, weights, positions, attention, exact GELU,
output correction and numerical tolerances. No new training or reserved-tail use.

All engines include normal API work and finite-input/output checks in the measured
call. Prepare resident input buffers before timing. Compile only the pure tensor
graph; put input/output validation around it, inside the measured call. Otherwise
Python boolean validation can break compilation and quietly leave eager work.
Require full-graph compilation with no silent fallback, fixed shapes and inference
mode. Record compile/setup and first-call costs separately. The
[PyTorch compile contract](https://docs.pytorch.org/docs/2.15/generated/torch.compile.html)
describes full-graph behavior; successful compilation and numerical parity still
need verification for this model.

ONNX uses the CPU provider, sequential execution, full graph optimization, one
intra-op/inter-op thread and disabled spinning. Exported/optimized graph hashes
and conversion parity are required. [ONNX Runtime thread controls](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
explain why default thread settings must be overridden. Retain unsupported
operators or export failures as outcomes rather than omitting that competitor.

For each batch separately, compare the native median of nine cell p50s and p99s
to the smallest corresponding baseline medians. Both ratios must be <=1.20 to
meet the working target. Current eager Torch, compiled Torch and ONNX CPU must
all complete with parity before a complete comparison can be claimed. Report
Python-wrapper cost, memory, setup, every trial and worst observed calls alongside
the aggregate; percentile samples are not hard deadlines or confidence bounds.

## Budget and next implementation boundary

Allow one acceptance run after correctness admission, a 180-second compile limit
per cell and a 45-minute total run cap. Preserve failures and partial runs. A poor
performance result is not a reason to rerun or tune until it becomes favourable.
A concrete correctness/environment defect can justify a separately recorded fix
and new run. Change the budget in a new protocol before running if admission
shows it is insufficient.

The consolidated baseline provides existing native/eager execution and source
identity checks. The compiled/ONNX adapters and bounded per-cell correctness runner are now
implemented. The host lock, fixed saved-model admission and timed comparison remain
the next implementation milestone. Do not describe the existing compatibility
runner as supporting these engines. If current runtimes remove the local advantage,
use that evidence to decide whether integration, memory or specialized workloads
justify further custom code before another kernel optimization.
