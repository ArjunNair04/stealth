# Python interface and bounded streaming

Python can now call the same checked C++ components and complete forecaster. A
stream retains the latest raw observations and calculates the same full window
as an ordinary prediction. This phase does not train a model or change its design.

## What each piece does

| Piece | Purpose |
| --- | --- |
| `bindings/module.cpp` | Checks NumPy inputs and calls the existing native layers. |
| `python/stealth_cpu/` | Public Python imports and package documentation. |
| `pyproject.toml`, `requirements-python-dev.txt` | Optional package build and declared, pinned development dependencies. |
| `include/stealth/window_stream.hpp`, `src/window_stream.cpp` | Retain bounded history, stage a new row, commit it only after a successful calculation. |
| `tools/check_python_package.py` | Build a source archive, build its wheel, audit both, and test an offline installation in a fresh environment. |
| `tests/python/test_bindings.py` | Independent answers, strict buffers, ownership, batching, exceptions and concurrent sessions. |
| `tests/window_stream_tests.cpp` | Sliding windows, warmup, reset, rejected rows, lifetime, bounds and shared-model concurrency. |
| `examples/python_components.py` | Hand-checkable dense, forecast, batch and stream example. |

The detailed guarantees are in [the contract](python-stream-contract.md).

## Install locally

From the permanent checkout, create an isolated environment and install the
explicit build requirements, then the package:

```sh
python3 -m venv local/python-env
local/python-env/bin/python -m pip install -r requirements-python-dev.txt
local/python-env/bin/python -m pip install --no-build-isolation .
local/python-env/bin/python examples/python_components.py
```

On Windows use `local/python-env/Scripts/python.exe` and `python` instead of
`python3`. The native compiler, CMake and Ninja must already be available as
described in the main setup guide. No build command silently installs Torch. The
CMake Python option defaults to OFF, so native-only builds remain unchanged.
The package is built locally; no PyPI publication or portable wheel distribution
is claimed. A wheel targets its specific Python, OS and CPU architecture.

The installed development environment used for this milestone is under
`local/python-infra-env/`. Test-only isolated installations are under
`local/python-package-*/runtime/`. These remain ignored, and can be recreated.

## Use a model

```python
import numpy as np
from stealth_cpu import ForecastModel

model = ForecastModel.load("model.stlf")
session = model.session()
# rows must be an aligned, C-contiguous, native float32 NumPy array [time,features].
rows = np.asarray(your_observations, dtype=np.float32, order="C")
trace = session.predict(rows)
forecast = trace[-1]

# Each member is an independent sequence; this API evaluates members sequentially.
batch_trace = session.predict_batch(np.stack([rows, rows]))

stream = model.window_stream()
for row in rows:
    value = stream.push(row)  # None until a full window; then one float.
```

The diagnostic prefix outputs are point forecasts at successive historical
origins. They are not a future temperature curve. The actual benchmark uses the
last output of a complete 96-row input. Loading a numerical file alone does not
establish its feature order, training provenance or predictive usefulness.

Reusable interfaces also expose `Dense`, `FeedForward`, `gelu`, `layer_norm`,
`softmax`, `residual_add` and `causal_attention`. FeedForward retains its defined
prenormalization and residual behavior. An arbitrary model builder, trained
exporter and automatic differentiation are not supplied by this wrapper.

## Memory and concurrency

Parameters are copied and retained by the native model. Sessions/streams retain
model ownership even if the original Python model variable is deleted. They
never return views into reusable internal workspaces. Inputs are not modified;
readonly inputs work. Wrong dtypes, non-contiguous/unaligned buffers, invalid
shapes and non-finite inputs are rejected explicitly. Caller mutation of an input
while its call is active is unsupported.

Single-sequence and stream calls snapshot input into preallocated native storage.
Batches allocate a bounded snapshot for that call. Results are fresh owned NumPy
arrays; stream results are Python floats. Python object/output allocation and
input copying are explicit costs. Only the native stream/forward operations carry
the tested no-C++-heap-allocation guarantee.

A session or stream permits one active operation. Overlapping calls are rejected;
separate sessions/streams may run concurrently on an immutable shared model.
The GIL is released only during forecast/stream native execution. Primitive
convenience calls keep the GIL and allocate workspaces per call. No claim is made
for free-threaded Python or concurrent external NumPy writes.

## Streaming semantics

A three-row example produces `None, None, prediction`. The next accepted row
replaces the oldest. Every window resets position markers to zero, matching the
frozen forecaster. Invalid input or a numerical failure leaves the accepted
history and output unchanged. Reset empties logical history. This implementation
recalculates the window; it is a correctness baseline for later cache experiments.

Separate raw streams are supported. Clock alignment, missing observations,
revisions, asynchronous feeds and trade decisions remain future adapter work.

## Evidence and remaining gates

Six native stream groups first failed against an explicit implementation stub.
Python tests first failed because the extension did not exist; installed tests
then exercised real bindings and independent reference equations. The complete
native Debug/Release suites, memory sanitizers, a dedicated thread-sanitizer run,
installed consumers and isolated source-to-wheel installation passed locally:
50 Debug groups, 50 Release groups, 49 sanitizer groups, 22 installed Python tests
and 23 public-data tests. A 5,000-cycle stream/error/reset run performed zero
intercepted C++ heap allocations; all six stream-construction allocation failures
preserved an existing usable stream. Compact evidence is recorded in
`docs/evidence/python-stream-v1/verification.json`.

Deliberately retaining the wrong old rows and accepting the wrong dtype must be
caught by their focused tests, followed by restoration and successful reruns.
The existing full-model tolerance remains 5e-5 absolute plus 2e-4 relative; no
expected answers or tolerances were loosened.

Next: establish pinned PyTorch framework parity with untrained diagnostic weights,
then freeze and run the ETTm1 training/export protocol. Compare trained native
predictions, then benchmark matching workloads against established runtimes.
Training configuration/RNG/checkpoint tooling, trained artifact provenance,
caching, model-swap orchestration and market-feed adapters remain separate work.
