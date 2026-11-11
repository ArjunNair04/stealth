# Stealth

Stealth is a small C++20 engine for running time-series forecasts on a CPU.
Train a model in PyTorch, export its weights, then run it through C++ or a thin
NumPy interface. Inference does not need PyTorch or a GPU.

The longer-term aim is to combine observations that arrive at different speeds:
for example, minute bars alongside daily and monthly factors. The project already
has an offline feature index that keeps track of when each observation became
available and how it was revised. Live feeds and trading deployment are future work.

## What works today

- A one-block transformer with multihead attention, layer normalization, exact
  GELU, an MLP and a forecast head.
- PyTorch training, checkpoints and export to a checked native model format.
- Single forecasts, batches and bounded streams, through C++ or Python.
- Reusable native workspaces, packed weights and an AVX path selected at runtime.
- Tests for arithmetic, invalid inputs, memory allocation, ownership, replay and
  package installation on Linux, Windows and Intel/ARM macOS.

This is a deliberately small inference engine. It does not yet offer arbitrary
model architectures, quantized weights or automatic multicore scheduling. Streams
reuse storage but recompute the current window. See [the architecture](docs/architecture.md)
for the supported interfaces and [the tests](docs/testing.md) for their guarantees.

## What we have measured

The latest completed Linux study used three saved ETTm1 models on one Xeon CPU,
with one inference thread. Lower times are better.

| Measurement | Result |
| --- | --- |
| Eight forecasts: previous native code → updated code | 1.797 ms → 1.584 ms; **11.84% lower median latency** |
| Eight forecasts: slow-call latency (p99) | **12.20% lower** in the paired comparison |
| Eight forecasts: updated native vs eager PyTorch 2.6 | 1.584 ms vs 1.708 ms |
| Sixteen forecasts: native vs eager PyTorch 2.6 | 3.170 ms vs 2.788 ms; PyTorch is faster |

These results describe that workload and machine. The change uses more scratch
memory; single-forecast latency is essentially unchanged. The
[full report](docs/native-batching-v1.md) includes all trials, failed candidates,
assembly findings and a script to verify the published evidence.

The next comparison adds compiled PyTorch and ONNX Runtime. Its
[runner and fixed experiment](docs/runtime-comparison.md) are implemented; the
controlled Linux measurement is still pending. We do not yet claim an advantage
over those engines.

A faster forecast is not necessarily a better forecast. On the
[bike-rental example](docs/bike-demand-results-v1.md), the model's mean absolute
error was 43.82 rentals, compared with 87.15 for the last observed count and 47.90
for ridge regression. Ridge did better on squared error. The ETTm1 and
[BTC return](docs/btc-return-results-v1.md) studies failed their prediction gates.
No validated trading alpha is claimed.

## Try the native build

You need a C++20 compiler, CMake 3.25+ and Ninja 1.11+.
The [setup guide](docs/development.md) covers each platform.

```sh
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
./build/dev/stealth_forecast_demo
```

On Windows, use a Visual Studio developer shell and add `.exe` to the executable
name. The demo uses fixed example weights to show the calculation; it is not a
trained forecasting service.

Use the `release` preset for measurements and `sanitize` for memory and undefined
behavior checks on GCC/Clang. Build outputs stay in `build/`. Native configuration
does not download dependencies.

## Use it from Python

Follow the [Python setup](docs/python-stream-v1.md) to build the local package.
After [training and exporting a model](docs/training-v1.md):

```python
import numpy as np
from stealth_cpu import ForecastModel

model = ForecastModel.load("model.stlf")
session = model.session()
rows = np.asarray(your_observations, dtype=np.float32, order="C")
forecast = session.predict_last(rows)
```

Each row contains the model's features in their recorded order. A session owns
reusable working memory; separate sessions can share one immutable model. Python
calls still pay for input checks, copying and result allocation.

For a native consumer, install a release build with
`cmake --install build/release --prefix "$PWD/install/release"`, then use
`find_package(stealth CONFIG REQUIRED)` and link `stealth::core`.
Rebuild binaries for each operating system and CPU architecture; the 0.x ABI may change.

## Find your way around

| I want to… | Start here |
| --- | --- |
| Understand the project without reading every research note | [Project guide](docs/index.md) |
| Build, test or contribute | [Development](docs/development.md) · [Testing](docs/testing.md) |
| Understand the model and its inputs | [Model walkthrough](docs/forecast-v1.md) · [Feature timing](docs/signal-timeline-v1.md) |
| Inspect the speed work | [Batching results](docs/native-batching-v1.md) · [How the optimizations work](docs/performance.md) |
| Run the next comparison | [Runtime comparison](docs/runtime-comparison.md) |
| See what comes next | [Direction](docs/direction.md) · [Work queue](docs/tasks.md) |
| Review the implementation history | [Baseline review](docs/review-baseline.md) |

The main code lives in `include/stealth/` and `src/`; Python bindings are in
`bindings/` and `python/`. `examples/` contains runnable demos, `tests/` contains
component and integration checks, and `tools/` contains training and experiment
scripts. Local datasets and generated reports stay under ignored `local/`.

The repository is public for review. A redistribution licence has not yet been selected.
