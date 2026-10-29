# Stealth

Portable C++20 CPU forecasting research project. Stealth is a working name.
The native library provides checked numeric components, causal attention and a
complete one-block point forecaster with a versioned weight-file loader. It has
scalar and packed layouts, component/integration tests and an installable CMake
package. The first task uses 96 ETTm1 readings to predict one temperature an hour
ahead. The optional Torch reference, tested training/checkpoint workflow and native
export are implemented; see [the training guide](docs/training-v1.md).

Start with the [complete model walkthrough and runnable example](docs/forecast-v1.md).
The [attention walkthrough](docs/attention-v1.md) and
[performance guide](docs/performance.md) explain the underlying components.
Independent diagnostic and framework parity tests are in place. Public-data
learning results are [recorded here](docs/training-results-v1.md); the simple
persistence baseline beats all three first-run models. Comparative latency remains
a separate experiment. The optional
Python wrapper and bounded raw-window baseline are implemented.
The CLI reports build metadata and inspects model files.

## Start locally

Install a C++20 compiler, CMake 3.25+ and Ninja 1.11+ using [the setup guide](docs/development.md).
Then, from this repository:

```sh
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
./build/dev/stealth --build-info
./build/dev/stealth_attention_demo
./build/dev/stealth_forecast_demo
```

On Windows, use a Visual Studio developer shell and append `.exe` to executable names.
Open the repository folder in VS Code to use the checked-in CMake tasks and debugger.

## Build modes

- `dev`: debugging, warnings and tests.
- `release`: optimised build for measurement, with tests.
- `sanitize`: address/undefined-behavior checks on GCC or Clang, macOS/Linux.

For any mode, configure, build and test with its preset name. Compilation outputs
stay under `build/` in this checkout. There are no automatic dependency downloads,
Python imports, GPU dependencies or machine-specific CPU instruction flags.

## Install for another native project

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
cmake --install build/release --prefix "$PWD/install/release"
```

PowerShell equivalent for the last step:

```powershell
cmake --install build/release --prefix "$PWD/install/release"
```

A consumer uses `find_package(stealth CONFIG REQUIRED)` and links `stealth::core`.
Set `CMAKE_PREFIX_PATH` to the installed prefix. The test suite installs and moves
this package before building a separate consumer, including a path containing spaces.

Source and build instructions are portable; compiled binaries must match the
operating system, architecture and C++ runtime. Rebuild on each device. The CI
matrix checks Linux GCC/Clang, Windows MSVC, and Intel/ARM macOS.

## Project layout

```text
include/stealth/  Public C++ API
src/             Native implementation
app/             Command-line program
examples/        Runnable attention and forecast diagnostics
cmake/           Version and package configuration
tests/           Component, integration, allocation and installation tests
bench/           Optional native latency benchmark
tools/           Optional fixture, benchmark and diagnostic tools
docs/            Contracts, measured evidence, setup and work queue
```

[Numeric contract](docs/numeric-contract.md) · [Performance](docs/performance.md)
· [Development guide](docs/development.md) · [Architecture and Python interface](docs/architecture.md)
· [Tasks](docs/tasks.md) · [Test-first plan](docs/testing.md)

The repository is public for review. No redistribution licence has been selected.

## Public-data benchmark preparation

The first learning dataset is ETTm1: electricity loads and equipment temperature
at 15-minute intervals. A pinned-source downloader, strict data checks,
training-only scaling and chronological window tests are implemented. The user
selected a one-hour point target and one block first. The native implementation
is checked against independent equations and the Torch reference. The frozen
training recipe is in `benchmarks/ettm1/training_v1.json`. See the
[plain-language guide](docs/public-benchmark.md) and
[data contract](docs/benchmark-contract.md). The
[first training result](docs/training-results-v1.md) and
[follow-up diagnosis](docs/forecast-diagnostics-results-v1.md), and
[correction-model comparison](docs/correction-results-v1.md) retain all outcomes: persistence still wins on test MAE. No forecast or runtime advantage is
claimed from the completed correctness work.

## Optional Python and raw-window interface

The `stealth_cpu` package now calls the same native components and forecaster.
It includes owned prediction sessions, a bounded sequential batch interface and
a raw-window stream with reset and failure preservation. See the
[installation and component guide](docs/python-stream-v1.md) and
[runnable Python example](examples/python_components.py). The native CMake build
remains independent of Python. The optional [training workflow](docs/training-v1.md)
exports learned weights into this engine. The [first runtime comparison](docs/runtime-results-v1.md) is complete: ordinary
eager Torch is faster locally, while the standalone native engine uses less
process memory. The [first profiled attention optimization](docs/attention-results-v1.md)
reduces native full-model latency by 10.36% in a paired local experiment, with
byte-identical native predictions and unchanged owned heap workspace. Torch
remains faster in the fresh comparison. Cached streaming and current-platform
deployment comparisons remain upcoming work.


The [four-row matrix execution follow-up](docs/dense-rows-results-v1.md) reuses
packed weights across observations and integrates bounded tiles into the MLP and
transformer. It reduces paired native latency a further 12.98%, preserves checked
predictions exactly and adds 17.25 KiB of preallocated workspace. The failed first
candidate is retained. [Current-runtime comparison requirements](docs/runtime-frontier.md)
set the next benchmark target; leading-runtime parity has not been achieved.
