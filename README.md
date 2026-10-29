# Stealth

Portable C++20 CPU forecasting research project, with optional Python inference
and Torch training. The permanent checkout contains the complete one-block model,
export/load workflow and tested numerical components. Training runs in Torch;
inference can run through the C++ library or its NumPy interface without Torch.

Start with [the consolidated baseline and review guide](docs/review-baseline.md).
[PR #18](https://github.com/ArjunNair04/stealth/pull/18) preserves and brings together
the twelve implementation PRs. The [active work queue](docs/tasks.md) separates
completed engineering from the remaining performance and forecasting decisions.
[Direction and exit gates](docs/direction.md) set the path from mixed-frequency
observations to one trustworthy prediction pipeline.

| What works | Current limit |
| --- | --- |
| Portable native builds, installation and Python packaging | Rebuild binaries for each device; the 0.x ABI is provisional |
| Dense/MLP, multihead attention, layer norm, exact GELU and one-block forecasts | This is a specific inference architecture; dropout and new depth are not implemented |
| Torch training, checkpoints, native export and checked loading | Saved ETTm1 models still lose to the last-value baseline on primary old-test error |
| Full traces, latest-only sessions, sequential batches and bounded raw streams | Streams recompute the full window; market-clock integration remains open |
| Measured native optimizations | Latest-only local p50: 0.181 ms C++ versus 0.287 ms eager Torch 2.2.2; Torch wins at batch eight |

The local latency result covers one Intel Mac and the saved one-block workload.
Current compiled and independent runtime comparisons, sustained-load testing,
research novelty and financial usefulness remain unestablished. See
[all latest-only measurements](docs/latest-results-v1.md) and
[all correction-model outcomes](docs/correction-results-v1.md).

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

## Run and evaluate

- [Mixed-frequency signal timing reference](docs/signal-timeline-v1.md)

- [Model walkthrough and native example](docs/forecast-v1.md)
- [Python installation and component example](docs/python-stream-v1.md)
- [Latest-only session API](docs/latest-forecast-v1.md)
- [Pinned public-data preparation](docs/public-benchmark.md)
- [Training, restart and native export](docs/training-v1.md)
- [Current-runtime comparison requirements](docs/runtime-frontier.md)
- [Linux comparison preparation](docs/frontier-linux.md)

Use `stealth --inspect-model path/to/model.stlf` to inspect the validated format
version, output mode and dimensions. Before benchmarking an installed Python
wheel, run `python tools/build_identity.py --check-extension`: stale or non-Release
builds are rejected. Rebuild native consumers and wheels after updating this
baseline because build metadata adds a public field; model files are unchanged.
