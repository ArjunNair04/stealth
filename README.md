# Stealth

Portable C++20 numeric foundation for a CPU forecasting research project.
Stealth is a working name. The native library now provides aligned FP32 storage,
checked dense projections, softmax, layer normalization, GELU, residual addition
and a composed feed-forward block. It includes a scalar reference, packed weight
layout, component/integration tests and an installable CMake package.

See [the first native milestone](docs/native-v1.md) for behavior and test evidence,
and [the performance walkthrough](docs/performance.md) for measured results and
how the optimizations work. This is an early numeric library; a full transformer,
trained forecast, streaming engine and Python wrapper remain future milestones.
The CLI currently reports build metadata.

## Start locally

Install a C++20 compiler, CMake 3.25+ and Ninja 1.11+ using [the setup guide](docs/development.md).
Then, from this repository:

```sh
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
./build/dev/stealth --build-info
```

On Windows, use a Visual Studio developer shell and `./build/dev/stealth.exe`.
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
cmake/           Version and package configuration
tests/           Component, integration, allocation and installation tests
bench/           Optional native latency benchmark
tools/           Optional fixture, benchmark and diagnostic tools
docs/            Contracts, measured evidence, setup and work queue
```

[Numeric contract](docs/numeric-contract.md) · [Performance](docs/performance.md)
· [Development guide](docs/development.md) · [Architecture and Python plan](docs/architecture.md)
· [Tasks](docs/tasks.md) · [Test-first plan](docs/testing.md)

The repository is public for review. No redistribution licence has been selected.
