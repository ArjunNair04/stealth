# Stealth

Portable C++20 development foundation for a CPU forecasting research project.
Stealth is a working name. The current release provides a native library, a CLI,
a reproducible build, tests and an installable CMake package. Forecasting and a
Python wrapper are future milestones; no performance or trading claim is made.

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
- `release`: optimised build for later measurement, with tests.
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
tests/          Native build and installation tests
tools/          Optional developer diagnostics
docs/           Setup, architecture and work queue
```

[Development guide](docs/development.md) · [Architecture and Python plan](docs/architecture.md)
· [Tasks](docs/tasks.md)

The repository is private. No redistribution licence has been selected.
