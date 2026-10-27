# Development setup

The source lives in your normal local checkout. Build products and local reports
also stay there. No Codex runtime or temporary directory is required.

## Requirements

- CMake 3.25 or newer (presets schema 6).
- Ninja 1.11 or newer.
- A C++20 compiler and standard library. Recommended baseline: Apple Clang 16+,
  GCC 12+, Clang 16+, or Visual Studio 2022 with Desktop development with C++.
- Git. Python 3.10+ is optional for `tools/doctor.py`; native builds do not need it.

## macOS, Intel or Apple Silicon

Install Apple's Command Line Tools if `xcode-select -p` fails:

```sh
xcode-select --install
```

With an existing Homebrew installation:

```sh
brew install cmake ninja
```

Use `command -v cmake` and `command -v ninja` to check your shell path. Homebrew
normally uses `/usr/local` on Intel and `/opt/homebrew` on Apple Silicon. Neither
prefix is baked into the shared build. Official CMake installers are another option:
https://cmake.org/download/

The local setup verified on 2026-10-27 has macOS 14.8.7 x86_64, Apple Clang 16.0.0,
Homebrew LLVM 22.1.7, CMake 4.4.3 and Ninja 1.13.2. See the ignored `local/` folder
for verification output from this computer. Homebrew warns that this macOS release
is outside its supported tier; CMake and Ninja nevertheless installed successfully.

If multiple compilers are installed, choose deliberately. CMake caches a compiler
on first configuration. Use a new build directory to change it. For example:

```sh
cmake -S . -B build/apple-debug -G Ninja -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_BUILD_TYPE=Debug
cmake --build build/apple-debug
ctest --test-dir build/apple-debug --output-on-failure
```

`CMakeUserPresets.json` is ignored and can hold personal compiler paths. The shared
`CMakePresets.json` never holds absolute paths to a developer's machine.

## Linux (Ubuntu/Debian)

```sh
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build
```

Check the resulting versions; older distributions may require a newer CMake from
Kitware or a current compiler. Select Clang with `CXX=clang++ cmake --preset dev`
on the first configuration, or create a separate build directory.

## Windows

Install Visual Studio 2022 Build Tools with the **Desktop development with C++**
workload, including MSVC and the Windows SDK. Install CMake and Ninja and add them
to PATH. Use **Developer PowerShell for VS 2022** (or x64 Native Tools Command
Prompt), so the compiler, linker and SDK environment are available.

```powershell
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
.\build\dev\stealth.exe --build-info
```

Alternatively, CMake's Visual Studio generator can configure without a developer shell:

```powershell
cmake -S . -B build/vs -G "Visual Studio 17 2022" -A x64
cmake --build build/vs --config Debug
ctest --test-dir build/vs -C Debug --output-on-failure
```

## Daily workflow

```sh
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

Use `release` for measurement. Use `sanitize` to investigate memory and undefined
behavior errors; those builds are slower and do not represent production latency.
On macOS, leak detection availability depends on the sanitizer runtime.

The core library contains no third-party dependencies. Do not silently install
packages or download code during configuration. Future dependencies must be
explicit, versioned, and justified in review. Do not enable fast-math or
`-march=native` in the portable reference build.

## VS Code

Recommended extensions are listed in `.vscode/extensions.json`. CMake Tools selects
presets and supplies IntelliSense configuration to the C/C++ extension. The default
build task configures and builds `dev`. The test task builds before running CTest.
Choose **Stealth native (macOS/Linux)** or **Stealth native (Windows)** in Run and Debug.
CodeLLDB supplies the macOS/Linux debugger; the C/C++ extension supplies Windows debugging.
The launch task passes `--build-info`; set a breakpoint in `app/main.cpp`.

The editor configuration is scoped to this project and does not change your global
compiler or shell settings. Formatting uses the installed `clang-format` on PATH.

## Diagnostics and installation

```sh
python3 tools/doctor.py
python3 tools/doctor.py --json
cmake --install build/release --prefix "$PWD/install/release"
./install/release/bin/stealth --build-info
```

The diagnostic command reports tools and the currently selected shell compiler;
it does not install or modify anything. The program's `--build-info` identifies
the compiler actually used for that build.

CI verifies clean builds on Linux, Windows and both Mac architectures. A platform
is verified only when its job passes; the mere presence of a workflow is not proof.
The install test checks that a relocated package can be consumed through
`find_package`, without relying on source-tree headers or paths.

## References

- CMake presets: https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html
- CMake compile requirements: https://cmake.org/cmake/help/latest/command/target_compile_features.html
- Ninja: https://ninja-build.org/
- CMake installation: https://cmake.org/download/
