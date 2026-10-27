# Infrastructure verification

Verified on 2026-10-27. Native source commit: `28b10617713c627e691720fcd5be3a6c14f47bd5`.
This report covers the development foundation, not a forecasting implementation.

## Local Mac

- macOS 14.8.7, Intel x86_64; Apple Clang 16.0.0.
- CMake 4.4.3 and Ninja 1.13.2 installed through the existing Homebrew setup.
- Debug: 5/5 CTest checks passed with warnings treated as errors.
- Release: 5/5 checks passed with warnings treated as errors.
- Address/undefined-behavior sanitizers: 4/4 checks passed.
- Installed release CLI executed successfully.
- Both ordinary build modes installed the library, moved its installation prefix,
  and compiled/executed a separate C++20 consumer against the relocated package.
- LLDB stopped at main with source-level debug information, continued and exited 0.
- VS Code CMake/C++ extensions were already present; CodeLLDB 1.12.3 was installed.

## Clean hosted builds

[GitHub Actions run](https://github.com/ArjunNair04/stealth/actions/runs/40597051569)
completed successfully for all five jobs:

| Environment | Debug | Release | Relocated consumer |
| --- | --- | --- | --- |
| Linux GCC | Pass | Pass | Pass |
| Linux Clang | Pass | Pass | Pass |
| macOS ARM | Pass | Pass | Pass |
| macOS Intel | Pass | Pass | Pass |
| Windows MSVC | Pass | Pass | Pass |

The Linux Clang job also passed sanitizer checks. This establishes source/build
portability for the current starter. It does not establish identical numerical
results for future models or binary compatibility between devices.

The subsequent completion commit changes documentation only. Local machine paths
and diagnostic inventories remain under ignored local/, with build outputs in
build/ and the user-local package in install/.
