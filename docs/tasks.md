# Work queue

## 1. Local C++ infrastructure (complete)

- [x] Choose a permanent standalone checkout and private repository identity.
- [x] Inspect the installed compiler, SDK, editor and GitHub account.
- [x] Install CMake and Ninja through the existing local package manager.
- [x] Create a C++20 library, CLI, CMake presets and editor configuration.
- [x] Verify debug, release, sanitizer and relocated-install builds locally.
- [x] Push the initial repository and verify the cross-platform CI matrix.

Done means another clean checkout can configure, build, test and install using
only a documented compiler, CMake and Ninja. Record actual passed configurations.

## 2. Native numeric primitives

Define contiguous FP32 buffers, shapes and errors; implement only the operations
needed by the agreed small model. Check mathematical fixtures, malformed shapes,
non-finite input policy and allocation behavior. Keep a clear scalar reference.

## 3. Small reference model and native parity

Freeze model layout and weight serialization, train/export a deterministic
reference, and verify native forward outputs before adding caching. Specify
position, normalization and reset behavior before testing incremental equivalence.

## 4. Optional Python wrapper

After native primitives are stable, add pybind11/scikit-build-core packaging and
isolated-install tests. Verify Python/native parity, buffer lifetime, invalid
inputs and explicitly documented copy behavior. Native builds remain independent.

## 5. CPU performance study and QTS adapter

Measure the same workload across native and established runtimes. Record CPU,
compiler, threads, warm-up, memory and latency distributions. Then define a
separate, preregistered QTS pilot. Do not infer alpha from synthetic correctness.

## GitHub tasks

- [Set up and verify portable local C++ development](https://github.com/ArjunNair04/stealth/issues/1)
- [Implement the first native numeric primitives](https://github.com/ArjunNair04/stealth/issues/2)
- [Establish a small reference model and native forward parity](https://github.com/ArjunNair04/stealth/issues/3)
- [Add an optional Python wrapper after the native layers](https://github.com/ArjunNair04/stealth/issues/4)
- [Measure the CPU performance frontier and define the QTS boundary](https://github.com/ArjunNair04/stealth/issues/5)

See [verification](verification.md) for the completed infrastructure checks.
