# Windows allocation-failure repair, 11 November 2026

The Windows Debug job timed out while deliberately failing a signal-index
construction allocation. Forecast, batching, hot-allocation, model-load and stream
checks had already completed. This failure also appeared on the parent branch;
it was not introduced by the batch execution change.

## Reproduction and cause

Commit `449736907e77ec23801f9fba587c5cd3b59acefd` added flushed phase markers
without reducing any allocation assertions or workload counts. The preserved
[Windows CI excerpt](evidence/windows-allocation-v1/before-ci-excerpt.txt) stops at
allocation 2 of 13 in populated signal-index construction. The test timeout is
now 120 seconds instead of CTest's implicit 1,500 seconds; this bounds a broken
test, not inference latency.

MSVC's Debug standard library allocates iterator proxies even for empty vectors.
Its default vector constructor is `noexcept`; a failed proxy allocation therefore
terminates instead of reaching our `bad_alloc` handler. The
[matching Microsoft vector implementation](https://github.com/microsoft/STL/blob/vs-2022-17.14/stl/inc/vector#L652)
and [proxy allocator](https://github.com/microsoft/STL/blob/vs-2022-17.14/stl/inc/xmemory#L1198)
explain the observed boundary. This is a library-specific failure mode; the
same allocation test had passed on Linux and macOS.

## Repair

`SignalIndex` uses the zero-count vector constructor, which permits exceptions,
for its two outer members. It reserves channel storage once and constructs each
inner vector with a zero count. This also avoids the non-throwing default and
move constructors while filling the outer container. The index still starts
empty, validates the same inputs and builds the same receipt history.

Iterator debugging remains enabled. No allocation is exempted from injection,
no exception is swallowed, and no numerical tolerance or inference kernel changes.
The test now covers empty and populated tapes; after every injected failure it
queries the old index and successfully constructs and queries a fresh one.
An unexpected `terminate` in this test emits a diagnostic and exits unsuccessfully
instead of depending on interactive CRT error handling.

## Verification

The repair source commit is
`46995d51d456a07f8f8fa6325b2903238c83eb18`. All sixteen CI jobs across its push
and pull-request runs pass: Windows MSVC, Linux GCC/Clang, Intel/ARM Mac, both
Torch CPU references and the compiled Torch/ONNX correctness suite. The
[verification receipt](evidence/windows-allocation-v1/verification.json) binds
job URLs, source hashes, local log hashes and package identity. The
[Windows success excerpt](evidence/windows-allocation-v1/after-ci-excerpt.txt)
retains the available summary from a truncated returned log; the full job is linked.

Local validation passes 73 Debug groups, 73 Release groups, 72 sanitizer groups,
86 offline tests and all 36 wrapper tests against the rebuilt package. Source
archive to wheel installation, package relocation and extension source identity
also pass. The earlier failed CI and local environment-selection failures remain
recorded. No dependencies were downloaded: packaging used the installed build
environment and cached NumPy wheel, with the existing AppleClang compiler; the
offline suite used the project's supported Python 3.10 environment.

This is an internal repair/review, not independent approval of the PR stack.
Review also inspected batch/window separation, bounded workspace, moved state,
Python ownership and failure publication; no additional required change was found
in that focused pass. The [active queue](tasks.md#immediate-order-after-the-windows-repair)
puts the complete current-runtime measurement harness before further kernels.

The existing native batching evidence remains tied to its original binaries.
Its independent public verifier still reproduces all accepted/rejected gates.
This repair changes cold feature-index construction, not forecast mathematics or
steady-state inference, and establishes no additional speed improvement.
