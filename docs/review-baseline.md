# Consolidated baseline review, 29 October 2026

This is the review entry point for the complete Stealth research prototype.
[PR #18](https://github.com/ArjunNair04/stealth/pull/18) brings the twelve earlier
implementation PRs together against main. Every original commit and result is
preserved; [the stack record](review-stack.json) pins their original heads.
The working checkout remains the permanent standalone Stealth repository.

## What this review establishes

This is an internal integration review of numerical/failure boundaries, model
loading, Python ownership, training/export, benchmark provenance and packaging.
There were zero submitted GitHub reviews or inline review threads across the
original stack at the review snapshot. This work is not independent approval,
a complete security audit or a deployment certification.

| Finding | Resolution | Evidence |
| --- | --- | --- |
| Benchmark source revision could describe current C++ while timing an older installed wheel | Embed a source fingerprint in native/wheel build metadata; reject missing, stale or non-Release wheel metadata before loading/timing | Old installed wheel rejected; focused mismatch tests and deliberately removed-guard test |
| CLI labelled version-2 correction models as version 1 | Report validated output mode and corresponding format version | Both serialized formats tested through the CLI; version-2 test failed first |
| Overview documents described completed stages as pending | Rewrite the overview, architecture boundary and active queue | Current evidence links remain explicit; earlier queue preserved as history |
| Twelve dependent PRs obscured the complete package | One baseline PR retains every original commit and review link | Ancestry checks cover all twelve heads; originals remain recoverable |

The fingerprint covers native sources/headers, bindings, Python module, CMake
files, package configuration, CLI/benchmark sources and the identity helper. Its
canonical path/hash representation is independently checked in Python and CMake.
It works from source archives without Git. It detects accidental stale-source use;
it does not authenticate a compiler or replace build flags and binary hashes.
Changing source files triggers CMake reconfiguration. Existing timing reports
remain tied to their original measured commits; no timing improvement is claimed
from these fixes. Rebuild native consumers because `BuildInfo` gains a field.

## Reviewer route

1. Read [architecture](architecture.md), [model contract](forecast-contract.md) and
   [latest-only semantics](latest-forecast-v1.md). They define what is being tested.
2. Inspect the focused consolidation fixes relative to `3b46023`, then the native
   numerical/loader code in `src/` and ownership boundary in `bindings/module.cpp`.
3. Inspect component/failure/causality tests, training/export tests and the isolated
   installation test. Fixtures and JSON receipts account for much of the full diff;
   they are evidence, not additional runtime features.
4. Read [forecast outcomes](correction-results-v1.md) and [latency outcomes](latest-results-v1.md)
   separately. The last-value forecast still wins on primary old-test error. The
   local single-request native win coexists with a batch-eight deficit.

## Validation

The consolidated code passes 59 Debug groups, 59 Release groups and 58 sanitizer
groups, including relocated consumers where applicable. The offline Python suite
has 44 tests; Torch has 47; the wrapper has 24. The twelve runtime/CLI contracts
also pass under sanitizers. Independent fixtures regenerate exactly. The rebuilt
package preserves all 9,216 saved full-trace values and 96 final outputs bitwise.
Source-archive to wheel to isolated runtime installation is checked separately.
The [verification record](evidence/baseline-review-v1.json) records the tested code,
source/build identity, checks, package/log hashes and completed CI evidence.

Build and source-identity checks:

```sh
cmake --preset release -DSTEALTH_BUILD_BENCHMARKS=ON
cmake --build --preset release
ctest --preset release
python tools/build_identity.py --check-extension
python -m unittest discover -s tests -v
```

Use the declared Python build/runtime environment for the extension check;
[development](development.md) and [Python installation](python-stream-v1.md)
provide setup. Native CMake itself remains independent of Python. The consolidated
PR remains open for review rather than implying approval through green tests.

## Next decision

Use the [Linux comparison preparation](frontier-linux.md) and a confirmed host
before another performance claim. New depth, dropout, optimizers, event feeds or
hand-tuned kernels are outside this consolidation milestone. Discuss a bounded
forecast-quality experiment with the user separately; keep the reserved tail
untouched until its final evaluation design is fixed.
