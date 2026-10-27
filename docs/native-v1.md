# First native layers: implementation and evidence

A model is a chain of small calculations. This milestone builds and checks those
calculations individually, then connects them into one feed-forward layer. It is
the numerical foundation for a later transformer.

The implementation is in `include/stealth/numeric.hpp`, `src/numeric.cpp`,
`include/stealth/feed_forward.hpp` and `src/feed_forward.cpp`. The public
[numeric contract](numeric-contract.md) was frozen before implementation.

## What is implemented

| Component | Purpose | Main checks |
| --- | --- | --- |
| AlignedBuffer | Own a bounded, contiguous array of FP32 values | Alignment, zero initialization, ownership moves, size limits |
| Dense | Mix input values using learned weights and a bias | Hand answers, independent double formulas, packed/scalar agreement, all output tail sizes |
| Softmax | Turn scores into probabilities over an active prefix | Sum and range, shift invariance, exact zero mask suffix, large values, invalid masks |
| Layer normalization | Put each input vector on a common scale | Population variance, constant inputs, epsilon, scale/shift, extreme finite values |
| Exact GELU | Smoothly adjust activations | Independent normal-CDF formula, generated values, extremes |
| Residual addition | Add the original input back | Finite addition, overflow rejection, aliasing |
| FeedForward | Compose normalization, two projections, GELU and a residual | Independent composed fixture, late failure, recovery, repeated workspace reuse |

The block computes:

```text
input --> normalize --> widen --> GELU --> narrow --> add original input --> output
```

This is a pre-norm feed-forward residual block. Attention and sequence handling
are separate layers that have not yet been implemented.

## How calls fail safely

Construction validates and copies model weights, then allocates storage. Invalid
parameters raise standard C++ exceptions. Inference uses a small error code and
caller-owned scratch space, with no exceptions or explicit heap allocation.

Every hot operation checks dimensions, overlap and finite inputs. If an operation
reports an error, the caller's output remains unchanged. Intermediate scratch may
change; callers must treat it as disposable. The composed block publishes only
after all stages pass. A later valid call can reuse the same workspace.

Inputs must refer to live storage. This API cannot discover dangling pointers.
Input/output overlap is rejected, including partial overlap. The entire provided
scratch span must be separate from inputs and outputs. Each workspace has one
writer at a time; a model must not be moved or destroyed while callers use it.
The repeat test interleaves two workspaces sequentially. Parallel thread stress
and ThreadSanitizer coverage remain future checks.

## Verification completed

At code commit `cb8bb3108b4dcd9d5a6ccdfb32115f11787f76db`:

- Local Debug: 20/20 CTest groups passed.
- Local Release: 20/20 passed, with assertions independent of `NDEBUG`.
- Local AddressSanitizer + UndefinedBehaviorSanitizer: 19/19 passed. The install
  consumer is intentionally omitted from sanitizer builds.
- Fixture regeneration reproduced the header and provenance manifest exactly.
- Installation, moving the install prefix, and compiling/running an external
  consumer passed in both Debug and Release. The consumer exercises native maths.
- All five [CI jobs passed](https://github.com/ArjunNair04/stealth/actions/runs/40601121512):
  Linux GCC, Linux Clang, Intel macOS, ARM macOS and Windows MSVC. Each ran Debug,
  Release, install relocation and a benchmark smoke run. Linux Clang also ran
  sanitizers. The GCC/Clang compile-policy test is not applicable to MSVC, which
  uses `/fp:precise`; Windows therefore has 19 CTest groups per build.

These are test groups, not independent proofs of correctness. A passing suite does
not establish zero defects, arbitrary floating-point accuracy or deployment safety.
The [machine-readable receipt](evidence/native-v1/tests.json) records the scope.

### Test-first development and deliberate faults

The numeric contract and component tests preceded implementation. The initial
numeric stub failed compilation because warnings-as-errors rejected an unused
private field. After implementation, the corrected stub was replayed: all nine
component groups failed for missing behavior. This is a recorded red replay,
not a claim that the original red run reached the tests.

All four feed-forward groups failed against their stub before that block was
implemented. The unsafe compiler-flag test also failed before its guard existed.
The final code then passed each group.

Three deliberate mutations checked whether existing tests could detect plausible
mistakes. Each failed the expected test, after which the original source was
restored and rebuilt:

| Deliberate defect | Detecting test | Observed failure |
| --- | --- | --- |
| Add one to a scalar dense output | `numeric.dense` | 3 observed where 2 was expected |
| Write ones in the masked softmax suffix | `numeric.softmax` | Expected masked zero failed |
| Replace the residual stage with GELU | `feed_forward.parity` | Composed reference mismatch |

The mutation receipts include original-source hashes. Raw local execution logs
remain under ignored `local/`; they are not published with device paths.

### Independent expected answers

Small component cases have hand-computed answers. Generated component cases use
separate double-precision formulas. The composed fixture comes from
`tools/generate_numeric_fixture.py`, using only Python's standard library and
explicit equations; it does not call the C++ library. Its source/output hashes
are in `tests/fixtures/feed_forward.json`.

The scalar and packed comparisons additionally check the optimization. They do
not replace the independent mathematical fixtures. Declared tolerances cover the
bounded tested inputs; bitwise agreement across machines is not promised.

### Memory and repeated calls

The allocation executable intercepts ordinary and aligned C++ `new`/`new[]`.
A positive control detects two intentional allocations. Across 20,000 cycles of
block inference, softmax, non-finite rejection and bad-shape rejection, it then
observed zero intercepted allocations. This does not instrument every operating
system or library allocator.

A separate test runs 20,000 block calls, alternating two preallocated workspaces,
and checks each output. Moved objects, insufficient scratch, overflow, overlapping
buffers, invalid dimensions and late failures also have explicit tests.

## Reproduce and isolate a failure

From a clean checkout with the documented compiler, CMake and Ninja:

```sh
cmake --preset dev -DSTEALTH_WARNINGS_AS_ERRORS=ON
cmake --build --preset dev
ctest --preset dev
ctest --preset dev -L component
ctest --preset dev -L integration
ctest --preset dev -R '^numeric.softmax$'
python3 tools/generate_numeric_fixture.py --check
```

Repeat configure/build/test with `release` and, on supported platforms, `sanitize`.
Python is needed only to regenerate/check the independent fixture and record
benchmark receipts; native builds and checked-in-fixture tests do not import it.

Git separates the frozen contract, components, composition, and benchmark/build
safeguards into reviewable commits on `codex/native-primitives`. Main remains the
previous verified baseline until the pull request is merged. A future regression
should start with a failing case and retain its fixture, then use the relevant
CTest group and Git history to locate the change.

## Next acceptance gate

Freeze a tiny causal-attention model and independent fixtures, implement attention
components, then check full-block/model outputs and future-input invariance.
Weight-file validation and ordinary forward parity must precede cached streaming.
Python bindings can follow the stabilized native API. See the [work queue](tasks.md)
and [testing plan](testing.md) for later gates.

No weights loader, trained model, attention, cache, market feed, order routing,
QTS adapter or financial result is part of this milestone.
