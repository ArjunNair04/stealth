# Test-first development and initial objectives

Updated: 2026-10-29. These acceptance requirements distinguish completed checks
from open work. Primitive, attention/block, model loading, Torch training/export,
Python wrapper and raw-window stream tests are implemented. The measured attention
and [four-row execution](dense-rows-v1.md) changes add exact native parity,
allocation, failure and performance gates. See the linked result records for
actual outcomes; test requirements alone are not evidence of deployment readiness.
Current-runtime frontier comparisons, cached streaming and market-feed integration
remain open.

## First milestone

Deliver a small native CPU forecaster with independently checked arithmetic,
versioned trained weights, reference agreement and causal behavior. Then verify
bounded streaming updates and expose the same implementation to Python. A useful
forecast, a correct implementation and a deployable trading system require
separate evidence.

## Development loop

For each behavior, specify inputs, outputs, errors and a tiny example first.
Write a test that fails for the intended reason, implement the smallest change
that passes it, then improve the code while keeping the tests green. Record the
relevant failing/passing check in the change description; main stays passing.
Fixes begin with a regression case that reproduces the defect.

Expected outputs must come from hand calculations, independently expressed
high-precision formulas, a pinned reference implementation, or a mathematical
property. Do not generate expected answers by calling the implementation under
test. Preserve checked-in reference fixtures with their generator, configuration,
seed and hashes; changing them requires explaining why the previous answer was
wrong. Never regenerate fixtures merely to hide a mismatch.

Use a few deliberate faults to confirm important tests can detect the intended
failure, such as a transposed weight matrix or a removed causal mask. Remove the
fault before integration. Coverage can expose untested paths, but a target test
count or 100% line coverage is not the acceptance criterion.

## Objective 1: freeze the smallest contract

Before numeric implementation, define FP32 inputs/weights, row-major layout,
shapes, ownership, permitted aliasing, resource limits and invalid-input behavior.
Specify a single-observation API boundary and separate per-model weights from
per-stream mutable state. The initial inference engine has one writer per stream;
parallel sharing requires a later explicit contract.

Discuss model choices with the user before changing them. The user selected one
block first and a one-hour point forecast. The first complete specification is
now frozen in forecast-contract.md and benchmarks/ettm1/one_hour.json: width 64,
four heads, hidden width 256, fixed sinusoidal positions, pre-norm, exact GELU,
final layer normalization and a scalar head. Diagnostic shapes may be smaller.
Every window starts its positions at zero; bounded cached episodes remain a
separate implementation. The earlier two-block baseline was an unfrozen proposal.

## Objective 2: reliable native arithmetic

Implement only the required buffers and operations: dense linear projections,
stable softmax, layer normalization, the selected activation and residual addition.
Keep the scalar reference clear; optimization follows correctness.

Required evidence:

- Tiny hand-calculated examples, independently calculated higher-precision cases,
  and reproducible generated cases over the supported shapes and value ranges.
- Invalid dimensions, dimension-product overflow, aliasing, allocation limits,
  zero/constant inputs, large finite values and NaN/infinity behavior.
- Softmax non-negativity, sum-to-one and constant-shift invariance within declared
  floating-point tolerances; define fully masked rows explicitly before coding.
- Debug/release and memory/undefined-behavior sanitizer passes; tests must remain
  active when compiler assertions are disabled.

Done: each required primitive has checked normal and failure behavior, and its
consumer cannot silently receive an invalid numerical result.

## Objective 3: a trustworthy complete model

Test weight loading before model execution. Validate format/version, byte order,
shape/count/size consistency, resource bounds, integrity hash, truncated files,
non-finite values and unsupported fields. Parse into a candidate model; a failed
load must not partially replace an existing valid model. A hash detects accidental
change; it is not proof of trusted origin.

Compare individual layers and the full model against pinned PyTorch CPU inference
using the same weights and preprocessing, with dropout disabled. Use fixed
untrained diagnostic weights first, then a reproducible public-data training/export
example. Use ETTm1 for the requested public-data starting point; retain synthetic
cases for diagnostic tests. Keep native correctness tests runnable from compact
exported fixtures without installing Python; run a separate reference-generation/consistency job
in its declared Python environment.

Set per-operation and end-to-end absolute/relative tolerances before accepting
results. Report the worst discrepancy and its input. Investigate a mismatch before
changing tolerance. Bitwise equality across compilers and processors is not a
portable requirement; seeds alone do not guarantee identical training runs.

Causal tests must establish that changing future observations cannot change an
earlier prediction, and that appending a suffix preserves prefix outputs within
tolerance. Preprocessing must obey the same information boundary. Return labels
start after the information cutoff and the feasible execution time when a
financial adapter is introduced.

Done: the native model matches the reference over the frozen fixtures, rejects
malformed weights and passes causal invariance tests on supported platforms.

## Objective 4: correct bounded streaming

Before caching, freeze how each episode starts, advances, reaches capacity and
resets. Compare every cached step to full-prefix evaluation with the same inputs,
weights, positions, normalization and attention rules.

Test first/last supported context positions, rejected overflow, reset versus a
fresh instance, repeated reset, replay, long runs of bounded episodes, and
interleaved independent streams. Invalid input must leave state unchanged. Track
memory/allocation budgets where they are part of the contract. A rolling cache is
not assumed equivalent to recomputing the latest raw window.

For the later multi-source adapter, specify event identity, availability and
receipt order before implementing it. Test duplicate deliveries, conflicting
revisions, delayed events, missing sources, holidays/daylight-saving transitions
and stale data. An observation with an old effective time cannot be inserted into
an earlier decision before it was received. These integration tests become
required when that adapter is introduced, not prerequisites for scalar arithmetic.

Done: streaming is numerically equivalent under its declared semantics, bounded
in memory, and isolated between streams. Multi-source readiness requires its own
clock and failure tests.

## Objective 5: a thin optional Python interface

Once the early native layers and public API are stable, a wrapper may proceed
without waiting for all streaming work. Use the same native implementation.

Test a fresh isolated installation, native/Python agreement, exceptions, dtype,
shape, contiguity, readonly inputs, object lifetime and permitted copy behavior.
Reject unsupported buffers explicitly. If the GIL is released, test ownership and
concurrent calls under the documented policy; do not imply that one mutable stream
is safe for concurrent mutation. The standalone C++ build must still pass without
Python or binding dependencies installed.

Done: Python reliably calls the checked core; no numerical implementation is
duplicated in Python.

## Test execution and performance

Every native change runs deterministic behavior/regression tests in debug and
release. Keep the existing Linux GCC/Clang, Windows MSVC and Intel/ARM Mac matrix.
Relevant changes also run sanitizers and package-consumer checks. Add bounded
randomized/property tests as operations appear. Add malformed-input fuzzing when
the weight loader exists, and longer streaming soak tests when state exists;
preserve minimized failures and random seeds for replay. Native dependencies
remain explicit. Reference and wrapper tests use separate pinned environments.

Performance measurements use release builds, fixed workloads, controlled threads,
warm-up, repeated samples and a named machine. Record median/tail latency and
memory, and re-run correctness after each optimization. Do not make noisy shared
CI wall-clock timing a hard universal latency test. A dedicated comparable machine
and measured variability are needed for performance-regression thresholds.

## Later evidence required for deployment

Engineering gates: sustained runs, feed outages and malformed input, backpressure,
recovery/replay, model version rollback, monitoring and predictable resource use.

Research gates: simple baselines, frozen variants and costs, chronological splits
with overlapping-label leakage controlled, untouched validation, uncertainty and
multiple-testing control. Synthetic learning demonstrations do not prove alpha.

Trading gates: forward shadow observation, executable timing/cost/borrow evidence,
position/exposure limits, stale-feed abstention, duplicate-order prevention,
reconciliation and a tested stop mechanism. Passing software tests does not
replace these gates or authorize live orders.

## Numerical references

- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
