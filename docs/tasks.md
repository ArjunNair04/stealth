# Active work queue

The user approved consolidation and stronger runtime comparisons before further
kernel optimization. [Baseline review](review-baseline.md) records findings and
verification; [historical tasks](tasks-history.md) preserve the earlier sequence.

## 1. Consolidate the existing foundation

Implementation PRs #6–#17 are preserved in the single baseline PR #18. Two review
findings are fixed: stale installed wheels can no longer enter the benchmark,
and model inspection distinguishes format versions and output modes. Overview
documentation now describes the implemented training/export/inference workflow.
Review and merge are separate from successful builds; independent approval is
still outstanding. No model architecture or inference arithmetic changed.

## 2. Establish the current-runtime comparison

Prepare a supported common CPU host, exact package/build lock and identical raw
validation windows. Compare native core, the wrapper, current eager Torch,
compiled Torch and ONNX Runtime CPU. Latest-only batch one is primary; batch eight
is reported separately. [Linux preparation](frontier-linux.md) records the fixed
decision criteria and the host-lock requirements. Current host access, conversion
and compiled/ONNX runner validation are still required before timing.

Stop further native kernel work until this comparison identifies a useful target.
Retain unsupported/export/compile failures as outcomes. Do not substitute an old
compatibility result or shared-CI timing for a supported-host performance result.

## 3. Establish forecasting usefulness

Discuss one bounded modelling experiment with the user before changing the
objective or architecture. Require simple baselines, validation-only choices and
a protected final evaluation. The saved correction models still miss persistence
on the primary old-test measure. The old test is exploratory; 12,080 reserved tail
rows remain unused for model selection and scoring.

## Later gates

Multi-source clock/revision/missingness contracts, latest-only stream failure
policy, sustained load, concurrency, recovery and a QTS adapter remain separate
milestones. No financial deployment or research novelty follows from passing
software tests. No new features are required to finish the present comparison.

## GitHub tracking

- [Consolidated baseline PR #18](https://github.com/ArjunNair04/stealth/pull/18)
- [Runtime and QTS boundary issue #5](https://github.com/ArjunNair04/stealth/issues/5)
- [Model/reference issue #3](https://github.com/ArjunNair04/stealth/issues/3)
- [Python interface issue #4](https://github.com/ArjunNair04/stealth/issues/4)
