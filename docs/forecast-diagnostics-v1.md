# Diagnosing the first forecast result

The first transformer lost to keeping the latest temperature. Before changing
its architecture, this study asks whether a small weighted sum of the available
history helps, and where the frozen transformer's errors occur.

The protocol is `benchmarks/ettm1/diagnostics_v1.json`. It is committed before the
new outcomes. The original test period has already been inspected, so this is a
retrospective diagnostic comparison. Rows 57,600 onward remain reserved: source
integrity and clocks may be verified, but no features, targets, fitted statistics
or predictions from those rows enter the analysis. No significance, general
forecasting advantage or trading conclusion is claimed from this exercise.

## Four small linear comparisons

Each model sees the same 96 observations available to the transformer and predicts
the same temperature four quarter-hour steps ahead. Two feature sets use either
96 temperature readings (97 fitted parameters including intercept), or all seven
measurements (673 fitted parameters). Both sets try two targets: the future
level itself, or its change from the latest temperature. For a change prediction,
we add the latest temperature back to obtain the final level forecast.

**Ridge regression** fits a weighted sum while discouraging unnecessarily large
weights. The exact objective is mean squared training error plus lambda times the
sum of squared coefficients. The fitted intercept is unpenalized. The four fixed
penalties are 0.0001, 0.01, 1 and 100. Each of the four families selects its penalty
using validation MAE, with the smaller penalty winning an exact tie. All 16
training/validation results are retained; only those four validation-selected
models are scored on the already-inspected test period. There is no extra fitting
on validation data and no test-based family/seed winner used as a new deployment
choice.

Observations and the original training-only feature scales are rounded to FP32
as in the reference workflow. Linear fitting then uses FP64, with training-window
centering and an eigendecomposition of the mean covariance. One decomposition is
shared across penalties/targets for each feature set. Invalid or materially
negative eigenvalues fail; tiny roundoff negatives are clamped at zero. The real
recipe uses strictly positive penalties. The unregularized full-rank case is
included only in arithmetic tests.

Learning a change is not an inherently richer linear model: without shrinkage,
including the current temperature among the inputs makes the level/change forms
equivalent. With shrinkage, the level form is pulled toward the training target
mean, while the change form is pulled toward a constant correction to the latest
reading. The intercept remains free, so the latter is not forced to zero change.
The independent unit test checks the unregularized equivalence. Any observed
benefit here is evidence about these linear priors, not proof that a transformer
head change will help.

## Inspecting the original model

The runner verifies all three original sealed runs and selected model hashes.
It does not retrain, recalibrate or choose among their seeds. It recomputes final
checkpoint train/validation/test forecasts and checks that the original test MAE
and RMSE replay within an absolute 1e-6 allowance. Reported training errors here
use the final selected model; they differ conceptually from the earlier loss
averaged while weights were changing during an epoch.

Diagnostics include MAE, RMSE, signed bias, the size of predicted changes,
predicted/actual change correlation, the fraction of forecasts worse than
persistence, target-calendar-month summaries, and actual-movement groups.
Movement thresholds are the 50th/90th percentiles of absolute one-hour changes
in training targets only. Buckets use actual future changes retrospectively for
description; they cannot be used as a live selection rule. Constant-change
correlations are undefined and recorded as null, as are empty buckets.

## Running and checking

Use the existing explicit Torch reference environment. No new dependency or native
kernel is introduced. Commit the frozen source/protocol before a real run:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
local/torch-reference-env/bin/python tools/diagnose_ettm1.py \
  --raw local/datasets/ettm1-v1 --prepared local/datasets/ettm1-prepared-v1 \
  --prior-runs local/ettm1-training-v1 --output local/forecast-diagnostics-v1
```

Run again into a different new directory to check deterministic replay. Outputs
include every candidate and sealed choice, local fitted coefficients and forecast
arrays, aggregate diagnostics and a completion manifest of hashes. Existing
outputs are refused and failures retain an incomplete directory. Original runs
are never overwritten. Raw/derived arrays and coefficients remain local; only
compact aggregates and provenance should be committed.

Tests check scalar answers, an independently augmented least-squares solution,
constant/rank-deficient designs, level/change equivalence, feature and target
clocks, reserved-tail isolation, validation-only selection, known error summaries,
empty/constant groups, deterministic complete replay and failure preservation.
The transformer architecture and native deployment format remain unchanged. A
follow-up model design will be discussed with the user after these findings.
