# What the first forecast diagnosis found

Keeping the latest temperature still wins on the primary measure: average
absolute error. Four small linear models get closer than the three original
transformers, but none beats that baseline on the already inspected test period.
The temperature-only models nearly match it and have slightly lower RMSE, which
puts more weight on large mistakes. That secondary improvement does not reverse
the primary result.

This study keeps the original task: read one day of observations and predict one
temperature an hour ahead. The original transformer weights are unchanged. Each
linear family chooses its penalty on validation data before test scoring. All
16 training/validation candidates, all four selected linear models and all three
original transformer seeds are retained in the [aggregate record](evidence/forecast-diagnostics-v1/results.json).

## Measured errors

| Method | Train MAE | Validation MAE | Old-test MAE | Old-test RMSE |
| --- | ---: | ---: | ---: | ---: |
| Keep latest temperature | 0.739592 | 0.601602 | 0.417916 | 0.588700 |
| Repeat yesterday | 2.559112 | 1.969966 | 1.529571 | 1.967430 |
| Temperature history → level | 0.718064 | 0.586868 | 0.418508 | 0.577408 |
| Temperature history → change | 0.717660 | 0.584999 | 0.418576 | 0.578858 |
| All seven inputs → level | 0.695970 | 0.620361 | 0.474536 | 0.630723 |
| All seven inputs → change | 0.701552 | 0.579310 | 0.443812 | 0.600132 |
| Original transformer, seed 7 | 0.708169 | 0.681501 | 0.497620 | 0.660156 |
| Original transformer, seed 19 | 0.745069 | 0.652982 | 0.581457 | 0.737400 |
| Original transformer, seed 43 | 0.726615 | 0.740554 | 0.642388 | 0.796845 |

Errors use the original OT units. Counts are 34,461 training and 11,517 forecasts
each for validation and the old test period. Training scores here use the final
selected weights on every training window; they are not the loss averaged while
weights were changing during training. No test-based seed or family winner is
selected for deployment.

Selected ridge penalties were 0.0001 for temperature-to-level, 0.01 for
temperature-to-change, 0.0001 for all-inputs-to-level, and 1 for
all-inputs-to-change. The temperature models have 97 fitted parameters and the
all-input models 673, including the intercept; the transformer has 50,689
trainable parameters. This is an accuracy comparison, not a measured speed result.

## Where the deficit appears

The test period is quieter: the average absolute one-hour change falls from
0.739592 in training to 0.601602 in validation and 0.417916 in test. This makes
keeping the current reading a stronger baseline. It describes these periods;
it does not establish the cause of every model error.

Using movement cutoffs fixed from training data, 8,154 of the 11,517 test windows
(70.8%) have an actual change of at most 0.493 units. On those quiet windows,
persistence MAE is 0.210279. The three transformer MAEs are 0.347367, 0.482697
and 0.532718. On the 219 windows with changes larger than 1.688 units,
persistence MAE is 2.070265 and the transformer errors are 1.926663, 1.678748
and 1.954732. Their gains on larger changes do not cover their losses on quiet
windows. These groups use the future outcome for explanation, so they are not
an available live trading or forecast-selection rule.

The transformer also develops offsets. Mean signed test errors are +0.191055,
-0.304543 and +0.477285 for seeds 7, 19 and 43, versus +0.002272 for
persistence. Positive means the prediction is too high on average. The differing
signs across seeds caution against one blanket correction. No correction was
fitted to the test outcomes.

Extra load measurements do not show a consistent benefit in this comparison.
For the change models, adding them improves validation MAE from 0.584999 to
0.579310 but worsens old-test MAE from 0.418576 to 0.443812. For the level
models, the all-input variant is worse on both validation and old test. This
is evidence about these fitted models and periods; it does not prove the extra
signals are intrinsically useless.

## What to test next

A focused next model idea is to start from the latest temperature and let the
same small transformer learn a correction. With a zero-initialized correction
head, the initial forecast would exactly match persistence. This gives it a
useful starting answer and asks it to learn when a change is justified.

This is a proposal for discussion with the user, not an implemented architecture
change. It is motivated by the baseline strength and the observed offsets, but
the linear change model did not beat persistence on test MAE. There is no promise
that changing the transformer head will fix the problem. A follow-up should keep
the objective, one-block size and comparison budget explicit, choose settings
using development data, and freeze its final evaluation rule before touching
reserved outcomes. More blocks and attention optimizations can be separate tests.

## Evidence and limits

The frozen study code is `96c0469e353fc753709207356f00060035458148`.
Two independent full runs produced byte-identical selections, coefficients,
predictions, aggregates and manifests. All five original baseline/transformer
MAE and RMSE pairs reproduced exactly. The 12,080 rows after index 57,599
remain excluded from feature construction, fitting and scoring. Source integrity
checks can read the full pinned CSV without using those later outcomes.

All 31 training/reference/experiment tests and 23 public-data tests pass locally.
Five native platform jobs and both pinned Torch CPU jobs pass at the frozen code
commit. Independent arithmetic, validation-only selection, reserved-tail
isolation, overwrite refusal, incomplete-run preservation, metric overflow and
original-score mismatch checks are covered. See [verification](evidence/forecast-diagnostics-v1/verification.json)
and [the protocol and rerun guide](forecast-diagnostics-v1.md).

The original test period has already been inspected; this is retrospective
exploration. Overlapping windows are not independent samples, and no significance
claim is made. Neither trading usefulness nor an inference-speed advantage is
established here. The original experiment remains preserved. Raw data, fitted
coefficients, checkpoints and prediction arrays stay local; only aggregates and
hashes are published. Source attribution and rights are recorded in the
[original learning report](training-results-v1.md).
