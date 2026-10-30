# BTC next-hour return v1: completed result

The CPU train/export/native-replay workflow completed. The forecast-quality gate
failed: the primary learned model did not improve on zero return on the frozen
primary metric. This is a negative exploratory result for this input set, model
and training budget, not a conclusion that all return forecasting is impossible.

The user-selected target, features, baselines, seeds, epoch budget, splits and
comparison rules were committed at `4030594d9de380a5d6d2115c0a9c5eac88638523`
before fitting or scoring. The [protocol](../benchmarks/btc/return_v1.json),
[method](btc-return-study-v1.md), [complete aggregate scores](evidence/btc-return-v1/scores.json)
and [verification record](evidence/btc-return-v1/verification.json) preserve the run.

## Forecast accuracy

There were 13,103 training examples, 4,344 validation examples and 5,160 evaluation
examples. The evaluation labels end during July 2025–January 2026. All candidates
were evaluated on the same hours. Lower error is better; one basis point is 0.01
percentage points. MSE was the preregistered primary metric.

| Predictor | Selected epoch | MSE (basis points squared) | RMSE (basis points) | MAE (basis points) |
| --- | ---: | ---: | ---: | ---: |
| Zero return | — | 1723.8995 | 41.5199 | 27.3113 |
| Training mean | — | 1725.8613 | 41.5435 | 27.3240 |
| Previous hour's return | — | 3413.3206 | 58.4236 | 39.5537 |
| Linear ridge, penalty 10 | — | 1728.0035 | 41.5693 | 27.3423 |
| Transformer, seed 7 (primary) | 3 | 1724.5187 | 41.5273 | 27.3093 |
| Transformer, seed 19 | 0 | 1725.8613 | 41.5435 | 27.3240 |
| Transformer, seed 43 | 0 | 1725.8613 | 41.5435 | 27.3240 |

All three seeds completed three learning epochs. Seeds 19 and 43 selected epoch 0,
the initial training-mean forecast, because their learned checkpoints did not
improve validation MSE. Those two selected models therefore contain no useful
learned prediction in this study; they must not be advertised as successful
transformer forecasts. The ridge penalty was selected on validation, not evaluation.

The primary seed's MSE was 0.0359% worse than zero. Its paired mean squared-error
difference was +0.6192 basis-points-squared, with the fixed approximate 95% daily
block-bootstrap interval [-0.4069, +1.7756]. The interval crosses zero, so the
experiment provides no clear improvement. Its slightly smaller MAE does not
reverse the primary-metric conclusion.

The learned forecast's standard deviation was 0.1239 basis points, versus 41.5167
for actual returns; its prediction/return correlation was 0.00239. It remained
close to a constant forecast. These diagnostics describe this fitted model; they
do not identify a new rule to tune against the evaluation period.

## Inference and reproducibility

All 28,512 latest predictions across validation/evaluation and three seeds passed
native/Torch parity. Sixteen complete traces per split per seed were additionally
checked through both native kernels. Maximum absolute discrepancy was
`4.76837158203125e-7` basis points. The fresh-process replay regenerated the full
5,160-row ledger, including all three native forecasts and baselines, byte-for-byte.
All 37 sealed artifacts passed hash verification. This is correctness evidence,
not a new latency benchmark.

The local training/reference suite passed 90 tests with two optional Linux-only
tests skipped; all 79 offline tests passed. All 16 GitHub checks passed at the
frozen implementation commit, including the real optional Linux adapters in their
separate CI job. No native kernel or packaging code changed.

Raw data, trained weights, epoch checkpoints, host details and the full prediction
ledger stay under ignored local storage. Original run: `local/btc-return-v1/`;
independent replay: `local/btc-return-v1-replay/`. The committed evidence contains
aggregate scores and hashes only. No dependencies were installed for this run.

## What follows

The current pipeline is sufficient to test a financial prediction end to end.
This initial own-price/volume feature set has not earned a forecasting edge. The
next research choice is one additional signal family with a defensible availability
clock, followed by a new frozen protocol and an evaluation policy that acknowledges
all previously inspected history. Keep this failed result as the baseline.

The history was reused from QTS, its original publication delay is unknown, and
the closing-price target is idealized. No fees, orders, fills, capacity or trading
P&L were modeled. Evidence remains E0. No predictive improvement, research novelty,
alpha, deployment readiness or current-runtime speed advantage is claimed.
