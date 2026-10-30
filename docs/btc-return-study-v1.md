# BTC next-hour return: frozen exploratory study v1

Completed: see the [recorded results](btc-return-results-v1.md). The primary
forecast-improvement gate failed; native parity and deterministic replay passed.

The user selected the next-hour BTC return on 2026-10-30. The question is whether
Bitcoin's own recent price and volume history helps predict its price change
over the next hour. This uses the existing one-block model and the previously
admitted hourly archive. It does not wait for Beaker and does not add native layers.

The stable hypothesis is `btc_own_history_next_hour_return_v1`; the experiment is
`btc_hourly_return_v1`. [The complete recipe](../benchmarks/btc/return_v1.json) is
committed before any real-data fitting or scoring. The separate QTS repository and
its canonical records are unchanged. This is a Stealth forecasting experiment,
with no portfolio, order, fill or economic alpha conclusion.

## Inputs and answer

Each observation contains five quantities from a completed BTCUSDT hourly bar:
its close-to-close simple return in basis points, log high/low range in basis
points, log(1 + quote volume), and sine/cosine of its UTC ending hour. One basis
point is 0.01 percentage points. The time features represent a clock smoothly,
so midnight remains close to 23:00. All features depend on the current or earlier
bar. ETH is admitted by the source reader but not used as a predictor.

The model sees 96 observations, or four days. It predicts
`10000 * (next_hour_close / current_close - 1)`. A prediction of +10 means +0.10%.
This predicts a return directly rather than estimating a future price level and
subtracting two large numbers.

The architecture stays at one pre-norm block, width64, four heads, MLP width256,
exact GELU, fixed window-relative positions and final layer normalization. The
existing scalar level-output mode represents the return. The body is seeded in
the usual way; initially zero head weights and a fitted bias produce the training
label mean. This constant candidate is eligible at epoch0. If all learning epochs
are worse on validation, selecting epoch0 is recorded as no useful learned model.

The feature scaler is fitted only to rows actually used in training inputs and
stored as FP32 with the exported weights. Training labels are FP32; reported
forecast errors use the original FP64 calculated returns. No test scaling or
outcome-dependent feature selection is permitted.

## Clocks and evidence limit

A bar labelled10:00 ends at11:00. Its close becomes an input only at the end of
that bar. The ledger records information end, hypothetical decision one nanosecond
later, target start at that information end, and target end one hour later.
The one-nanosecond offset only expresses ordering; it is not a measured feed delay.
The target starts at an idealized closing price just before that decision, so it
must not be interpreted as a tradeable entry price or a return earned by an order.

The original archive receipt remains in the source records. It cannot be used to
pretend those files were received historically. This study bypasses the live
feature-availability bridge and calls native inference on completed historical
windows under that explicit idealized observation assumption. Real publication
and receipt delays require a future, separately timed experiment.

The archive has been used in QTS before. Evaluation is chronologically separated
from this fit, but it is not an untouched program holdout. All scientific claims
remain E0. USDT quote, one venue, revised archives, missing executable bid/ask and
source-use rights are inherited limitations. No trading P&L is computed.

## Comparison and stopping rules

Training labels end by2025-01-01 UTC; validation labels end during January–June2025;
evaluation labels end during July2025–January2026. A label must end strictly after
its split start and at or before its split end. Past input context may cross a
split boundary. The first return needs two closes and the first prediction needs
96 feature rows, so early training rows are warmup, not filled values.

Baselines are zero return, the training label mean, the previous hour's return,
and ridge regression using the same flattened96x5 observations. The ridge penalty
is chosen from0.01,0.1,1,10 using validation MSE only; its intercept is unpenalized
and its centering/scaling uses training only. All candidate validation losses are
retained. The neural model uses seeds7,19,43, each with exactly three training
epochs. Lowest validation MSE chooses among epochs0–3; earliest wins ties. All
seeds and the linear baseline are sealed before test samples are scored.

The primary loss is MSE in squared basis points; MAE, RMSE, bias, prediction spread
and correlation aid interpretation. Seed7 is the preregistered primary seed;
19 and43 are sensitivity checks, not opportunities to pick a winner. The primary
seed versus zero comparison receives an approximate95% paired circular24-hour
block-bootstrap interval with2000 replicates and seed20261030. This describes
uncertainty under that resampling assumption, not a confirmatory alpha test.

The descriptive gate requires every seed to beat both zero and the selected ridge
baseline on test MSE, and the primary seed's loss-difference interval to lie below
zero. Regardless of outcome, stop after the fixed run. Do not extend training,
change features, select another seed or reuse the same evaluation as confirmation.
A failure is a result to preserve, not a reason to hide an experiment.

## Train, verify and replay

Use the existing declared Torch/native environment. No dependencies are installed
by these tools. Set `QTS_DATA_ROOT` to the QTS checkout, then run from Stealth:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  local/torch-reference-env/bin/python tools/train_btc_returns.py train \
  --csv "$QTS_DATA_ROOT/data/derived/binance_spot_1h_v3/20260918T012100Z/hourly.csv" \
  --output local/btc-return-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  local/torch-reference-env/bin/python tools/train_btc_returns.py replay \
  --csv "$QTS_DATA_ROOT/data/derived/binance_spot_1h_v3/20260918T012100Z/hourly.csv" \
  --run local/btc-return-v1 --output local/btc-return-v1-replay
```

The real CLI requires clean committed code, the pinned source digest and a matching
Release native extension. Runs create new directories; failures and partial
checkpoints are retained. Completion records hash the recipe, scaler, baseline,
selections, checkpoints, exported models, scores and timestamped predictions.
Every validation/test latest prediction is compared against Torch at the existing
5e-5 absolute +2e-4 relative tolerance. Sixteen complete traces per split are also
checked through both native kernels. Fresh-process replay recomputes the full
saved ledger and requires byte equality.

Tests use synthetic bars, hand-computed targets and a tiny real training cycle.
They cover source gaps, feature causality, exact split boundaries, train-only
scaling, an independently solved ridge case, deterministic uncertainty, selected
model invariance to changed test prices, export parity, replay, tampering and
failure without a completion seal. CI already discovers these tests in the Torch
reference matrix. The native engine and packaging are unchanged.
