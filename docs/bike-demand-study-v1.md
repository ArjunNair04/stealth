# Bike rentals: an understandable forecasting demonstration

Imagine a bike station counter. At the end of each hour, we know how many rentals
just happened. We show the model the last two days of counts and the clock, then
ask how many rentals will happen in the next hour. A morning rush provides a
pattern we can inspect. Whether this model learns it is an empirical question.

The user requested an easier public task after the BTC experiment. The proposed
hourly objective was presented for consultation, with hourly as the stated default.
This is a separate study; the BTC result and ETTm1 reserved tail remain intact.

## Source and clocks

Hadi Fanaee-T (2013), *Bike Sharing*, UCI Machine Learning Repository,
[DOI 10.24432/C5W894](https://doi.org/10.24432/C5W894),
[official source](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The archive contains Capital Bikeshare's hourly and daily counts for 2011–2012.
The pinned hourly file contains 17,379 rows and 165 absent hourly labels. The
source page's instance count differs; the actual pinned file controls this study.
Source acquisition and byte hashes precede model fitting. Raw downloads stay in
ignored `local/`; the protocol includes URL, attribution, license and hashes.

The source date/hour labels do not establish UTC, DST treatment or publication
latency. We use a naive source calendar and assume each count becomes available
when its hour ends. At the boundary labelled 09:00, for example, the latest input
is the count for 08:00–09:00; the target is the count for 09:00–10:00. This is a
retrospective learning exercise, not a live availability guarantee.

`casual + registered = cnt` is checked for source integrity. Neither component is
an input. Future observed counts and weather are excluded. Each input row has its
observed count and four calendar values: sine/cosine pairs for the following
hour and weekday. These calendar values come from arithmetic on the current
clock, not from inspecting a future row. Two values let the model understand that
23:00 and 00:00 are neighbours on a clock. Weekday convention is Monday=0.

## Frozen design

[The machine-readable recipe](../benchmarks/bike/demand_v1.json) controls the run.
The existing one-block model reads 48 observed hours, uses width 64/four heads/a
256-unit GELU MLP and predicts one count. Its initial output repeats the latest
count; learning adjusts that starting guess. No native architecture changes.

Require 48 consecutive observed input hours and the next observed target hour.
Exclude windows crossing missing hours. Count all excluded target hours, including
the initial history warmup, separately for each split. This restricts evaluation
to intact stretches and may exclude difficult periods; no missing count is invented.

- Training labels: 2011. Fit scaling on unique rows used as training inputs.
- Validation labels: January–June 2012. Select the earliest minimum validation
  mean absolute error (MAE), including the untrained persistence candidate.
- Evaluation labels: July–December 2012. Input context may precede a split boundary.
- Seeds 7, 19, 43; eight epochs each; CPU AdamW, one thread, deterministic algorithms.
  Training minimizes squared error; selection and the primary score use MAE.
- Baselines: latest count, same hour yesterday, training weekday/hour average,
  and ridge regression on exactly the same flattened 48x5 inputs. Select the
  ridge penalty on validation MAE only. An unseen training calendar cell falls
  back to the training label mean.

MAE means "how many rentals the prediction missed by, on average." The
preregistered learning gate requires every seed to reduce MAE at least 10% versus
repeating the latest count, plus a negative upper bound for seed 7's paired
absolute-error difference interval. Resample whole calendar days 2000 times for
that interval. It is approximate: longer seasonal dependence is not represented.
Report all baselines and seeds even when that gate passes. Passing does not imply
best-in-class forecasting or improvement over ridge.

No output clipping or rounding. Report negative forecasts as a model limitation.
The visual example is fixed to the first evaluation week, July 1–7, before scoring;
it is not selected for looking good. Keep absent predictions as gaps in the plot.
No additional epochs, feature search or architecture changes after evaluation.

## Files and reproducibility

- `tools/bike_demand.py` checks source bytes/schema/counts/calendar, constructs
  causal windows, fits the input scaler, and estimates the paired interval.
- `tools/train_bike_demand.py` trains, selects, exports and runs native inference.
  It reuses existing optimizer/checkpoint/numerical helpers without modifying
  earlier frozen experiments.
- `tests/training/test_bike_demand.py` covers hand-calculated clocks, gap exclusion,
  scaler isolation, future-data perturbations, failure preservation, real tiny
  training, native parity, exact replay and artifact tampering.

The source hash, protocol hash, clean code commit, training environment and native
build identity bind each run. Every epoch has a pickle-free checkpoint. Selection
is written before test samples are constructed. A completion manifest hashes all
artifacts; failures leave a failure record and cannot receive a completion seal.
All selected validation/test predictions are checked against packed C++; sampled
full traces are checked with both native kernels. Existing tolerance remains
`5e-5 + 2e-4 * abs(reference)` in rental units. A separate process must reproduce
the entire evaluation ledger byte-for-byte.

Use the existing declared training environment; install nothing during the run.
Download the official ZIP with TLS verification, retain it, extract only `hour.csv`
and compare its SHA256 with the recipe. For the local source layout used here:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
local/torch-reference-env/bin/python tools/train_bike_demand.py train \
  --csv local/datasets/bike-v1/hour.csv --output local/bike-demand-v1
local/torch-reference-env/bin/python tools/train_bike_demand.py replay \
  --csv local/datasets/bike-v1/hour.csv --run local/bike-demand-v1 \
  --output local/bike-demand-v1-replay
```

Outputs are immutable and local. Use a new output path for independent replication.
Public summaries and figures credit the source; private environment reports and
raw datasets are not committed. This experiment checks useful learning and the
Python-to-C++ route. The controlled Linux runtime comparison remains a separate
performance gate.

The optional `tools/plot_bike_demand.py --run local/bike-demand-v1 --output
local/bike-demand-v1-figures` command renders PNG, SVG and PDF from sealed outputs.
It requires an already installed Matplotlib environment, independently of Torch;
it verifies the plotted artifact hashes and inserts gaps for missing forecasts.
