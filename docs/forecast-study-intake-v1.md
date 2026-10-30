# First financial forecasting study: intake and proposed design

Status on 2026-10-30: the framework can run a bounded offline forecasting study.
The user requested useful forecasting work while UCL Beaker access is deferred.
The user subsequently chose BTC next-hour return. The original volatility proposal
below is retained as unexecuted design history. The accepted task and frozen
protocol are recorded in [BTC return study v1](btc-return-study-v1.md).

## A small, tangible question

The recommended question is: **How much will Bitcoin move over the next day?**
For example, after observing the completed hour ending at noon, predict the size
of price movements during the following 24 hours. Upward and downward movements
both count. This could eventually inform risk sizing, but does not itself say
whether to buy or sell.

A second choice is the next-hour return: whether the price rises or falls, and by
how much. This requires a different label, loss and baseline; do not silently run
both and select whichever looks better. A fresh non-financial problem is also an
option if the user prefers a clean public forecasting benchmark first.

Volatility measured from intraday returns has an established forecasting
literature; this is a useful application and comparison problem, not a novelty
claim. See [Andersen, Bollerslev, Diebold and Labys](https://www.nber.org/papers/w8160).
Our proposed use of hourly observations is a coarse movement measure, not the
same measurement precision as minute or tick observations.

## Available data and the admission result

A read-only search of QTS metadata found a checksum-backed Binance spot archive
for BTCUSDT and ETHUSDT covering 2023-07-01 inclusive through 2026-02-01 exclusive.
It contains 22,704 consecutive hourly bars per asset. Seven years of minute FX
archives are another candidate, with a separate calendar/duplicate audit needed.
The hourly archive offers the smaller first integration step; it does not satisfy
the eventual minute-bar requirement.

The pinned QTS normalized CSV has SHA256
`b7c060733b729162dbe7de96231df8435de9e91bb2afacf72da828f2ed28d2e4`.
The enclosing QTS data manifest has SHA256
`405148943dec824db7876cc03d2bf1f548a6143fbb67b3ef5e9ebc48988a3803`.
All 132 listed artifact hashes and all 62 ZIP/provider-checksum pairs were checked
against the local retained files. No raw dataset is copied into the public repo.

[hourly_bars.py](../tools/hourly_bars.py) is a standard-library reader of that
explicit normalized schema. It requires the expected hash, symbol set and exact
coverage. It rejects schema changes, missing/extra hours, duplicates, reversed
per-symbol order, clock-unit mismatches, invalid OHLC, non-finite numbers, negative
volumes, invalid trade counts, inconsistent taker totals and wrong instrument
identity. Input bytes, rows and line lengths are bounded. Source URLs are checked
as provenance strings; the reader never fetches them.

The real local admission accepted all 45,408 rows: 26,400 source timestamps use
milliseconds and 19,008 use microseconds. The [source documentation](https://github.com/binance/binance-public-data)
specifies the spot timestamp change on 2025-01-01 and warns archives can be revised.
The reader checks the original open/close integers against the normalized UTC
interval. A 10:00 bar finishes at 11:00; its close is never a 10:00 observation.

The `known_at` field is retained as `received_ns`, the actual later archive
receipt. It is not backdated to historical bar completion. Neither matching
checksums nor this reader establishes original historical publication, an
independent raw-trade reconstruction, execution rights or tradability. Public
archive accessibility also does not establish redistribution/production rights.

## Reproduce the admission locally

Set `QTS_DATA_ROOT` to the existing QTS checkout. Run from Stealth with an unused
output directory; no packages or downloads are required:

```sh
python3 tools/hourly_bars.py \
  --csv "$QTS_DATA_ROOT/data/derived/binance_spot_1h_v3/20260918T012100Z/hourly.csv" \
  --sha256 b7c060733b729162dbe7de96231df8435de9e91bb2afacf72da828f2ed28d2e4 \
  --symbols BTCUSDT ETHUSDT \
  --start 2023-07-01T00:00:00Z --end 2026-02-01T00:00:00Z \
  --output local/hourly-admission-new-run
python3 -m unittest discover -s tests -p test_hourly_bars.py -v
```

The CLI checks the derived CSV hash supplied by the caller. The separate recorded
132-artifact check additionally rehashed each artifact listed in the pinned QTS
manifest and compared every ZIP digest with its adjacent provider `.CHECKSUM`.
Neither action verifies that the source's price records were economically correct.

## Proposed experiment, subject to the user's choice

Keep the existing one-block architecture: width 64, four attention heads, MLP
width 256, exact GELU and a scalar output. Do not expand depth or add kernels for
this first application. Begin with BTC inputs only; ETH remains a separately
admitted future extension, not a feature chosen after reading outcomes.

For the volatility choice, a concrete candidate is:

- Input memory: the previous 96 completed hourly observations. Candidate features
  are hourly log return, trailing 24-hour log realized variance, and UTC time-of-day
  sine/cosine. Every rolling feature uses only the past.
- Target: log of the sum of squared hourly log returns over the *next* 24 hours.
  A fixed variance floor must be registered before fitting. In plain language,
  the model predicts the next day's total price movement on a compressed scale.
- Label alignment: a trailing 24-hour variance series shifted forward by 24 hours
  gives the future target. This fits the existing scalar head and scaler contract;
  no custom native output layer is needed. Report log-scale error first, and keep
  any variance-scale conversion and its bias explicit.
- Timing: only completed bars enter the input. Any assumed publication delay must
  be fixed and visibly labelled as an assumption. Actual archive receipt cannot
  support a historical live replay. Forecast origins, information cutoff and target
  interval must be recorded separately.
- Baselines: repeat the last day's variance, an exponentially weighted forecast,
  and a small linear model with the same permitted history. Include a time-of-day
  baseline if using hourly origins. Apply the same origin set to every method.
- Proposed periods: fit through 2024-12-31; select using January–June 2025;
  evaluate once on July 2025–January 2026 after choices are frozen. Labels must end
  within their own split; input history may precede it. These are reused QTS data,
  so the evaluation is chronological and separated from this fit, not an untouched
  program holdout.
- Freeze primary loss, baseline definitions, floor, optimizer, training budget,
  seeds, checkpoint selection and comparison rules before reading outcomes. Report
  every seed and baseline. Hourly origins have overlapping 24-hour targets: account
  for that dependence in uncertainty, or preregister daily origins as primary.
- Run CPU training, export the selected checkpoint, check Torch/native predictions,
  and retain a timestamped forecast ledger with deterministic replay. Scores and
  runtime are separate measurements. Beaker is not required to establish forecast
  parity or run a bounded local learning experiment.

No predictive outcome was inspected to choose these proposed periods or settings
in this turn. Source integrity/coverage checks do read the archive, but do not
compute features, labels, forecast losses or returns. The existing ETTm1 reserved
tail remains untouched.

## Test evidence and remaining boundary

The 13 new test methods cover independently specified timestamp values, the 2025
unit transition, complete grids, multiple assets, corruption, malformed fields,
resource limits, immutable records, fresh-process replay and output overwrite
rejection. Invalid input creates no success report. The initial test run failed on the missing reader before implementation.
The full local standard-library suite passed all 79 tests. Only synthetic market
rows are committed. CI runs these checks on the existing
five-platform native matrix, without a market-data or network dependency.

A portable [metadata audit](evidence/hourly-admission-v1/audit.json) and
[verification record](evidence/hourly-admission-v1/verification.json) are committed.
The original admission reports live under ignored `local/hourly-admission-v1/` and its
independent replay directory. They contain metadata and hashes, not numerical
price history. The reader is an offline Python tool, not a live feed connector or
an addition to the installed native API.

Next, agree the objective and freeze the experiment. Then implement the causal
feature/label builder with boundary tests, fit simple baselines, train the current
model and verify exported inference. A future new-data evaluation, live receipt
capture, recoverable operation and trading controls remain separate milestones.
QTS canonical research records were not changed for this Stealth data boundary.
