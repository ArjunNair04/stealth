# First public benchmark: ETTm1

We now have a real dataset prepared for the first learning experiment. The next
choice is what to ask the model to predict. No model has been trained or scored
by this preparation step.

## What the data means

Think of a notebook with a new line every 15 minutes. Each line has six electricity
load measurements and the temperature of an electrical transformer's oil. The
transformer in this dataset name is electrical equipment; it is not a neural
network. We can use earlier notebook entries to predict a later temperature.

The [original ETDataset repository](https://github.com/zhouhaoyi/ETDataset)
provides the source. We pinned revision
`1d16c8f4f943005d613b5bc962e9eeb06058cf07`, including its README and license,
with hashes in [source.json](../benchmarks/ettm1/source.json). The actual CSV has
69,680 rows, from 1 July 2016 through 26 June 2018, with exactly 900 seconds between
adjacent timestamps. This differs from the README's approximate row count and
its loose reference to minute data. Source clock timezone is unspecified.

Attribution: Haoyi Zhou and ETDataset contributors, *Informer: Beyond Efficient
Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021; data collected
with Beijing Guowang Fuda Science & Technology Development Company.
The source [license](https://github.com/zhouhaoyi/ETDataset/blob/1d16c8f4f943005d613b5bc962e9eeb06058cf07/LICENSE)
is CC BY-ND 4.0. Original files and preprocessing outputs stay in ignored local
storage. This repository publishes our preparation code and source references.

## The three boxes

Put the earliest observations in a practice box, later observations in a checking
box, and the final benchmark observations in an exam box. Never move exam answers
into the practice box to make the model look better.

| Partition | Zero-based rows, end excluded | Observations |
| --- | --- | ---: |
| Training | 0 to 34,560 | 34,560 |
| Validation | 34,560 to 46,080 | 11,520 |
| Test | 46,080 to 57,600 | 11,520 |
| Unused suffix | 57,600 to 69,680 | 12,080 |

These row boundaries follow the upstream Informer minute-loader convention: blocks
of 30-day months, not exact calendar months. We inspect structure and validity in
all boxes, but have not computed prediction errors or target summaries outside
training. Test data is locally accessible, not protected by a security barrier;
keeping outcomes untouched is an experiment protocol.

Scaling puts differently sized readings on comparable scales. Its averages and
spread are fitted using the training box only, then frozen. A later forecast may
use past observed values from validation/test as its history, but may not refit
those statistics or see the future values it must predict.

## The first model discussion

The proposal sent to the user is: use the last day's 96 readings, including the
past temperature, to predict one temperature four steps (one hour) ahead. A second
option is to predict all 96 temperatures in the next day. The window code supports
both; the experiment target and architecture remain uncommitted choices.

For example, with the most recent reading at 10:00, a four-step point forecast
answers for 11:00. It cannot use the 10:15, 10:30, 10:45 or 11:00 measurements as
inputs. Predicting the next day's curve requires many answers and gives us a
broader forecasting task, with more training and diagnostic work.

Before training, agree the horizon, input channels, positions and output meaning.
Start the comparison with simple controls: repeat the latest temperature, repeat
the corresponding time yesterday, and a fitted linear predictor. Report forecast
error separately from inference latency and memory. Use the same inputs and
trained weights for native-versus-reference timing. The proposed one-hour task
would be a development benchmark, not a replication of published long-horizon
leaderboard scores. Architecture, baselines, metrics, seed list and stopping
rules must be frozen before looking at held-out model outcomes.

## What the new files do

| File | Purpose |
| --- | --- |
| `benchmarks/ettm1/source.json` | The shopping list: exact source files, fingerprints, clocks and partitions. |
| `tools/prepare_public_benchmark.py` | Fetches/checks that list, prepares the training scaler and verifies replay. |
| `tests/test_public_benchmark.py` | Tries broken files and time mistakes, using invented tiny data. |
| `local/datasets/ettm1-v1/` | The original downloaded files and retrieval receipt, ignored by Git. |
| `local/datasets/ettm1-prepared-v1/` | The structural audit, scaler and reproducibility manifest, ignored by Git. |

The Python utility prepares data. It is not the planned Python wrapper around
C++ inference. The native library still has no Python or network dependency.
The detailed engineering rules are in [the contract](benchmark-contract.md).

## Run it yourself

From the permanent checkout, with Python 3.10 or later and no extra packages:

```sh
python3 -m unittest discover -s tests -p test_public_benchmark.py -v
python3 tools/prepare_public_benchmark.py fetch --output local/datasets/my-ettm1
python3 tools/prepare_public_benchmark.py prepare --raw local/datasets/my-ettm1 --output local/datasets/my-prepared-ettm1
python3 tools/prepare_public_benchmark.py check --raw local/datasets/my-ettm1 --prepared local/datasets/my-prepared-ettm1
```

Use a new destination each time. On Windows, `python` may be the interpreter name.
If Python's certificate store fails on macOS, an existing system CA file can be
supplied explicitly using `--cafile /etc/ssl/cert.pem` on the fetch command. Do not
disable certificate verification. The actual local download used that existing
CA store after the default store rejected TLS before HTTP.

A failed fetch retains a failed/incomplete receipt and any partial file. Successful
preparation verifies the originals before parsing, checks every timestamp and
numeric field, and writes its completion manifest last. Replay recomputes all
three output files and compares their bytes. Its tool hash binds the run to the
exact preparation implementation; a changed implementation needs a new run.

## Verification recorded on 28 October 2026

- Nineteen initial test methods failed against deliberate unimplemented stubs
  before implementation (42 errors, including parameterized subtests).
- Twenty-one final offline tests pass. They cover finite values, clocks, limits,
  split boundaries, training-only statistics, interruptions, immutable output
  paths, pinned-source integrity and byte-identical replay.
- Seed 20261028 exercises 200 sets of split/window dimensions with both target
  modes and all three partitions.
- Two temporary faults were detected by assertions: allowing future rows to affect
  the scaler, and using the last input as the forecast target. Both were removed;
  the full suite passed again.
- The real public snapshot was fetched, prepared and checked. Independent output
  directories contain byte-identical audit, scaler and manifest files.

Compact provenance and test evidence are in
[verification.json](evidence/public-benchmark-v1/verification.json).
All benchmark tests are configured to run offline in each of the five existing CI jobs. CI does not
need the real data, Python packages, dataset credentials or network downloads for
these tests. No training, financial result or forecasting speed claim is made.

## Next native work and QTS

Once the complete model layout is agreed, implement the versioned weight loader
and malformed-file tests, then the complete model and pinned reference parity.
Only then train/export the chosen public-data example. Keep performance ideas in
[the optimization notebook](performance.md#deferred-optimization-notebook).

A read-only QTS check found 28 local HistData M1 archives, covering AUDUSD, EURUSD,
GBPUSD and NZDUSD for 2019–2025. Their catalog describes bid-price data and records
known clock/provenance and execution limitations. The equity archive is largely
hourly. This is an initial inventory, not a new completeness audit. No QTS file or
research record was changed. A later adapter should select an audited subset,
define return labels and costs, and keep raw market data out of this public repo.
