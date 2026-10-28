# Public benchmark data contract

Frozen data-engineering contract, 28 October 2026. The subsequently agreed
forecast target and model architecture are in forecast-contract.md. This is preparation,
not a trained-model result or a leaderboard reproduction.

## Source and ownership

Use ETTm1 at the revision, byte lengths and SHA-256 hashes in
`benchmarks/ettm1/source.json`. Download the original CSV, README and license into
a new ignored local directory. The repository states CC BY-ND 4.0. Retain its
attribution/license locally; publish our code and manifest, not transformed data.
A digest detects changes, not trusted authorship. No dependency installation,
authentication or QTS writes are needed. TLS verification stays enabled; an
explicit existing CA file may be supplied where Python's default store fails.

Fetch refuses an existing destination. A failed download retains an incomplete
receipt and any downloaded bytes; only a final complete receipt represents a
successful fetch. A retry uses a new directory. Read and download limits are
12,000,000 bytes per file, 100,000 CSV rows and 2,048 bytes per line. Source-pinned
length/hash checks happen before CSV parsing. No archive extraction is involved.

## CSV and clocks

Require the exact eight-column header and seven finite FP32-representable numbers
per row. Dates must use the exact `YYYY-MM-DD HH:MM:SS` form. All adjacent rows
must be precisely 900 seconds apart: reject duplicates, reversals, gaps, malformed
rows, missing values and non-finite/oversized numbers. Do not silently repair.
Require the pinned row count and first/last clocks. Preserve the source clock as
unspecified; do not invent UTC or historical publication/receipt timestamps.

The actual file has 69,680 rows. This differs from the README's approximate
70,080-row statement. Its cadence is 15 minutes, not one minute.

## Chronological partitions

Zero-based, half-open row intervals: train [0,34560), validation [34560,46080),
test [46080,57600). The remaining 12,080 rows are unused. These are the upstream
Informer minute-loader's 12/4/4 blocks of 30-day months, not calendar months:
https://github.com/zhouhaoyi/Informer2020/blob/main/data/data_loader.py

Fit feature means and population standard deviations using training rows only;
use scale 1 for constant training columns. Reject non-finite results. Validation,
test and unused observations must have no influence on these statistics. Convert
one row at a time when consumed; do not materialize every overlapping window.

A window ends at exclusive input index e, so its latest observation is e-1.
Context C uses [e-C,e). A point forecast H steps ahead targets e+H-1. A curve
forecast of H steps targets [e,e+H). C and H must be integers in [1,1024].
Require e >= C, e >= partition start and e+H <= partition end. Thus validation
may read earlier training history, but every target stays in its own partition
and no forecast origin precedes the final training observation. No shuffling of
partitions. Past validation/test observations may enter later windows as they
become available; weights/scaler remain frozen. This is rolling-origin evaluation.

## Prepared snapshot and replay

Prepare offline from the verified originals. Emit a structural audit, training
scaler and a manifest binding source/config/tool/artifact hashes to a new output
directory. Write the manifest last; incomplete outputs are not accepted. Replay
recomputes the outputs and requires byte-identical files. Never overwrite a run.
The data audit may inspect schema, clocks and finite-value validity throughout;
it must not calculate test errors, correlations, plots or other target summaries.

## Acceptance checks

Write intended failing component tests for parsing, scaling and causal windows,
then integration tests for verified fetch, preparation, failure isolation and
exact replay. Include byte/row/line bounds, HTTP/hash/size failures, malformed
CSV/timestamps, split boundaries, future-data perturbation, constant features and
seeded window properties. Native builds remain usable without Python. This tool
uses only Python's standard library and is not the planned native Python wrapper.

Model loading, training, native/reference agreement and speed comparisons on the
same trained model remain later gates. QTS bars need their own clock, coverage,
cost and out-of-sample design before a returns experiment.
