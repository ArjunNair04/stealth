# Does a larger forecaster earn its cost?

A wider model has more space to learn patterns, but it must do more arithmetic
for every forecast. This study measures both sides on the existing next-hour
bike-rental task. It changes width, not the forecasting question.

The user approved comparing approximately 51k, 200k and 793k parameters after the
model-capacity discussion. The earlier illustration used seven inputs and 96
history steps. This experiment uses the established bike task's five inputs and
48 steps for every size: respectively 50,561, 199,425 and 792,065 **trainable**
parameters, plus ten fitted scaler values. One block, four attention heads and
an MLP four times as wide stay fixed; each head becomes wider with the model.

## Frozen design

The [size protocol](../benchmarks/bike/size_v1.json) pins the original
[demand protocol](../benchmarks/bike/demand_v1.json) by SHA-256. The runner creates
a separate child recipe for each width; the original recipe and saved run remain
unchanged. All three sizes are freshly trained with seeds 7, 19 and 43, eight
epochs, the same AdamW settings, batches, training windows, fitted scaler and
validation-based checkpoint selection. Epoch zero remains eligible. There is no
outcome-dependent extension, clipping, feature search or optimizer search.

The source is Hadi Fanaee-T's UCI Bike Sharing dataset (2013), CC BY 4.0,
[DOI 10.24432/C5W894](https://doi.org/10.24432/C5W894). Existing clock, missing-hour
and split rules apply. July–December 2012 has already been inspected in the
original study. Every result here is exploratory reuse of that evaluation set;
this is not fresh generalization evidence. Equal training steps are also not
equal computation, or a claim that each size has its best hyperparameters.

Report every seed, selected epoch, MAE, RMSE, negative predictions and the
median/range across seeds. Retain the original persistence, seasonal, calendar
and ridge baselines. No significance-based promotion is attached to this size
search; the inherited child demonstration gates are only descriptive checks
against persistence. No financial conclusion follows from bike demand.

## Native measurements and verification

For every exported model, check native/Torch latest predictions on the complete
validation and evaluation splits and full traces on 16 fixed windows per split
with both native kernels. Preserve the original numerical tolerances. A separate
fresh process replays all evaluation ledgers and requires byte equality.

After training and reference preparation finish, time 27 serial native workers:
three sizes × three seeds × three trials. Cyclic order gives every size each
position once per seed. Each worker uses the same first 32 eligible validation
windows, 64 warmups and 1,000 measured batch-one latest forecasts. Preserve all
raw times; report the median of nine worker p50 and p99 values per size, plus
ranges. These are preloaded native calls; loading, Python conversion and feed
handling are excluded. They are not directly comparable to the earlier 96-step
ETTm1 latency number.

Record measured native weight, position and one-workspace buffers separately
from process RSS, and retain source, binary, model, protocol and data hashes.
Study wall time includes training, checkpoint I/O, validation, baselines and
native parity; it is not a clean training-throughput benchmark. No builds,
training or profiling run alongside formal latency workers. Desktop timing has
no CPU affinity/frequency isolation and implies no guaranteed deadline. Linux
performance remains deferred at the user's request.

The parent completion seal requires every child study. Verification checks all
nested artifact hashes, child configurations and summary consistency. Failed
runs retain their failure record and checkpoints; new runs require new folders.
Tests exercise changed protocols, incomplete/duplicate timing grids, repeated
prediction checksums, memory inconsistencies, a tiny real train/export/replay
and native timing cycle, tampering and partial-run failure.

## Reproduce with the existing declared environment

Commit the frozen code and protocol before a real run. Build the Release native
benchmark and install the matching Python extension. Use one-thread BLAS settings
for every command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1`. No new dependency is required.

```sh
local/torch-reference-env/bin/python tools/bike_size.py train \
  --csv local/datasets/bike-v1/hour.csv --output local/bike-size-v1
local/torch-reference-env/bin/python tools/bike_size.py replay \
  --csv local/datasets/bike-v1/hour.csv --run local/bike-size-v1 \
  --output local/bike-size-v1-replay
local/torch-reference-env/bin/python tools/bike_size.py benchmark \
  --csv local/datasets/bike-v1/hour.csv --run local/bike-size-v1 \
  --binary build/release/stealth_runtime_bench --output local/bike-size-v1-latency
```

A useful larger model must earn its additional measured cost. This experiment
provides the trade-off; it does not silently replace the existing model or change
the agreed output/loss. Any count-aware redesign needs a separate discussion.
