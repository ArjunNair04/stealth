# Forecasting a correction to the latest reading

If the latest temperature is 20, the model now starts with an answer of 20.
It learns a correction: +0.2 would produce 20.2, while -0.1 would produce 19.9.
The user approved this comparison after the original transformer and four small
linear baselines failed to beat persistence on the already inspected test period.

The inputs, objective and body stay fixed: seven measurements, 96 quarter-hour
rows, one temperature four steps ahead, one block, width 64, four heads and
hidden width 256. The combined change is latest-target anchoring plus a
zero-initialized head. This experiment does not separately identify the effects
of those two choices. It adds no parameters and does not claim a speed gain.

## Numerical and file contract

`ForecastConfig.output` is `level` by default, preserving the original equations.
The new `correction` mode returns, at each historical origin t:

```
float32(double(head[t]) * target_scale + raw[t, target])
```

It uses the raw target at that origin, not the last row of the whole sequence.
The target mean is not added. Scaling and the transformer body follow the
[original numerical contract](forecast-contract.md). Native output is staged and
published only when every row succeeds; overflow preserves the caller's entire
output. Workspace identity includes the output mode. Rolling windows still reset
positions and recompute; caching is not added.

Torch consumes the same seeded initialization draws as the old model, then zeros
head weight and bias. Thus the starting body matches its paired original seed.
Raw predictions with a zero head exactly equal the input target, without a
normalization/inverse-normalization round trip. `standardized=True` still returns
an absolute standardized forecast: the head plus
`(double(raw_target)-target_mean)/target_scale`, rounded to FP32. The existing loss,
optimizer, clipping and accumulation code is reused unchanged. The head receives
gradients on the first backward pass; body gradients are initially zero and
become available after the head changes. AdamW can still apply weight decay to
parameters with zero gradients.

Level exports remain byte-identical version 1. Correction exports use the same
bounded payload layout and SHA-256 trailer, with these header differences:

| Offset | Correction model value |
| --- | --- |
| 0 | `STLFC02` followed by a zero byte |
| 8 | u32 version 2 |
| 56 | u32 output mode 1 (correction) |
| 60 | u32 reserved 0 |

The loader accepts only the exact version-1/level and version-2/correction
combinations. Unknown or conflicting magic/version/mode values are rejected.
Older readers reject the new format instead of silently producing level forecasts.
Python config accepts `output="correction"`; C++ uses
`ForecastOutput::correction`. Old checkpoint config records without an output
field still mean level. New checkpoints include the mode explicitly, and sealed
run manifests bind the complete checkpoint metadata and exported file hashes.

## Frozen comparison

The task is `benchmarks/ettm1/one_hour_correction_v1.json`; the recipe is
`benchmarks/ettm1/training_correction_v1.json`. Commit them before reading any new
learning outcome. Keep seeds 7/19/43, exactly six epochs, batch 64, accumulation 2,
AdamW settings, clipping, data/scaler, validation selection and parity tolerances
identical to v1. Every seed selects the earliest trained epoch with the lowest
validation MAE. Epoch zero establishes the persistence starting point; it is not
added to checkpoint selection, keeping that rule matched to v1.

All seeds must finish and have their exports sealed before old-test scoring.
Report every seed against its original counterpart and persistence, with MAE
primary and RMSE secondary. The old test is exploratory because it has already
been inspected. Rows 57,600 onward remain excluded from features, training and
scoring; source integrity checks may read the pinned file. There is no new
holdout confirmation, IID significance claim, test-based seed choice, trading
claim or outcome-dependent extension of the training budget.

From the existing training environment, run each seed into a new local directory:

```sh
local/torch-reference-env/bin/python tools/train_correction_ettm1.py train \
  --raw local/datasets/ettm1-v1 --prepared local/datasets/ettm1-prepared-v1 \
  --seed 7 --output local/ettm1-correction-v1/seed7
```

Repeat for seeds 19 and 43, then evaluate all three together:

```sh
local/torch-reference-env/bin/python tools/train_correction_ettm1.py evaluate \
  --raw local/datasets/ettm1-v1 --prepared local/datasets/ettm1-prepared-v1 \
  --runs local/ettm1-correction-v1/seed7 local/ettm1-correction-v1/seed19 local/ettm1-correction-v1/seed43 \
  --output local/ettm1-correction-v1/evaluation
```

Use one CPU thread per process. The runner enforces deterministic Torch settings,
checks the original source/scaler hashes, refuses dirty source and existing
outputs, and retains incomplete runs on numerical failure. Training may use
separate independent processes per seed; these runs are not latency benchmarks.
Use the original study commit/environment for exact replay. Raw data, checkpoints,
exports and prediction arrays stay local; publish aggregates and provenance only.

## Verification

Tests start with independent known outputs: zero correction equals each origin's
raw target, and a constant correction adds exactly its scaled value. The original
double-equation fixture supplies a nonzero-head oracle. Checks cover both native
kernels, full task shape, gradients, future-input independence, prefixes, rolling
streams, reset, wrong workspaces, late overflow and output preservation.

Version tests retain every legacy fixture, reject conflicting rehashed headers,
and exercise every truncation/one-byte corruption of a small correction artifact.
Allocation interception covers correction stream/error/reset calls. Miniature
training verifies native export, all-seed evaluation, and exact resumed weights,
optimizer/RNG arrays and history. Recipe tests reject changed budgets/architecture
and prove that invalid values in the unused tail never enter model tensors.
