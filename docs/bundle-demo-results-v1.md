# CPU training and feature replay demo: recorded result

The complete path now runs: verified public observations → PyTorch CPU training →
versioned feature/model bundle → checked replay → native C++ forecasts. The model
still trails persistence on primary error in this small demonstration.

The recipe was frozen in commit `9c96d88628ee599b0ae92c3af01d44fbeba27eee` before
training. It used the existing agreed one-block, four-head correction model and
only the first 3,072 ETTm1 observations for fitting and scoring. Seed 7 ran exactly
two epochs on one CPU thread. Epoch 2 was selected by validation MAE: 1.041214
at epoch 1, then 0.984746 at epoch 2. No training budget or settings changed after
seeing the outcome.

## Prediction results

| Method | Mean absolute error | Root mean squared error | Forecasts |
| --- | ---: | ---: | ---: |
| Native model through feature bridge | 0.889411 | 1.330052 | 509 |
| PyTorch model | 0.889411 | 1.330052 | 509 |
| Repeat latest temperature | 0.863330 | 1.353475 | 509 |
| Temperature at target time on previous day | 2.041966 | 2.662695 | 509 |

Errors use the dataset's original OT units. Lower is better. Native MAE is 3.02%
worse than the latest-value baseline. Its slightly lower RMSE does not change the
primary-metric result. This was a small engineering demo on an already inspected
development region, with overlapping windows; it is not new validation evidence.

## Integration evidence

- All 509 feature-bridge outputs exactly equal direct native latest-only inference
  on the corresponding raw windows. This catches order, scaling and window mistakes.
- Maximum absolute difference from PyTorch across those forecasts is
  `0.000003814697265625`, below the unchanged `5e-5 + 2e-4 * abs(reference)` tolerance.
- The ledger contains 95 warm-up decisions followed by 509 forecasts, with selected
  feature provenance and decision/target clocks. A fresh process reloads the saved
  bundle and reproduces the entire ledger byte-for-byte.
- 20 new tests are included in 67 passing training/reference tests. All 66 offline
  tests and 24 Python-wrapper tests also pass locally. In-memory mutation checks
  caught disabled gap resets, premature state commits and bypassed metadata checks.
- The inference modules import no Torch. C++ numerical code and packaging were
  unchanged; the installed extension's source fingerprint matches this checkout.

The run took 17.38 seconds for training/setup/export selection and 19.39 seconds
through evaluation and ledger creation on the existing local CPU environment
(Python 3.10.6, Torch 2.2.2, NumPy 1.26.4, macOS x86_64). These are one-run elapsed
times, not inference benchmarks or comparisons with current competing runtimes.

## Reproduction and scope

[The contract and commands](model-bundle-v1.md) describe the fixed two-epoch run and
fresh-process replay. Local outputs are under `local/ettm1-bundle-demo-v1/` and
`local/ettm1-bundle-demo-replay-v1/`. They preserve two complete checkpoints,
optimizer/RNG state, training-only scaler, selected weights, bundle, prediction
ledger, scores and hashes. The public repository contains [aggregate evidence](evidence/bundle-demo-v1/results.json),
not the source observations, transformed dataset or model artifacts.

Bundle identifier:
`be20a4395f8e4bea0488a47af41a4675ba792d3a2c5d33ec5931d7a4decace53`.
The aggregate evidence records the code/data/protocol/extension and artifact hashes.
Exact training reproducibility across different processors or Torch releases is
not promised; parity tolerances define the portable numerical check.

Availability clocks in this demo are invented and labelled synthetic. No real
market connector or source timing audit was added. The protected 12,080-row tail
remains unused for fitting, prediction or scoring. No financial hypothesis,
strategy, alpha conclusion or live deployment was attempted. Linux/current-runtime
comparison and a separately agreed predictive study remain the next gates.
