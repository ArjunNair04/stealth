# First ETTm1 learning result

The model reads the previous day of seven electricity/temperature measurements
and predicts one temperature an hour ahead. The architecture and six-epoch,
three-seed recipe were fixed before training. Each seed used its best validation
epoch; all seeds completed before the final period was scored.

| Method | Selected epoch | Test MAE | Test RMSE |
| --- | ---: | ---: | ---: |
| Keep latest temperature | — | 0.417916 | 0.588700 |
| Repeat target-time temperature from yesterday | — | 1.529571 | 1.967430 |
| One-block model, seed 7 | 3 | 0.497620 | 0.660156 |
| One-block model, seed 19 | 3 | 0.581457 | 0.737400 |
| One-block model, seed 43 | 6 | 0.642388 | 0.796845 |

Errors are in the original OT units, over 11,517 final-period forecasts.
0 of the three fixed seeds beat the persistence baseline on the primary MAE metric.
This single split does not establish a general advantage. No parameters or
training budget were changed after seeing the test results.

The useful engineering result is a complete working path from Torch training
through saved optimizer/RNG state to a checked native model file. Replaying
seed 7 from epoch 1 to epoch 2 reproduced every weight, optimizer tensor,
shuffle state and reported training/validation metric exactly.

Both native kernels pass the predeclared absolute 5e-5 plus relative 2e-4
comparison on 32 fixed windows from each split, checking all 96 origin
outputs per window. This sampling is explicitly narrower than testing every
possible input. The final-period forecast scores above are Torch scores.

Local training used Torch 2.2.2, Python 3.10.6 and NumPy 1.26.4 with one CPU
thread per process. The older Torch pin is required by the Intel Mac wheel
support boundary. Linux CI additionally exercises Torch 2.13.0. These are
correctness results; no inference speed advantage has been measured here.

The frozen experiment code is `5022cc7fed671749650b53c1b5ef3fdd67254894`.
The subsequent checkpoint-header hardening changes cold-file validation,
not the equations or training recipe. Use the recorded commit and environment
for exact replay. The [training guide](training-v1.md) gives the commands.

The compact [result record](evidence/training-v1/results.json) contains every
epoch/seed, hashes and parity errors. [Verification](evidence/training-v1/verification.json)
records the regression checks. Original data, checkpoints and trained binaries
remain under ignored `local/ettm1-training-v1/`; no dataset redistribution or
trading-readiness claim accompanies these results.

Data source: [ETDataset](https://github.com/zhouhaoyi/ETDataset), pinned to
revision `1d16c8f4f943005d613b5bc962e9eeb06058cf07`. Its CC-BY-ND 4.0
notice is retained in the local source snapshot. This one-hour point task
does not reproduce the commonly reported multi-horizon ETT paper scores.
