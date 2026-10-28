# Correction-model learning result

Starting from the latest temperature improved all three runs compared with their
original counterparts. It still did not beat keeping the latest temperature on
the primary measure, average absolute error, in the already inspected test period.
The best old-test score among the three correction runs is only about 1.6% worse
than persistence, but that descriptive observation is not a test-based seed
selection or a new deployment choice.

The model starts with the latest reading and learns an addition or subtraction.
Its zero-initialized head exactly matches persistence before learning. The same
one-block body, seven inputs, 96-row history, one-hour target, seeds, six-round
budget, optimizer and validation rule were retained. This is a combined test of
anchoring and zero-head initialization; their individual effects are not isolated.

## All paired results

| Method | Selected round | Validation MAE | Old-test MAE | Old-test RMSE |
| --- | ---: | ---: | ---: | ---: |
| Keep latest temperature | — | 0.601602 | 0.417916 | 0.588700 |
| Original, seed 7 | 3 | 0.681501 | 0.497620 | 0.660156 |
| Correction, seed 7 | 1 | 0.574392 | 0.448029 | 0.603548 |
| Original, seed 19 | 3 | 0.652982 | 0.581457 | 0.737400 |
| Correction, seed 19 | 1 | 0.576592 | 0.424493 | 0.582040 |
| Original, seed 43 | 6 | 0.740554 | 0.642388 | 0.796845 |
| Correction, seed 43 | 3 | 0.581595 | 0.498065 | 0.658973 |

MAE is average absolute error. RMSE gives larger mistakes more weight. All errors
are in original OT units, over 11,517 validation forecasts and 11,517 old-test
forecasts. Test scores are from Torch; the exported C++ models are checked on the
fixed sampled windows described below. Every seed and every training round is
retained in the [aggregate record](evidence/correction-v1/results.json).

The correction models reduce paired old-test MAE by about 10.0%, 27.0% and 22.5%
for seeds 7, 19 and 43. Nevertheless, their errors remain about 7.2%, 1.6% and
19.2% above persistence. Seed 19 has slightly lower test RMSE than persistence
(0.582040 versus 0.588700). That secondary improvement does not reverse the
primary MAE result, and the other two correction seeds have higher RMSE.

All three selected correction checkpoints improve validation MAE relative to
persistence and their original paired models. This did not carry through to an
old-test MAE win. Rounds one, one and three were selected using validation alone;
all six rounds were completed for every seed. Later training losses kept falling
while validation performance fluctuated. No training extension, new penalty,
calibration or seed selection was made after scoring the old test.

## What now works in the package

The correction route is available in both the Torch reference and C++ engine,
including the Python wrapper, batch sessions and rolling raw-window streams.
Its explicit version-2 model files cannot be mistaken for the original level
forecasts. Old model exports and checkpoints retain their original meaning.
Both output modes preserve the existing shape, finite-value, ownership and
failure contracts. See [the model and experiment contract](correction-model-v1.md).

The model still has 50,689 trainable parameters. This change adds no parameters
or inference buffers. Correction stream/error/reset tests observe no intercepted
C++ heap allocations. These are engineering results, not a measured latency
advantage over another runtime.

## Verification and reproducibility

The frozen experiment code is `34ca3ed152862d1d548b47ca058e52a218cbf84e`.
Local checks pass: 52 native debug groups, 52 release groups, 51 sanitizer groups,
40 training/reference/experiment tests, 22 wrapper tests and 23 data tests.
They cover independent correction answers, nonzero-head equation fixtures,
gradients, future-input independence, file corruption, late overflow, output
preservation, allocation checks, old checkpoint compatibility and relocated
installation. All five native platform jobs and both pinned Torch CPU jobs pass
at the frozen commit; an initial ARM runner-capacity cancellation passed on retry.

The real seed-7 round-one checkpoint was independently resumed through round two.
Every parameter, optimizer tensor, random state and reported training/validation
metric matched exactly. Two independent full scoring passes produced
byte-identical score, start and completion records. Persistence and seasonal
baseline scores also match the original experiment exactly.

For every selected export, both native kernels pass the unchanged absolute 5e-5
plus relative 2e-4 tolerance on 32 fixed windows from each of train, validation
and test, checking all 96 origin outputs per window. The largest observed
absolute discrepancy is 3.814697265625e-6. Sampling is narrower than verifying
every possible input. [Verification and provenance](evidence/correction-v1/verification.json)
record the source, package and log hashes.

This is exploratory evidence on the previously inspected test period. The
12,080 reserved tail rows remain unused for feature construction, training and
scoring. Overlapping forecasts are dependent; no significance, trading or
production-readiness claim follows. Original results remain preserved, and raw
data, learned files and prediction arrays remain local. The source attribution
and rights are in the [original learning report](training-results-v1.md).

A useful next engineering stage is a controlled CPU latency and memory comparison
using the same saved weights and inputs across C++ and Torch. Further model
changes should have their own frozen comparison and discussion with the user.
