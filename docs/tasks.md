# Active work queue

[Direction and exit gates](direction.md) define the shortest path to a trustworthy
mixed-frequency prediction pipeline. The user selected Linux for the next runtime
comparison and UCL Beaker as the candidate host; interactive login is deferred.
The user expects minute bars alongside daily, weekly and potentially monthly
factors. Exact sources, freshness thresholds and prediction targets remain open.

## Completed foundation

- Baseline PR #18 consolidates #6–#17; their commits and branches are preserved.
  The older PRs are closed as superseded. Independent review/merge remain open.
- The numerical, model loading, Torch training/export and Python/raw-window
  infrastructure pass their recorded checks. Forecast quality still trails
  persistence; the local speed result is limited to its measured workload.
- The [signal timeline reference](signal-timeline-v1.md) assembles bounded offline
  observations by receipt, publication, effective time and revision. It provides
  explicit readiness/age/provenance and schema order, with component and CLI demo
  checks. It is not a feed connector or part of the installed native package.

## Feature-to-model contract and bounded CPU demo

The [versioned bundle and replay bridge](model-bundle-v1.md) bind feature order,
preprocessing, model weights and the decision grid. Missing/stale/withdrawn inputs
produce an explicit no-forecast outcome. A gap resets history; invalid calls and
native arithmetic failures preserve history. The Python reference uses the
existing latest-only native session and keeps one bounded raw window.

The new two-epoch, one-seed [ETTm1 demo protocol](../benchmarks/ettm1/bundle_demo_v1.json)
uses only the first 3,072 observations for fitting and scoring. It is an engineering
demonstration of training/export/replay, with invented availability clocks, not a
new predictive study. The original six-epoch studies and reserved tail stay intact. The [completed demo](bundle-demo-results-v1.md)
produced 509 native forecasts and byte-identical fresh-process replay; primary MAE
was 3.02% worse than persistence. This completes the requested integration demo,
not the search for a useful predictor.

## Next local implementation

Keep the feature bridge as the correctness reference. Before choosing a financial
prediction task, agree the source, target, availability clocks, cadence/calendar
and freshness policy with the user. Live ingestion, persistent crash recovery,
trading-universe mapping and production model activation are separate work.
Current-runtime adapters remain the next performance gate; no new kernel tuning
is justified by this integration demo.

## Parallel gate: current-runtime comparison

Implement and validate the compiled Torch and ONNX CPU adapters. After Beaker
access resumes, inspect CPU queue/toolchain compatibility and obtain a permitted
CPU allocation, then lock the environment and execute the [fixed comparison](frontier-linux.md).
Keep Stealth in a separate checkout/environment. Shared login-node or noisy CI
measurements cannot establish the latency frontier. Preserve failures and partial
runs; do not tune kernels or rerun poor performance into a favourable result.

## Then: one useful predictive experiment

Discuss a single target, horizon and target universe with the user. Use simple
baselines, chronological splits, validation-only choices and a protected final
evaluation. The saved ETTm1 correction models still miss persistence on primary
old-test error; the old test is exploratory, and 12,080 reserved tail rows remain
unused for model selection/scoring. Financial work must pass QTS data and timing
gates and starts as research, with no live order execution.

## Tracking

- [Baseline PR #18](https://github.com/ArjunNair04/stealth/pull/18)
- [Runtime and QTS boundary issue #5](https://github.com/ArjunNair04/stealth/issues/5)
- [Model/reference issue #3](https://github.com/ArjunNair04/stealth/issues/3)
- [Python interface issue #4](https://github.com/ArjunNair04/stealth/issues/4)
- [Historical tasks and preserved outcomes](tasks-history.md)
