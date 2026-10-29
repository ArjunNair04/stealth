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

## Next local task: feature-to-model contract

Specify and test a versioned bundle that binds feature order/schema, preprocessing
and weights. Add a checked bridge to existing inference sessions and a replay
ledger. Missing/stale/withdrawn inputs must produce an explicit no-forecast result.
Define gap and reset behavior before sending rows to WindowStream; dropping a
minute silently must not make a spaced window look contiguous. Use invented
fixtures first and test component behavior before integration. Consult the user
before changing model inputs, objective, architecture or missing-data treatment.

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
