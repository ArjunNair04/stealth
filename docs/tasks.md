# Active work queue

[Direction and exit gates](direction.md) define the shortest path to a trustworthy
mixed-frequency prediction pipeline. The [Beaker Linux validation](beaker-linux-v1.md)
now passes native, sanitizer, Python and training/reference gates. A scheduled,
CPU-pinned comparison beats eager Torch 2.6 at batch one and loses at batch eight.
The current compiled/ONNX comparison remains open; Linux access itself is resolved.
The user expects minute bars alongside daily, weekly and potentially monthly
factors. Exact sources, freshness thresholds and prediction targets remain open.

## Completed foundation

- Baseline PR #18 consolidates #6–#17; their commits and branches are preserved.
  The older PRs are closed as superseded. Independent review/merge remain open.
- The numerical, model loading, Torch training/export and Python/raw-window
  infrastructure pass their recorded checks. ETTm1 and BTC studies miss their forecast gates; the later bike study improves
  MAE over persistence and ridge. Local speed evidence is workload-specific.
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
During the earlier Beaker deferral, the user requested a useful forecasting
application. The [hourly-data admission and study proposal](forecast-study-intake-v1.md)
now provide a checked local candidate and a concrete target choice. The user chose BTC next-hour return. The [frozen return study](btc-return-study-v1.md)
is now complete with a negative predictive result and successful native replay. Current-runtime
adapters remain the parallel performance gate; no new kernel tuning is justified
by the integration demo or by data admission.

## Parallel gate: current-runtime comparison

The [compiled Torch and ONNX CPU adapters](runtime-adapters-v1.md) now have real
Linux CI parity checks and a bounded per-cell admission runner. The complete
90-cell timed orchestrator and aggregate gate are still pending. Beaker CPU
allocation and the native toolchain are now tested. Its CentOS 7 environment has
only been admitted for the separate Torch 2.6 compatibility run. Inspect a modern
container/runtime environment, lock it, then execute the [fixed comparison](frontier-linux.md).
Keep Stealth in a separate checkout/environment. Shared login-node or noisy CI
measurements cannot establish the latency frontier. Preserve failures and partial
runs. The separately authorized local audit has its own frozen baseline and gates;
do not rerun poor performance into a favourable result.

## First financial forecasting experiment: completed

The user selected BTC next-hour return. The [frozen study](btc-return-study-v1.md)
trained all three seeds for three epochs, selected on validation only, and scored
5,160 evaluation hours. The [recorded result](btc-return-results-v1.md) fails its
forecast-improvement gate: the primary learned model's MSE is 0.036% worse than
zero return, and the other two seeds selected the initial constant. All 28,512
validation/evaluation latest predictions passed native/Torch parity, and the saved
ledger replayed byte-for-byte in a fresh process. No alpha or trading result.

The next research decision is one additional signal family with an explicit
availability contract and a new frozen experiment. Preserve this result and avoid
using the same evaluation period as independent confirmation. Model expansion
needs evidence from that research. The later local audit changes execution only;
current-runtime frontier claims still require the controlled Linux comparison.
The ETTm1 reserved tail remains unused.

## Tracking

- [Baseline PR #18](https://github.com/ArjunNair04/stealth/pull/18)
- [Runtime and QTS boundary issue #5](https://github.com/ArjunNair04/stealth/issues/5)
- [Model/reference issue #3](https://github.com/ArjunNair04/stealth/issues/3)
- [Python interface issue #4](https://github.com/ArjunNair04/stealth/issues/4)
- [Historical tasks and preserved outcomes](tasks-history.md)

## Structured public learning demonstration: completed

The user requested a simpler task after BTC. The [bike-demand study](bike-demand-study-v1.md)
trained the existing one-block model to predict next-hour rentals from 48 observed
hours and known calendar information. [All three seeds passed the frozen learning
gate](bike-demand-results-v1.md). Primary MAE is 43.82 rentals versus 87.15 for the
latest count and 47.90 for ridge; ridge has the better RMSE. Native/Torch parity
passes on 24,198 latest predictions, and fresh-process replay is byte-identical.
The first-week plot and all results are public with UCI attribution.

The primary model still gives 292 negative forecasts and misses some demand peaks.
Before further model changes, discuss a count-aware output/loss and which kind of
error matters most. Any follow-up needs a new frozen protocol; the existing test
months are now inspected. Controlled Linux runtime benchmarking remains the
separate performance priority when Beaker access becomes available.

## Mixed-speed native feature milestone: implemented

[SignalIndex and its frozen conformance workload](native-signal-index-v1.md)
replace repeated full-tape scans with per-channel native receipt indexes while
preserving the independent Python oracle. Snapshot queries use caller-owned
memory; construction and Python conversion allocate. The finite index supports
up to 256 channels and 100,000 deliveries, exact revision/provenance rules,
arbitrary historical queries and concurrent readers.

The adversarial diagnostic produces 98 identical native/oracle decisions:
60 forecasts, 17 warm-up decisions and 21 unavailable decisions. All 60 forecasts
match eager Torch within the existing tolerance; fresh-process ledgers are
byte-identical. New direct native, binding, integration, allocation and failure
tests accompany the implementation. Actual compiled/ONNX conformance is wired
into the optional Linux job; its CI result is separate from local evidence.

Next: finish the fair end-to-end runtime harness and controlled Linux measurement.
The feature adapter used by standard runtimes must receive the same clocks,
corrections, warm-up and gap rules. Live ingestion, bounded eviction and persistent
recovery remain separate unimplemented components. No new neural kernel tuning,
model architecture or financial objective is implied by this milestone.

## Local latency audit: completed

Three portable native changes reduce paired single-forecast median latency by
27.65% on the fixed Intel Mac workload, with unchanged native owned memory and
bitwise predictions. Immutable provenance reuse separately reduces the full Python
feature-to-forecast probe by 11.74% using the same optimized native core. Two
marginal candidates were removed. All local correctness, sanitizer, wrapper and
integration gates pass; [full evidence and limits](local-latency-v1.md) are recorded.

Next local performance candidates: benchmark broader shapes; specify a separate
buffered feature adapter to avoid repeated tuple-to-NumPy conversion; then evaluate
optional matrix backends from profiles. Keep the existing replay bridge as oracle.
Linux/Beaker timing remains deferred until the user provides access.

## Assembly-guided optimization: completed

The [second local audit](assembly-latency-v1.md) accepts a runtime-guarded AVX dense
specialization after assembly inspection, a 108-shape probe and exact parity.
The final complete-model comparison reduces median paired latest latency by
29.43% over the already optimized baseline; the portable fallback remains tested.
Tiny products stay portable. Cached FP64 softmax exponentials failed the primary
gate and were removed, with their evidence retained.

Next focused experiment: group independent output tiles in single-row dense
calls to shorten dependency stalls without changing accumulation order. Then
measure normalization across independent rows. Keep public validation and the
Python replay oracle intact. Controlled Linux/Beaker performance remains deferred.

## Model capacity comparison: analyzed; large-family numerical gate failed

The [completed diagnostic comparison](bike-size-results-v1.md) trained widths
64/128/256 across three seeds under the frozen eight-epoch bike recipe. Native
median p50 is 53.302/147.513/492.298 µs, with 0.340/1.071/3.715 MB of owned buffers.
The medium model's median MAE improves only 2.56% for 2.77 times the inference
cost; the largest family has a weaker median than the medium family.

The small and medium children pass their numerical gates. One large-model
near-zero output differs from standard Torch by 0.0000986643 rentals, exceeding
the frozen allowance. Both native kernels agree exactly, and changing only
Torch dense accumulation to the native order reproduces that result. The failed
parent and child remain unsealed; no tolerance was relaxed. All 37,458 evaluation
predictions replay byte-identically in a fresh process. The final local training
suite passes 114 tests with three optional Linux skips, including a regression
for preserving checkpoints and earlier results after a late parity failure.

Keep width 64 as the efficient default for this inspected task. Next numerical
work: specify an independently justified error budget for cross-runtime FP32
accumulation and near-zero outputs, then validate it on new cases. Preserve this
failure as evidence. Further predictive changes still need the user's input on
outputs/loss and a new evaluation design. Linux performance remains deferred.
