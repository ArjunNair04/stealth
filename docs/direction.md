# Direction: one trustworthy prediction pipeline

The aim is to turn a changing collection of observations into reproducible CPU
forecasts. A minute-level decision may use a minute bar plus the latest daily,
weekly and monthly factors that were actually available. The signal universe
(inputs) can differ from the trading universe (assets we might eventually trade).
That mapping must be explicit; a feature is not automatically a trade.

We have enough numerical building blocks to pursue this path. We do not need to
build a general deep-learning framework before testing useful predictions.
PyTorch continues to own training, automatic differentiation and its optimiser/
random-number machinery; the existing training scripts determine which options
are exposed. C++ owns the supported inference model. New layers, outputs, depth,
dropout, batching optimisations and specialised kernels need a concrete experiment
or measured bottleneck before expanding the implementation.

## Work order and exit gates

| Stage | Concrete deliverable | Gate before advancing | Current state |
| --- | --- | --- | --- |
| 1. Know what was available | Mixed-frequency feature snapshots with publication/receipt clocks, revisions, age, missingness and fixed order | Independent timing examples, future-information invariance, duplicate/conflict tests and deterministic replay | Offline reference and checked native replay implemented; no real connector |
| 2. Bind features to inference | A versioned model bundle records feature schema, preprocessing and weights; replay feeds the existing session through a checked bridge | Wrong feature order/scaler/schema fails; unavailable inputs produce an explicit no-forecast outcome; gaps, restart and rejected rows preserve declared state | Implemented in tools/model_bundle.py and tools/feature_inference.py; small CPU demo completed with exact replay and documented accuracy limits |
| 3. Reproduce one predictive study | One data snapshot, chronological experiment, simple baselines and saved prediction ledger | Frozen target/timing, leakage tests, reproducible train/export/replay and protected evaluation | BTC return study completed with native replay; primary model misses zero baseline; reused history remains exploratory and independent evaluation remains open |
| 4. Measure runtime value | Same outputs and inputs through native, eager/compiled Torch and ONNX CPU | Numerical parity, exact environment/host record, complete fixed comparison, preserved failures | Linux selected; compiled/ONNX adapters pass fixture CI; Beaker access, host lock and full timed comparison pending |
| 5. Assess one financial hypothesis | One source/feature family, signal-to-trading-universe mapping and QTS evaluation | Point-in-time identities, realistic decision/fill timing, costs, simple baselines, uncertainty and an untouched evaluation | No alpha experiment started |

Stages 2 and 4 can proceed as separate bounded workstreams. Cluster access does
not block the feature contracts. New kernel tuning waits for stage 4. Stage 5
starts once one complete pipeline is trustworthy; it does not wait for every
possible neural-network feature. Paper trading and deployment add their own
monitoring, recovery, risk and forward-testing gates afterwards.

## Mixed-frequency rules

Slow factors keep their latest known value until a newer eligible observation
arrives or their declared age limit expires. Their observation age and publication
age are distinct: an old economic period may have a newly released reading.
Revision history stays intact. We do not interpolate using a later release, reset
freshness on a repeated delivery, silently replace missing inputs with zero, or
assume a source updated merely because a minute elapsed.

Decision times are supplied explicitly. The reference does not choose a trading
cadence, exchange calendar or allowed missing-input model. UTC epoch nanoseconds
are an exact ordering format, not a claim that a source has nanosecond precision.
An adapter must conservatively establish when coarser observations were available.
Actual freshness thresholds, market holidays, release schedules and source rights
are selected with each source; the demo's thresholds are invented examples.

## What counts as enough framework

Before the first financial study, require one adapter with auditable timestamps,
one versioned feature/model bundle, a deterministic replay path with explicit
failure outcomes, and a chronological evaluation against simple baselines.
Require evidence for that specific pipeline. Additional connectors and model
architectures can then reuse the contracts without changing old experiments.

The user selected BTC next-hour return and the first bounded study is complete.
The next discussion will choose one additional signal family, with its timing
contract, before a separately frozen study. Do not infer these from the availability
of minute bars. QTS data remains in QTS and requires its existing clock, coverage,
identity and licensing gates; no current-survivor snapshot becomes validated
alpha simply because a new model consumes it.

## Current limits and evidence

[The baseline review](review-baseline.md) covers the existing native/training/
packaging foundation and its limitations. [The signal contract](signal-timeline-v1.md)
covers the new offline component. [The Linux protocol](frontier-linux.md) separates
prepared acceptance rules from measurements still to be made. Passing component
tests establishes their stated behavior, not infallibility, forecast usefulness,
research novelty or deployability.
