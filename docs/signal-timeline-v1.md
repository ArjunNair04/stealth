# Signal timing reference, version 1

Status: implemented and locally tested; cross-platform CI is tracked on the PR. This is an offline,
standard-library reference component under tools/, not a native hot path or a
market connector. It does not alter the model or publish a trading signal.

A decision sees only information that reached us before that decision. Imagine
a monthly reading about January arriving in February: February decisions can
use it; January decisions cannot. A minute decision can combine this reading
with daily and weekly factors and a more recent minute bar.

## Input and identity

A SignalSpec fixes a feature name, source, entity, field and two age limits in
nanoseconds: max_observation_age_ns and max_publication_age_ns. Both are explicit;
there are no guessed freshness defaults. Their ordered list defines a SHA256
schema identifier. Names and source/entity/field triples must be unique.

An Observation contains that source/entity/field, a stable event_id, revision,
effective_at_ns, available_at_ns, received_at_ns and a finite value. None means
an explicit withdrawal, not an accidentally missing measurement. Revisions are
nonnegative integers; gaps are allowed. A changed effective time needs an explicit
upstream correction policy; it cannot silently change under the same event_id.

All times are nonnegative signed-64-bit UTC epoch nanoseconds, represented as
integers. Require effective <= available <= received. This first contract covers
observed point values, including period-end factors released later. It excludes
announced future values, forecasts with a separate target time, interval events,
unresolved local times and ambiguous clock ordering. An adapter must resolve
source time zones, precision, calendars and licensing before creating records.
No date-string conversion or market calendar is provided here. The separately
labelled ETTm1 engineering demo uses invented integer clocks in a synthetic domain;
they must never be interpreted as historical availability evidence.

## Selection at one decision

1. Only records with received_at_ns strictly less than decision_at_ns are visible.
   Equality is deliberately excluded because timestamp equality does not prove
   receipt occurred before a decision. The validated clock ordering also excludes
   unavailable or future-effective observations.
2. For each event, choose its highest visible revision. A delayed older revision
   cannot undo a newer one. Revision publication clocks must be nondecreasing.
3. Choose the event with the latest effective time for the requested feature.
   An old-period correction cannot displace a newer-period observation. Two event
   identities for one feature/effective time are ambiguous and rejected.
4. Apply the observation-age and publication-age limits. Equality to a limit is
   allowed; exceeding either yields stale. Receipt time is never used to refresh
   old information. A withdrawn latest event yields withdrawn, with no fallback
   to an older event. No visible event yields missing.

Identical payloads for the same event/revision are duplicate deliveries; retain
the earliest receipt. Different values or publication clocks for that identity
are conflicts and rejected. Values are finite binary64 numbers and duplicate
values compare exactly, including the sign of zero. Different input order gives
the same snapshot.
Unknown channels are rejected. Inputs are immutable and supplied by the caller.
The complete input batch is validated before a timeline is constructed.

## Output and failure boundary

Snapshots preserve schema order, selection provenance, observation/publication/
receipt ages, and valid/stale/missing/withdrawn states. Non-valid values are None,
not zeros or NaNs. A snapshot is ready only when every feature is valid.
require_values(schema_id=...) rejects a mismatched expected schema or a non-ready
snapshot. The [model bundle and replay bridge](model-bundle-v1.md) now enforce feature-schema
binding and checked float32 conversion; equal feature counts alone do not establish
matching semantics.

The reference bounds features to 256 and caller-supplied records to at most
100,000. The caller may choose a smaller max_records; duplicate deliveries count
toward this limit. Snapshot selection scans stored records and keeps one selected
record per feature. This is the correctness oracle for a future indexed/streaming
implementation, not a low-latency guarantee or an unbounded feed service.

## Acceptance cases

Hand-calculated mixed-frequency snapshots; publication/receipt boundaries;
delayed arrivals and reversed delivery; revisions and withdrawals; duplicate and
conflict handling; per-feature age limits; fixed feature order and schema mismatch;
invalid clocks/nonfinite values/ambiguous identities; bounded consumption; and
unchanged old snapshots after adding future receipts. Compare generated cases
against a separately expressed brute-force selector. Test the runnable example
through its command-line output as an integration check.

No real feeds, QTS datasets, model choices, retraining, latency claims or alpha
results are part of this component. Trading-universe mapping, execution clocks,
costs and portfolio decisions remain downstream of feature assembly.

## Run the reference

```sh
python3 examples/signal_timeline_demo.py
python3 -m unittest discover -s tests -p test_signal_timeline.py -v
```

The four demo decisions show a missing monthly release, its first availability,
a new minute reading, and finally an expired minute reading alongside a corrected
monthly value. The example uses invented clocks and values. The installed
stealth_cpu package and native hot path are unchanged.

Verification: 22 focused tests, including four-frequency assembly, 110 generated
cutoffs checked against an independently expressed selector and a subprocess demo.
The full offline suite has 66 passing tests. Three deliberate in-memory faults
(admitting future receipts, preferring older revisions and refreshing publication
age from receipt) were each detected. Signed-zero duplicate and Unicode-control
regressions failed before their fixes. These checks establish the stated finite
replay contract; durable ingestion, calendars, live recovery and inference
integration remain separate work.
