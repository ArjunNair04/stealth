# Feature/model bundle and CPU demo

A bundle is the model's recipe card: which inputs it expects, in which order,
how those numbers are scaled, which weights to use, and which clock defines one
step. The replay bridge checks that card before feeding observations to C++.
PyTorch still owns training; native C++ still computes the forecasts.

This is an optional research workflow under `tools/`. It requires the declared
NumPy dependency and the installed `stealth_cpu` extension. The bundle loader and
bridge do not import Torch. They are not yet part of the installed Python package.
The native library and model file format are unchanged.

## Bundle contract

`create_bundle` writes a new directory containing `model.stlf` and a final
`manifest.json`. It never replaces an existing directory. The manifest binds:

- Ordered signal specifications, including source/entity/field and freshness
  limits, plus the timeline's feature-schema digest.
- Model bytes by SHA256; the exact native configuration and target feature name.
- The FP32 means and scales already embedded in the native model. The bridge sends
  raw inputs; scaling occurs once inside native inference.
- Positive fixed cadence, grid origin, `utc` or explicitly `synthetic` clock
  domain, strict receipt cutoff, gap behavior and target-clock interpretation.

The bundle identifier is the SHA256 of the canonical manifest bytes. The caller
pins this identifier from its trusted run record when calling `load_bundle`.
The loader reads bounded files, validates the native blob, and checks the entire
manifest against it. Unknown fields, malformed/nonfinite/duplicate JSON keys,
feature/scaler/config mismatches and changed bytes fail. It loads a candidate
without modifying an existing active model. Digests detect changes; they do not
authenticate a publisher. Changing a schema deliberately requires a new bundle
and a new trusted identifier.

Manifest size is capped at 128 KiB; the native blob is capped at 16 MiB and uses
the existing native dimension bounds. Files are read once into bytes. No filename
from the manifest is used to open a path. Manifest-last writing identifies a
complete bundle, but does not promise crash-durable storage or atomic live-model
activation.

## Replay behavior

`FeatureReplay.advance(timeline, decision_at_ns)` checks the bundle's exact feature
schema and grid, then asks the timeline what was known strictly before that
cutoff. All observations are point values under the timeline contract.

| Situation | Result | History afterward |
| --- | --- | --- |
| Valid contiguous row, less than full context | `warming_up`, no forecast | Append row |
| Valid contiguous row, full context | `forecast` | Keep latest context rows |
| Missing, stale or withdrawn required feature | `unavailable`, no forecast | Empty; consume this decision time |
| Forward jump over one or more grid steps | Reset reason `gap` | Start a fresh window with this valid row |
| Wrong schema, off-grid/repeated/reversed time, FP32 overflow or native error | Exception | Prior history and decision time preserved |
| Explicit `reset()` | New episode | Empty history and no last decision |

With a 96-row context, publication resumes only after 96 contiguous valid rows.
Slow features may carry forward while their explicit age limits allow it. Later
revisions affect new decision rows, never rewrite previously consumed rows.
No interpolation, zero filling, missing-row compression or exchange calendar is
inferred. Market closures require a separately agreed policy.

The forecast target is `decision_at_ns + horizon * interval_ns` on this decision
grid. This is not automatically a source observation clock or a tradable order
clock. An adapter must establish the label relationship, source timezone and
availability before a real study. Target clock overflow fails before mutation.

One replay owns one native session and a bounded window; concurrent use of the
same replay, including reset, is rejected. Every candidate prefix is numerically
checked before committing, including warm-up prefixes. A successful prediction
uses `predict_last`; it does not compute unused earlier output heads. This Python
oracle still allocates and scans the offline timeline. It makes no hot-path or
latency claim. Restart is explicit reset plus replay from the beginning of the
needed history; no crash-persistent state snapshot is provided yet.

## Small CPU training demonstration

The user previously selected seven ETTm1 inputs, the preceding 96 quarter-hour
observations, and a single temperature forecast one hour ahead. This demo retains
the agreed one-block correction model: width 64, four heads, hidden width 256.
It starts from the latest temperature and learns a correction.

The new frozen `benchmarks/ettm1/bundle_demo_v1.json` uses seed 7, exactly two epochs,
one CPU thread and AdamW with the existing training machinery. Its row boundaries
are 2,048 for training, 2,560 for validation and 3,072 for demo evaluation. Scaling
is fitted only on the first 2,048 rows. The earliest lowest-validation-MAE epoch
is selected before evaluation; both persistence and previous-day baselines are
reported. No outcome-dependent extension or parameter search is allowed.

This produces 1,949 training windows and 509 windows in each later split. Targets
begin at row 99, 2,051 and 2,563 respectively (zero-based); each target is four
steps after the last input. The first three targets of each later split are
omitted by the existing window contract. Overlap is allowed in input histories,
not in training targets across split boundaries. The final 12,080 reserved rows
are not fitted, predicted or scored. Source integrity and structural verification
still check the whole file.

ETTm1 does not establish historical publication/receipt clocks or a timezone.
The demo invents receipt at `row_index * interval_ns` and decision one tick later,
labels the bundle `synthetic`, and uses a one-tick age limit. This exercises exact
ordering and missing-input behavior without claiming actual historical
availability. Source date strings are retained separately in local examples.

Use the existing declared training environment; these commands install nothing:

```sh
local/torch-reference-env/bin/python tools/bundle_demo.py train \
  --raw local/datasets/ettm1-v1 --output local/ettm1-bundle-demo-v1
local/torch-reference-env/bin/python tools/bundle_demo.py replay \
  --raw local/datasets/ettm1-v1 --run local/ettm1-bundle-demo-v1 \
  --output local/ettm1-bundle-demo-replay-v1
```

Commit the code/protocol first. Outputs must be new ignored `local/` directories.
The run preserves checkpoints, optimizer/RNG state, all epochs, scaler, model
bundle, per-decision feature provenance, predictions, baselines, environment,
code/data/protocol hashes and an artifact seal. Failures remain recorded. The
second command reloads the bundle in a fresh process and requires a byte-identical
ledger. It does not retrain the model. The CLI uses the training environment for
verification convenience; the inference modules themselves need no Torch.

## Verification and limits

Component tests cover bundle integrity, preprocessing, schema order, clock and
range checks. Integration tests compare independently known zero-correction
answers and nontrivial Torch/native outputs; they exercise slow factors, strict
receipt boundaries, missing/stale/withdrawn inputs, gaps, resets, revisions,
future-input invariance, busy sessions, genuine native arithmetic failure and
transactional recovery. A miniature train/export/replay runs entirely offline.
The existing training CI discovers these tests on both declared CPU Torch versions.

```sh
local/torch-reference-env/bin/python -m unittest discover -s tests/training -v
python3 -m unittest discover -s tests -v
```

The demo is pipeline evidence on an already developmental slice. It does not
establish useful predictions, financial alpha, current-runtime superiority or
deployment readiness. [The completed run and its limits](bundle-demo-results-v1.md) are recorded separately.
