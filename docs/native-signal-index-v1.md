# Native signal index v1: frozen contract

This milestone replaces repeated full-history scans with an immutable C++ index.
It answers: "At this decision time, what did we actually know?" It preserves the
Python signal timeline as an independent selection oracle. The first version
indexes a finite validated tape; it does not accept live appends or implement a
crash-durable journal. Building an index allocates; valid and rejected C++ snapshot
calls use caller-owned output storage and must allocate nothing.

## Scope and rules

The low-level schema is an ordered list of observation/publication age limits.
Records use a channel slot and an event number. String identity, feature names,
schema hashing and model binding remain in the existing checked Python boundary.
C++ callers own that mapping; matching dimensions alone is not schema validation.

There are 1–256 channels and at most 100,000 input deliveries, including duplicates.
Times and revisions are nonnegative signed 64-bit integers. Effective time must
not exceed publication time, which must not exceed receipt time. Values are finite
doubles; an absent value is an explicit withdrawal. Forecast conversion to FP32
remains checked by the existing replay bridge.

For each channel/event, effective time is fixed and publication time cannot
decrease with revision number. Different events cannot share a channel/effective
time. A duplicate revision must have exactly the same payload, including the sign
of zero, and keeps its earliest receipt. Any conflict rejects the whole tape,
even if that record would only become visible in the future.

Snapshots include only receipts strictly before the supplied decision. The winner
per channel is the largest (effective time, revision) pair. A late older revision
cannot undo a newer one; a correction to an older event cannot displace a newer
event. Missing, withdrawn, stale and valid are distinct. Equality to either age
limit is valid. Withdrawn takes priority over stale. Invalid decision/output size
leaves every output slot untouched. Snapshot order is arbitrary and read-only;
separate output buffers allow concurrent queries.

## Implementation and acceptance plan

Validate and canonicalize once, then store winner changes in receipt order for
each channel. Binary search the last change strictly before the decision. Storage
is O(deliveries + channels); query work is O(sum log(channel changes + 1)).
Historical forecasts are never rewritten by later corrections.

Before implementation: hand-derived boundary/correction tests, independent scan
comparisons, invalid lineage/duplicate tests, permutation and future invariance,
restart reconstruction, concurrent readers and allocation interception.
Run debug, release, sanitizers and relocated consumer checks. Python integration
must compare complete snapshots and forecast decisions, not just model outputs.
Use the existing model and weights; no new training or architecture is required.

## Performance and limits

This is an algorithmic change, not a measured speed claim. Python conversion and
replay still allocate. No lock-free, branch-free, bounded live-state, crash recovery,
or end-to-end zero-allocation claim follows from the C++ snapshot test.

A later controlled comparison must use the same tape, features, warm-up, gap rules,
weights and outputs for native, Torch eager/compiled and ONNX CPU. Count feature
assembly, marshaling and inference separately and together; retain unavailable
decisions and failures. Report setup cost, memory, p50/p99 and variability. Compare
a thin checked adapter plus the standard runtimes before claiming a useful niche.
Shared CI establishes correctness only; Beaker access and the host lock remain open.

## Implemented evidence and use

The C++ API is `stealth::temporal::SignalIndex` in
`include/stealth/signal_index.hpp`. Supply numeric limits/records at construction,
then a preallocated `std::span<SignalFeature>` to `snapshot`. It owns copies of its
inputs. Replacing an index after a failed construction can leave the old instance
active; allocation-failure tests exercise every intercepted construction allocation.
All callers must preserve the channel/event identity mapping.

The optional repository adapter `tools/native_signal_timeline.py` subclasses
the existing immutable timeline interface and can be passed directly to
`FeatureReplay.advance`. It preserves named schema checks, readiness, provenance,
FP32 checks and gap/warm-up rules. Its constructor shares the Python validator,
then validates the numeric tape in C++; its queries never call the Python scan.
Native validation is tested directly as well. The adapter retains the Python
canonical tape and allocates Python results. It remains a repository tool; the
installed wheel contains the internal numeric binding. That private boundary
accepts exact built-in lists/tuples and numeric values, preventing custom container
callbacks from changing the size between validation and copying. The named
repository adapter still accepts bounded iterables through its existing validator.

With the declared training dependencies and a rebuilt Release wheel:

```sh
python tools/temporal_conformance.py --output local/temporal-check-1
```

The output directory must be new. It contains the frozen synthetic tape, model,
bundle, complete oracle/native decision ledgers, external predictions, source and
artifact hashes, and a report. Failures retain partial artifacts and a failure
record. `--engine torch_compiled_inductor` and `--engine onnxruntime_cpu` require
the existing optional frontier environment. No dependencies are installed by the
runner. Compilation is required to generate a kernel; there is no eager fallback.

The fixed tape has 269 deliveries, 222 unique revisions, seven cadences and 98
decision times. It exercises delayed and out-of-order revisions, duplicates,
missing/stale/withdrawn inputs, restoration and a separate decision gap. The native
index stores 218 winner changes. Native and scan-based replay produce identical
ledgers: 60 forecasts, 17 warm-up decisions and 21 unavailable decisions. Local
eager Torch 2.2.2 agrees on all 60 forecasts; maximum absolute difference is
9.5367431640625e-7 against the pre-existing 5e-5 + 2e-4 * abs(expected) tolerance.
Fresh processes reproduce the tape, model and native/oracle ledgers byte-for-byte.

This harness checks external runtimes on the identical complete forecast windows.
Warm-up arithmetic and decision control still run through the existing native
bridge. It is not yet an independent end-to-end serving implementation for each
runtime, nor a comparison of their total processing cost.

### Validation record

- The new C++ test first failed because its header was absent; Python adapter and
  conformance tests first failed on their missing modules.
- Six native groups cover direct cases, invalid tapes, deterministic generated
  scan comparisons, bounds, future invariance and concurrent readers. Capacity
  tests include 100,000 distinct records and 100,000 duplicate deliveries.
- Allocation interception records zero allocations for 10,000 native query/error
  cycles and injects failure at every index-construction allocation.
- Seven installed-wrapper tests compare complete native/Python snapshots and query
  concurrency, validate bindings and prove queries bypass the Python scan.
- Six local integration tests cover complete decisions, eager parity, independent
  process reconstruction and injected mismatch failures. One further test runs
  actual Inductor and ONNX on the same windows in optional Linux CI.
- Local debug and Release suites each pass all 65 tests, including the relocated
  consumer. The sanitizer suite has 64 tests because package relocation is
  excluded under instrumentation; all 64 pass after the fixture correction.
- All 79 offline tests, 107 of 110 training/reference tests (three optional Linux
  tests skipped), and all 31 installed-wrapper tests pass. The source distribution
  builds a wheel and passes a fresh offline NumPy-only runtime installation.

Two development failures are retained in the record. Initially the gap fixture
overlapped a stale-input interval, so it correctly produced an unavailable result;
the gap was moved to a separately valid interval without relaxing its assertion.
AddressSanitizer then caught a use-after-free in the test's own
`vector.assign(count, records[0])` setup. The repeated value is now copied before
resizing, and the sanitizer suite was rerun. A subsequent review added a failing-then-passing regression for custom Python
containers and restricted the private binding to ordinary lists/tuples before any
copy or allocation. No production tolerance or validation was weakened. Raw build/failure logs remain under ignored local output directories.

Pinned local artifacts:
- Tape SHA-256: `d935fa2629fbbe159661cc9b0f30c1f028875a5c8a4f36c95e52951d4e224790`.
- Model SHA-256: `7e9b8b9574c9b90b490d7f4cebca83b443350d0552f5e8b798629be490394375`.
- Both decision ledgers: `4cc60d6c5f49a5c1d1a80bda30d6fa7b505c51c74a6f15785731d1b1817cc624`.

No training run, model architecture, existing experiment, model file format or
numerical kernel changed in this milestone.
