# Local latency audit, 30 October 2026

The existing model now does the same calculation with less surrounding work.
On the local Intel Mac, typical single-forecast native latency fell from 185 to
136 microseconds. The median reduction across paired trials is 27.65%; the ratio
of the two table medians is 26.33%. Predictions remain bit for bit identical.
The Python feature adapter also saves work by reusing immutable provenance records.

The user authorized local optimization while keeping controlled Linux/Beaker
performance work deferred. This updates the earlier decision to postpone kernel
work. Model design, weights, training, forecasting objectives and data splits were
held fixed. The reserved ETTm1 tail was not used.

## What changed and why

1. Dense projections consumed 61% of the initial native profile. Full eight-output
   tiles now have constant bounds, with a separate partial-tile path. AppleClang
   replaces four per-tile bias `memcpy` calls with vector loads in the full four-row
   path. Ordered FP32 accumulation and the scalar reference remain unchanged.
2. Finite-value checks now reduce validity across a span, allowing SIMD scans.
   Every value is still checked before arithmetic or publication. This scans the
   complete span instead of returning at the first invalid value: invalid-input
   latency can increase, while the existing worst-case work bound is unchanged.
3. Normalization computes independent channels together and masks unrepresentable
   doubles to zero **in scratch only**, before narrowing. Any invalid channel
   rejects the whole output. Mean/variance reduction order, FP64 intermediate
   arithmetic, FP32 outputs and disabled FMA contraction remain unchanged.
4. `NativeSignalTimeline` builds a read-only lookup of canonical observations at
   construction. Queries reuse those immutable records instead of rebuilding and
   revalidating their identifiers and timestamps. The independent Python scan
   oracle and `FeatureReplay` state machine are unchanged. Snapshot/decision
   construction still allocates in Python.

These are portable C++20/Python changes. No AVX-only build flags, new dependency,
quantization, approximate activation or removed safety check was introduced.

## Native measurement

The [frozen protocol](../benchmarks/ettm1/local_latency_v1.json) was committed before
kernel changes. It uses the same 32 saved validation windows, three trained model
seeds, three paired trials, batch sizes one/eight, and latest/full output modes:
72 serial worker processes. Baseline and candidate order alternates. Every worker
checks every requested output before warm timings. Sampling is separate from timing.

Workload: seven features, 96 observations, width 64, four heads, hidden width 256.
Intel Core i5-10600, macOS 14.8.7, AppleClang 16, Release `-O3`, one inference
thread, no affinity/frequency control. Times below are microseconds per call;
batch-eight calls contain eight sequential windows. Each value is the median of
nine worker quantiles, not a quantile of a pooled sample.

| Output / batch | Before p50 | After p50 | Before p99 | After p99 | Median paired p50 reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Latest / 1 | 185.145 | 136.399 | 287.107 | 194.262 | 27.65% |
| Latest / 8 | 1,481.795 | 1,089.371 | 1,967.474 | 1,391.900 | 25.50% |
| Full / 1 | 1,480.221 | 1,315.521 | 1,985.300 | 1,706.927 | 11.41% |
| Full / 8 | 11,695.256 | 10,339.819 | 13,104.066 | 12,056.561 | 11.93% |

All frozen gates pass. Each seed's median latest/batch-one reduction is positive:
26.64%, 29.26%, 27.65% for seeds 7, 19, 43. The primary paired p99 improves 28.20%.
Individual tails still vary: one latest/batch-eight pair worsens 17.20%, and one
full/batch-one pair worsens 4.22%. All pairs and raw timed samples are retained in
[native evidence](evidence/local-latency-v1/native.json). No hard deadline follows
from these warm desktop measurements, and other shapes/CPUs need separate timings.

Native owned memory is unchanged: 204,604 parameter bytes, 24,576 position bytes
and 222,240 workspace bytes per session, excluding caller buffers and bookkeeping.
The existing allocation-interception tests still pass on successful and selected
failure paths. Cross-window batching remains sequential; this does not maximize
all CPU cores or establish a new frontier against current inference engines.

## Python feature path

A separately [frozen probe](../benchmarks/temporal/bridge_latency_v1.json) compares
the old and new adapters using the **same optimized native extension**. It uses
one saved model, seven invented streams, 7,700 observations and 1,000 warmed
forecast decisions per worker; three pairs run in alternating order. It times the
whole `FeatureReplay.advance` call, including snapshot selection, input conversion,
native prediction and decision construction. Construction is excluded.

Median worker p50 falls from 245.575 to 216.445 microseconds: 11.74% median paired
reduction. Median paired p99 improves 14.66%, but the third pair's p99 worsens
14.23%. Every predicted value matches bit for bit. These percentages cannot be
added to the native improvement; the probes have different boundaries and inputs.
All six workers and raw samples are in [bridge evidence](evidence/local-latency-v1/bridge.json).

The lookup trades setup memory for less repeated work. For this 7,700-record tape,
its dictionary and tuple keys occupy 726,200 bytes, excluding integer objects,
shared observations and allocator overhead. This replaces a smaller event-name
tuple. Lookup size is bounded by the existing 100,000-record admission limit;
there is no growing query cache. The native allocation-free query contract is
unchanged; Python is still an allocating convenience interface.

## Rejected candidates and remaining work

Two additional candidates passed numerical tests but missed the 5% incremental
selection threshold. Four-row input preparation saved 2.84% and added 852 bytes of
workspace for the measured model; it was removed. Contiguous row stores removed
scalar lane extractions in this compiler, but saved only 2.13%; that change was
also removed. [Every development cell](evidence/local-latency-v1/development.json)
and both rejected patches remain recorded. Incremental development runs explicitly
name their intermediate baselines; only the final run compares the accepted code
with the original frozen baseline from a clean commit.

The final sampled native profile attributes about 62% to dense projections, 15%
to normalization and 10% to attention. Percentages describe shares of remaining
work, not absolute speed changes. Python profiling still finds full-window NumPy
conversion and row validation costs. Next candidates are a separately tested
buffered replay adapter, broader shape coverage, and optional CPU-specific matrix
backends with portable fallbacks. Caching earlier keys/values requires proving
position/reset equivalence. Threading, pinning and queue tuning require actual
arrival rates and a deployment latency budget. Linux work stays deferred until
the user makes that environment available.

## Verification and reproduction

Local results: 66 Debug and 66 Release native tests pass, including relocated
consumers; 65 sanitizer tests pass. The 83 offline tests, 32 isolated installed
wrapper tests, and 107 training/reference/integration tests pass; three optional
Linux-engine tests skip locally. The integration suite includes adversarial
revisions, duplicate deliveries, stale/missing/withdrawn inputs, concurrent
readers, transactional failures and fresh-process replay. This change's new
provenance test failed before implementation and passes after it.

The saved-model audit checks 9,216 full outputs plus all 9,216 prefix predictions
against the original installed extension, bit for bit. The maximum discrepancy
from the separate high-precision reference in the frozen native benchmark is
1.90735e-6, within the unchanged 5e-5 absolute / 2e-4 relative tolerances. Tests also
cover non-binary dense inputs, row/column tile edges, NaNs/infinities in every
scan position, signed zero, subnormals and overflow in every normalization channel.
See [hashes and verification](evidence/local-latency-v1/manifest.json).

Preserve the old Release executable before rebuilding the candidate, prepare the
unchanged inputs/models with the existing latest-output benchmark, then use a
fresh ignored output directory:

```sh
python3 tools/compare_native_latency.py \
  --baseline local/baseline/stealth_runtime_bench \
  --candidate build/release/stealth_runtime_bench \
  --inputs local/runtime-latest-v1 --models local/ettm1-correction-v1 \
  --output local/my-new-latency-comparison
```

For the bridge, save `tools/native_signal_timeline.py` from baseline commit
`57ea560` and use `tools/benchmark_feature_bridge.py --help`. Both arms require the
same current Release extension; source and input hashes are checked. The exact
small parity-capture script is preserved in the evidence directory; copy it to
`local/local-latency-v1/capture_predictions.py` and run it with a new output path
in each baseline/candidate Python environment. It uses the saved local paths above.
Do not run timing alongside builds, tests or profilers. Fresh reruns are evidence
about that environment, not an instruction to replace an inconvenient result.
