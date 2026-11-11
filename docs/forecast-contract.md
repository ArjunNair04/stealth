# One-block point forecaster: version 1 contract

The user selected one temperature one hour ahead and one transformer block first
on 28 October 2026. The task is fixed in `benchmarks/ettm1/one_hour.json`: 96 rows,
seven measurements per row, target OT four 15-minute steps after the latest input.
For a latest reading at 10:00, use the 96 readings from the preceding day's 10:15
through today's 10:00 inclusive; predict 11:00. No intervening future rows enter
input. Later two-block comparisons need a new recorded model/experiment version.

## Model and arithmetic

The first model has width 64, four heads and hidden width 256. Normalize each raw
feature using frozen FP32 means/scales fitted on training rows only. Compute
`(double(x)-mean)/scale`, check finite FP32 range, then round to FP32. Apply a
learned biased input projection, then add a fixed sinusoidal position vector.
Use the existing pre-norm causal attention/feed-forward block, final layer norm,
and a biased scalar linear head. The head predicts standardized future OT;
return `double(head)*target_scale + target_mean`, checked and rounded to FP32.
No dropout is applied. Exact GELU and epsilon 1e-5 follow the existing core.

Positions restart at zero for every full window. For position p and channel pair
2i/2i+1 use sin/cos of `p / 10000^(2i/width)`, computed in double and stored in FP32
at construction. Add to the input projection with checked double addition then
FP32 rounding. This uses the positional formula from
[Attention Is All You Need](https://arxiv.org/abs/1706.03762); the full forecaster
here is our small diagnostic architecture, not a reproduction of that paper.

The benchmark consumes the final row's single forecast. Native `forward` also
returns the same one-hour forecast at each earlier position for diagnostic
prefix/causality tests. These are different historical forecast origins, not a
curve of future temperatures. Short prefixes are diagnostic; the benchmark always
uses 96 input rows and scores only the final output. Rolling a window resets its
positions and recomputes it; no cache-equivalence claim is made.

## Native API and bounds

`ForecastConfig` records features, width, heads, hidden width, maximum context,
horizon, target feature and epsilon. A single block is fixed by format version 1.
Diagnostic shapes may vary: features 1..64, width 1..512, heads divide width,
hidden 1..2048, context/horizon 1..1024, target index below features, epsilon finite
and positive. Validate dimensions before products and allocations. The ETTm1
config remains the specific 7/64/4/256/96/4/6 model above.

`ForecastModel(config, flat_parameters, kernel)` validates and owns parameters.
`ForecastWorkspace(config)` owns reusable caller state; one writer per workspace.
`forward(raw_inputs, tokens, outputs, workspace)` requires row-major inputs
[tokens,features], outputs[tokens], and 1..context tokens. Inputs/output must be
disjoint. No allocation, logging, file access or parameter mutation in forward.
Output remains entirely unchanged on any failure. Workspace scratch is disposable.
Reject non-finite input, invalid/moved state, mismatched workspace and numerical
overflow. Models are immutable; concurrent publishing/reloading is not provided.

The ETTm1 layout has 50,689 trainable FP32 values and 14 fitted scaler values.
Packed kernel storage can be padded. Fixed positions add 96*64 FP32 values.
Workspace memory is separate. No latency claim accompanies this count.

## Canonical parameter order

All matrices are row-major [output,input]. Parameters are concatenated as:

1. input mean[F], input scale[F] (finite, strictly positive scales);
2. input weight[W,F], input bias[W];
3. attention scale[W], shift[W], fused Q/K/V weight[3W,W], bias[3W];
4. attention output weight[W,W], bias[W];
5. feed-forward scale[W], shift[W], up weight[H,W], bias[H], down weight[W,H], bias[W];
6. final scale[W], shift[W], scalar head weight[1,W], bias[1].

Total values: `4W² + 2WH + 13W + H + WF + 2F + 1`. Reject wrong counts or any
non-finite value before constructing a model. Zero or negative input scales fail.
Normalization gamma values may be zero or negative.

## File layout and cold loading

Version 1 consists of a 64-byte header, flat FP32 payload, and a 32-byte SHA-256
trailer of all preceding bytes. Integers and FP32 bit patterns are little-endian;
parse them explicitly without unaligned pointer casts or host-byte-order assumptions.
Maximum file size is 16 MiB. Unknown versions, byte-order markers, header lengths,
reserved fields, inconsistent sizes/counts, invalid shapes/scales, non-finite
weights, truncation, trailing bytes and digest mismatches are errors.

| Byte offset | Field |
| --- | --- |
| 0 | Eight bytes: `STLFC01` followed by zero |
| 8 | u32 version = 1 |
| 12 | u32 byte-order marker = 0x01020304 |
| 16 | u32 header length = 64 |
| 20,24,28,32 | u32 features, width, heads, hidden |
| 36,40,44 | u32 context, horizon, target feature index |
| 48 | FP32 epsilon |
| 52 | u32 parameter count |
| 56,60 | u32 reserved = 0 |
| 64 | Parameter payload, 4*count bytes |
| 64+4*count | SHA-256 digest, 32 bytes |

`load_forecaster(bytes)` and `load_forecaster_file(path)` return a fully constructed
candidate in a unique pointer. `active = load_forecaster(...)` only replaces the
previous model if loading succeeds. Exceptions leave the previous model usable;
allocation failure is propagated. Loading is a cold operation and can allocate.
The file reader bounds allocation before reading and rejects short/extra bytes.
SHA-256 follows [FIPS 180-4](https://csrc.nist.gov/pubs/fips/180-4/upd1/final).
The digest detects changes; it does not authenticate the source or license a model.
This is not a cryptographic certification or a general-purpose security API.

## Required evidence before training

Failing tests first; independent Python double-equation fixture, explicit known
values, scalar/packed agreement, prefix/future invariance, last-row failure
preservation and recovery, moved state, no hot allocations, repeated calls and
portable installed consumer. SHA known answers and hashlib-generated padding
boundary cases; every truncation and byte mutation of a compact valid artifact;
rehashed malformed headers/payloads; seeded malformed-input replay under sanitizers.

Full-model diagnostic tolerance is absolute 5e-5 plus relative 2e-4, set before
implementation. The reference evaluates equations in double while native dense
accumulates FP32. Investigate failures rather than widening this allowance.
Pinned PyTorch parity, trained export and the frozen training/evaluation protocol
remain separate gates. No test/validation forecast errors are computed here.


Training exports also require a provenance manifest binding the numerical file's
hash to the task id, feature order, cadence, scaler fit, data snapshot, code and
training protocol. These research/export checks are implemented in the optional
[Torch training workflow](training-v1.md); the [first frozen results](training-results-v1.md)
include all seeds, baseline comparisons, trained parity and restart evidence.

The optional correction output has a distinct version-2 file contract; see
[the correction model](correction-model-v1.md). Version-1 equations and exports
remain unchanged.


## Independent window batches

`ForecastModel::forward_batch` accepts contiguous `[batch,tokens,features]`
input and returns `[batch,tokens]` or `[batch]` for latest-only output. Each window
restarts position zero and attends only within itself. A `ForecastBatchWorkspace`
is constructed with the exact model configuration and a positive capacity up to
1024 windows. Calls may use any smaller positive batch; normal feature/context
limits and the 16,777,216-element input limit still apply.

The workspace stages all output before publication. Invalid shapes, overlapping
input/output, nonfinite input, arithmetic overflow and moved-from workspace
errors leave caller output unchanged, including a failure in the last tile.
Successful and failing native calls allocate no heap memory. One caller owns a
workspace at a time; immutable model weights may be shared across workspaces.

Activation storage covers at most four windows and matrix/normalization row
scratch covers at most 32 rows. Output staging grows with declared batch capacity
and context. `bytes()` reports all owned native workspace buffers, excluding
inputs, outputs, object/allocator overhead and executable code.

The Python session uses the same native API, caches scratch lazily and grows it
only when required. Existing batches larger than 1024 are split into bounded
calls; an exception discards the unpublished Python output. `workspace_bytes`
includes cached batch scratch. Python continues to allocate owned output and
snapshot mutable NumPy inputs before releasing the GIL. Its single-window calls
retain the original execution path.
