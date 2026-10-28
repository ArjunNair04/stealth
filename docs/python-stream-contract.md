# Python and bounded raw-window infrastructure contract

This phase adds inference infrastructure, not training, new model architecture or
latency claims. Preserve the frozen one-block task and numerical tolerances.

## Native WindowStream

One immutable shared model, one writer per stream, one raw row per push. Capacity
is the model context. Until full, accept valid rows but leave the one-element
output untouched. Once full, drop the oldest row and recompute the latest window
with positions starting at zero. This is full recomputation, not a KV cache.
Validate each candidate prefix before committing history. Every error preserves
accepted history, count and caller output. Reset empties logical history; it does
not erase retained buffer bytes. Allocate bounded storage at construction only.
The stream retains its model's lifetime. Separate streams may share an immutable
model concurrently; a stream itself must have exactly one caller at a time.
Rows are already ordered, complete observations; timestamps, gaps, revisions and
asynchronous market sources belong to a future adapter. Do not infer feed readiness.

## Python boundary

Optional pybind11/scikit-build-core package, import stealth_cpu. Native builds
remain Python-free. Only native-endian, aligned, C-contiguous NumPy float32 arrays
are accepted; no implicit dtype/layout conversions. Readonly inputs are valid.
Owned model parameters are copied during construction. Returned arrays own their
memory. No borrowed workspace views escape. Callers must not mutate arguments
while a call is active.

Expose Dense, FeedForward, softmax, layer_norm, gelu, residual_add, causal_attention,
ForecastConfig, ForecastModel, ForecastSession and WindowStream. Primitive calls
hold the GIL. Forecast sessions/streams snapshot input into owned preallocated
buffers (a bounded per-call snapshot for batches), release the GIL only for native work, and reject concurrent use of the
same mutable session. Separate sessions can run concurrently on a shared model.
Python prediction allocates an output array and explicitly copies input/output;
the native allocation guarantee does not apply to the Python wrapper.

Session predict takes [T,F], 1<=T<=context, and returns the T historical-origin
point forecasts. predict_batch takes [B,T,F] and returns [B,T], sequentially through
one workspace. This first batch interface is a convenience API, not a fused tensor
kernel. Bound total batch input/output elements by max_elements before allocating;
reject empty batches. A failed batch returns no partial result. Streams return
None until full, then a Python float. No automatic differentiation is exposed.

Tests precede native behavior. Require independent fixture agreement, invalid
buffers and errors, caller-input preservation, lifetime and repeated-call checks,
separate-session concurrency, stream reset/rolling equivalence, no native hot
allocations, source/wheel isolation and clean-install verification. No training or
open-source comparative performance experiment is part of this milestone.
