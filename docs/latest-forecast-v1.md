# Latest-forecast execution contract

The user selected one temperature forecast one hour ahead, using one block.
The existing forward path returns every origin's forecast for diagnostics. Add
an explicit latest-only mode while keeping the full trace as the default.

For this one-block architecture, all embedded inputs and their K/V projections
are needed for the last query. Earlier attention outputs, post-attention MLP
rows and readout rows cannot affect the last answer and may be skipped. Keep all
QKV projections in this first implementation; do not change their arithmetic or
packing. This is pruning unused inference work, not changing weights, positions,
causal dependencies, preprocessing or the forecasting objective. A future second
block would need earlier first-block outputs and must revisit this schedule.

`SequenceOutput::all` remains the default for native attention, block and model
forward calls. `SequenceOutput::last` requires one row's output (one scalar for
the forecaster). Invalid selectors/shapes, non-finite inputs, aliasing, moved
state and errors in required calculations preserve the caller's output. All raw
inputs and full QKV arrays are still validated. Errors in skipped diagnostic-only
calculations are not evaluated: latest mode may succeed where the full trace
would overflow in an unused earlier MLP row. This distinction is explicit and
covered by a regression. Existing streams retain their default full-trace failure
policy in this phase.

Python sessions add `predict_last([T,F]) -> float` and
`predict_last_batch([B,T,F]) -> [B]`, with the same copy, shape, ownership and
busy-session rules. Batch evaluation remains sequential in native code. Reuse
existing preallocated workspaces, so this first latest-only API claims less
computation, not reduced owned memory.

Before implementation, test latest against the independently generated full
trace for every fixture prefix, all supported kernel choices and irregular tile
shapes. Include invalid modes, lengths, aliasing, moved state, required overflow,
unused-row overflow, repeated mixed all/last calls and zero allocations. Compare
all 9,216 origins in the saved validation windows using latest-prefix inference
against the preserved full traces. No native bit may change on those cases.

The ordinary Torch benchmark must also prune unused work: only the final query
attends to all available K/V rows, with no causal mask needed for that last query,
and only the final residual/MLP/readout is evaluated. Test all/last agreement and
preserve the reference tolerance. Do not compare optimized native latest-only
latency to unoptimized Torch full-trace latency and call it a runtime advantage.

Freeze a new output-mode benchmark before timing: same weights/windows, one
thread, all three seeds, three trials, batch one/eight, native core/Python and
ordinary Torch, both full trace and latest only. Keep outputs, versions, hashes,
parity, setup/first-call time, clock cost, raw timings and memory scopes explicit.
Report within-engine work savings separately from matched latest-only engine
comparisons. All source-tail rows and QTS data remain unused.
