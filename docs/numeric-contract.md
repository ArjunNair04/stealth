# Native numeric contract v1

Scope frozen before implementation: owned FP32 buffers; single-observation dense,
softmax, layer normalization, exact GELU and residual operations; a pre-norm
feed-forward residual block; component/integration/allocation tests and a measured
reference-versus-packed benchmark. This is a numeric foundation, not a complete
transformer, trained model, trading system or Python wrapper.

## Storage and errors

Use contiguous row-major FP32 storage with 64-byte alignment for owned buffers.
A buffer is move-only, may be empty, and cannot exceed 16,777,216 floats (64 MiB).
Validate size products before allocating. Model dimensions must be nonzero;
weights are copied and validated once during construction. Construction throws
invalid_argument for invalid values/shapes and length_error for capacity/overflow;
allocation failure retains standard bad_alloc semantics.

Hot calls are noexcept and return a small Error enum. Their spans must refer to
live, valid storage; a numeric API cannot validate arbitrary dangling pointers.
Input and output must be disjoint. The entire provided scratch span must be
disjoint from every input and output. Read-only inputs may overlap each other.
Reject all output/input partial and exact aliasing. Scratch must have at least
as many elements as the output, and may be modified on failure. Output remains
unchanged for every reported error. No heap allocation, locks, logging or system
calls occur in a primitive or block forward call after construction.

Validate shapes and limits before reading elements. Reject NaN/infinity inputs.
Finite-input arithmetic that overflows FP32 returns numerical_error. Do not
silently clamp inputs. Use standard IEEE floats without fast-math; disable FMA
contraction for the native reference/packed kernels to retain operation order.
Bitwise portability across compilers is not promised.

## Operations

Dense: y[o] = bias[o] + sum_i weights[o,i] * x[i]. Input weights have shape
[out,in]. The scalar kernel accumulates in FP32 in increasing input order.
The packed kernel interleaves blocks of 8 output rows; it preserves each output's
input order, pads only the final output block with zeros, and ignores padded
lanes. Each Dense instance owns one chosen layout, not both. This trades a small
padding cost for contiguous independent output lanes that a compiler can vectorize.
All inference inputs/outputs are checked; immutable weights are not rescanned.

Softmax: an explicit active-prefix length encodes the simple causal mask. It must
be in [1,n]; an all-masked row is rejected. All inputs, including inactive suffix
values, must be finite. Subtract the active maximum, use double intermediates,
normalize the active prefix and write exact zeros to inactive output positions.

Layer normalization: population variance across the input vector, positive finite
epsilon, elementwise scale and shift. Use two-pass double mean/variance and
intermediates before checked conversion to FP32. Constant inputs yield the shift.

GELU: exact x * Phi(x), evaluated as 0.5*x*erfc(-x/sqrt(2)) in double before
conversion. Residual addition uses checked double addition before conversion.

Feed-forward: y = x + down(GELU(up(layer_norm(x)))). Model parameters are immutable.
Each caller supplies a separate workspace containing normalized, expanded,
activated, projected and scratch spans. A workspace has one writer at a time.
One model may be reused with independent workspaces; sharing one workspace across
simultaneous calls is unsupported. A failed call does not change caller output.
No persistent stream/cache state exists in this tranche.

## Frozen checks

Hand-calculated fixtures lead. Independent double formulas and a Python standard-
library-generated composed fixture supply additional expected values. Primitive
checks use abs 2e-6 + rel 2e-5; dense and composed comparisons use abs 2e-5 + rel
5e-5 on the declared generated finite inputs. Properties use explicit bounds.
Reject non-finite actual/expected values before tolerance comparisons. Extreme
inputs have separate finite/failure expectations; they do not justify loosening
normal-range tolerances. Packed and scalar outputs are compared on all tail sizes.

Tests cover overflow/limits, moves, alignment, shapes, partial/exact aliasing,
non-finite values, output preservation on failure, mask limits, constant inputs,
large finite inputs, generated sizes and composed failures. Replace global C++
allocation entry points in one test executable to detect accidental allocations
inside warmed forward calls. Run deterministic long repeats with preallocated
workspaces. This checks C++ heap entry points, not arbitrary future OS/library
allocation mechanisms. Run debug/release/sanitizers and the platform matrix.

Benchmark both layouts with identical shape, checks and arithmetic; alternate
timing order, warm up, consume results and report per-call p50/p95/p99 plus clock
measurement overhead. No latency threshold in shared CI. Record the CPU/compiler,
flags, trial count and limitations. Fast is a measured local result, not a global
optimality or HFT-deployment claim.
