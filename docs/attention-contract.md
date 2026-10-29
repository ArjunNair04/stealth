# Causal attention contract v1

This phase implements bounded full-sequence causal attention and one composed
pre-norm transformer block, plus an executable causal-average demonstration.
The preceding numeric contract still applies. This is the next component gate
within issue #3, not completion of trained-model, weight-file or streaming gates.

## Shapes and capacity

FP32 input/output sequences are contiguous row-major [tokens,width]. Width is
1..512, context capacity is 1..1024, and heads must divide width with 1..width
heads. Feed-forward hidden width is 1..2048. A call has 1..capacity tokens.
Attention splits contiguous channels into heads of width/heads channels.

Workspaces are allocated once. Calls allocate no C++ heap storage and return
Error without exceptions. Scratch is private to each workspace. All external
input/output spans must be valid live storage and disjoint; readonly inputs may
overlap. On any reported failure the entire caller output is unchanged. Reject
non-finite values anywhere in the supplied sequence before publishing anything.
Future-input invariance applies to finite valid inputs; an invalid suffix rejects
the whole call. Models are immutable during calls; each workspace has one writer.

## Attention equations

For query token t and head h, score(t,j,h) = dot(q[t,h], k[j,h]) / sqrt(head_width)
for j in 0..t. Self-attention includes the current token. No future score or value
participates. Use double dot accumulation, then checked FP32 scores; a score
outside finite FP32 returns numerical_error. Apply the existing stable softmax
to the active score row. Sum probabilities times values in double and check the
FP32 conversion. Concatenate heads in their original channel order.

Compute one score/probability row at a time, not a full tokens-by-tokens matrix.
Scratch memory is O(capacity*width + capacity), compute O(tokens^2*width).
Weights and workspace are reused, but there is no persistent history or KV cache.
Reusing a workspace must give the same answer as a fresh workspace.

## Composed block

For every row, normalize the original input, apply one fused dense projection
whose output is [Q channels, K channels, V channels], then causal attention.
Apply a dense output projection, add the original input, then run the existing
pre-norm feed-forward residual block. All rows are staged; publish only when the
whole sequence succeeds. Dropout is absent in this inference implementation.

The block consumes already embedded vectors. Positional encoding is NOT added
inside the block: a future full-model input stage must specify and apply it once.
No feature preprocessing, position scheme, output forecasting head, weight-file
format, training/export, Python binding or financial interpretation is supplied
by this component phase. Those remain explicit full-model gates in testing.md.

## Frozen verification

Write hand-calculated attention tests and shape/failure cases before execution
code. Independent Python standard-library double equations generate a compact
multihead/block fixture and provenance. Tolerances: attention abs 2e-5 + rel 5e-5;
composed block abs 3e-5 + rel 1e-4 on the bounded fixture/generated inputs.
Do not relax tolerance after observing results. Scalar and packed dense backends
must agree with the independent block fixture.

Test every prefix, future edits, appended suffixes, unequal head counts, one token,
context boundaries, malformed dimensions, non-finite values, finite overflow,
partial/exact aliasing, moved objects, output preservation, recovery after failure,
and independent workspace reuse. Deliberately remove the causal restriction and
verify that a causal test fails. Check allocation interception and relocated
package use, Debug/Release, sanitizers and the existing five-platform CI matrix.
A pinned PyTorch reference check is a later full-model gate; this fixture is
explicitly an independent equation reference, not a PyTorch parity claim.
