# Four-row dense execution contract

The accepted attention implementation is the baseline. Fresh three-seed sampling
puts dense projections at 34.6–36.2%, attention at 33.2–33.7% and GELU at
21.0–22.1%. Sampling attributes stacks rather than isolated elapsed time. Raw
traces remain local; [sanitized counts](evidence/dense-rows-v1/profile.json) bind
this choice to the saved models and baseline executable.

The candidate reuses a weight tile for four independent observations. Each result
retains FP32 products/accumulation, original input order, bias initialization and
rounding. No fused multiply-add, approximate activation or model change is added.
One-row APIs remain available. Row/column remainders retain ordinary calculations.

`Dense::forward_rows(input, rows, output, scratch)` accepts contiguous row-major
inputs and output with positive bounded rows. Input/output sizes must be exact;
all products are checked before multiplication. Scratch must cover the entire
output. All inputs, output and the entire scratch span are disjoint. Validate all
inputs and stage all results before publishing any output. Invalid state, shapes,
non-finite input, insufficient scratch, overlap and overflow fail without changing
caller output. The existing single-row call uses the same numerical contract.

`FeedForwardWorkspace(width, hidden, capacity=1)` preallocates bounded row storage.
`FeedForward::forward_rows` validates all rows, normalizes each independently,
uses the dense row path, applies exact GELU, projects back and publishes the
complete residual only after all stages pass. A single-row workspace retains its
old buffer size. Workspace capacity never implies mutable model parameters.

The transformer processes at most four rows together in QKV projection and in
its post-attention projection/feed-forward stages. Attention itself is unchanged.
The complete block continues to stage caller output. Workspace growth is bounded
by the four-row tile, rather than allocating every hidden activation for the
whole sequence. Parameter/model-file layouts remain unchanged; native consumers
and Python wheels are rebuilt together after header changes.

Tests precede implementation: independent tiny matrix answers, comparison with
ordinary row calls and scalar weights, tile boundaries and remainders, row
isolation, finite extremes, overflow after an earlier valid row, all alias pairs,
invalid capacity, moved-from state and zero hot allocations. Complete-model
fixtures, causal-prefix tests and saved trained outputs gate integration. A
wrong-row mutation must be detected. No tolerance is relaxed.

The [frozen experiment](../benchmarks/ettm1/dense_rows_v1.json) alternates the
preserved and candidate binaries for all three saved models, three trials and
batch one/eight. It requires 5% median paired batch-one p50 reduction, improvement
for each seed in median and at most 10% median paired p99 increase. Memory and all
tails are reported. Repeat the four-engine benchmark after acceptance. Failed
candidates stay recorded. Shared CI establishes correctness, never stable latency.
