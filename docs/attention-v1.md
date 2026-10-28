# Attention: a runnable example and a map of the files

Attention lets a new observation combine information from observations already
available. The first example gives each available observation equal importance,
so the answer is simply the average seen so far. A learned model would calculate
different importance scores; these diagnostic examples have not been trained.

From the project folder, run:

```sh
./build/release/stealth_attention_demo
```

On Windows, append `.exe`. To build it first, configure and build the `release`
preset as in the README. The example is enabled by default and installed alongside
the CLI; `STEALTH_BUILD_EXAMPLES=OFF` omits it.

```text
Step  Input  Answer  Changed input  New answer
   1   10.0    10.0           10.0        10.0
   2   20.0    15.0           20.0        15.0
   3   30.0    20.0           30.0        20.0
   4   40.0    25.0          400.0       115.0
```

Changing the last observation leaves all earlier answers unchanged. This is the
central rule of causal attention: earlier answers cannot read later information.
The example calls the real native attention component and returns failure if its
known answers or this rule are broken.

## What the files do

| File | Simple job |
| --- | --- |
| `src/numeric.cpp` | The calculator pieces: mix, rescale, activate and add numbers. |
| `src/feed_forward.cpp` | Connect some calculator pieces into a small processing block. |
| `src/attention.cpp` | Let each observation look at itself and earlier observations. |
| `src/transformer_block.cpp` | Connect attention to the processing block. |
| `examples/attention_demo.cpp` | Run the little number example above. |
| `tests/attention_tests.cpp` | Check attention by itself, including future-data leakage. |
| `tests/transformer_tests.cpp` | Check the connected block and isolate failures. |
| `tools/generate_attention_fixture.py` | Make an independent answer sheet from equations. |
| `tools/record_benchmarks.py` | Run the stopwatch and preserve each measurement. |
| `CMakeLists.txt` | Tell the compiler which pieces to assemble. |

Public headers in `include/stealth/` describe how another C++ program calls each
piece. Tests use the same library that the example and installed consumers use.

## What was added

The [contract](attention-contract.md) was committed before execution code. Causal
multihead attention accepts row-major query, key and value arrays. A query is the
current observation's lookup; a key describes an available observation; its value
is the information to combine. Multiple heads perform separate lookups on channel
groups, and their answers are concatenated.

The connected block normalizes each input row, computes Q/K/V together through
one dense projection, performs attention, projects the result, adds the original
input, then runs the existing feed-forward residual block. It stages all results
and publishes only after the entire call succeeds. It does not add positional
encoding; that belongs once at the future full-model input boundary.

Weights are immutable during execution and copied at construction. Each caller
preallocates a private workspace. Supported limits are 1–1,024 tokens, width
1–512, heads dividing width, and feed-forward hidden width 1–2,048. Readonly inputs
may overlap; output may not overlap them. Invalid data anywhere rejects the entire
call. The caller's output remains unchanged, including a failure on the final row.

Only one attention score/probability row is held at a time. Workspace grows
linearly with context length at fixed width; attention computation still grows
quadratically. This is full-sequence evaluation, with no persistent cache or
streaming state. Workspace reuse is not a streaming-memory implementation.

## Verification

Code commit: `3bf9eed0025e72843ae065d8aee5fb363acd6e10`.

- All four attention test groups failed against the stub before implementation.
  All five transformer groups then failed against their block stub before wiring.
- Hand answers and independently generated double-precision equation fixtures
  pass for scalar and packed paths. Both fixture generators reproduce their
  checked-in output and provenance exactly. These are not PyTorch parity tests.
- Prefix evaluation and finite future-input edits preserve earlier outputs. The
  component test also exercises the maximum 1,024-token boundary.
- Removing the causal restriction deliberately caused BOTH the component and
  integration causality tests to fail. The original code was restored and rebuilt.
- Local Debug: 30/30 groups; Release: 30/30; AddressSanitizer and
  UndefinedBehaviorSanitizer: 29/29. Package relocation is omitted in sanitizers.
- Existing allocation interception detected zero C++ heap allocations in 4,000
  new attention/block/error-path cycles, with the positive control still passing.
  The original 20,000 feed-forward cycles also continue to pass. This does not
  intercept every operating-system or library allocator.
- Another 4,000 block calls alternate two independent workspaces and compare each
  result. This is sequential reuse, not parallel thread stress.
- Relocated installed consumers build and run the attention and block APIs.

The [test receipt](evidence/attention-v1/tests.json) records platform CI status and
[the implementation CI run](https://github.com/ArjunNair04/stealth/actions/runs/40654886326)
contains job logs. Windows omits the GCC/Clang-specific fast-math compile test.
These tests do not constitute a zero-defect guarantee or a deployment gate.

Useful focused commands:

```sh
ctest --preset dev -L attention
ctest --preset dev -L transformer
ctest --preset dev -R '^transformer.failures$'
python3 tools/generate_attention_fixture.py --check
```

## Initial sequence timings

Three preserved desktop trials, each with 1,000 measured calls per kernel and
1,000 warm-up calls. One inference thread, AppleClang 16.0.0.16000026 Release
`-O3 -ffp-contract=off`, Intel Core i5-10600, macOS 14.8.7. Calls alternate
scalar/packed order and include validation and publication. No affinity or
frequency control; clock overhead is reported rather than subtracted.

One block: width 64, four heads, feed-forward width 256. Times below are for
recomputing the entire sequence, not for adding one cached observation. Ranges
are minima/maxima of the three trial quantiles, not confidence intervals.

| Sequence length | Packed median | Packed p99 | Median speedup over scalar | Workspace |
| --- | ---: | ---: | ---: | ---: |
| 16 observations | 0.222–0.230 ms | 0.306–0.339 ms | 2.78–2.80× | 30,656 bytes |
| 64 observations | 1.104–1.127 ms | 1.402–1.488 ms | 2.42–2.44× | 104,960 bytes |

Both models own 199,936 parameter bytes. Scalar and packed benchmark outputs were
identical for these fixed inputs; the independent reference comparisons use the
frozen tolerances. [Raw trials and hashes](evidence/attention-v1/manifest.json)
are preserved. There is no established-runtime, cold-cache, concurrent-load or
end-to-end market-system comparison.

## What remains before a complete forecaster

Specify the input features, positions, model stack and prediction head. Then test
versioned weight loading, establish pinned PyTorch agreement, and train/export a
deterministic synthetic example. Only after ordinary full-model parity should
cached streaming be introduced. Python bindings and a QTS data adapter remain
separate gates. Issue #3 stays open until those full-model requirements pass.


CI follow-up, 28 October 2026: all five jobs passed against final attention head
`3e8437f76c01f6529faf06356ee27a8871e80dc3`, including the previously queued ARM
macOS job. [Final-head run](https://github.com/ArjunNair04/stealth/actions/runs/40655586794).
The original evidence snapshot remains unchanged as a record of publication time.
