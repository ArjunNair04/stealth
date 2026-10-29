# First measured attention optimization

The change is accepted: the paired full-model batch-one p50 reduction is 10.36%,
with byte-identical native model outputs. [The results](attention-results-v1.md)
include all 36 paired cells, the 72-cell Torch rerun, tails and verification.

The saved-model profile puts attention at 38.8–39.4% of sampled stacks, dense
projections at 32.4–34.6%, GELU at 17.0–20.2%, and layer normalization at
5.4–5.7%. Softmax accounts for 12.4–13.4 percentage points within attention.
These are three separate five-second, one-millisecond stack samples, one for each
saved model, using the exact preserved runtime-v1 executable. Sampling has
measurement overhead and attributes call stacks; it does not measure isolated
kernel latency. [Profile evidence](evidence/attention-tiling-v1/profile.json)
contains sanitized counts and hashes. Raw system traces remain local.

## Change and numerical contract

Attention blends the value rows using one weight per earlier observation. The
old loop computes one channel's entire sum before moving to its neighbour. The
new loop computes eight neighbouring sums together. This gives the compiler
independent work to combine and reads adjacent values from memory together.
Each sum still visits observations in the same order, uses FP64 products and
accumulation, and rounds to FP32 only at the same final boundary. Query/key
scores, softmax, output staging, finite checks and model weights are unchanged.

`AttentionWorkspace(capacity, width, heads, AttentionKernel::reference)` retains
the scalar path. The default is `AttentionKernel::tiled`; heads smaller than
eight and incomplete final tiles use the scalar loop. The implementation is
portable C++20, with no intrinsics, new libraries, approximate math or fast-math.
The eight local accumulators need 64 bytes of bounded temporary storage; no
heap workspace is added. Reported owned workspace bytes are unchanged. The
workspace object has an additional kernel selector; rebuild native consumers
and Python wheels together rather than mixing old and new object layouts.
Model-file bytes and forecasting architecture do not change.

Tests require bitwise scalar/tiled equality, independent constant-value answers,
multiple heads, widths around tile boundaries, maximum context/width, causality,
invalid-kernel rejection, numerical failures and unchanged caller output.
Allocation interception exercises full tiles and tails. A deliberate wrong-channel
mutation must fail before the correct implementation is restored. Whole trained
model outputs are compared against the preserved old native Python wheel before
speed testing. No tolerance is relaxed.

## Frozen performance experiment

[The acceptance protocol](../benchmarks/ettm1/attention_tiling_v1.json) compares the
preserved baseline executable with the candidate in alternating process order.
It uses the same three models, 32 validation windows, three trials, batch one/eight,
64 warm-ups and 1,000/200 timed calls as runtime-v1. Both binaries pass reference
parity before each timing cell. Sampling is disabled during timing. The primary
gate is at least 5% median paired batch-one p50 reduction, improvement for every
seed in median, and no more than 10% median paired p99 increase. Every individual
tail remains reported; these desktop samples do not establish a deadline.
Additional gates are unchanged numerical behavior, no added owned heap workspace,
zero hot allocations, full regression tests and rebuilt-package parity.

```sh
python tools/compare_native.py --baseline local/native-profile-v1/baseline-runtime \
  --inputs local/runtime-v1-complete --models local/ettm1-correction-v1 \
  --output local/attention-tiling-comparison-new
```

The recorder requires clean Git and refuses existing outputs. Reproduce baseline
builds from the runtime-v1 commit on the same machine; this run additionally
requires its preserved binary hash, so changing the baseline requires a new
protocol. Broader platform timings require their own baseline/protocol record.
Use the ordinary runtime recorder after acceptance to compare the installed
Python path and Torch again. Timings, metadata and model/data hashes are saved;
weights, data windows and raw predictions stay local. The reserved data tail,
training objective and QTS research records remain untouched.

On macOS, sample only the benchmark process started for this experiment:
`/usr/bin/sample <pid> 5 1 -mayDie -file <local-report-path>` while the native
runner executes 10,000 calls with the fixed inputs. Linux/Windows can use their
own sampling profilers; the core and correctness suite do not depend on this
macOS tool. `tools.native_profile.summarize_profile` reads the saved macOS call
graph and rejects unknown/inconsistent formats. Do not add nested softmax counts
to attention's inclusive count.
