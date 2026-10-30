# Assembly-guided latency audit, 30 October 2026

The previous pass left a native forecast near 136 microseconds. The final paired
comparison in this pass measures **136.617 → 97.141 microseconds** on the same
Intel Mac workload: 28.90% less latency when comparing the two table medians,
or 29.43% median reduction across paired trials. All three saved models improve.
The predictions remain bit for bit identical and owned native workspace is unchanged.

This is an incremental comparison against `82ee5ae`, not against the project's
original implementation. The final measured source commit is `96e874c`, with a
clean working tree when measurement started. Models, inputs, architecture,
forecast objectives and tolerances stayed fixed. No training or reserved-tail
inspection was needed. Controlled Linux/Beaker performance work remains deferred.

## What the assembly exposed

Think of a CPU vector instruction as carrying several numbers in one trip. The
portable matrix loop was already carrying four numbers at a time; the local CPU
can carry eight. The retained specialization uses that wider instruction only
when the compiler runtime reports usable AVX support. The default library remains
runnable through its portable path on other supported machines.

For each input dimension of a four-row, eight-output tile, the generated Release
assembly changes as follows. Both loops perform exactly the same 32 FP32 products
and 32 FP32 additions, in the same order within each output.

| Instructions per inner-loop iteration | Portable | AVX |
| --- | ---: | ---: |
| Vector width | 128 bits / 4 floats | 256 bits / 8 floats |
| Vector multiplies | 8 | 4 |
| Vector additions | 8 | 4 |
| Weight loads | 2 × 16 bytes | 1 × 32 bytes |
| Scalar input loads plus broadcasts | 4 loads + 4 shuffles | 4 broadcasts |
| Additional register copies inside loop | 3 | 0 |
| Accumulator spills inside loop | 0 | 0 |

Weight/input byte traffic per iteration is unchanged. These are static instruction
counts, not measured CPU cycles. The [assembly evidence](evidence/assembly-latency-v1/assembly.json)
contains both actual loops, the remaining single-row dependency chain, and full
assembly hashes. No fused multiply-add instruction appears in the specialization.

The code uses function-specific [compiler target attributes](https://clang.llvm.org/docs/AttributeReference.html#target)
and [AVX intrinsics](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html).
The CPU guard uses the compiler runtime's usable-feature test; LLVM's
[implementation](https://github.com/llvm/llvm-project/blob/main/compiler-rt/lib/builtins/cpu_model/x86.c)
checks the necessary CPU and operating-system vector-state support. Detection is
a cached feature-bit read, not a CPUID instruction per forecast. The compiled
assembly confirms that the guard itself contains no AVX instructions.

All shape/alias/input checks run before dispatch. Every backend writes disposable
scratch, and the common finite-output check publishes only after the whole call
succeeds. Partial tiles use bounded loads/stores; caller buffers need not be
aligned. The scalar reference and portable packed kernel remain available.

The specialization is compiled on x86-64 GCC/Clang outside the MSVC ABI. ARM and
MSVC retain the portable path. `-DSTEALTH_ENABLE_AVX=OFF` disables it explicitly.
There is no whole-library `-march=native`, AVX, FMA or fast-math flag. This does not
yet provide ARM SIMD or MSVC-specific acceleration, and physical execution on a
non-AVX x86 machine was not tested locally.

## Final complete-model comparison

The [frozen protocol](../benchmarks/ettm1/assembly_latency_v1.json) retains the
previous grid: 32 saved validation windows, three models, three paired trials,
batch one/eight, latest/full output, and 72 serial fresh worker processes.
Each worker checks every requested prediction against the independent reference
before timing. Baseline/candidate order alternates. Builds, tests and profiling
were kept separate from timing.

Workload: 96 observations, seven features, width 64, four heads, hidden width 256.
Intel Core i5-10600, macOS 14.8.7, AppleClang 16, Release `-O3`, one inference
thread, no affinity or frequency control. Times are microseconds per call. Each
table entry is the median of nine worker quantiles, not a pooled quantile.
Batch-eight calls still contain eight sequential windows.

| Output / batch | Before p50 | After p50 | Before p99 | After p99 | Median paired p50 reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Latest / 1 | 136.617 | 97.141 | 214.276 | 133.533 | 29.43% |
| Latest / 8 | 1,092.705 | 785.197 | 1,373.317 | 1,008.855 | 28.57% |
| Full / 1 | 1,332.786 | 1,198.721 | 1,759.889 | 1,686.700 | 10.34% |
| Full / 8 | 10,483.175 | 9,191.474 | 11,945.093 | 10,938.172 | 11.75% |

All frozen gates pass. Per-seed median primary reductions are 29.43%, 32.18%
and 27.88% for seeds 7, 19 and 43. Median paired primary p99 improves 33.27%.
Individual secondary tails vary: the worst latest/batch-eight p99 pair increases
8.60%, and the worst full/batch-one pair increases 19.70%. All pairs are retained
in [native evidence](evidence/assembly-latency-v1/native.json). Warm desktop timings
are not execution deadlines or a performance claim about every CPU or shape.

Owned memory remains 204,604 parameter bytes, 24,576 position bytes and 222,240
workspace bytes per session. The specialization adds executable code and stack
usage; unchanged owned heap workspace does not mean every memory cost is zero.
The native hot-allocation tests continue to pass. This pass did not separately
remeasure the Python feature-to-forecast boundary.

## Small shapes changed the dispatch rule

A separate [108-shape probe](evidence/assembly-latency-v1/shape-protocol.json) tested
six input widths, six output widths and three row counts in three alternating
pairs. Every output matched the ordered scalar reference bitwise. Timings group
16 warmed calls per sample, so these are diagnostic throughput measurements,
not the latency of one isolated call.

The initial AVX rule slowed one-row 1→8 and 3→8 products by about 3 ns, or 13–14%.
The dispatch overhead outweighed the saved arithmetic. The final rule keeps fewer
than 16 outputs with fewer than four rows on the portable path; fewer than eight
outputs always use it. The original results remain preserved.

In the final probe, the dominant four-row 64→192 projection falls from 3,106.56 to
1,591.12 ns per call, a 48.76% median paired reduction. The two tiny cases return
to within about 1.3% of the portable path. The worst median paired regression
across all 108 shapes is 4.52%; individual pairs vary more, including 42.90% in
one already-portable shape. This is evidence for a simple dispatch boundary on
this machine, not proof of a universal crossover. See all
[initial/final shape cells](evidence/assembly-latency-v1/shapes.json).

## A promising-looking change was rejected

Attention computes each softmax exponential twice. A separate candidate saved the
FP64 exponential and reused it, keeping the same sum/division order. It passed
all 67 Release tests after the test's explicit memory expectation was updated to
account for its declared extra scratch space.

It saved only 2.04% on the primary latest/batch-one workload, slightly regressed
seed 7, and added 768 workspace bytes. It failed the predeclared 5% gate and was
removed. Full-history batch-one output improved 8.39%, so it could merit a
separately specified full-trace option later. Both the
[rejected patch](evidence/assembly-latency-v1/exp-candidate.patch) and all compact
[development cells](evidence/assembly-latency-v1/development.json) remain recorded.
No unsuccessful experiment was rerun into acceptance.

## Remaining opportunities, now more specific

Separate five-second samples attribute the final native work approximately as
follows: dense projections 49.7%, normalization 20.4%, attention including softmax
12.6%, GELU 3.7%, other 13.5%. These are shares of remaining work, not before/after
absolute cost estimates. [Profile counts](evidence/assembly-latency-v1/profile.json)
retain both baseline and candidate samples; raw local system reports stay ignored.

1. **Multiple output tiles for single-row dense calls.** The single-row AVX loop
   still updates one vector accumulator repeatedly. Even when the compiler
   unrolls it, each addition depends on the previous addition. Processing several
   output tiles together would create independent chains without changing the
   order within any dot product. The shape probe also shows much smaller gains
   for long single-row products than for four-row products. This is the next
   focused matrix experiment, with register pressure and small shapes as gates.
2. **Normalization across rows.** Mean and variance remain ordered FP64 reductions,
   which the compiler correctly declines to reassociate. Independent rows can be
   processed together while retaining each row's reduction order. Any new row API
   must preserve transactional failures and retain an independent reference.
3. **Repeated validation/copies inside the composed pipeline.** Public boundaries
   should remain checked. A privately owned workspace may permit shared validation
   or fewer intermediate copies, but only after proving the exact failure and
   ownership invariants. The present change removes none of those checks.
4. **Python window conversion.** The previous pass isolated repeated tuple-to-NumPy
   conversion. A bounded array-based adapter remains a separate experiment, using
   the existing replay state machine as oracle. Native gains cannot simply be
   added to Python timing percentages.

Skipping older query projections or caching keys/values is not treated as a free
optimization: the current implementation checks intermediate arithmetic, and
rolling windows restart their positional indices. Those changes require an
explicit equivalence/failure contract before implementation. Threading and queue
policy similarly need an actual workload and target host. Linux timing remains
user-gated; routine correctness CI is separate.

## Verification and reproduction

67 Debug, 67 Release, 66 sanitizer and 67 forced-portable native tests pass,
including relocated consumers where applicable. All 84 offline tests, 34 wrapper
tests under each of NumPy 1.26.4 and 2.2.6, and 107 Torch/reference/integration
tests pass; three optional Linux-engine tests skip locally. A fresh source
archive builds a wheel and passes its isolated NumPy-only installation checks.

The direct specialization test failed before implementation because usable AVX
was detected but the stub did no work. It now checks exact ordered-reference
parity across row/column tails and unaligned caller spans. New hand cases detect
accidental FMA contraction and preserve signed zero, subnormal and range endpoints.
Existing tests cover non-finite input, overflow in the last tile/row, aliasing,
moved state, output preservation, allocation bounds and concurrent sessions.

The old and new installed packages agree bitwise on 9,216 full outputs and every
one of their 9,216 prefix predictions. Maximum discrepancy from the independent
high-precision benchmark reference remains 1.9073486328125e-6, within unchanged
5e-5 absolute / 2e-4 relative tolerances. See
[verification and hashes](evidence/assembly-latency-v1/manifest.json).

Preserve the baseline binary before rebuilding, then use a fresh output directory:

```sh
python3 tools/compare_native_latency.py \
  --protocol benchmarks/ettm1/assembly_latency_v1.json \
  --baseline local/assembly-latency-v1/baseline-runtime \
  --candidate build/release/stealth_runtime_bench \
  --inputs local/runtime-latest-v1 --models local/ettm1-correction-v1 \
  --output local/new-assembly-comparison
```

For shape diagnostics, compile the preserved `shape_probe.cpp` twice against
Release libraries built with AVX enabled/disabled. Use `-std=c++20 -O3
-ffp-contract=off`, the repository's include directory and each static library;
run three alternating process pairs. The saved protocol defines the full grid.
For assembly, use the exact command in `compile_commands.json`, replacing `-c
-o object` with `-S -fverbose-asm -o assembly`; do not add host-wide ISA flags.
