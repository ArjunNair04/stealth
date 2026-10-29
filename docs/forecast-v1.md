# One block, one temperature, one hour ahead

The first complete native forecasting model is implemented. Its arithmetic and
file loading are checked with invented inputs and untrained weights. Learning
from the prepared ETTm1 data is the next phase; no forecasting skill is established.

## The model in plain language

At 10:00, collect the latest 96 quarter-hour readings: six electricity-load numbers
and the observed temperature on each line. Ask for one temperature at 11:00.

```mermaid
flowchart LR
    A[96 rows of seven readings] --> B[Put readings on comparable scales]
    B --> C[Mix each row into 64 numbers]
    C --> D[Add a position marker]
    D --> E[One attention and feed-forward block]
    E --> F[Read the final row]
    F --> G[One temperature one hour ahead]
```

Scaling uses averages and spreads learned from the training period, then frozen.
The input projection learns mixtures of the seven readings. Position markers
preserve order within the window. Four attention heads learn four ways of
combining earlier information; we do not assign them human meanings such as
"trend" or "volatility." The feed-forward part transforms the combined information.
A final normalization and small linear head produce a standardized temperature,
which is converted back to the target's original scale.

The user chose one block first, with two blocks left for a later comparison.
Width 64, four heads and hidden width 256 are the compact initial dimensions.
There are 50,689 trainable values plus 14 fitted scaler values. This is a baseline
architecture for understanding and measurement; no research novelty is claimed.

The exact task is [one_hour.json](../benchmarks/ettm1/one_hour.json). The model,
parameter order and versioned file format are fixed in
[the native contract](forecast-contract.md). These choices replace the earlier
unfrozen two-block proposal. Training settings and evaluation controls still need
to be frozen before outcomes are inspected.

## What was added

| File | Job |
| --- | --- |
| `include/stealth/forecaster.hpp`, `src/forecaster.cpp` | Connect raw readings, scaling, positions, the block and the final forecast. |
| `include/stealth/model_file.hpp`, `src/model_file.cpp` | Load a bounded versioned weight file and return a fully built candidate model. |
| `src/sha256.*` | Detect changes to file contents using an internal SHA-256 implementation. |
| `tools/generate_forecast_fixture.py` | Calculate independent diagnostic answers and an example artifact with Python's standard library. |
| `tests/forecast_tests.cpp` | Check complete-model answers, causal prefixes, dimensions, failure isolation and reuse. |
| `tests/model_file_tests.cpp` | Try truncated, corrupted and malformed files, plus independent checksum examples. |
| `tests/allocation_tests.cpp` | Check repeated inference allocation behavior and simulated failures during loading. |
| `examples/forecast_demo.cpp` | Run the complete load-and-predict path on a tiny diagnostic example. |

All native calls use the existing core. No Python runtime, framework, GPU or
platform SDK was added to CMake. The optional Python generator is a reference
tool, not the planned Python wrapper.

## Try the complete path

From the checkout after building:

```sh
./build/release/stealth_forecast_demo
python3 tools/generate_forecast_fixture.py --check --artifact local/my-diagnostic.stlf
./build/release/stealth --inspect-model local/my-diagnostic.stlf
```

Use a fresh artifact filename: the generator refuses to overwrite an existing
file. On Windows append `.exe` to native executable names and use `python` if that
is the installed interpreter name. The example and CLI are also installed under
`install/release/bin/` by the normal installation command.

The demo uses five invented rows and a reduced width-four model so failures are
easy to inspect. It prints `Final diagnostic output: 6.92691` and checks the full
result against independent equations before returning success. That number is a
software fixture, not an ETTm1 prediction or a trained result.

## Tested behavior

Eleven initial test groups failed against explicit stubs before inference,
checksum and loader implementation. A twelfth group later added an independent
reference at the full 96-by-seven input shape, width 64 and hidden width 256.
No native outputs were used to create expected answers.

Local verification:

- Debug: 44/44 CTest groups; Release: 44/44; ASan/UBSan: 43/43. The sanitizer
  configuration excludes installation/relocation as designed.
- Twenty-three standard-library data tests, including the selected one-hour task.
  The 10:00 example begins at the preceding day's 10:15 and targets 11:00.
- Independent equation fixtures reproduce exactly. The full task-shape fixture's
  largest local absolute discrepancy is 3.09944e-6 at zero-based origin 19, within
  the predeclared 5e-5 + 2e-4*abs(reference) tolerance. Scalar and packed outputs
  agree exactly on this fixture.
- Valid prefix extension and future edits preserve earlier predictions exactly.
  A failure on the final inverse-scaled output preserves every caller output.
- SHA-256 agrees with Python hashlib at 17 lengths, including padding boundaries
  and 1 MiB, plus the fixed `abc` known answer.
- Every shorter prefix and every one-byte bit flip of the 984-byte valid fixture
  is rejected. Rehashed invalid fields/weights exercise semantic checks beyond
  the digest. There are 1,500 additional seeded mutation cases (seed 0x20261028).
- File tests cover actual reads, extra bytes, a missing file, a short file and the
  16 MiB size limit. Maximum valid dimensions and context are exercised separately
  on Debug, Release and sanitizers.
- Simulated allocation failure at each of 21 cold-load allocation points preserves
  the previous model, for both scalar and packed layouts. The old model still
  produces its expected answers afterward.
- Four thousand added model/error-path cycles perform zero intercepted C++ heap
  allocations, with the existing positive control. This does not intercept every
  possible C/OS allocator. Separate interleaved workspaces are tested sequentially;
  no concurrent-writer claim is made.
- Deliberate removal of positions, premature output publication and skipped
  integrity verification were each detected by the appropriate tests. All faults
  were removed and verification rerun.
- The relocated installed consumer constructs/runs the forecaster and links the
  file-loader API without Python. The CLI inspects a generated file successfully.

Compact source/fixture/test provenance is stored with this milestone under
`docs/evidence/forecast-v1/`. Earlier evidence directories are historical snapshots
of their own phases and are not rewritten when files evolve.

## Memory and optimization

For the agreed configuration, the packed implementation reports these owned
buffer sizes on the local build:

| Buffer | Bytes |
| --- | ---: |
| Parameters and fitted scalers | 204,604 |
| Fixed position table | 24,576 |
| Reusable per-caller workspace | 204,576 |

These exclude input/output arrays, object bookkeeping and allocator overhead.
All allocation and weight packing happen during construction/loading. Model
parameters are immutable; every caller owns a workspace. Forward uses staged
outputs and explicit error checks. There is no caching, explicit AVX/NEON backend
or fixed-shape template specialization yet. This phase records no trained-model
latency claim; previous timings measured smaller components/blocks.

## What comes next

1. Declare a pinned Python reference environment and compare the same model,
   preprocessing, weights and positions against PyTorch CPU.
2. Freeze optimizer, seeds, stopping rules, simple baselines and evaluation metrics,
   then train/export the ETTm1 one-hour example. Preserve unsuccessful outcomes.
3. Bind trained exports to a manifest containing feature order, cadence, task id,
   data/scaler/code hashes and training provenance. The numerical file alone does
   not carry those research records or authenticate its creator.
4. The thin Python wrapper and bounded raw-window tests are now implemented in
   [the next infrastructure milestone](python-stream-v1.md). Cached streaming remains
   separate: resetting positions for a new 96-row window is not assumed equivalent
   to retaining an old KV cache.
5. Compare a second block and later optimizations on forecast error, median/tail
   latency and memory. Keep future QTS returns evaluation a separate experiment.
