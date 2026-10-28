# Work queue

## 1. Local C++ infrastructure (complete)

- [x] Choose a permanent standalone checkout and repository identity (now public).
- [x] Inspect the installed compiler, SDK, editor and GitHub account.
- [x] Install CMake and Ninja through the existing local package manager.
- [x] Create a C++20 library, CLI, CMake presets and editor configuration.
- [x] Verify debug, release, sanitizer and relocated-install builds locally.
- [x] Push the initial repository and verify the cross-platform CI matrix.

Done means another clean checkout can configure, build, test and install using
only a documented compiler, CMake and Ninja. Record actual passed configurations.

## Test-first acceptance rules

Follow [the testing plan](testing.md): specify behavior, observe an intended failing
test, implement it, then refactor. Independently checked expected answers and
failure cases are required. Infrastructure is verified; model correctness and
trading readiness remain unestablished.

## 2. Native numeric primitives (implemented on codex/native-primitives)

- [x] Freeze bounded FP32 storage, shapes, errors and output-preservation rules.
- [x] Implement dense, softmax, layer normalization, GELU and residual components.
- [x] Retain scalar dense and add an output-packed layout with tail-lane checks.
- [x] Compose a feed-forward block with an independent Python math fixture.
- [x] Verify component failures, integration failures, moves, repeated calls and allocations.
- [x] Run Debug, Release, sanitizers, relocation and all five platform CI jobs.
- [x] Record three local latency trials and explain the speed/memory tradeoffs.

See [implementation evidence](native-v1.md) and [performance](performance.md).
The task remains open until the implementation PR is merged. This establishes
primitive and feed-forward correctness on tested inputs, not full-model parity.

## 3. Small reference model and native parity (training workflow implemented)

The user selected one temperature one hour ahead and one block first. The frozen
configuration is seven inputs, 96 context rows, width 64, four heads and hidden
width 256. See [the model contract](forecast-contract.md) and
[implementation evidence](forecast-v1.md).

Native input scaling, projection, positions, the causal block, final normalization
and scalar readout are implemented on `codex/one-block-forecaster`. The bounded
versioned weight loader preserves the previous model on failure. Independent
double-equation fixtures cover reduced and full task shapes; malformed files,
causality, output isolation, allocation failure and installed consumers are tested.

The optional Torch CPU reference, framework/native parity, numerical gradient
checks, AdamW training, accumulation, exact epoch restart, bounded pickle-free
checkpoints and provenance-bound native export are implemented. See the
[training guide](training-v1.md). The first fixed ETTm1 run is complete: all three trained exports pass sampled
native parity, and exact full-size epoch restart is verified. None beats
persistence on test MAE; [all results](training-results-v1.md) are retained.
Issue #3 stays open through review and merge.
Synthetic learning fixtures remain software diagnostics.

The diagnostic study is complete: all four validation-selected linear baselines
also miss persistence on old-test MAE. Temperature-only fits nearly match it;
quiet-window errors and seed-dependent bias explain where the transformers lose.
All original scores reproduce exactly and two full runs are byte-identical.
The old test is exploratory; the remaining source tail stays unused. See
[the findings](forecast-diagnostics-results-v1.md) and
[the diagnostic contract](forecast-diagnostics-v1.md). A correction-from-latest
transformer is proposed for discussion, not implemented or approved yet.

Gate cached streaming separately after ordinary forward parity; test reset,
capacity and stream isolation. A shifted 96-row window resets its positions and
must not silently reuse an incompatible cache. Two-block variants need their own
recorded comparison, not an untracked change to this first architecture.

## 4. Optional Python wrapper and bounded raw streams (implemented)

The `stealth_cpu` package calls the same native core. It exposes reusable numeric
components, full-model sessions, bounded sequential batching and raw-window
streams. Strict NumPy buffer rules, explicit copies, lifetime, output ownership,
failures and separate-session concurrency are tested. Native builds retain no
Python dependency. Source archive -> wheel -> isolated runtime installation is
checked across the platform matrix. See [implementation evidence](python-stream-v1.md).

`WindowStream` stages each complete ordered row before accepting it, preserves
history on failure, waits for full context before publishing, and recomputes the
latest window with positions reset to zero. Reset, rolling/reference agreement,
bounds, allocations and concurrent independent streams are tested. This completes
the raw-window baseline; KV caching and market-feed clock alignment remain separate.

## 5. CPU performance study and QTS adapter (partial measurement only)

Scalar-versus-packed primitive, feed-forward and full-sequence block microbenchmarks
are recorded. Established-runtime,
streaming, cold-start, sustained-load and end-to-end comparisons remain pending.

Measure the same workload across native and established runtimes. Record CPU,
compiler, threads, warm-up, memory and latency distributions. Then define a
separate, preregistered QTS pilot. Do not infer alpha from synthetic correctness.
Keep deterministic correctness checks separate from hardware-specific timing
thresholds, and require new operational/financial gates before deployment.

## GitHub tasks

- [Set up and verify portable local C++ development](https://github.com/ArjunNair04/stealth/issues/1)
- [Implement the first native numeric primitives](https://github.com/ArjunNair04/stealth/issues/2)
- [Establish a small reference model and native forward parity](https://github.com/ArjunNair04/stealth/issues/3)
- [Add an optional Python wrapper after the native layers](https://github.com/ArjunNair04/stealth/issues/4)
- [Measure the CPU performance frontier and define the QTS boundary](https://github.com/ArjunNair04/stealth/issues/5)

See [verification](verification.md) for infrastructure and [native v1](native-v1.md) for the first numerical layers.

## Public benchmark foundation

ETTm1 source pinning, strict parsing, training-only scaling, chronological window
indices and immutable preparation/replay are implemented on
`codex/public-benchmark`. See [the data contract](benchmark-contract.md) and
[the benchmark guide](public-benchmark.md). The user has now selected the one-hour point target and one-block architecture.
Native model/loading and the frozen training workflow are recorded above.
