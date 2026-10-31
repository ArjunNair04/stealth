# Native, Python and training boundaries

The native library is the shared core. The CLI and optional Python extension
call the same implementation.

```text
C++ command-line program ----> stealth::core <---- Python extension
```

The core exposes build metadata and the [native numeric layer](native-v1.md).
Owned FP32 buffers and immutable model parameters are prepared during construction.
Each inference caller owns its output and reusable workspace. Dense projections
can use either the scalar or packed layout through the same public interface.

The core provides [causal multihead attention and a pre-norm transformer
block](attention-v1.md), with fused Q/K/V projection and one attention-score row
at a time. The [complete forecaster](forecast-v1.md) adds frozen feature scaling,
a learned input projection, fixed positions, final normalization and a scalar
head. Weight files are bounded, versioned and digest-checked before construction.
The CLI inspects validated model versions and dimensions. Torch reference parity,
training/checkpoints and native export are implemented. Latest-only inference
has a measured local single-request advantage; incremental caching and a current
runtime comparison remain open. Forecast quality still trails persistence.

## Portability

Use standard C++20 and CMake targets. Keep platform-specific profiling or kernels
behind optional backends. Avoid Apple Accelerate, AVX, CUDA or any particular
Python runtime as requirements of the baseline build. Source portability does not
mean a single binary runs on every CPU or operating system.

The static core is built with position-independent code and is linked
into the optional Python extension. Public includes are separated from generated/private
headers; consumers link the exported `stealth::core` target. API/ABI compatibility
is not promised during the initial 0.x research phase.

## Python and raw-window milestone

The optional pybind11/scikit-build-core package `stealth_cpu` is implemented.
Its native-only CMake default remains OFF. It exposes components, owned forecast
sessions, bounded native batching and raw-window streams. Strict float32
buffer validation, explicit copies and result ownership keep Python lifetimes
separate from native workspaces. Separate sessions share immutable models and
release the GIL during inference; concurrent reuse of one session is rejected.

The raw stream recomputes a bounded latest window and does not assume cache
compatibility when positions restart. See [the contract](python-stream-contract.md)
and [installation/evidence](python-stream-v1.md). The separate Torch workflow owns training and automatic differentiation; the
native package owns inference. Public wheel publication remains open. API/ABI compatibility remains provisional in 0.x.

## Research boundary

The offline [signal timeline](signal-timeline-v1.md) now provides source/entity/
field identity, effective/publication/receipt clocks, revisions, ages and explicit
missing/stale/withdrawn states. It is a standard-library reference under tools/,
separate from the installed inference package. A schema-bound bridge and actual
source adapters remain open. Data collection, features, forecasts and portfolio
policy have separate contracts. Reuse QTS contracts through a later adapter;
do not import QTS datasets into this infrastructure repository.
