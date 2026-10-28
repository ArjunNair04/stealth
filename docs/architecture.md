# Native boundary and later Python wrapper

The native library is the shared core. The CLI is one caller. A later Python
extension will be another caller of the same implementation.

```text
C++ command-line program ----> stealth::core <---- future Python extension
```

The core exposes build metadata and the [native numeric layer](native-v1.md).
Owned FP32 buffers and immutable model parameters are prepared during construction.
Each inference caller owns its output and reusable workspace. Dense projections
can use either the scalar or packed layout through the same public interface.

The core now also provides [causal multihead attention and a pre-norm transformer
block](attention-v1.md), with a fused Q/K/V projection and one attention-score row
at a time. Inputs are embedded vectors. Position encoding, a forecasting head,
model serialization, training and incremental caching remain pending. Synthetic correctness and local latency results exist; there is no
forecast-quality or financial result. The CLI remains a build-information utility.

## Portability

Use standard C++20 and CMake targets. Keep platform-specific profiling or kernels
behind optional backends. Avoid Apple Accelerate, AVX, CUDA or any particular
Python runtime as requirements of the baseline build. Source portability does not
mean a single binary runs on every CPU or operating system.

The static core is built with position-independent code so it can later be linked
into a Python extension. Public includes are separated from generated/private
headers; consumers link the exported `stealth::core` target. API/ABI compatibility
is not promised during the initial 0.x research phase.

## Python milestone (planned)

Use pybind11 with scikit-build-core unless requirements reveal a better fit.
Keep Python optional: the C++ build and native CLI must remain usable without it.
Bindings should call native functions, preserve exception meanings and validate
array dtype, shape, strides and ownership. Release the GIL only while native code
runs without touching Python objects. Avoid implicit copies in measured paths;
explicitly document when a copy is required.

Deliver an isolated development install first. Later wheel builds must target
specific operating systems, architectures and supported Python versions. No
package has been installed or published for this milestone.

Reference: https://pybind11.readthedocs.io/en/stable/compiling.html

## Research boundary

Future signal records need source identity, effective and available timestamps,
age and missingness/quality state. Keep data collection, forecasts and portfolio
policy separate. Reuse QTS contracts through an adapter once a native model exists;
do not import QTS datasets into this infrastructure repository.
