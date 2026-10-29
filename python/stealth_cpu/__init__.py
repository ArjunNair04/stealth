"""CPU inference; arrays must be native float32, aligned and C-contiguous.

Returned arrays own their data. Parameters are copied on construction. Session
calls snapshot input and release the GIL during native work. Use one session per
concurrent caller. This interface does not provide automatic differentiation.
"""
from ._core import (
    Dense, FeedForward, ForecastConfig, ForecastModel, ForecastSession, WindowStream,
    build_info, causal_attention, gelu, layer_norm, residual_add, softmax, parameter_count,
)
__version__ = "0.1.0"
__all__ = ["Dense", "FeedForward", "ForecastConfig", "ForecastModel", "ForecastSession",
           "WindowStream", "causal_attention", "gelu", "layer_norm", "residual_add",
           "softmax", "parameter_count", "build_info"]
