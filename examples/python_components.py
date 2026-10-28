"""Runnable installed-package example with hand-checkable, untrained weights."""
import numpy as np
import stealth_cpu as s

weights = np.array([[1, 2], [-1, 3]], dtype=np.float32)
bias = np.array([1, -2], dtype=np.float32)
print('Dense result:', s.Dense(weights, bias).forward(np.array([2, 4], dtype=np.float32)))

config = s.ForecastConfig(features=1, width=1, heads=1, hidden=1, context=3, horizon=4, target=0)
parameters = np.zeros(s.parameter_count(config), dtype=np.float32)
parameters[0] = 20  # Target mean.
parameters[1] = 2   # Target scale.
parameters[-1] = 3  # Constant standardized output; 3*2+20 = 26.
model = s.ForecastModel(config, parameters)
session = model.session()
rows = np.array([[10], [12], [15]], dtype=np.float32)
predictions = session.predict(rows)
if not np.array_equal(predictions, [26, 26, 26]):
    raise RuntimeError("Diagnostic prediction mismatch")
print('Diagnostic predictions:', predictions)
print('Batch shape:', session.predict_batch(np.stack([rows, rows])).shape)
stream = model.window_stream()
print('Streaming outputs:', [stream.push(row) for row in rows])
print('These are invented, untrained diagnostic weights, not forecast-quality evidence.')
