"""Checked, bounded offline replay from timed features to native predictions.

This Python reference allocates and scans. It is not a live feed handler or an
allocation-free hot path. Each replay has one writer and one native session.
"""
from dataclasses import dataclass
import math
import threading

import numpy as np
from model_bundle import ModelBundle
from signal_timeline import SignalTimeline, Snapshot, MAX_TIME, _integer


# A NumPy 2 scalar comparison can round a Python float down to this endpoint.
# Keep the limit as a Python float so range admission precedes FP32 conversion.
_FP32_MAX = float(np.finfo(np.float32).max)


@dataclass(frozen=True)
class ForecastDecision:
    bundle_id: str
    schema_id: str
    decision_at_ns: int
    target_at_ns: int
    state: str
    reset_reason: str | None
    history_rows: int
    window_start_ns: int | None
    value: float | None
    snapshot: Snapshot


class FeatureReplay:
    def __init__(self, bundle: ModelBundle):
        if not isinstance(bundle, ModelBundle):
            raise TypeError('Expected a loaded ModelBundle')
        self._bundle = bundle
        self._session = bundle.model.session()
        self._rows = ()
        self._last = None
        self._lock = threading.Lock()

    @property
    def history_rows(self):
        return len(self._rows)

    @property
    def last_decision_ns(self):
        return self._last

    def reset(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Replay is busy')
        try:
            self._rows, self._last = (), None
        finally:
            self._lock.release()

    def _predict(self, rows):
        # Raw inputs; embedded native preprocessing runs exactly once.
        prediction = self._session.predict_last(np.asarray(rows, dtype=np.float32))
        if not math.isfinite(prediction):
            raise ValueError('Non-finite native forecast')
        return prediction

    def advance(self, timeline: SignalTimeline, decision_at_ns: int) -> ForecastDecision:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Replay is busy')
        try:
            return self._advance(timeline, decision_at_ns)
        finally:
            self._lock.release()

    def _advance(self, timeline, clock):
        bundle, config = self._bundle, self._bundle.model.config
        _integer(clock, 'decision_at_ns')
        if (clock < bundle.grid_origin_ns or
                (clock-bundle.grid_origin_ns) % bundle.interval_ns or
                clock + config.horizon*bundle.interval_ns > MAX_TIME):
            raise ValueError('Decision is outside the bundle clock grid or target range')
        if self._last is not None and clock <= self._last:
            raise ValueError('Decisions must be strictly increasing; reset before replaying')
        if (not isinstance(timeline, SignalTimeline) or timeline.schema_id != bundle.schema_id or
                timeline.specs != bundle.specs):
            raise ValueError('Feature schema mismatch')
        snapshot = timeline.snapshot(clock)
        gap = self._last is not None and clock-self._last != bundle.interval_ns
        if not snapshot.ready:
            candidate, value, state, reason = (), None, 'unavailable', 'unavailable'
        else:
            row = snapshot.require_values(schema_id=bundle.schema_id)
            if any(not math.isfinite(x) or abs(x) > _FP32_MAX for x in row):
                raise ValueError('Feature value exceeds finite FP32 range')
            prior = () if gap else self._rows
            candidate = (prior + (row,))[-config.context:]
            # Validate even warming prefixes, so an arithmetic failure never
            # commits an unusable row. Only a full context publishes a forecast.
            prediction = self._predict(candidate)
            ready = len(candidate) == config.context
            value, state = (prediction, 'forecast') if ready else (None, 'warming_up')
            reason = 'gap' if gap else None
        result = ForecastDecision(
            bundle.bundle_id, bundle.schema_id, clock, clock+config.horizon*bundle.interval_ns,
            state, reason, len(candidate),
            clock-(len(candidate)-1)*bundle.interval_ns if candidate else None, value, snapshot)
        # All validation, numerical execution and result construction precede commit.
        self._rows, self._last = candidate, clock
        return result
