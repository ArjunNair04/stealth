"""Finite native index with the named schema and snapshots of the Python oracle.

Construction reuses Python boundary validation and also validates numeric records
in C++. Queries use the native index. Python result conversion allocates; only the
C++ caller-buffer API promises allocation-free queries. No live append/journal.
"""
from dataclasses import dataclass, field
from stealth_cpu._core import _SignalIndex
from signal_timeline import (
    SignalTimeline, Observation, Feature, Snapshot, MAX_RECORDS, _integer,
)


@dataclass(frozen=True, init=False, eq=False)
class NativeSignalTimeline(SignalTimeline):
    _index: object = field(repr=False)
    _event_names: tuple[str, ...] = field(repr=False)

    def __init__(self, specs, records, *, max_records=MAX_RECORDS):
        # Retain the existing strict schema/identifier boundary and canonical tape.
        super().__init__(specs, records, max_records=max_records)
        channels = {spec.key: i for i, spec in enumerate(self.specs)}
        events = sorted({(*r.key, r.event_id) for r in self._records})
        identities = {key: i for i, key in enumerate(events)}
        limits = [(s.max_observation_age_ns, s.max_publication_age_ns) for s in self.specs]
        rows = [(channels[r.key], identities[(*r.key, r.event_id)], r.revision,
                 r.effective_at_ns, r.available_at_ns, r.received_at_ns, r.value)
                for r in self._records]
        object.__setattr__(self, '_index', _SignalIndex(limits, rows))
        object.__setattr__(self, '_event_names', tuple(key[-1] for key in events))

    @property
    def index_entry_count(self):
        return self._index.index_entry_count

    def snapshot(self, decision_at_ns):
        _integer(decision_at_ns, 'decision_at_ns')
        features = []
        for spec, (state, raw, observed_age, published_age, receipt_age) in zip(
                self.specs, self._index.snapshot(decision_at_ns)):
            if raw is None:
                features.append(Feature(spec, state, None, None, None, None, None))
                continue
            event, revision, effective, available, received, value = raw
            record = Observation(*spec.key, self._event_names[event], revision,
                                 effective, available, received, value)
            features.append(Feature(spec, state, value if state == 'valid' else None,
                                    record, observed_age, published_age, receipt_age))
        return Snapshot(decision_at_ns, self.schema_id, tuple(features))
