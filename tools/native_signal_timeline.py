"""Finite native index with the named schema and snapshots of the Python oracle.

Construction reuses Python boundary validation and also validates numeric records
in C++. Queries use the native index. Python result conversion allocates; only the
C++ caller-buffer API promises allocation-free queries. No live append/journal.
"""
from dataclasses import dataclass, field
from types import MappingProxyType
from stealth_cpu._core import _SignalIndex
from signal_timeline import (
    SignalTimeline, Feature, Snapshot, MAX_RECORDS, _integer,
)


@dataclass(frozen=True, init=False, eq=False)
class NativeSignalTimeline(SignalTimeline):
    _index: object = field(repr=False)
    _observations: object = field(repr=False)

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
        # Canonical records are already validated and immutable. Retain a
        # read-only lookup rather than reconstructing provenance on every query.
        # Its size is bounded by record_count; no mutable per-query cache.
        object.__setattr__(self, '_observations', MappingProxyType({
            (identities[(*r.key, r.event_id)], r.revision): r for r in self._records}))

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
            event, revision, _, _, _, value = raw
            record = self._observations[event, revision]
            features.append(Feature(spec, state, value if state == 'valid' else None,
                                    record, observed_age, published_age, receipt_age))
        return Snapshot(decision_at_ns, self.schema_id, tuple(features))
