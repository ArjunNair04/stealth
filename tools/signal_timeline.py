"""Bounded, offline oracle for point-in-time feature assembly.

No network, model dependency, calendar inference, imputation or live-feed state.
See docs/signal-timeline-v1.md before adapting external observations.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable

MAX_TIME = 2**63 - 1
MAX_FEATURES = 256
MAX_RECORDS = 100_000


def _integer(value: int, label: str, minimum: int = 0, maximum: int = MAX_TIME) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")


def _identifier(value: str, label: str) -> None:
    if (type(value) is not str or not 1 <= len(value) <= 128 or
            value != value.strip() or not value.isprintable()):
        raise ValueError(f"{label} must be a nonempty, bounded identifier without control characters")


@dataclass(frozen=True)
class SignalSpec:
    name: str
    source: str
    entity: str
    field: str
    max_observation_age_ns: int
    max_publication_age_ns: int

    def __post_init__(self) -> None:
        for name in ("name", "source", "entity", "field"):
            _identifier(getattr(self, name), name)
        _integer(self.max_observation_age_ns, "max_observation_age_ns")
        _integer(self.max_publication_age_ns, "max_publication_age_ns")

    @property
    def key(self) -> tuple[str, str, str]:
        return self.source, self.entity, self.field


@dataclass(frozen=True)
class Observation:
    source: str
    entity: str
    field: str
    event_id: str
    revision: int
    effective_at_ns: int
    available_at_ns: int
    received_at_ns: int
    value: float | None

    def __post_init__(self) -> None:
        for name in ("source", "entity", "field", "event_id"):
            _identifier(getattr(self, name), name)
        for name in ("revision", "effective_at_ns", "available_at_ns", "received_at_ns"):
            _integer(getattr(self, name), name)
        if not self.effective_at_ns <= self.available_at_ns <= self.received_at_ns:
            raise ValueError("Require effective_at_ns <= available_at_ns <= received_at_ns")
        if self.value is not None:
            if type(self.value) not in (int, float):
                raise ValueError("value must be a finite number or an explicit None withdrawal")
            try:
                value = float(self.value)
            except (ValueError, OverflowError) as exc:
                raise ValueError("value must be finite") from exc
            if not math.isfinite(value):
                raise ValueError("value must be finite")
            object.__setattr__(self, "value", value)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.source, self.entity, self.field


@dataclass(frozen=True)
class Feature:
    spec: SignalSpec
    state: str
    value: float | None
    observation: Observation | None
    observation_age_ns: int | None
    publication_age_ns: int | None
    receipt_age_ns: int | None


class InputsNotReady(ValueError):
    """At least one required feature is missing, stale or withdrawn."""


@dataclass(frozen=True)
class Snapshot:
    decision_at_ns: int
    schema_id: str
    features: tuple[Feature, ...]

    @property
    def values(self) -> tuple[float | None, ...]:
        return tuple(feature.value for feature in self.features)

    @property
    def ready(self) -> bool:
        return bool(self.features) and all(feature.state == "valid" for feature in self.features)

    def require_values(self, *, schema_id: str) -> tuple[float, ...]:
        """Publish a row only under the caller's expected schema and all-valid inputs."""
        if schema_id != self.schema_id:
            raise ValueError("Feature schema mismatch")
        if not self.ready:
            reasons = ", ".join(f"{f.spec.name}={f.state}" for f in self.features if f.state != "valid")
            raise InputsNotReady(reasons)
        return tuple(float(feature.value) for feature in self.features)


@dataclass(frozen=True, init=False)
class SignalTimeline:
    """Immutable finite replay input; snapshot uses a simple linear scan.

    Records may arrive in any iterable order. Their recorded receipt clocks, not
    iterable order, define availability. Every delivery counts toward max_records.
    """
    specs: tuple[SignalSpec, ...]
    schema_id: str
    _records: tuple[Observation, ...]

    def __init__(self, specs: Iterable[SignalSpec], records: Iterable[Observation], *,
                 max_records: int = MAX_RECORDS) -> None:
        _integer(max_records, "max_records", 1, MAX_RECORDS)
        ordered = []
        names, keys = set(), set()
        for item in specs:
            if len(ordered) >= MAX_FEATURES:
                raise ValueError("Feature limit exceeded")
            if not isinstance(item, SignalSpec):
                raise TypeError("Expected SignalSpec")
            if item.name in names or item.key in keys:
                raise ValueError("Duplicate feature name or channel")
            ordered.append(item)
            names.add(item.name)
            keys.add(item.key)
        if not ordered:
            raise ValueError("At least one feature is required")

        unique, event_clocks, point_ids = {}, {}, {}
        for count, record in enumerate(records, start=1):
            if count > max_records:
                raise ValueError("Record limit exceeded")
            if not isinstance(record, Observation):
                raise TypeError("Expected Observation")
            if record.key not in keys:
                raise ValueError("Observation channel is absent from feature schema")
            event = (*record.key, record.event_id)
            if event in event_clocks and event_clocks[event] != record.effective_at_ns:
                raise ValueError("An event cannot change effective time across revisions")
            event_clocks[event] = record.effective_at_ns
            point = (*record.key, record.effective_at_ns)
            if point in point_ids and point_ids[point] != record.event_id:
                raise ValueError("Ambiguous event identities for one feature/effective time")
            point_ids[point] = record.event_id
            identity = (*event, record.revision)
            previous = unique.get(identity)
            if previous is not None:
                previous_value = None if previous.value is None else previous.value.hex()
                current_value = None if record.value is None else record.value.hex()
                if (previous.effective_at_ns, previous.available_at_ns, previous_value) != (
                        record.effective_at_ns, record.available_at_ns, current_value):
                    raise ValueError("Conflicting deliveries for one event/revision")
                if previous.received_at_ns <= record.received_at_ns:
                    continue
            unique[identity] = record

        # Sort by event identity then revision, checking publication lineage.
        publication = {}
        canonical = tuple(unique[identity] for identity in sorted(unique))
        for record in canonical:
            event = (*record.key, record.event_id)
            if event in publication and publication[event] > record.available_at_ns:
                raise ValueError("Revision publication clocks must be nondecreasing")
            publication[event] = record.available_at_ns
        schema = {"version": 1, "features": [asdict(item) for item in ordered]}
        digest = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":"),
                                           ensure_ascii=True).encode("ascii")).hexdigest()
        object.__setattr__(self, "specs", tuple(ordered))
        object.__setattr__(self, "schema_id", digest)
        object.__setattr__(self, "_records", canonical)

    @property
    def record_count(self) -> int:
        return len(self._records)

    def snapshot(self, decision_at_ns: int) -> Snapshot:
        _integer(decision_at_ns, "decision_at_ns")
        latest = {}
        for record in self._records:
            if record.received_at_ns >= decision_at_ns:
                continue
            previous = latest.get(record.key)
            if previous is None or (record.effective_at_ns, record.revision) > (
                    previous.effective_at_ns, previous.revision):
                latest[record.key] = record
        features = []
        for spec in self.specs:
            record = latest.get(spec.key)
            if record is None:
                features.append(Feature(spec, "missing", None, None, None, None, None))
                continue
            observation_age = decision_at_ns - record.effective_at_ns
            publication_age = decision_at_ns - record.available_at_ns
            receipt_age = decision_at_ns - record.received_at_ns
            if record.value is None:
                state = "withdrawn"
            elif (observation_age > spec.max_observation_age_ns or
                  publication_age > spec.max_publication_age_ns):
                state = "stale"
            else:
                state = "valid"
            features.append(Feature(spec, state, record.value if state == "valid" else None,
                                    record, observation_age, publication_age, receipt_age))
        return Snapshot(decision_at_ns, self.schema_id, tuple(features))
