"""Versioned feature/scaler/model binding for the optional research workflow.

The native model remains independently loadable. This module needs NumPy and the
installed stealth_cpu extension, but does not import PyTorch. Hashes detect a
changed artifact; the caller obtains expected_id from its trusted run record.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import stealth_cpu as native
from signal_timeline import SignalSpec, SignalTimeline, MAX_TIME, _integer

MAX_MANIFEST = 128 * 1024
MAX_MODEL = 16 * 1024 * 1024
CONFIG_FIELDS = ('features', 'width', 'heads', 'hidden', 'context', 'horizon',
                 'target', 'epsilon', 'output')


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _read(path, limit):
    with Path(path).open('rb') as source:
        blob = source.read(limit + 1)
    if len(blob) > limit:
        raise ValueError('Bundle artifact size limit exceeded')
    return blob


def _json(blob):
    def pairs(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError('Duplicate JSON key')
            result[name] = value
        return result

    def real(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('Non-finite JSON number')
        return number

    def invalid(value):
        raise ValueError('Non-finite JSON constant')

    try:
        result = json.loads(blob, object_pairs_hook=pairs, parse_float=real, parse_constant=invalid)
        if type(result) is not dict:
            raise ValueError('Bundle manifest must be an object')
        return result
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('Invalid bundle JSON') from exc


def _manifest(blob, specs, *, interval_ns, grid_origin_ns, clock_domain):
    if type(blob) is not bytes or len(blob) > MAX_MODEL:
        raise ValueError('Expected bounded model bytes')
    # Validate the SAME byte string before interpreting its embedded scaler.
    model = native.ForecastModel.from_bytes(blob, kernel='packed')
    config = {name: getattr(model.config, name) for name in CONFIG_FIELDS}
    schema = SignalTimeline(specs, ())
    if config['features'] != len(schema.specs):
        raise ValueError('Feature count differs from model')
    _integer(interval_ns, 'interval_ns', 1)
    _integer(grid_origin_ns, 'grid_origin_ns')
    if grid_origin_ns + config['horizon'] * interval_ns > MAX_TIME:
        raise ValueError('Forecast target clock exceeds supported range')
    if clock_domain not in ('utc', 'synthetic'):
        raise ValueError('clock_domain must be utc or synthetic')
    count = config['features']
    scaler = np.frombuffer(blob, dtype='<f4', count=2*count, offset=64)
    manifest = dict(
        schema_version=1, model_file='model.stlf', model_sha256=_sha(blob),
        features=[asdict(s) for s in schema.specs], feature_schema_id=schema.schema_id,
        config=config, target_feature=schema.specs[config['target']].name,
        preprocessing=dict(kind='embedded_standardization_fp32_v1',
                           mean=scaler[:count].tolist(), scale=scaler[count:].tolist()),
        timing=dict(interval_ns=interval_ns, grid_origin_ns=grid_origin_ns,
                    clock_domain=clock_domain, gap_policy='reset_and_warm_up',
                    decision_cutoff='received_strictly_before',
                    target_clock='decision_grid_plus_horizon'))
    raw = canonical(manifest)
    if len(raw) > MAX_MANIFEST:
        raise ValueError('Bundle manifest size limit exceeded')
    return model, schema, raw


@dataclass(frozen=True)
class ModelBundle:
    model: object
    specs: tuple[SignalSpec, ...]
    schema_id: str
    bundle_id: str
    interval_ns: int
    grid_origin_ns: int
    clock_domain: str
    manifest_bytes: bytes


def create_bundle(path, model_bytes, specs, *, interval_ns, grid_origin_ns, clock_domain):
    """Validate first, create a new directory, then write the manifest last.

    Never replaces an existing bundle. A partial write has no valid manifest and
    cannot load; preserve it for diagnosis instead of silently repairing it.
    """
    _, _, raw = _manifest(model_bytes, specs, interval_ns=interval_ns,
                          grid_origin_ns=grid_origin_ns, clock_domain=clock_domain)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    with (path/'model.stlf').open('xb') as target:
        target.write(model_bytes)
    with (path/'manifest.json').open('xb') as target:
        target.write(raw)
    return _sha(raw)


def load_bundle(path, *, expected_id):
    """Load a candidate without modifying any existing model or replay state."""
    if (type(expected_id) is not str or len(expected_id) != 64 or
            any(c not in '0123456789abcdef' for c in expected_id)):
        raise ValueError('Expected a pinned lowercase SHA256 bundle identifier')
    path = Path(path)
    raw = _read(path/'manifest.json', MAX_MANIFEST)
    if _sha(raw) != expected_id:
        raise ValueError('Bundle identifier mismatch')
    manifest = _json(raw)
    try:
        if type(manifest['features']) is not list or len(manifest['features']) > 256:
            raise ValueError('Invalid feature list')
        specs = tuple(SignalSpec(**s) for s in manifest['features'])
        timing = manifest['timing']
        # Fixed names: never open a path supplied by the manifest.
        blob = _read(path/'model.stlf', MAX_MODEL)
        if _sha(blob) != manifest['model_sha256']:
            raise ValueError('Model digest mismatch')
        model, schema, expected = _manifest(
            blob, specs, interval_ns=timing['interval_ns'],
            grid_origin_ns=timing['grid_origin_ns'], clock_domain=timing['clock_domain'])
        # Exact canonical bytes also reject extra fields and bool-for-int aliases.
        if raw != expected:
            raise ValueError('Bundle metadata differs from its native model or feature schema')
    except (KeyError, TypeError) as exc:
        raise ValueError('Malformed bundle metadata') from exc
    return ModelBundle(model, schema.specs, schema.schema_id, expected_id,
                       timing['interval_ns'], timing['grid_origin_ns'],
                       timing['clock_domain'], raw)
