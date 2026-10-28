#!/usr/bin/env python3
"""Verified, offline-replayable ETTm1 preparation; Python standard library only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import ssl
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
SOURCE = json.loads((ROOT / 'benchmarks/ettm1/source.json').read_text(encoding='utf-8'))
MAX_BYTES = 12_000_000
MAX_ROWS = 100_000
MAX_LINE = 2_048
MAX_CONTEXT = 1_024
FP32_MAX = float.fromhex('0x1.fffffep127')
COLUMNS = ('date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT')
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def integer(value: int, low: int, high: int, name: str) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'Invalid {name}')


def finite_row(row: Sequence[float], width: int) -> tuple[float, ...]:
    if len(row) != width or width == 0:
        raise ValueError('Invalid numeric row width')
    values = tuple(float(x) for x in row)
    if any(not math.isfinite(x) or abs(x) > FP32_MAX for x in values):
        raise ValueError('Non-finite value or FP32 magnitude exceeded')
    return values


def split_ranges(n: int, ends: dict[str, int]) -> dict[str, tuple[int, int]]:
    integer(n, 1, MAX_ROWS, 'row count')
    if set(ends) != {'train', 'validation', 'test'}:
        raise ValueError('Invalid split names')
    start = 0
    result = {}
    for name in ('train', 'validation', 'test'):
        end = ends[name]
        integer(end, start + 1, n, 'split boundary')
        result[name] = (start, end)
        start = end
    return result


@dataclass(frozen=True)
class Frame:
    timestamps: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]


def parse_csv(data: bytes, expected: dict) -> Frame:
    """Reject malformed observations; do not fill, reorder or repair them."""
    if len(data) > MAX_BYTES:
        raise ValueError('CSV byte limit exceeded')
    integer(expected['rows'], 1, MAX_ROWS, 'expected row count')
    if tuple(expected['columns']) != COLUMNS or expected['interval_seconds'] != 900:
        raise ValueError('Unsupported CSV contract')
    try:
        text = data.decode('ascii')
        # Check physical lines first so csv.reader cannot allocate a huge field.
        if any(len(line) > MAX_LINE for line in text.splitlines()):
            raise ValueError('CSV line limit exceeded')
        reader = csv.reader(io.StringIO(text, newline=''), strict=True)
        if tuple(next(reader, ())) != COLUMNS:
            raise ValueError('CSV header mismatch')
        timestamps, values = [], []
        previous = None
        for row in reader:
            if len(timestamps) >= expected['rows'] or len(row) != len(COLUMNS):
                raise ValueError('CSV row count or width mismatch')
            clock = datetime.strptime(row[0], DATE_FORMAT)
            if clock.strftime(DATE_FORMAT) != row[0]:
                raise ValueError('Noncanonical timestamp')
            if previous is not None and clock - previous != timedelta(seconds=900):
                raise ValueError('CSV duplicate, gap or reversed clock')
            values.append(finite_row(row[1:], len(COLUMNS)-1))
            timestamps.append(row[0])
            previous = clock
        if (len(timestamps) != expected['rows'] or timestamps[0] != expected['first']
                or timestamps[-1] != expected['last']):
            raise ValueError('CSV extent mismatch')
        return Frame(tuple(timestamps), tuple(values))
    except (UnicodeError, csv.Error, OverflowError) as exc:
        raise ValueError('Invalid CSV encoding or value') from exc


@dataclass(frozen=True)
class Scaler:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def transform(self, row: Sequence[float]) -> tuple[float, ...]:
        values = finite_row(row, len(self.mean))
        return finite_row(tuple((x-m)/s for x, m, s in zip(values, self.mean, self.scale)), len(values))


def fit_scaler(rows: Sequence[Sequence[float]], training_end: int) -> Scaler:
    """Only the training prefix can influence fitted statistics."""
    integer(training_end, 1, len(rows), 'training boundary')
    width = len(rows[0])
    train = tuple(finite_row(rows[i], width) for i in range(training_end))
    means, scales = [], []
    for j in range(width):
        mean = math.fsum(row[j] for row in train) / training_end
        variance = math.fsum((row[j]-mean)**2 for row in train) / training_end
        scale = math.sqrt(variance)
        if not math.isfinite(mean) or not math.isfinite(scale):
            raise ValueError('Invalid training statistics')
        if scale == 0:
            if any(row[j] != train[0][j] for row in train):
                raise ValueError('Variance underflow')
            scale = 1.0
        means.append(mean); scales.append(scale)
    return Scaler(tuple(means), tuple(scales))


@dataclass(frozen=True)
class Window:
    input_start: int
    input_end: int
    target_start: int
    target_end: int


def windows(n: int, ends: dict[str, int], split: str, context: int,
            horizon: int, mode: str) -> Iterator[Window]:
    """Yield bounded index descriptors, never materialized overlapping tensors."""
    ranges = split_ranges(n, ends)
    integer(context, 1, MAX_CONTEXT, 'context')
    integer(horizon, 1, MAX_CONTEXT, 'horizon')
    if split not in ranges or mode not in ('point', 'curve'):
        raise ValueError('Unknown split or target mode')
    start, stop = ranges[split]
    for end in range(max(context, start), stop-horizon+1):
        first_target = end+horizon-1 if mode == 'point' else end
        yield Window(end-context, end, first_target, end+horizon)


def validate_source() -> None:
    if SOURCE['schema_version'] != 1:
        raise ValueError('Unsupported source manifest')
    files = SOURCE['files']
    if len(files) != 3 or {f['name'] for f in files} != {'ETTm1.csv', 'LICENSE', 'README.md'}:
        raise ValueError('Invalid source filenames')
    for record in files:
        integer(record['bytes'], 1, MAX_BYTES, 'source bytes')
        sha = record['sha256']
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Invalid source hash')
        if not record['url'].startswith('https://'):
            raise ValueError('Source URL must use HTTPS')
    split_ranges(SOURCE['csv']['rows'], SOURCE['split_ends'])


def read_bounded(path: Path, limit: int) -> bytes:
    with path.open('rb') as stream:
        result = stream.read(limit+1)
    if len(result) > limit:
        raise ValueError('File size limit exceeded')
    return result


def verify_bytes(data: bytes, record: dict) -> None:
    if len(data) != record['bytes'] or digest(data) != record['sha256']:
        raise ValueError(f"Source size/hash mismatch: {record['name']}")


def fetch_snapshot(output: Path, cafile: str | None = None) -> None:
    validate_source()
    output = Path(output)
    # Build the verified TLS context before creating output; never disable TLS checks.
    context = ssl.create_default_context(cafile=cafile)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'schema_version': 1, 'status': 'incomplete',
               'source_manifest_sha256': digest(canonical(SOURCE)), 'files': []}
    receipt_path = output/'receipt.json'
    receipt_path.write_bytes(canonical(receipt))
    try:
        for record in SOURCE['files']:
            request = urllib.request.Request(record['url'], headers={'User-Agent': 'stealth-public-benchmark/1'})
            partial = output/(record['name']+'.partial')
            with urllib.request.urlopen(request, context=context, timeout=45) as response:
                if hasattr(response, 'geturl') and response.geturl() != record['url']:
                    raise ValueError('Unexpected source redirect')
                # Retain bounded partial bytes even when a midstream read fails.
                count = 0
                with partial.open('xb') as stream:
                    while True:
                        chunk = response.read(min(65536, record['bytes']+1-count))
                        if not chunk:
                            break
                        stream.write(chunk)
                        count += len(chunk)
                        if count > record['bytes']:
                            raise ValueError('Download exceeds pinned byte length')
            data = read_bounded(partial, record['bytes'])
            verify_bytes(data, record)
            partial.rename(output/record['name'])
            receipt['files'].append({**record, 'retrieved_at': datetime.now(timezone.utc).isoformat()})
            receipt_path.write_bytes(canonical(receipt))
        receipt['status'] = 'complete'
        receipt_path.write_bytes(canonical(receipt))
    except Exception as exc:
        receipt['status'] = 'failed'
        # Error text can contain local paths. Keep the portable exception class only.
        receipt['error_type'] = type(exc).__name__
        receipt_path.write_bytes(canonical(receipt))
        raise


def verified_frame(raw: Path) -> Frame:
    validate_source()
    receipt = json.loads(read_bounded(raw/'receipt.json', 65536))
    if (receipt.get('schema_version') != 1 or receipt.get('status') != 'complete'
            or receipt.get('source_manifest_sha256') != digest(canonical(SOURCE))):
        raise ValueError('Incomplete or mismatched source receipt')
    observed = receipt.get('files', [])
    if len(observed) != len(SOURCE['files']):
        raise ValueError('Incomplete source receipt')
    data = None
    for record, actual in zip(SOURCE['files'], observed):
        if any(actual.get(key) != value for key, value in record.items()):
            raise ValueError('Source receipt mismatch')
        received = datetime.fromisoformat(actual['retrieved_at'])
        if received.utcoffset() is None:
            raise ValueError('Receipt must include an explicit UTC offset')
        value = read_bounded(raw/record['name'], record['bytes'])
        verify_bytes(value, record)
        if record['name'] == SOURCE['csv']['name']:
            data = value
    if data is None:
        raise ValueError('No CSV in verified source')
    return parse_csv(data, SOURCE['csv'])


def prepared_artifacts(raw: Path) -> dict[str, bytes]:
    frame = verified_frame(raw)
    ranges = split_ranges(len(frame.values), SOURCE['split_ends'])
    end = SOURCE['split_ends']['train']
    scaler = fit_scaler(frame.values, end)
    audit = {'schema_version': 1, 'dataset_id': SOURCE['dataset_id'],
             'rows': len(frame.values), 'columns': list(COLUMNS),
             'first': frame.timestamps[0], 'last': frame.timestamps[-1],
             'interval_seconds': 900, 'timezone': SOURCE['csv']['timezone'],
             'split_ends': SOURCE['split_ends'],
             'partitions': {name: {'start_row': a, 'end_row_exclusive': z,
                                    'first': frame.timestamps[a], 'last': frame.timestamps[z-1]}
                            for name, (a, z) in ranges.items()},
             'unused_rows': len(frame.values)-SOURCE['split_ends']['test'],
             'forecast_results_computed': False}
    scale_record = {'schema_version': 1, 'training_rows': end, 'columns': list(COLUMNS[1:]),
                    'mean': scaler.mean, 'scale': scaler.scale, 'variance_ddof': 0,
                    'constant_column_scale': 1.0}
    artifacts = {'audit.json': canonical(audit), 'scaler.json': canonical(scale_record)}
    artifacts['manifest.json'] = canonical({
        'schema_version': 1, 'status': 'complete',
        'source_manifest_sha256': digest(canonical(SOURCE)),
        'tool_sha256': digest(Path(__file__).read_bytes()),
        'source_files': SOURCE['files'],
        'artifacts': {name: {'sha256': digest(value), 'bytes': len(value)}
                      for name, value in artifacts.items()},
        'purpose': 'Data preparation only; no training or forecast score.'})
    return artifacts


def prepare_snapshot(raw: Path, output: Path) -> None:
    output = Path(output)
    if output.exists():
        raise FileExistsError('Prepared destination already exists')
    artifacts = prepared_artifacts(Path(raw))
    output.mkdir(parents=True, exist_ok=False)
    # Manifest is inserted last, so partial output is never a completed snapshot.
    for name, data in artifacts.items():
        with (output/name).open('xb') as stream:
            stream.write(data)


def check_snapshot(raw: Path, prepared: Path) -> None:
    expected = prepared_artifacts(Path(raw))
    for name, data in expected.items():
        if read_bounded(Path(prepared)/name, len(data)) != data:
            raise ValueError(f'Prepared replay mismatch: {name}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fetch = commands.add_parser('fetch', help='Fetch original pinned files into a new directory')
    fetch.add_argument('--output', type=Path, required=True)
    fetch.add_argument('--cafile', help='Optional existing CA file; certificate checks stay enabled')
    prepare = commands.add_parser('prepare', help='Validate originals and fit training-only scaling')
    prepare.add_argument('--raw', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    check = commands.add_parser('check', help='Recompute and require byte-identical prepared files')
    check.add_argument('--raw', type=Path, required=True)
    check.add_argument('--prepared', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'fetch':
        fetch_snapshot(args.output, args.cafile)
    elif args.command == 'prepare':
        prepare_snapshot(args.raw, args.output)
    else:
        check_snapshot(args.raw, args.prepared)
    print(f'{args.command}: passed (no forecast scores computed)')


if __name__ == '__main__':
    main()
