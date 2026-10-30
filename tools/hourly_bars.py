#!/usr/bin/env python3
"""Offline admission of the QTS normalized Binance spot hourly CSV contract.

No download, fitting, labels or assumed historical publication times. Raw bytes
stay outside this repository. This validates a pinned derived artifact, not an
independent reconstruction of the exchange's original trades.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Sequence

MAX_BYTES = 64 * 1024 * 1024
MAX_ROWS = 200_000
MAX_LINE = 4096
HOUR_NS = 3_600_000_000_000
MICROSECOND_CHANGE_NS = 1_735_689_600_000_000_000
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
COLUMNS = tuple(('symbol,base_asset,quote_asset,instrument,bar_start,bar_end,'
                 'source_open_time,source_close_time,source_time_unit,open,high,low,close,'
                 'base_volume,quote_volume,trade_count,taker_buy_base_volume,'
                 'taker_buy_quote_volume,known_at,source_sha256,source').split(','))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp_ns(text: str, *, receipt: bool = False) -> int:
    pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
    if receipt:
        pattern += r'(?:\.\d{1,6})?'
    if not isinstance(text, str) or not re.fullmatch(pattern + 'Z', text):
        raise ValueError('Timestamp must be explicit UTC in the declared format')
    delta = datetime.fromisoformat(text[:-1] + '+00:00') - EPOCH
    ns = (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000
    if not 0 <= ns <= 2**63 - 1:
        raise ValueError('Timestamp outside supported range')
    return ns


def check_digest(text: str) -> None:
    if not isinstance(text, str) or re.fullmatch('[0-9a-f]{64}', text) is None:
        raise ValueError('Invalid SHA256 digest')


@dataclass(frozen=True)
class BarContract:
    sha256: str
    symbols: tuple[str, ...]
    start: str
    end: str  # Exclusive: final bar must end here.

    def __post_init__(self) -> None:
        check_digest(self.sha256)
        if (type(self.symbols) is not tuple or not 1 <= len(self.symbols) <= 16
                or any(not isinstance(s, str) or re.fullmatch('[A-Z0-9]{2,16}USDT', s) is None
                       for s in self.symbols)
                or len(set(self.symbols)) != len(self.symbols)):
            raise ValueError('Expected distinct, explicit spot USDT symbols')
        start, end = timestamp_ns(self.start), timestamp_ns(self.end)
        if start % HOUR_NS or end % HOUR_NS or end <= start:
            raise ValueError('Coverage must use ascending whole UTC hours')
        if (end - start) // HOUR_NS * len(self.symbols) > MAX_ROWS:
            raise ValueError('Coverage exceeds row limit')


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    start_ns: int
    end_ns: int
    received_ns: int  # Archive receipt, never backdated to bar close.
    open: float
    high: float
    low: float
    close: float
    base_volume: float
    quote_volume: float
    trade_count: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float
    source_sha256: str
    source: str


def unsigned(text: str) -> int:
    if re.fullmatch('[0-9]{1,19}', text) is None:
        raise ValueError('Expected an unsigned integer')
    value = int(text)
    if value > 2**63 - 1:
        raise ValueError('Integer exceeds supported range')
    return value


def number(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError('Non-finite numerical field')
    return value


def parse_bars(data: bytes, contract: BarContract) -> tuple[Bar, ...]:
    """Require exact coverage per symbol; reject gaps instead of filling them."""
    if type(data) is not bytes or not data or len(data) > MAX_BYTES:
        raise ValueError('CSV byte limit exceeded or empty input')
    if not isinstance(contract, BarContract):
        raise ValueError('An explicit BarContract is required')
    contract.__post_init__()
    if digest(data) != contract.sha256:
        raise ValueError('CSV digest mismatch')
    lines = data.decode('utf-8').splitlines()
    if any(not line or len(line) > MAX_LINE for line in lines):
        raise ValueError('Blank or oversized CSV line')
    if len(lines) - 1 > MAX_ROWS:
        raise ValueError('CSV row limit exceeded')
    reader = csv.reader(io.StringIO(data.decode('utf-8'), newline=''), strict=True)
    if tuple(next(reader)) != COLUMNS:
        raise ValueError('Unexpected hourly CSV schema/order')
    start, end = timestamp_ns(contract.start), timestamp_ns(contract.end)
    expected_next = dict.fromkeys(contract.symbols, start)
    bars = []
    for line_number, fields in enumerate(reader, 2):
        try:
            if len(fields) != len(COLUMNS):
                raise ValueError('Wrong row width')
            row = dict(zip(COLUMNS, fields))
            symbol = row['symbol']
            if (symbol not in expected_next or row['base_asset'] != symbol[:-4]
                    or row['quote_asset'] != 'USDT' or row['instrument'] != 'spot'):
                raise ValueError('Unexpected symbol or instrument identity')
            begin, finish = timestamp_ns(row['bar_start']), timestamp_ns(row['bar_end'])
            if begin != expected_next[symbol] or finish != begin + HOUR_NS or finish > end:
                raise ValueError('Gap, duplicate, ordering or coverage mismatch')
            unit, factor = ('milliseconds', 1_000_000) if begin < MICROSECOND_CHANGE_NS else ('microseconds', 1000)
            if (row['source_time_unit'] != unit
                    or unsigned(row['source_open_time']) * factor != begin
                    or (unsigned(row['source_close_time']) + 1) * factor != finish):
                raise ValueError('Source clock does not match normalized interval')
            received = timestamp_ns(row['known_at'], receipt=True)
            if received < finish:
                raise ValueError('Archive receipt predates completion of the bar')
            op, hi, lo, cl = (number(row[key]) for key in ('open','high','low','close'))
            if not (0 < lo <= op <= hi and lo <= cl <= hi):
                raise ValueError('Invalid OHLC bounds')
            base, quote, taker_base, taker_quote = (number(row[key]) for key in (
                'base_volume','quote_volume','taker_buy_base_volume','taker_buy_quote_volume'))
            if not (0 <= taker_base <= base and 0 <= taker_quote <= quote):
                raise ValueError('Invalid volume or taker-volume bounds')
            trades = unsigned(row['trade_count'])
            check_digest(row['source_sha256'])
            prefix = f'https://data.binance.vision/data/spot/monthly/klines/{symbol}/1h/'
            # Source is provenance, never followed or downloaded by this tool.
            if not row['source'].startswith(prefix) or re.fullmatch(
                    r'[A-Za-z0-9_-]+\.zip', row['source'][len(prefix):]) is None:
                raise ValueError('Unexpected source archive path')
            bars.append(Bar(symbol, begin, finish, received, op, hi, lo, cl,
                            base, quote, trades, taker_base, taker_quote,
                            row['source_sha256'], row['source']))
            expected_next[symbol] = finish
        except (ValueError, OverflowError) as exc:
            raise ValueError(f'Invalid hourly row {line_number}: {exc}') from exc
    if any(value != end for value in expected_next.values()):
        raise ValueError('Missing expected hourly coverage')
    return tuple(bars)


def audit(bars: Sequence[Bar], contract: BarContract) -> dict:
    """Metadata only; does not report prices, labels or predictive outcomes."""
    return {
        'schema_version': 1,
        'kind': 'normalized_binance_spot_hourly_admission',
        'data_sha256': contract.sha256,
        'reader_sha256': digest(Path(__file__).read_bytes()),
        'symbols': list(contract.symbols), 'start': contract.start, 'end_exclusive': contract.end,
        'rows': len(bars),
        'rows_per_symbol': {s: sum(b.symbol == s for b in bars) for s in contract.symbols},
        'source_time_units': {
            'milliseconds': sum(b.start_ns < MICROSECOND_CHANGE_NS for b in bars),
            'microseconds': sum(b.start_ns >= MICROSECOND_CHANGE_NS for b in bars)},
        'archive_digests': sorted({b.source_sha256 for b in bars}),
        'earliest_archive_receipt_ns': min(b.received_ns for b in bars),
        'latest_archive_receipt_ns': max(b.received_ns for b in bars),
        'checks': ['pinned_bytes', 'exact_symbol_hour_grid', 'source_clock_units',
                   'bar_end_exclusive', 'archive_receipt_after_bar_end',
                   'finite_positive_ohlc_bounds', 'nonnegative_volumes_and_taker_bounds',
                   'integer_trade_count', 'spot_identity_and_source_shape'],
        'historical_availability_verified': False,
        'raw_archives_reconstructed': False,
        'forecast_experiment_run': False,
        'scientific_evidence_ceiling': 'E0',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--symbols', nargs='+', required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory already exists; choose a new run directory')
    contract = BarContract(args.sha256, tuple(args.symbols), args.start, args.end)
    with args.csv.open('rb') as source:
        data = source.read(MAX_BYTES + 1)
    bars = parse_bars(data, contract)
    report = audit(bars, contract)
    encoded = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'audit.json').write_bytes(encoded)
    print(f'Accepted {len(bars)} hourly bars; metadata-only audit saved.')


if __name__ == '__main__':
    main()
