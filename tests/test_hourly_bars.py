"""Synthetic, independently specified cases for the external hourly-bar boundary."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import hourly_bars as h

HEADER = ('symbol,base_asset,quote_asset,instrument,bar_start,bar_end,'
          'source_open_time,source_close_time,source_time_unit,open,high,low,close,'
          'base_volume,quote_volume,trade_count,taker_buy_base_volume,'
          'taker_buy_quote_volume,known_at,source_sha256,source').split(',')


def rows():
    # Two adjacent UTC hours across the documented milliseconds -> microseconds change.
    return [dict(zip(HEADER, [
        'BTCUSDT', 'BTC', 'USDT', 'spot', start, end, raw_start, raw_end, unit,
        '100', '110', '90', '105', '10', '1000', '5', '4', '400',
        '2026-01-01T01:02:03.000004Z', 'a' * 64,
        'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/example.zip'
    ])) for start, end, raw_start, raw_end, unit in [
        ('2024-12-31T23:00:00Z', '2025-01-01T00:00:00Z',
         '1735686000000', '1735689599999', 'milliseconds'),
        ('2025-01-01T00:00:00Z', '2025-01-01T01:00:00Z',
         '1735689600000000', '1735693199999999', 'microseconds')]]


def encode(records):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=HEADER, lineterminator='\n')
    writer.writeheader(); writer.writerows(records)
    return stream.getvalue().encode()


def contract(data):
    return h.BarContract(hashlib.sha256(data).hexdigest(), ('BTCUSDT',),
                         '2024-12-31T23:00:00Z', '2025-01-01T01:00:00Z')


class HourlyBarTests(unittest.TestCase):
    def parse(self, records):
        data = encode(records)
        return h.parse_bars(data, contract(data))

    def test_independent_clock_values_and_receipt_are_preserved(self):
        result = self.parse(rows())
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].start_ns, 1735686000000000000)
        self.assertEqual(result[0].end_ns, 1735689600000000000)
        self.assertEqual(result[1].end_ns, 1735693200000000000)
        self.assertEqual(result[0].received_ns, 1767229323000004000)
        self.assertGreater(result[0].received_ns, result[1].end_ns)
        self.assertEqual(result[0].close, 105.)
        self.assertEqual(result[0].source_sha256, 'a' * 64)
        with self.assertRaises(AttributeError): result[0].close = 99.

    def test_digest_is_mandatory(self):
        data = encode(rows())
        with self.assertRaisesRegex(ValueError, 'digest'):
            h.parse_bars(data + b'\n', contract(data))

    def test_gap_duplicate_reverse_and_wrong_coverage(self):
        base = rows()
        for records in (base[:1], base[1:], base + base[:1], base[::-1], []):
            with self.subTest(records=records), self.assertRaises(ValueError):
                self.parse(records)

    def test_clock_unit_boundary_and_interval_errors(self):
        changes = [
            ('source_time_unit', 'microseconds'),
            ('source_open_time', '1735686000001'),
            ('source_close_time', '1735689600000'),
            ('bar_start', '2024-12-31T23:00:00'),
            ('bar_start', '2024-12-31T23:00:00+00:00'),
            ('bar_start', '2024-12-31T23:01:00Z'),
            ('bar_end', '2025-01-01T00:01:00Z'),
            ('known_at', '2024-12-31T23:30:00Z'),
            ('known_at', '2026-01-01T01:02:03'),
            ('source_open_time', '1.735686e12'),
        ]
        for key, value in changes:
            records = rows(); records[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.parse(records)

    def test_nonfinite_nonpositive_and_impossible_ohlc(self):
        for key, value in [('open','0'), ('low','-1'), ('high','99'),
                           ('close','111'), ('open','89'), ('low','111'),
                           ('close','NaN'), ('high','inf'), ('open','1e999')]:
            records = rows(); records[0][key] = value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                self.parse(records)

    def test_volume_trade_count_and_taker_bounds(self):
        for key, value in [('base_volume','-1'), ('quote_volume','nan'),
                           ('trade_count','1.5'), ('trade_count','-1'),
                           ('trade_count','999999999999999999999'),
                           ('taker_buy_base_volume','11'),
                           ('taker_buy_quote_volume','1001')]:
            records = rows(); records[0][key] = value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                self.parse(records)
        records = rows()
        for key in ('base_volume','quote_volume','trade_count',
                    'taker_buy_base_volume','taker_buy_quote_volume'):
            records[0][key] = '0'
        self.assertEqual(len(self.parse(records)), 2)

    def test_symbol_instrument_and_provenance(self):
        for key, value in [('symbol','ETHUSDT'), ('base_asset','ETH'),
                           ('quote_asset','USD'), ('instrument','future'),
                           ('source_sha256','x'*64), ('source','https://example.com/a'),
                           ('source','https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1h/a.zip')]:
            records = rows(); records[0][key] = value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                self.parse(records)

    def test_schema_extra_missing_duplicate_and_blank_rows(self):
        good = encode(rows())
        variants = [good.replace(b'symbol,',b'symbol,symbol,',1),
                    good.replace(b'symbol,',b'',1),
                    good.replace(b'BTCUSDT,BTC,',b'BTCUSDT,',1),
                    good.replace(b'BTCUSDT,BTC,',b'BTCUSDT,BTC,extra,',1),
                    good + b'\n']
        for data in variants:
            with self.subTest(data=data[:80]), self.assertRaises(ValueError):
                h.parse_bars(data, contract(data))

    def test_contract_bounds_and_duplicate_symbols(self):
        sha = 'a' * 64
        for args in [(sha,(), '2025-01-01T00:00:00Z','2025-01-01T01:00:00Z'),
                     (sha,('BTCUSDT','BTCUSDT'),'2025-01-01T00:00:00Z','2025-01-01T01:00:00Z'),
                     ('x',('BTCUSDT',),'2025-01-01T00:00:00Z','2025-01-01T01:00:00Z'),
                     (sha,('BTCUSDT',),'2025-01-01T01:00:00Z','2025-01-01T00:00:00Z'),
                     (sha,('BTCUSDT',),'2025-01-01T00:00:01Z','2025-01-01T01:00:01Z')]:
            with self.subTest(args=args), self.assertRaises(ValueError): h.BarContract(*args)

    def test_resource_bounds(self):
        data=encode(rows())
        with patch.object(h, 'MAX_BYTES', len(data)-1):
            with self.assertRaises(ValueError): h.parse_bars(data, contract(data))
        with patch.object(h, 'MAX_LINE', 10):
            with self.assertRaises(ValueError): h.parse_bars(data, contract(data))
        with patch.object(h, 'MAX_ROWS', 1):
            with self.assertRaises(ValueError): h.parse_bars(data, contract(data))

    def test_multiple_symbols_independent_grid(self):
        records=rows(); second=rows()
        for row in second:
            row['symbol']='ETHUSDT'; row['base_asset']='ETH'
            row['source']=row['source'].replace('BTCUSDT','ETHUSDT')
        data=encode([records[0],second[0],records[1],second[1]])
        c=h.BarContract(hashlib.sha256(data).hexdigest(), ('BTCUSDT','ETHUSDT'),
                        '2024-12-31T23:00:00Z','2025-01-01T01:00:00Z')
        self.assertEqual(len(h.parse_bars(data,c)),4)
        bad=encode(records + second[:1])
        c=h.BarContract(hashlib.sha256(bad).hexdigest(), c.symbols,c.start,c.end)
        with self.assertRaises(ValueError): h.parse_bars(bad,c)

    def test_invalid_csv_never_creates_success_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'input.csv'; output = root / 'audit'
            original = encode(rows()); source.write_bytes(original + b'corrupt')
            result = subprocess.run([
                sys.executable, str(Path(h.__file__)), '--csv', str(source),
                '--sha256', contract(original).sha256, '--symbols', 'BTCUSDT',
                '--start', '2024-12-31T23:00:00Z', '--end', '2025-01-01T01:00:00Z',
                '--output', str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('digest mismatch', result.stderr)
            self.assertFalse(output.exists())

    def test_fresh_process_audit_replay_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/'input.csv'; path.write_bytes(encode(rows()))
            argv=[sys.executable,str(Path(h.__file__)), '--csv',str(path),
                  '--sha256',contract(path.read_bytes()).sha256,'--symbols','BTCUSDT',
                  '--start','2024-12-31T23:00:00Z','--end','2025-01-01T01:00:00Z']
            reports=[]
            for name in ('a','b'):
                p=subprocess.run(argv+['--output',str(root/name)],capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr)
                reports.append((root/name/'audit.json').read_bytes())
            self.assertEqual(*reports)
            report=json.loads(reports[0])
            self.assertFalse(report['historical_availability_verified'])
            self.assertFalse(report['forecast_experiment_run'])
            self.assertEqual(report['rows'],2)
            self.assertEqual(report['source_time_units'],{'milliseconds':1,'microseconds':1})
            self.assertNotIn('105',json.dumps(report))
            p=subprocess.run(argv+['--output',str(root/'a')],capture_output=True)
            self.assertNotEqual(p.returncode,0)
            self.assertEqual((root/'a'/'audit.json').read_bytes(),reports[0])


if __name__ == '__main__': unittest.main()
