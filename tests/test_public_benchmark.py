"""Offline, independent tests. No external datasets or packages are required."""
import csv
import hashlib
import importlib.util
import io
import json
import math
import random
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

TOOL = Path(__file__).resolve().parents[1] / 'tools' / 'prepare_public_benchmark.py'
spec = importlib.util.spec_from_file_location('public_benchmark', TOOL)
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)
COLUMNS = ['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']


def fixture(n=24):
    # All numbers are invented, not copied or adapted from ETTm1.
    rows = [[(datetime(2020, 1, 1) + timedelta(minutes=15*i)).strftime('%Y-%m-%d %H:%M:%S')]
            + [str(i + j) for j in range(7)] for i in range(n)]
    return rows


def encoded(rows, columns=COLUMNS):
    out = io.StringIO(newline='')
    writer = csv.writer(out, lineterminator='\n')
    writer.writerow(columns)
    writer.writerows(rows)
    return out.getvalue().encode('ascii')


def contract(rows):
    return {'name': 'ETTm1.csv', 'columns': COLUMNS, 'rows': len(rows),
            'first': rows[0][0], 'last': rows[-1][0], 'interval_seconds': 900,
            'timezone': 'unspecified_source_clock'}


def source_fixture():
    rows = fixture()
    files = {'ETTm1.csv': encoded(rows), 'LICENSE': b'INVENTED TEST LICENSE\n',
             'README.md': b'INVENTED TEST SOURCE\n'}
    source = {'schema_version': 1, 'dataset_id': 'invented_test',
              'repository': 'https://example.com/test', 'revision': '0'*40,
              'license': 'TEST-ONLY', 'attribution': 'Original test values',
              'distribution_policy': 'Test only', 'csv': contract(rows),
              'split_ends': {'train': 12, 'validation': 18, 'test': 24},
              'files': [{'name': name, 'url': 'https://example.com/'+name,
                         'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
                        for name, value in files.items()]}
    return source, files


class ParserTests(unittest.TestCase):
    def test_exact_values_and_clocks(self):
        rows = fixture()
        frame = b.parse_csv(encoded(rows), contract(rows))
        self.assertEqual(frame.timestamps, tuple(row[0] for row in rows))
        self.assertEqual(frame.values[3], tuple(float(3+j) for j in range(7)))

    def test_missing_nonfinite_out_of_fp32_and_text(self):
        for value in ('', 'NA', 'nan', 'inf', '-inf', '3.5e38', 'word'):
            rows = fixture(); rows[-1][-1] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                b.parse_csv(encoded(rows), contract(rows))

    def test_schema_and_row_width(self):
        rows = fixture()
        cases = [encoded(rows, COLUMNS[::-1]), encoded(rows, COLUMNS[:-1]),
                 encoded([rows[0][:-1]] + rows[1:]), encoded([rows[0]+['8']] + rows[1:]),
                 encoded(rows) + b'\n', encoded(rows) + b'\xff']
        for data in cases:
            with self.subTest(data=data[:30]), self.assertRaises(ValueError):
                b.parse_csv(data, contract(rows))

    def test_timestamp_failures(self):
        for index, timestamp in ((1, '2020-01-01 00:00:00'),
                                  (1, '2020-01-01 00:30:00'),
                                  (1, '2019-12-31 23:45:00'),
                                  (1, '2020-1-1 00:15:00'),
                                  (1, '2020-01-01T00:15:00Z'),
                                  (1, '2020-02-30 00:15:00')):
            rows = fixture(); expected = contract(rows); rows[index][0] = timestamp
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                b.parse_csv(encoded(rows), expected)

    def test_truncation_and_wrong_endpoint(self):
        rows = fixture(); expected = contract(rows)
        for data in (encoded(rows[:-1]), encoded(rows[1:]), encoded(fixture(25))):
            with self.assertRaises(ValueError): b.parse_csv(data, expected)
        expected['last'] = '2020-01-02 00:00:00'
        with self.assertRaises(ValueError): b.parse_csv(encoded(rows), expected)

    def test_resource_limits(self):
        rows = fixture()
        with self.assertRaises(ValueError): b.parse_csv(b'x'*12_000_001, contract(rows))
        rows[0][1] = '0'*2049
        with self.assertRaises(ValueError): b.parse_csv(encoded(rows), contract(rows))
        expected = contract(fixture()); expected['rows'] = 100_001
        with self.assertRaises(ValueError): b.parse_csv(encoded(fixture()), expected)


class ScalerTests(unittest.TestCase):
    def test_hand_computed_population_variance_and_constant(self):
        # Training first column [1,3] => mean2, population std1. Second is constant.
        s = b.fit_scaler(((1., 5.), (3., 5.), (999., -999.)), 2)
        self.assertEqual(s.mean, (2., 5.)); self.assertEqual(s.scale, (1., 1.))
        self.assertEqual(s.transform((4., 5.)), (2., 0.))
        t = b.fit_scaler(((0.,), (2.,), (4.,)), 3)
        self.assertAlmostEqual(t.scale[0], math.sqrt(8/3), places=14)

    def test_future_suffix_cannot_change_fit_or_prefix(self):
        rows = tuple((float(i), float(i+5)) for i in range(20))
        changed = rows[:8] + tuple((1e20, -1e20) for _ in rows[8:])
        first = b.fit_scaler(rows, 8); second = b.fit_scaler(changed, 8)
        self.assertEqual(first, second)
        self.assertEqual(first.transform(rows[2]), second.transform(rows[2]))

    def test_reject_invalid_training_input(self):
        for rows, end in (((), 0), (((1.,),), 0), (((1.,),), 2),
                          (((1.,), (math.nan,)), 2), (((1.,), (math.inf,)), 2),
                          (((1.,), (2., 3.)), 2), (((), ()), 2), (((1.,),), True)):
            with self.subTest(rows=rows, end=end), self.assertRaises(ValueError):
                b.fit_scaler(rows, end)

    def test_transform_rejects_bad_width_and_nonfinite(self):
        s = b.fit_scaler(((1., 2.), (3., 4.)), 2)
        for row in ((1.,), (1., math.inf), (math.nan, 2.), (1e308, 1e308)):
            with self.assertRaises(ValueError): s.transform(row)


class WindowTests(unittest.TestCase):
    def test_point_hand_computed_boundaries(self):
        ends = {'train': 12, 'validation': 18, 'test': 24}
        train = list(b.windows(24, ends, 'train', context=4, horizon=2, mode='point'))
        valid = list(b.windows(24, ends, 'validation', context=4, horizon=2, mode='point'))
        self.assertEqual((train[0].input_start, train[0].input_end,
                          train[0].target_start, train[0].target_end), (0, 4, 5, 6))
        self.assertEqual(len(train), 7)
        self.assertEqual((valid[0].input_start, valid[0].input_end,
                          valid[0].target_start, valid[0].target_end), (8, 12, 13, 14))
        self.assertEqual((valid[-1].input_end, valid[-1].target_end), (16, 18))
        self.assertEqual(len(valid), 5)

    def test_curve_and_no_crossing_targets(self):
        ends = {'train': 12, 'validation': 18, 'test': 24}
        valid = list(b.windows(24, ends, 'validation', context=4, horizon=2, mode='curve'))
        self.assertEqual((valid[0].target_start, valid[0].target_end), (12, 14))
        for w in valid:
            self.assertLess(w.input_end-1, w.target_start)
            self.assertGreaterEqual(w.target_start, 12)
            self.assertLessEqual(w.target_end, 18)

    def test_seeded_index_properties_and_unused_suffix(self):
        rng = random.Random(20261028)
        for _ in range(200):
            tr = rng.randint(5, 40); va = tr+rng.randint(5, 20); te = va+rng.randint(5, 20)
            context = rng.randint(1, tr); horizon = rng.randint(1, 5)
            ends = {'train': tr, 'validation': va, 'test': te}
            for name, start, stop in (('train', 0, tr), ('validation', tr, va), ('test', va, te)):
                for mode in ('point', 'curve'):
                    a = list(b.windows(te, ends, name, context, horizon, mode))
                    self.assertEqual(a, list(b.windows(te+10, ends, name, context, horizon, mode)))
                    self.assertEqual(len(a), max(0, stop-horizon-max(context, start)+1))
                    for w in a:
                        self.assertEqual(w.input_end-w.input_start, context)
                        self.assertGreaterEqual(w.input_end, start)
                        self.assertGreaterEqual(w.target_start, start)
                        self.assertLessEqual(w.target_end, stop)
                        self.assertLess(w.input_end-1, w.target_start)
                        self.assertEqual(w.target_end-(w.input_end-1), horizon+1)
                        self.assertEqual(w.target_end-w.target_start, 1 if mode=='point' else horizon)

    def test_reject_bad_arguments_and_partitions(self):
        good = {'train': 12, 'validation': 18, 'test': 24}
        for context, horizon in ((0, 1), (1, 0), (-1, 2), (1025, 1), (1, 1025), (True, 1), (1, 1.5)):
            with self.assertRaises(ValueError): list(b.windows(24, good, 'train', context, horizon, 'point'))
        for ends in ({'train': 0, 'validation': 18, 'test': 24},
                     {'train': 12, 'validation': 12, 'test': 24},
                     {'train': 12, 'validation': 18, 'test': 25},
                     {'train': 12, 'validation': 18},
                     {'train': True, 'validation': 18, 'test': 24}):
            with self.assertRaises(ValueError): list(b.windows(24, ends, 'train', 4, 2, 'point'))
        with self.assertRaises(ValueError): list(b.windows(24, good, 'unknown', 4, 2, 'point'))
        with self.assertRaises(ValueError): list(b.windows(24, good, 'train', 4, 2, 'unknown'))


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.files = source_fixture()

    def opener(self, request, **kwargs):
        url = request.full_url if hasattr(request, 'full_url') else request
        self.assertTrue(kwargs.get('context').check_hostname)
        return io.BytesIO(self.files[url.rsplit('/', 1)[-1]])

    def fetch(self, name='raw'):
        raw = self.root/name
        with patch.object(b, 'SOURCE', self.source, create=True):
            with patch('urllib.request.urlopen', side_effect=self.opener): b.fetch_snapshot(raw)
        return raw

    def test_verified_fetch_prepare_and_byte_identical_replay(self):
        raw = self.fetch()
        receipt = json.loads((raw/'receipt.json').read_text())
        self.assertEqual(receipt['status'], 'complete')
        self.assertEqual(len(receipt['files']), 3)
        one, two = self.root/'prepared1', self.root/'prepared2'
        with patch.object(b, 'SOURCE', self.source, create=True):
            b.prepare_snapshot(raw, one); b.prepare_snapshot(raw, two)
            b.check_snapshot(raw, one)
        self.assertEqual(sorted(p.name for p in one.iterdir()), ['audit.json', 'manifest.json', 'scaler.json'])
        for p in one.iterdir(): self.assertEqual(p.read_bytes(), (two/p.name).read_bytes())
        audit = json.loads((one/'audit.json').read_text())
        self.assertEqual(audit['rows'], 24)
        self.assertEqual(audit['split_ends'], self.source['split_ends'])
        self.assertEqual(audit['forecast_results_computed'], False)
        scaler = json.loads((one/'scaler.json').read_text())
        self.assertEqual(scaler['training_rows'], 12)
        self.assertEqual(scaler['mean'][0], 5.5)

    def test_interrupted_stream_retains_partial_bytes(self):
        class Interrupted:
            def __enter__(self): self.calls = 0; return self
            def __exit__(self, *args): return False
            def read(self, limit):
                self.calls += 1
                if self.calls == 1: return b'partial'
                raise OSError('simulated interrupted transport')
        raw = self.root/'interrupted'
        with patch.object(b, 'SOURCE', self.source), patch('urllib.request.urlopen', return_value=Interrupted()):
            with self.assertRaises(OSError): b.fetch_snapshot(raw)
        self.assertEqual((raw/'ETTm1.csv.partial').read_bytes(), b'partial')
        self.assertEqual(json.loads((raw/'receipt.json').read_text())['status'], 'failed')

    def test_receipt_identity_and_verified_csv_contract(self):
        raw = self.fetch()
        original = (raw/'receipt.json').read_bytes()
        for field, value in (('source_manifest_sha256', '0'*64), ('schema_version', 9), ('files', [])):
            receipt = json.loads(original); receipt[field] = value
            (raw/'receipt.json').write_bytes(json.dumps(receipt).encode())
            with patch.object(b, 'SOURCE', self.source), self.assertRaises(ValueError):
                b.prepare_snapshot(raw, self.root/'never-created')
        (raw/'receipt.json').write_bytes(original)
        # Correctly hashed bytes can still violate the data contract.
        rows = fixture(); rows[2][0] = rows[1][0]
        self.files['ETTm1.csv'] = encoded(rows)
        record = self.source['files'][0]
        record.update(bytes=len(self.files['ETTm1.csv']), sha256=hashlib.sha256(self.files['ETTm1.csv']).hexdigest())
        raw2 = self.fetch('valid-hash-bad-clock')
        with patch.object(b, 'SOURCE', self.source), self.assertRaises(ValueError):
            b.prepare_snapshot(raw2, self.root/'never-created')
        self.assertFalse((self.root/'never-created').exists())

    def test_existing_destinations_never_overwritten(self):
        raw = self.fetch(); out = self.root/'prepared'
        with patch.object(b, 'SOURCE', self.source, create=True):
            b.prepare_snapshot(raw, out)
            before = (out/'manifest.json').read_bytes()
            with self.assertRaises(FileExistsError): b.prepare_snapshot(raw, out)
            with patch('urllib.request.urlopen') as opener:
                with self.assertRaises(FileExistsError): b.fetch_snapshot(raw)
                opener.assert_not_called()
        self.assertEqual(before, (out/'manifest.json').read_bytes())

    def test_reject_bad_or_missing_raw_and_partial_receipt(self):
        raw = self.fetch()
        for change in ('hash', 'length', 'missing', 'receipt'):
            path = raw/'ETTm1.csv'; path.write_bytes(self.files['ETTm1.csv'])
            receipt = json.loads((raw/'receipt.json').read_text())
            if change=='hash': path.write_bytes(path.read_bytes().replace(b'0,1,2', b'9,1,2', 1))
            if change=='length': path.write_bytes(path.read_bytes()+b'\n')
            if change=='missing': path.unlink()
            if change=='receipt':
                receipt['status'] = 'failed'; (raw/'receipt.json').write_text(json.dumps(receipt))
            out = self.root/('out-'+change)
            with patch.object(b, 'SOURCE', self.source, create=True), self.assertRaises((ValueError, FileNotFoundError)):
                b.prepare_snapshot(raw, out)
            self.assertFalse(out.exists())

    def test_fetch_failures_retained_but_not_published_as_complete(self):
        for change in ('hash', 'short', 'long', 'transport'):
            def bad(request, **kwargs):
                if change=='transport': raise URLError('test network unavailable')
                value = self.files['ETTm1.csv']
                if change=='hash': value = b'x'*len(value)
                elif change=='short': value = value[:-1]
                elif change=='long': value = value+b'x'
                return io.BytesIO(value)
            raw = self.root/change
            with patch.object(b, 'SOURCE', self.source, create=True), patch('urllib.request.urlopen', side_effect=bad):
                with self.assertRaises((ValueError, URLError)): b.fetch_snapshot(raw)
            receipt = json.loads((raw/'receipt.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            with patch.object(b, 'SOURCE', self.source, create=True), self.assertRaises(ValueError):
                b.prepare_snapshot(raw, self.root/('prepared-'+change))

    def test_modified_prepared_outputs_and_missing_manifest_rejected(self):
        raw = self.fetch(); out = self.root/'prepared'
        with patch.object(b, 'SOURCE', self.source, create=True):
            b.prepare_snapshot(raw, out)
            original = (out/'scaler.json').read_bytes()
            (out/'scaler.json').write_bytes(original+b'\n')
            with self.assertRaises(ValueError): b.check_snapshot(raw, out)
            (out/'scaler.json').write_bytes(original)
            (out/'manifest.json').unlink()
            with self.assertRaises((ValueError, FileNotFoundError)): b.check_snapshot(raw, out)


if __name__ == '__main__': unittest.main()
