"""Independent checks for a complete comparison, retained failures and provenance."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import benchmark_frontier as frontier
from runtime_common import canonical, digest


class Comparison(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(frontier.PROTOCOL.read_text())
        self.cells = frontier.matrix(self.protocol)

    def records(self):
        records = []
        # ONNX wins p50; compiled Torch wins p99. The gate must compare each
        # percentile with its own best baseline, not select a convenient engine.
        medians = {'native_core': 120, 'native_python': 140, 'torch_eager': 200,
                   'torch_compiled_inductor': 110, 'onnxruntime_cpu': 100}
        tails = {'native_core': 240, 'native_python': 280, 'torch_eager': 400,
                 'torch_compiled_inductor': 200, 'onnxruntime_cpu': 300}
        for cell in self.cells:
            n = cell['samples']
            result = dict(engine=cell['engine'], batch=cell['batch'], output_mode='last',
                          times_ns=[medians[cell['engine']]] * (n // 2) +
                                   [tails[cell['engine']]] * (n - n // 2),
                          clock_times_ns=[0] * 1000, setup_ns=1, first_call_ns=1,
                          max_abs_difference=0., checksum=1.)
            records.append(dict(cell=cell, status='complete', result=result))
        return records

    def test_exact_workload_and_rotation(self):
        self.assertEqual(len(self.cells), 90)
        self.assertEqual(sum(c['samples'] for c in self.cells), 54000)
        self.assertEqual(len({frontier.cell_id(c) for c in self.cells}), 90)
        for i in range(0, 90, 5):
            self.assertEqual({c['engine'] for c in self.cells[i:i+5]}, set(frontier.ENGINES))
        self.assertNotEqual(self.cells[0]['engine'], self.cells[5]['engine'])
        self.assertEqual(len(frontier.matrix(self.protocol, admission=True)), 30)

    def test_gate_boundary_and_separate_best_baselines(self):
        report = frontier.aggregate(self.protocol, self.records())
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['timed_calls'], 54000)
        for batch in ('1', '8'):
            gate = report['batches'][batch]
            self.assertTrue(gate['passed'])
            self.assertEqual(gate['p50_ratio'], 1.2)
            self.assertEqual(gate['p99_ratio'], 1.2)
            self.assertEqual(gate['engines']['native_python']['cells'], 9)
        records = self.records()
        for r in records:
            if r['cell']['engine'] == 'native_core' and r['cell']['batch'] == 8:
                r['result']['times_ns'][-100:] = [241] * 100
        report = frontier.aggregate(self.protocol, records)
        self.assertTrue(report['batches']['1']['passed'])
        self.assertFalse(report['batches']['8']['passed'])

    def test_missing_failed_duplicate_and_unknown_cells(self):
        for records in (self.records()[:-1], self.records()[:0]):
            report = frontier.aggregate(self.protocol, records)
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(report['batches'], {})
        records = self.records()
        records[0] = dict(cell=records[0]['cell'], status='failed', error='timeout')
        self.assertEqual(frontier.aggregate(self.protocol, records)['status'], 'incomplete')
        records = self.records()
        with self.assertRaises(ValueError): frontier.aggregate(self.protocol, records + [records[0]])
        records[0]['cell']['seed'] = 999
        with self.assertRaises(ValueError): frontier.aggregate(self.protocol, records)

    def test_reject_bad_workers_instead_of_averaging_away_failures(self):
        for field, value in [('times_ns', [1]), ('times_ns', [math.nan] * 1000),
                             ('checksum', math.inf), ('batch', 8), ('output_mode', 'all'),
                             ('max_abs_difference', -1), ('setup_ns', True)]:
            records = self.records()
            records[0]['result'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                frontier.aggregate(self.protocol, records)

    def test_median_of_cells_not_pooled_samples(self):
        records = self.records()
        chosen = [r for r in records if r['cell']['batch'] == 1 and r['cell']['engine'] == 'native_core']
        for i, record in enumerate(chosen):
            record['result']['times_ns'] = [i + 1] * 1000
        result = frontier.aggregate(self.protocol, records)
        self.assertEqual(result['batches']['1']['engines']['native_core']['p50_ns'], 5)
        self.assertEqual(result['batches']['1']['engines']['native_core']['p99_ns'], 5)

    def test_protocol_is_frozen(self):
        frontier.validate_protocol(self.protocol)
        changed = copy.deepcopy(self.protocol)
        changed['absolute_tolerance'] = .01
        with self.assertRaises(ValueError): frontier.validate_protocol(changed)


class Artifacts(unittest.TestCase):
    def test_seal_detects_edit_missing_extra_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'result.json').write_bytes(canonical({'value': 1}))
            frontier.seal(root, 'complete')
            self.assertEqual(frontier.verify_seal(root)['status'], 'complete')
            (root / 'extra').write_text('unrecorded')
            with self.assertRaises(ValueError): frontier.verify_seal(root)
            (root / 'extra').unlink()
            original = (root / 'result.json').read_bytes()
            (root / 'result.json').write_text('changed')
            with self.assertRaises(ValueError): frontier.verify_seal(root)
            (root / 'result.json').write_bytes(original)
            receipt = json.loads((root / 'seal.json').read_text())
            receipt['files']['../escape'] = digest(b'x')
            (root / 'seal.json').write_bytes(canonical(receipt))
            with self.assertRaises(ValueError): frontier.verify_seal(root)

    @unittest.skipUnless(os.name == 'posix', 'POSIX process supervision')
    def test_matrix_preserves_failure_and_continues_other_competitors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cells = frontier.matrix(json.loads(frontier.PROTOCOL.read_text()))[:3]
            def runner(cell, output, timeout):
                output.mkdir()
                (output / 'stderr.log').write_text('diagnostic')
                if cell == cells[1]: raise RuntimeError('deliberate compiler failure')
                return {'fixture': True}
            records = frontier.execute_cells(cells, root, runner, budget=10)
            self.assertEqual([r['status'] for r in records], ['complete', 'failed', 'complete'])
            self.assertIn('deliberate', (root / frontier.cell_id(cells[1]) / 'record.json').read_text())
            self.assertTrue((root / frontier.cell_id(cells[2]) / 'record.json').exists())
            with self.assertRaises(FileExistsError): frontier.execute_cells(cells, root, runner, budget=10)

    def test_global_deadline_prevents_later_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            cells = frontier.matrix(json.loads(frontier.PROTOCOL.read_text()))[:2]
            with patch.object(frontier.time, 'monotonic', side_effect=[0, 11, 12]), \
                    patch.object(frontier, 'print'):
                records = frontier.execute_cells(cells, Path(tmp), None, budget=10)
            self.assertEqual([r['status'] for r in records], ['not_run', 'not_run'])

class HostLock(unittest.TestCase):
    def test_every_installed_package_needs_matching_wheel_bytes(self):
        import zipfile
        from frontier_host import wheel_lock
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel = root / 'example-1.0-py3-none-any.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('example-1.0.dist-info/METADATA', 'Name: Example\nVersion: 1.0\n')
            locked = wheel_lock(root, {'example': '1.0'})
            self.assertEqual(locked['example']['sha256'], digest(wheel.read_bytes()))
            for packages in ({'example': '2.0'}, {'example': '1.0', 'missing': '1.0'}):
                with self.assertRaises(ValueError): wheel_lock(root, packages)
            (root / 'duplicate.whl').write_bytes(wheel.read_bytes())
            with self.assertRaises(ValueError): wheel_lock(root, {'example': '1.0'})

    def test_admission_requires_every_original_record(self):
        protocol = json.loads(frontier.PROTOCOL.read_text())
        fixture = Comparison()
        fixture.setUp()
        originals = {frontier.cell_id(r['cell']): r for r in fixture.records()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for cell in frontier.matrix(protocol, admission=True):
                record = copy.deepcopy(originals[frontier.cell_id(cell)])
                record['cell'] = cell
                record['result']['times_ns'] = [1]
                records.append(record)
                directory = root / frontier.cell_id(cell)
                directory.mkdir()
                (directory / 'record.json').write_bytes(canonical(record))
            (root / 'admission.json').write_bytes(canonical(dict(status='admitted', records=records)))
            frontier.seal(root, 'admitted')
            self.assertEqual(len(frontier.admitted_records(root, protocol)), 30)
            # A new, internally hashed but incomplete summary still cannot pass.
            (root / 'seal.json').unlink()
            records.pop()
            (root / 'admission.json').write_bytes(canonical(dict(status='admitted', records=records)))
            frontier.seal(root, 'admitted')
            with self.assertRaises(ValueError): frontier.admitted_records(root, protocol)

    def test_legacy_host_cannot_be_called_a_frontier_run(self):
        from frontier_host import observe
        with patch('frontier_host.platform.system', return_value='Darwin'):
            with self.assertRaisesRegex(ValueError, 'Linux'):
                observe(ROOT / 'build/release/stealth_runtime_bench')


if __name__ == '__main__': unittest.main()
