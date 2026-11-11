"""Exercise actual timing subprocesses without treating CI latency as evidence."""
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from benchmark_frontier import run_cell
from runtime_common import input_bytes
from torch_forecaster import Config, Forecaster


@unittest.skipUnless(os.name == 'posix', 'POSIX supervisor')
class Worker(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifacts = self.root / 'artifacts'
        self.artifacts.mkdir()
        model = Forecaster(Config(2, 4, 2, 6, 3, 1, 1, output='correction'), [0, 0], [1, 1], seed=7)
        (self.artifacts / 'model-7.stlf').write_bytes(model.export_bytes())
        (self.artifacts / 'input.bin').write_bytes(input_bytes(list(range(24)), 4, 3, 2))
        # Zero correction head: the answer is independently known for every row.
        (self.artifacts / 'expected-7.bin').write_bytes(struct.pack('<12f', *range(1, 24, 2)))
        self.native = ROOT / 'build/release/stealth_runtime_bench'

    def run_engine(self, engine, name, *, timeout=60, graphs=None, batch=2):
        return run_cell(dict(seed=7, trial=1, batch=batch, samples=3, engine=engine), self.root / name,
                        artifacts=self.artifacts, native=self.native, warmup=2, timeout=timeout,
                        expected_graphs=graphs)

    def test_eager_timing_and_independent_checksum(self):
        result = self.run_engine('torch_eager', 'eager')
        self.assertEqual(result['windows_checked'], 4)
        self.assertEqual(result['max_abs_difference'], 0)
        # Groups are [5,11], [17,23]; consume indices 0,1,0 across three calls.
        self.assertEqual(result['checksum'], 5 + 23 + 5)
        self.assertEqual(len(result['times_ns']), 3)
        self.assertTrue(all(t > 0 for t in result['times_ns']))
        with self.assertRaises(FileExistsError): self.run_engine('torch_eager', 'eager')

    def test_all_three_available_engines_agree(self):
        if not self.native.is_file(): self.skipTest('Native benchmark has not been built')
        results = [self.run_engine(e, e) for e in ('native_core', 'native_python', 'torch_eager')]
        self.assertEqual({r['checksum'] for r in results}, {33.})
        self.assertTrue(all(r['max_abs_difference'] == 0 for r in results))

    def test_late_mismatch_never_yields_timing_result(self):
        path = self.artifacts / 'expected-7.bin'
        path.write_bytes(path.read_bytes()[:-4] + struct.pack('<f', 999))
        with self.assertRaisesRegex(RuntimeError, 'worker failed'):
            self.run_engine('torch_eager', 'wrong')
        self.assertEqual((self.root / 'wrong/stdout.log').read_text(), '')
        self.assertIn('parity', (self.root / 'wrong/stderr.log').read_text())

    def test_timeout_keeps_process_outcome(self):
        with self.assertRaisesRegex(RuntimeError, 'timeout'):
            self.run_engine('torch_eager', 'timeout', timeout=.001)
        result = json.loads((self.root / 'timeout/process.json').read_text())
        self.assertTrue(result['timed_out'])
        self.assertNotEqual(result['returncode'], 0)

    @unittest.skipUnless(os.environ.get('STEALTH_RUN_FRONTIER') == '1', 'explicit modern Linux environment')
    def test_actual_linux_host_probe(self):
        from frontier_host import observe
        original = os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0, {min(original)})
            host = observe(self.native)
            self.assertEqual(host['cpu_affinity'], [min(original)])
            self.assertEqual(host['packages']['torch'], '2.15.0+cpu')
            self.assertIn('flags', host['cpu_model_and_isa'])
        finally:
            os.sched_setaffinity(0, original)

    @unittest.skipUnless(os.environ.get('STEALTH_RUN_FRONTIER') == '1', 'explicit modern Linux environment')
    def test_real_compiled_and_onnx_timing_and_graph_binding(self):
        compiled = self.run_engine('torch_compiled_inductor', 'compiled', timeout=180)
        self.assertEqual(compiled['checksum'], 33.)
        self.assertGreater(compiled['adapter']['generated_kernel_count'], 0)
        exported = self.run_engine('onnxruntime_cpu', 'onnx', timeout=180)
        graphs = exported['adapter']['graph_sha256']
        repeated = self.run_engine('onnxruntime_cpu', 'onnx-repeat', timeout=180, graphs=graphs)
        self.assertEqual(repeated['checksum'], 33.)
        with self.assertRaisesRegex(ValueError, 'graph differs'):
            self.run_engine('onnxruntime_cpu', 'onnx-wrong-lock', timeout=180,
                            graphs={'exported': '0' * 64, 'optimized': '0' * 64})
