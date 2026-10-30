"""End-to-end finite-tape checks; no latency or predictive-quality claim."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from temporal_conformance import run, tape, CapturingReplay
from signal_timeline import SignalTimeline
from native_signal_timeline import NativeSignalTimeline


def setUpModule():
    torch.set_num_threads(1)


class TemporalConformanceTests(unittest.TestCase):
    def test_adversarial_tape_timing_and_all_states(self):
        specs, records, clocks = tape()
        oracle, native = SignalTimeline(specs, records), NativeSignalTimeline(specs, records)
        self.assertNotIn(501, clocks)
        self.assertNotIn(511, clocks)
        states = set()
        for t in clocks:
            a, b = oracle.snapshot(t), native.snapshot(t)
            self.assertEqual(a, b)
            states.update(f.state for f in a.features)
        self.assertEqual(states, {'missing', 'stale', 'withdrawn', 'valid'})
        # Delayed higher revision arrives first; older event never displaces a newer one.
        self.assertEqual(native.snapshot(121).features[0].observation.revision, 2)
        self.assertEqual(native.snapshot(122).features[0].observation.revision, 2)
        self.assertEqual(native.snapshot(131).features[0].observation.effective_at_ns, 130)

    def test_native_oracle_and_eager_forecasts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'run'
            report = run(path)
            self.assertTrue(report['passed'])
            self.assertGreater(report['forecasts'], 20)
            self.assertEqual(report['decisions'], 98)
            self.assertFalse(report['timing_is_benchmark'])
            ledger = [json.loads(line) for line in (path/'native.jsonl').read_text().splitlines()]
            self.assertEqual((path/'native.jsonl').read_bytes(), (path/'oracle.jsonl').read_bytes())
            gap = next(r for r in ledger if r['decision_at_ns'] == 521)
            self.assertEqual((gap['reset_reason'], gap['history_rows']), ('gap', 1))
            self.assertEqual(ledger[0]['state'], 'unavailable')
            self.assertEqual(sum(r['state'] == 'forecast' for r in ledger), report['forecasts'])
            with self.assertRaises(FileExistsError):
                run(path)

    def test_fresh_process_reconstruction_has_identical_ledger(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/'first', Path(tmp)/'restart']
            for path in paths:
                subprocess.run([sys.executable, str(root/'tools/temporal_conformance.py'),
                                '--output', str(path)], check=True, capture_output=True, timeout=90)
            for filename in ('tape.json', 'model.stlf', 'native.jsonl', 'oracle.jsonl'):
                self.assertEqual((paths[0]/filename).read_bytes(), (paths[1]/filename).read_bytes())

    def test_snapshot_mismatch_is_preserved_as_failure(self):
        original = NativeSignalTimeline.snapshot
        def corrupt(source, clock):
            result = original(source, clock)
            return replace(result, decision_at_ns=clock+1)
        with tempfile.TemporaryDirectory() as tmp, patch.object(NativeSignalTimeline, 'snapshot', corrupt):
            path = Path(tmp)/'bad'
            with self.assertRaisesRegex(AssertionError, 'snapshot'):
                run(path)
            self.assertTrue((path/'failure.json').exists())
            self.assertFalse((path/'report.json').exists())

    def test_forecast_mismatch_is_preserved_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                'temporal_conformance.external_predictions', side_effect=lambda blob, windows, engine, output: [1e9]*len(windows)):
            path = Path(tmp)/'bad'
            with self.assertRaises(AssertionError):
                run(path)
            self.assertTrue((path/'failure.json').exists())
            self.assertTrue((path/'external.jsonl').exists())
            self.assertFalse((path/'report.json').exists())

    def test_native_schema_and_rejected_calls_preserve_history(self):
        from temporal_conformance import diagnostic_blob
        from model_bundle import create_bundle, load_bundle
        specs, records, clocks = tape()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bundle'
            ident = create_bundle(path, diagnostic_blob(), specs, interval_ns=10,
                                  grid_origin_ns=1, clock_domain='synthetic')
            replay = CapturingReplay(load_bundle(path, expected_id=ident))
            native = NativeSignalTimeline(specs, records)
            replay.advance(native, 21)
            before = replay.history_rows, replay.last_decision_ns
            wrong = NativeSignalTimeline(tuple(reversed(specs)), records)
            for source, clock in ((wrong,31), (native,21), (native,32)):
                with self.assertRaises(ValueError):
                    replay.advance(source, clock)
                self.assertEqual((replay.history_rows,replay.last_decision_ns), before)
            with patch.object(replay, '_predict', side_effect=ArithmeticError('injected')):
                with self.assertRaises(ArithmeticError):
                    replay.advance(native,31)
            self.assertEqual((replay.history_rows,replay.last_decision_ns), before)
            expected = replay.advance(native,31)
            replay.reset()
            replay.advance(native,21)
            self.assertEqual(replay.advance(native,31), expected)


@unittest.skipUnless(os.environ.get('STEALTH_RUN_FRONTIER') == '1',
                     'optional Linux compiler/export dependency job')
class TemporalExternalEngines(unittest.TestCase):
    def test_actual_compiled_and_onnx_on_identical_decision_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            for engine in ('torch_compiled_inductor', 'onnxruntime_cpu'):
                with self.subTest(engine=engine):
                    report = run(Path(tmp)/engine, engine=engine)
                    self.assertTrue(report['passed'])
                    self.assertGreater(report['forecasts'],20)
                    self.assertEqual(report['engine'],engine)


if __name__ == '__main__':
    unittest.main()
