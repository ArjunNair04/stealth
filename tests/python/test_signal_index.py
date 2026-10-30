"""Installed binding and native index versus the unchanged Python scan oracle."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import random
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from signal_timeline import SignalSpec, Observation, SignalTimeline, MAX_TIME
from native_signal_timeline import NativeSignalTimeline
from stealth_cpu._core import _SignalIndex


def fixtures():
    specs = tuple(SignalSpec(str(i), 'test', 'sensor', str(i), 50+i*30, 40+i*20)
                  for i in range(3))
    rng = random.Random(721)
    records = []
    for spec in specs:
        for event in range(30):
            for revision in range(3):
                pub = event*10+revision*2
                record = Observation(*spec.key, str(event), revision, event*10, pub,
                                     pub+rng.randrange(50), None if rng.randrange(9) == 0
                                     else rng.uniform(-10, 10))
                records.append(record)
                if rng.randrange(5) == 0:
                    records.append(replace(record, received_at_ns=record.received_at_ns+5))
    rng.shuffle(records)
    return specs, records


class SignalIndexTests(unittest.TestCase):
    def test_complete_snapshots_match_independent_scan(self):
        specs, records = fixtures()
        oracle = SignalTimeline(specs, records)
        native = NativeSignalTimeline(specs, records)
        self.assertEqual(native.record_count, oracle.record_count)
        self.assertLessEqual(native.index_entry_count, native.record_count)
        self.assertEqual(native.schema_id, oracle.schema_id)
        for t in list(range(400, -1, -1))+[MAX_TIME]:
            self.assertEqual(native.snapshot(t), oracle.snapshot(t))
        with self.assertRaises(FrozenInstanceError):
            native._index = None
        for bad in (-1, MAX_TIME+1, 1.5, True):
            with self.assertRaises(ValueError):
                native.snapshot(bad)

    def test_empty_capacity_duplicates_and_input_ownership(self):
        specs, _ = fixtures()
        records = [Observation(*specs[0].key, 'x', 0, 0, 0, 0, -0.)]
        native = NativeSignalTimeline(specs, records)
        records.clear()
        self.assertEqual(native.snapshot(1).features[0].value.hex(), '-0x0.0p+0')
        self.assertEqual(native.snapshot(1).features[1].state, 'missing')
        with self.assertRaises(ValueError):
            NativeSignalTimeline(specs, native._records*2, max_records=1)
        self.assertEqual(NativeSignalTimeline(specs, ()).record_count, 0)

    def test_future_permutation_and_reconstruction(self):
        specs, records = fixtures()
        early = [r for r in records if r.received_at_ns < 100]
        a, b = NativeSignalTimeline(specs, early), NativeSignalTimeline(specs, records[::-1])
        for t in range(101):
            self.assertEqual(a.snapshot(t), b.snapshot(t))
        self.assertEqual(b.snapshot(200), NativeSignalTimeline(specs, records).snapshot(200))

    def test_shared_index_concurrent_queries(self):
        specs, records = fixtures()
        source = NativeSignalTimeline(specs, records)
        clocks = list(range(400))*4
        expected = [source.snapshot(t) for t in clocks]
        with ThreadPoolExecutor(max_workers=4) as executor:
            self.assertEqual(list(executor.map(source.snapshot, clocks)), expected)

    def test_native_binding_rejects_malformed_and_conflicting_tapes(self):
        valid = (0, 1, 0, 0, 0, 1, 2.)
        for limits, records in (
                ([], []), ([(0, -1)], []), ([(1, 1)]*257, []),
                ([(1, 1)], [valid]*100001), ([(1, 1)], [(0,)]),
                ([(1, 1)], [valid, (0, 1, 0, 0, 0, 1, 3.)]),
                ([(1, 1)], [(0, 1, 0, 0, 0, 1, float('nan'))]),
                ([(1, 1)], [(0, 1, 0, 0, 2, 1, 3.)]),
                ([(1, 1)], [(True, 1, 0, 0, 0, 1, 2.)]),
                ([(1, 1)], [(0, 1, 0, 0, 0, 1, True)]),
                ([(1, 1)], [(0, 2**64, 0, 0, 0, 1, 2.)])):
            with self.subTest(limits=limits[:1], record_count=len(records)):
                with self.assertRaises((ValueError, TypeError, OverflowError)):
                    _SignalIndex(limits, records)
        index = _SignalIndex([(10, 10)], [valid])
        for bad in (True, 1.5, -1, MAX_TIME+1):
            with self.assertRaises((ValueError, TypeError, OverflowError)):
                index.snapshot(bad)
        self.assertEqual(index.snapshot(1)[0][0], 'missing')
        self.assertEqual(index.snapshot(2)[0][0], 'valid')

    def test_native_adapter_does_not_scan_at_query_or_import_torch(self):
        from unittest.mock import patch
        specs, records = fixtures()
        native = NativeSignalTimeline(specs, records)
        expected = native.snapshot(200)
        with patch.object(SignalTimeline, 'snapshot', side_effect=RuntimeError('scan forbidden')):
            self.assertEqual(native.snapshot(200), expected)
        code = ("import sys; sys.path.insert(0, "+repr(str(Path(__file__).resolve().parents[2]/'tools'))+
                "); import native_signal_timeline; assert 'torch' not in sys.modules")
        subprocess.run([sys.executable, '-c', code], check=True)


if __name__ == '__main__':
    unittest.main()
