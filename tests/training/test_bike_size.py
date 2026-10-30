"""Independent size arithmetic, complete grid and sealed sweep integration."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import bike_size as sizing
import train_bike_demand as bike
import test_bike_demand as fixtures


def setUpModule():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class Sizing(unittest.TestCase):
    def test_only_capacity_changes(self):
        protocol, base = sizing.read_recipe()
        for width in (64, 128, 256):
            recipe = sizing.child_recipe(base, width)
            self.assertEqual(recipe['model']['width'], width)
            self.assertEqual(recipe['model']['hidden'], 4 * width)
            recipe['model'].update(width=64, hidden=256)
            recipe['experiment_id'] = base['experiment_id']
            self.assertEqual(recipe, base)
        self.assertEqual(protocol['widths'], [64, 128, 256])
        self.assertEqual(sizing.parameter_counts(sizing.child_recipe(base, 64)),
                         {'stored': 50571, 'trainable': 50561})
        self.assertEqual(sizing.parameter_counts(sizing.child_recipe(base, 256)),
                         {'stored': 792075, 'trainable': 792065})

    def test_changed_protocol_rejected(self):
        protocol, base = sizing.read_recipe()
        for key, value in [('widths', [64, 128]), ('hidden_multiplier', 2),
                           ('evidence', 'untouched'), ('base_protocol_sha256', '0' * 64)]:
            changed = copy.deepcopy(protocol); changed[key] = value
            with self.assertRaises(ValueError): sizing.validate_protocol(changed)
        changed = copy.deepcopy(protocol); changed['latency']['samples'] = 7
        with self.assertRaises(ValueError): sizing.validate_protocol(changed)

    def test_balanced_complete_worker_order(self):
        protocol, base = sizing.read_recipe()
        cells = sizing.worker_order(protocol, base['seeds'])
        self.assertEqual(len(cells), 27)
        self.assertEqual(len(set(cells)), 27)
        for seed in base['seeds']:
            for position in range(3):
                self.assertEqual({[w for t, s, w in cells if t == trial and s == seed][position]
                                  for trial in (1, 2, 3)}, {64, 128, 256})

    def test_aggregate_rejects_missing_duplicate_and_invalid(self):
        protocol, base = sizing.read_recipe()
        records = [dict(trial=t, seed=s, width=w,
                        latency={'p50_ns': w * 1000, 'p99_ns': w * 2000}, checksum=w+s,
                        parameter_bytes=w, position_bytes=2*w, workspace_bytes=3*w)
                   for t, s, w in sizing.worker_order(protocol, base['seeds'])]
        result = sizing.aggregate_latency(records, protocol, base['seeds'])
        self.assertEqual(result['64']['p50_us'], 64)
        self.assertEqual(result['64']['owned_bytes'], 384)
        for changed in (records[:-1], records + records[:1]):
            with self.assertRaises(ValueError): sizing.aggregate_latency(changed, protocol, base['seeds'])
        changed = copy.deepcopy(records); changed[0]['latency']['p99_ns'] = float('nan')
        with self.assertRaises(ValueError): sizing.aggregate_latency(changed, protocol, base['seeds'])
        changed = copy.deepcopy(records); changed[0]['checksum'] += 1
        with self.assertRaises(ValueError): sizing.aggregate_latency(changed, protocol, base['seeds'])
        changed = copy.deepcopy(records); changed[0]['workspace_bytes'] += 1
        with self.assertRaises(ValueError): sizing.aggregate_latency(changed, protocol, base['seeds'])

    def test_tiny_sweep_replay_tamper_and_no_overwrite(self):
        protocol, _ = sizing.read_recipe(); protocol['widths'] = [4, 8]
        base = fixtures.Study().tiny(); base['model']['heads'] = 2; base['seeds'] = [7]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            sizing.train(root, fixtures.frame(), protocol, base, {'fixture': True})
            sizing.verify(root, protocol, base)
            replay = Path(tmp) / 'replay'
            sizing.replay(root, replay, fixtures.frame(), protocol, base)
            self.assertTrue((replay/'complete.json').exists())
            for width in (4, 8):
                self.assertEqual((root/f'width-{width}/predictions.jsonl').read_bytes(),
                                 (replay/f'width-{width}.jsonl').read_bytes())
            with self.assertRaises(FileExistsError): sizing.train(root, fixtures.frame(), protocol, base, {})
            binary = sizing.ROOT/'build/release/stealth_runtime_bench'
            if sys.platform == 'win32': binary = binary.with_suffix('.exe')
            self.assertTrue(binary.exists(), 'Build the Release benchmark before this integration test')
            protocol['latency'].update(samples=3, warmup=1)
            # Use another small training run bound to these diagnostic settings.
            timed = Path(tmp)/'timed'
            sizing.train(timed, fixtures.frame(), protocol, base, {'fixture': True})
            with patch.object(sizing, 'clean_commit', return_value='fixture'):
                sizing.benchmark(timed, Path(tmp)/'timings', fixtures.frame(), protocol, base, binary)
            self.assertTrue((Path(tmp)/'timings/complete.json').exists())
            protocol['latency'].update(samples=1000, warmup=64)
            with (root/'width-8/seed-7/model.stlf').open('ab') as out: out.write(b'x')
            with self.assertRaises(ValueError): sizing.verify(root, protocol, base)

    def test_failure_never_seals_partial_sweep(self):
        protocol, base = sizing.read_recipe()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            with patch.object(bike, 'run_study', side_effect=ValueError('injected')):
                with self.assertRaises(ValueError): sizing.train(root, fixtures.frame(), protocol, base, {})
            self.assertTrue((root/'failure.json').exists())
            self.assertFalse((root/'complete.json').exists())


if __name__ == '__main__': unittest.main()
