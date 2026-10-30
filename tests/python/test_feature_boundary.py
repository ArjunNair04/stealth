"""Feature range admission must agree under NumPy 1 and NumPy 2."""
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from generate_forecast_fixture import fixture
from model_bundle import create_bundle, load_bundle
from feature_inference import FeatureReplay
from signal_timeline import SignalSpec, Observation, SignalTimeline


class FeatureBoundary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.specs = tuple(SignalSpec(str(i), 'fixture', 'sensor', str(i), 100, 100)
                           for i in range(7))
        path = Path(self.tmp.name)/'bundle'
        ident = create_bundle(path, fixture()[-1], self.specs, interval_ns=10,
                              grid_origin_ns=1, clock_domain='synthetic')
        self.bundle = load_bundle(path, expected_id=ident)
        # Independent exact IEEE-754 binary32 finite endpoint, as a Python float.
        self.limit = float.fromhex('0x1.fffffep+127')

    def timeline(self, value):
        records = [Observation(*spec.key, 'first', 0, 0, 0, 0, 1.0) for spec in self.specs]
        records.append(Observation(*self.specs[0].key, 'second', 0, 10, 10, 10, value))
        return SignalTimeline(self.specs, records)

    def test_just_outside_range_rejects_before_prediction_and_preserves_history(self):
        for sign in (-1, 1):
            replay = FeatureReplay(self.bundle)
            source = self.timeline(sign*math.nextafter(self.limit, math.inf))
            with patch.object(replay, '_predict', return_value=0.0) as predict:
                replay.advance(source, 1)
                before = replay.history_rows, replay.last_decision_ns
                predict.reset_mock()
                with self.assertRaisesRegex(ValueError, 'FP32 range'):
                    replay.advance(source, 11)
                predict.assert_not_called()
                self.assertEqual((replay.history_rows, replay.last_decision_ns), before)

    def test_finite_endpoints_and_adjacent_inside_values_reach_prediction(self):
        for value in (self.limit, -self.limit, math.nextafter(self.limit, 0.0),
                      -math.nextafter(self.limit, 0.0), 0.0, -0.0):
            replay = FeatureReplay(self.bundle)
            with patch.object(replay, '_predict', return_value=0.0) as predict:
                replay.advance(self.timeline(value), 11)
                predict.assert_called_once()
                self.assertEqual(replay.history_rows, 1)
