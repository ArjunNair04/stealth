"""Bundle integrity and transactional feature-to-native integration contracts."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from torch_forecaster import Config, Forecaster
from signal_timeline import SignalSpec, Observation, SignalTimeline, MAX_TIME
from model_bundle import create_bundle, load_bundle, canonical, MAX_MANIFEST
from feature_inference import FeatureReplay


def setUpModule():
    torch.set_num_threads(1)


def fixture(output='correction'):
    model = Forecaster(Config(2, 4, 2, 6, 3, 2, 1, output=output),
                       [10, 20], [2, 4], seed=7)
    specs = tuple(SignalSpec(n, 'fixture', 'sensor', n, 1, 1) for n in ('load', 'temperature'))
    return model, specs


def timeline(specs, count=8, missing=(), withdrawal=()):
    records = [Observation(s.source, s.entity, s.field, str(i), 0, i*10, i*10, i*10,
                           None if (i,j) in withdrawal else float(10+i+j*20))
               for i in range(count) for j,s in enumerate(specs) if (i,j) not in missing]
    return SignalTimeline(specs, records)


class BundleFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'bundle'
        self.model, self.specs = fixture()
        self.ident = create_bundle(self.path, self.model.export_bytes(), self.specs,
                                   interval_ns=10, grid_origin_ns=1, clock_domain='synthetic')

    def load(self):
        return load_bundle(self.path, expected_id=self.ident)

    def rewrite(self, mutate):
        obj = json.loads((self.path/'manifest.json').read_bytes())
        mutate(obj)
        raw = canonical(obj)
        (self.path/'manifest.json').write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()


class BundleTests(BundleFixture):
    def test_round_trip_and_embedded_scaler(self):
        bundle = self.load()
        self.assertEqual(bundle.bundle_id, self.ident)
        self.assertEqual(bundle.specs, self.specs)
        self.assertEqual(bundle.schema_id, SignalTimeline(self.specs, ()).schema_id)
        manifest = json.loads(bundle.manifest_bytes)
        self.assertEqual(manifest['preprocessing']['mean'], [10.,20.])
        self.assertEqual(manifest['preprocessing']['scale'], [2.,4.])
        self.assertEqual(manifest['config']['target'], 1)
        self.assertEqual(manifest['target_feature'], 'temperature')
        with self.assertRaises(Exception): bundle.interval_ns = 7
        with self.assertRaises(FileExistsError):
            create_bundle(self.path, self.model.export_bytes(), self.specs,
                          interval_ns=10, grid_origin_ns=1, clock_domain='synthetic')

    def test_wrong_pin_and_changed_manifest(self):
        with self.assertRaises(ValueError): load_bundle(self.path, expected_id='0'*64)
        self.rewrite(lambda m: m['timing'].update(interval_ns=20))
        with self.assertRaises(ValueError): self.load()

    def test_rehashed_mismatched_metadata_rejected(self):
        original = (self.path/'manifest.json').read_bytes()
        mutations = [lambda m: m['preprocessing']['scale'].__setitem__(1, 8.),
                     lambda m: m['preprocessing']['mean'].__setitem__(0, 11.),
                     lambda m: m['config'].update(context=4),
                     lambda m: m.update(target_feature='load'),
                     lambda m: m['features'].reverse(),
                     lambda m: m.update(model_file='../model.stlf'),
                     lambda m: m.update(schema_version=True),
                     lambda m: m.update(extra='ignored?'),
                     lambda m: m['timing'].update(gap_policy='skip'),
                     lambda m: m['timing'].update(interval_ns=True),
                     lambda m: m['timing'].update(clock_domain='guess_local_timezone')]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                (self.path/'manifest.json').write_bytes(original)
                ident = self.rewrite(mutate)
                with self.assertRaises((ValueError, TypeError)):
                    load_bundle(self.path, expected_id=ident)

    def test_model_corruption_and_partial_bundle(self):
        p = self.path/'model.stlf'
        raw = p.read_bytes()
        p.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        with self.assertRaises(ValueError): self.load()
        # Even a matching outer digest cannot bypass the native validation.
        ident = self.rewrite(lambda m: m.update(model_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        with self.assertRaises(ValueError): load_bundle(self.path, expected_id=ident)
        p.unlink()
        with self.assertRaises(FileNotFoundError): load_bundle(self.path, expected_id=ident)

    def test_json_duplicate_nonfinite_and_size(self):
        for raw in (b'{"schema_version":1,"schema_version":1}', b'{"x":NaN}', b'{"x":1e999}',
                    b'[]', b'{' + b' '*MAX_MANIFEST + b'}'):
            with self.subTest(raw=raw[:40]):
                (self.path/'manifest.json').write_bytes(raw)
                with self.assertRaises(ValueError):
                    load_bundle(self.path, expected_id=hashlib.sha256(raw).hexdigest())

    def test_inference_modules_do_not_import_torch(self):
        code = "import sys; sys.path.insert(0, 'tools'); import model_bundle, feature_inference; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[2],check=True)

    def test_creation_validates_before_writing(self):
        for options in ({'interval_ns':0}, {'grid_origin_ns':-1}, {'interval_ns':MAX_TIME},
                        {'clock_domain':'local'}):
            path = Path(self.tmp.name)/'bad'
            args = dict(interval_ns=10, grid_origin_ns=1, clock_domain='synthetic')
            args.update(options)
            with self.assertRaises(ValueError):
                create_bundle(path, self.model.export_bytes(), self.specs, **args)
            self.assertFalse(path.exists())
        with self.assertRaises(ValueError):
            create_bundle(Path(self.tmp.name)/'bad', self.model.export_bytes(), self.specs[:1],
                          interval_ns=10, grid_origin_ns=1, clock_domain='synthetic')


class ReplayTests(BundleFixture):
    def test_independent_persistence_answer_and_target_clock(self):
        replay = FeatureReplay(self.load())
        source = timeline(self.specs)
        results = [replay.advance(source, 1+i*10) for i in range(8)]
        self.assertEqual([r.state for r in results], ['warming_up']*2+['forecast']*6)
        self.assertEqual([r.value for r in results[2:]], [32.,33.,34.,35.,36.,37.])
        self.assertEqual(results[2].target_at_ns, 41)
        self.assertEqual(results[-1].window_start_ns, 51)
        self.assertEqual(results[-1].history_rows, 3)

    def test_raw_values_scaled_once_and_torch_parity(self):
        model, specs = fixture('level')
        path = Path(self.tmp.name)/'level'
        ident = create_bundle(path, model.export_bytes(), specs, interval_ns=10,
                              grid_origin_ns=1, clock_domain='synthetic')
        bundle = load_bundle(path, expected_id=ident)
        replay, source = FeatureReplay(bundle), timeline(specs)
        session = bundle.model.session()
        for i in range(8):
            result = replay.advance(source, 1+i*10)
            if i >= 2:
                raw = np.array([[10+k,30+k] for k in range(i-2,i+1)], dtype=np.float32)
                self.assertEqual(result.value, session.predict_last(raw))
                with torch.inference_mode(): expected = float(model(torch.from_numpy(raw[None]))[0,-1])
                self.assertAlmostEqual(result.value, expected, delta=5e-5+2e-4*abs(expected))

    def test_gap_resets_and_then_recovers(self):
        replay, source = FeatureReplay(self.load()), timeline(self.specs)
        for t in (1,11,21): replay.advance(source,t)
        gap = replay.advance(source,41)
        self.assertEqual((gap.state,gap.reset_reason,gap.history_rows), ('warming_up','gap',1))
        self.assertEqual(replay.advance(source,51).state,'warming_up')
        self.assertEqual(replay.advance(source,61).value,36.)

    def test_missing_stale_withdrawn_reset_explicitly(self):
        for source, clock, state in ((timeline(self.specs, missing=((0,1),)),1,'missing'),
                                     (timeline(self.specs, missing=((3,1),)),31,'stale'),
                                     (timeline(self.specs, withdrawal=((3,1),)),31,'withdrawn')):
            replay = FeatureReplay(self.load())
            for t in range(1,clock,10): replay.advance(source,t)
            result = replay.advance(source,clock)
            self.assertEqual((result.state,result.value,result.history_rows),('unavailable',None,0))
            self.assertEqual(result.snapshot.features[1].state,state)
            self.assertEqual(result.reset_reason,'unavailable')
            self.assertEqual(replay.advance(source,clock+10).state,'warming_up')

    def test_invalid_clock_or_schema_preserves_history(self):
        replay, source = FeatureReplay(self.load()), timeline(self.specs)
        replay.advance(source,1); replay.advance(source,11)
        wrong = SignalTimeline(tuple(reversed(self.specs)), ())
        for src, t in ((source,11),(source,1),(source,22),(source,True),
                       (source,MAX_TIME),(wrong,21)):
            with self.assertRaises(ValueError): replay.advance(src,t)
            self.assertEqual((replay.history_rows,replay.last_decision_ns),(2,11))
        self.assertEqual(replay.advance(source,21).value,32.)

    def test_fp32_overflow_and_inference_failure_preserve_history(self):
        replay, source = FeatureReplay(self.load()), timeline(self.specs)
        replay.advance(source,1); replay.advance(source,11)
        records = [Observation(s.source,s.entity,s.field,'huge',0,20,20,20,1e100) for s in self.specs]
        with self.assertRaises(ValueError): replay.advance(SignalTimeline(self.specs,records),21)
        self.assertEqual((replay.history_rows,replay.last_decision_ns),(2,11))
        with patch.object(replay, '_predict', side_effect=RuntimeError('simulated native rejection')):
            with self.assertRaises(RuntimeError): replay.advance(source,21)
        self.assertEqual((replay.history_rows,replay.last_decision_ns),(2,11))
        self.assertEqual(replay.advance(source,21).value,32.)

    def test_native_arithmetic_rejection_does_not_commit_warmup(self):
        model, specs = fixture()
        with torch.no_grad(): model.input_weight.fill_(3e38)
        path = Path(self.tmp.name)/'overflow'
        ident = create_bundle(path,model.export_bytes(),specs,interval_ns=10,
                              grid_origin_ns=1,clock_domain='synthetic')
        replay = FeatureReplay(load_bundle(path,expected_id=ident))
        with self.assertRaises((ValueError,RuntimeError,OverflowError)):
            replay.advance(timeline(specs),1)
        self.assertEqual((replay.history_rows,replay.last_decision_ns),(0,None))

    def test_reset_restart_and_future_invariance(self):
        bundle = self.load()
        replay, fresh = FeatureReplay(bundle), FeatureReplay(bundle)
        source = timeline(self.specs)
        prefix = timeline(self.specs,count=5)
        first = [replay.advance(source,t) for t in range(1,42,10)]
        replay.reset(); replay.reset()
        again = [replay.advance(prefix,t) for t in range(1,42,10)]
        restarted = [fresh.advance(source,t) for t in range(1,42,10)]
        self.assertEqual(first,again)
        self.assertEqual(first,restarted)

    def test_slow_factor_carry_and_late_revision(self):
        specs = (self.specs[0],replace(self.specs[1],max_observation_age_ns=100,
                                     max_publication_age_ns=100))
        path = Path(self.tmp.name)/'slow'
        ident = create_bundle(path,self.model.export_bytes(),specs,interval_ns=10,
                              grid_origin_ns=1,clock_domain='synthetic')
        replay = FeatureReplay(load_bundle(path,expected_id=ident))
        records = [Observation('fixture','sensor','load',str(i),0,i*10,i*10,i*10,10.+i) for i in range(6)]
        records += [Observation('fixture','sensor','temperature','day',0,0,0,0,30.),
                    Observation('fixture','sensor','temperature','day',1,0,25,31,40.)]
        source = SignalTimeline(specs,records)
        answers = [replay.advance(source,t) for t in (1,11,21,31,41)]
        self.assertEqual([r.value for r in answers[2:]], [30.,30.,40.])
        self.assertEqual(answers[-1].snapshot.features[1].observation.revision,1)

    def test_busy_reset_or_advance_rejected(self):
        replay = FeatureReplay(self.load())
        with replay._lock:
            with self.assertRaises(RuntimeError): replay.reset()
            with self.assertRaises(RuntimeError): replay.advance(timeline(self.specs),1)
        self.assertEqual(replay.history_rows,0)


if __name__ == '__main__': unittest.main()
