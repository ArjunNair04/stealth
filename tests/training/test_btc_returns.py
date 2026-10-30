"""Hand-computed labels, temporal falsifiers and tiny real train/export workflow."""
import copy
from dataclasses import replace
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from hourly_bars import Bar, HOUR_NS
import btc_returns as b
import train_btc_returns as run


def bars(n=60):
    closes=[100 + i + (i%3)/4 for i in range(n)]
    return tuple(Bar('BTCUSDT',i*HOUR_NS,(i+1)*HOUR_NS,1000*HOUR_NS,
                     c,c+1,c-1,c,2.,200.,3,1.,100.,'a'*64,'synthetic')
                 for i,c in enumerate(closes))


def setUpModule():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)


class ReturnData(unittest.TestCase):
    def test_hand_return_range_and_clock(self):
        raw=bars(4); raw=tuple(replace(x,close=c,open=c,high=c+1,low=c-1)
                               for x,c in zip(raw,[100,110,99,99]))
        f=b.features(raw)
        self.assertEqual(f.end_ns.tolist(),[2*HOUR_NS,3*HOUR_NS,4*HOUR_NS])
        np.testing.assert_allclose(f.values[:,0],[1000,-1000,0],rtol=0,atol=1e-10)
        self.assertAlmostEqual(float(f.values[0,1]),10000*math.log(111/109),places=11)
        self.assertAlmostEqual(float(f.values[0,2]),math.log1p(200),places=14)
        self.assertAlmostEqual(float(f.values[0,3]),.5,places=14)
        self.assertFalse(f.values.flags.writeable)
        self.assertFalse(f.end_ns.flags.writeable)

    def test_future_prices_do_not_change_past_features(self):
        raw=bars(); changed=raw[:30]+tuple(replace(x,close=x.close*3,open=x.open*3,
                                   high=x.high*3,low=x.low*3,quote_volume=999.) for x in raw[30:])
        np.testing.assert_array_equal(b.features(raw).values[:29],b.features(changed).values[:29])
        f=b.features(raw); g=b.features(changed)
        a=b.split(f,4,0,30*HOUR_NS); z=b.split(g,4,0,30*HOUR_NS)
        np.testing.assert_array_equal(a.x,z.x);np.testing.assert_array_equal(a.y,z.y)

    def test_exact_split_boundaries_and_context(self):
        f=b.features(bars(20))
        train=b.split(f,3,0,10*HOUR_NS);val=b.split(f,3,10*HOUR_NS,15*HOUR_NS)
        self.assertEqual(train.input_end_ns.tolist(),list(range(4*HOUR_NS,10*HOUR_NS,HOUR_NS)))
        self.assertEqual(val.input_end_ns.tolist(),list(range(10*HOUR_NS,15*HOUR_NS,HOUR_NS)))
        self.assertEqual(val.label_end_ns[0],11*HOUR_NS)
        np.testing.assert_array_equal(val.x[0],f.values[6:9].astype(np.float32))
        self.assertEqual(float(val.y[0]),float(np.float32(f.values[9,0])))
        self.assertEqual(train.label_end_ns[-1],10*HOUR_NS)

    def test_missing_reverse_mixed_symbol_and_invalid_prices_fail(self):
        base=bars()
        for raw in (base[:5]+base[6:],base[::-1],base[:2]+(replace(base[2],symbol='ETHUSDT'),)+base[3:],
                    (replace(base[0],close=0),)+base[1:],(replace(base[0],high=float('inf')),)+base[1:]):
            with self.assertRaises(ValueError):b.features(raw)

    def test_empty_invalid_split_and_context_fail(self):
        f=b.features(bars(20))
        for args in ((0,0,10*HOUR_NS),(3,10*HOUR_NS,0),(3,0,2*HOUR_NS),(True,0,10*HOUR_NS)):
            with self.assertRaises(ValueError):b.split(f,*args)

    def test_scaler_excludes_future_and_constant(self):
        f=b.features(bars()); a=b.split(f,4,0,30*HOUR_NS)
        mean,scale=b.fit_scaler(f,a)
        changed=f.values.copy();changed[29:]=1e12
        altered=b.FeatureFrame(f.end_ns.copy(),changed)
        m,s=b.fit_scaler(altered,a)
        np.testing.assert_array_equal(mean,m);np.testing.assert_array_equal(scale,s)
        self.assertEqual(float(scale[2]),1.)
        self.assertEqual(len(mean),5)

    def test_ridge_independent_small_solution(self):
        x=np.array([[[-1.]],[[1.]]],dtype=np.float32);y=np.array([-2.,2.],dtype=np.float32)
        model=b.fit_ridge(x,y,1.)
        # Standardized x has variance1; covariance(x,y)=2; coefficient=2/(1+1)=1.
        np.testing.assert_allclose(b.ridge_predict(model,x),[-1,1],atol=1e-14)
        for alpha in (0,-1,float('nan')):
            with self.assertRaises(ValueError):b.fit_ridge(x,y,alpha)

    def test_constant_predictor_and_loss_interval(self):
        s=b.scores(np.zeros(3),np.array([-1.,0.,1.]))
        self.assertAlmostEqual(s['mse'],2/3)
        self.assertEqual(s['correlation'],None)
        settings=dict(block_hours=2,replicates=50,seed=3,confidence=.95)
        a=b.loss_interval(np.ones(8),np.zeros(8),np.ones(8),settings)
        self.assertEqual(a['mean_difference'],-1)
        self.assertEqual(a['lower'],-1);self.assertEqual(a['upper'],-1)
        self.assertEqual(a,b.loss_interval(np.ones(8),np.zeros(8),np.ones(8),settings))


class Study(unittest.TestCase):
    def tiny(self):
        p=copy.deepcopy(run.read_recipe());p['epochs']=1;p['seeds']=[7,19]
        p['model'].update(width=4,heads=2,hidden=6,context=4)
        p['split_ends']={'train':'1970-01-02T06:00:00Z','validation':'1970-01-02T21:00:00Z','test':'1970-01-03T12:00:00Z'}
        p['batch_size']=4;p['evaluation_batch_size']=4;p['full_trace_parity_windows']=2
        p['uncertainty'].update(block_hours=2,replicates=50)
        return p

    def test_frozen_task(self):
        p=run.read_recipe()
        self.assertEqual(p['experiment_id'],'btc_hourly_return_v1')
        self.assertEqual(p['model']['horizon'],1)
        self.assertEqual((p['seeds'],p['epochs']),([7,19,43],3))
        self.assertEqual(p['model']['output'],'level')

    def test_test_suffix_cannot_change_selected_models_or_baselines(self):
        p=self.tiny(); raw=bars(); changed=raw[:45]+tuple(
            replace(x,open=x.open*1.1,high=x.high*1.1,low=x.low*1.1,close=x.close*1.1)
            for x in raw[45:])
        binding={'data_sha256':'a'*64,'protocol_sha256':'b'*64,'code_commit':'c'*40}
        with tempfile.TemporaryDirectory() as tmp:
            left=Path(tmp)/'left';right=Path(tmp)/'right'
            run.run_study(left,b.features(raw),p,binding)
            run.run_study(right,b.features(changed),p,binding)
            for name in ['baselines.json','selection.json','scaler.json']+[
                    f'seed-{seed}/model.stlf' for seed in p['seeds']]:
                self.assertEqual((left/name).read_bytes(),(right/name).read_bytes(),name)
            self.assertNotEqual((left/'predictions.jsonl').read_bytes(),(right/'predictions.jsonl').read_bytes())

    def test_training_failure_preserves_failure_without_evaluation_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            with patch.object(run,'train_epoch',side_effect=ValueError('injected training failure')):
                with self.assertRaises(ValueError):run.run_study(root,b.features(bars()),self.tiny(),{})
            self.assertTrue((root/'failure.json').exists())
            self.assertTrue((root/'seed-7/epoch-000/manifest.json').exists())
            self.assertFalse((root/'selection.json').exists())
            self.assertFalse((root/'predictions.jsonl').exists())
            self.assertFalse((root/'complete.json').exists())

    def test_tiny_complete_training_native_replay_and_tamper(self):
        p=self.tiny();f=b.features(bars());binding={'data_sha256':'a'*64,'protocol_sha256':'b'*64,'code_commit':'c'*40}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run'
            result=run.run_study(path,f,p,binding)
            self.assertEqual(set(result['models']),{'7','19'})
            self.assertEqual(result['baselines']['zero']['count'],15)
            self.assertEqual(len(result['models']['7']['validation_history']),2)
            self.assertTrue(run.verify_run(path)['status']=='complete')
            self.assertEqual(run.replay(path,f), (path/'predictions.jsonl').read_bytes())
            with self.assertRaises(FileExistsError):run.run_study(path,f,p,binding)
            with (path/'seed-7'/'model.stlf').open('ab') as out:out.write(b'x')
            with self.assertRaises(ValueError):run.verify_run(path)


if __name__=='__main__':unittest.main()
