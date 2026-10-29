"""Independent correction answers, legacy compatibility, gradients and full replay."""
from dataclasses import replace
from pathlib import Path
import copy
import hashlib
import json
import struct
import sys
import tempfile
import unittest
import numpy as np
import torch
import stealth_cpu as native
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from torch_forecaster import Config,Forecaster
from training_state import make_optimizer,save_checkpoint,load_checkpoint,canonical
from train_ettm1 import read_recipe,run_seed,evaluate_runs
import generate_forecast_fixture as fixture


def setUpModule():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)


class Correction(unittest.TestCase):
    def config(self): return Config(2,4,2,5,3,1,1,output='correction')

    def test_initial_identity_and_same_hidden_initialization(self):
        c=self.config(); model=Forecaster(c,[0,1000],[1,3],seed=7)
        original=Forecaster(replace(c,output='level'),[0,1000],[1,3],seed=7)
        for name,p in model.named_parameters():
            if name.startswith('head_'): self.assertEqual(int(torch.count_nonzero(p)),0)
            else: self.assertTrue(torch.equal(p,getattr(original,name)),name)
        x=torch.tensor([[[.1,.125],[.2,-2.5],[.3,4.25]],[[.9,1.],[.2,8.],[.1,3.]]])
        with torch.no_grad():
            np.testing.assert_array_equal(model(x).numpy(),x[:,:,1].numpy())
            expected=((x[:,:,1].double()-1000)/3).float()
            np.testing.assert_array_equal(model(x,standardized=True).numpy(),expected.numpy())
        for k in ('reference','packed'):
            n=native.ForecastModel.from_bytes(model.export_bytes(),kernel=k)
            self.assertEqual(n.config.output,'correction')
            np.testing.assert_array_equal(n.session().predict_batch(x.numpy()),x[:,:,1].numpy())
        config=native.ForecastConfig(2,4,2,5,3,1,1,output='correction')
        self.assertEqual(config.output,'correction')
        with self.assertRaises(ValueError): native.ForecastConfig(2,4,2,5,3,1,1,output='unknown')
        with self.assertRaises(ValueError): replace(c,output='unknown')

    def test_independent_fixture_correction_and_legacy_export(self):
        p,x,y,_,blob=fixture.fixture()
        c=Config(7,4,2,7,5,4,6); raw=np.array(x,np.float32).reshape(5,7)
        legacy=Forecaster.from_flat(c,np.array(p,np.float32))
        self.assertEqual(legacy.export_bytes(),blob)
        model=Forecaster.from_flat(replace(c,output='correction'),np.array(p,np.float32))
        expected=np.array(y)-p[6]+raw[:,6]
        with torch.no_grad(): actual=model(torch.from_numpy(raw)[None]).numpy()[0]
        np.testing.assert_allclose(actual,expected,atol=5e-5,rtol=2e-4)
        for k in ('reference','packed'):
            n=native.ForecastModel.from_bytes(model.export_bytes(),kernel=k)
            np.testing.assert_allclose(n.session().predict(raw),expected,atol=5e-5,rtol=2e-4)

    def test_version_mode_mismatches_fail(self):
        model=Forecaster(self.config(),[0,0],[1,1]);blob=model.export_bytes()
        self.assertEqual(blob[:8],b'STLFC02\0')
        self.assertEqual(struct.unpack_from('<I',blob,8)[0],2)
        self.assertEqual(struct.unpack_from('<I',blob,56)[0],1)
        for offset,value in ((8,1),(56,0),(56,2),(60,1)):
            changed=bytearray(blob[:-32]);struct.pack_into('<I',changed,offset,value)
            with self.assertRaises(ValueError): native.ForecastModel.from_bytes(bytes(changed)+hashlib.sha256(changed).digest())
        changed=bytearray(blob[:-32]);changed[6]=ord('1')
        with self.assertRaises(ValueError): native.ForecastModel.from_bytes(bytes(changed)+hashlib.sha256(changed).digest())

    def test_gradient_identity_head_wakes_body(self):
        model=Forecaster(self.config(),[0,0],[1,2],seed=7)
        x=torch.tensor([[[.1,.2],[.3,.4],[.5,.6]]],requires_grad=True)
        model(x)[0,-1].backward()
        expected=torch.zeros_like(x);expected[0,-1,1]=1
        self.assertTrue(torch.equal(x.grad,expected))
        self.assertEqual(float(model.head_bias.grad),2.)
        self.assertGreater(float(model.head_weight.grad.abs().sum()),0.)
        for name,p in model.named_parameters():
            if not name.startswith('head_'): self.assertEqual(int(torch.count_nonzero(p.grad)),0,name)
        with torch.no_grad(): model.head_weight.add_(model.head_weight.grad*.01)
        model.zero_grad(set_to_none=True);x.grad=None
        model(x)[0,-1].backward()
        self.assertGreater(float(model.input_weight.grad.abs().sum()),0.)
        # Independent finite difference with a nonzero head tests the added input route.
        analytic=float(x.grad[0,-1,1]);delta=.001
        with torch.no_grad():
            plus=x.detach().clone();minus=plus.clone();plus[0,-1,1]+=delta;minus[0,-1,1]-=delta
            numerical=float((model(plus)[0,-1]-model(minus)[0,-1])/(2*delta))
        self.assertAlmostEqual(analytic,numerical,delta=.003+abs(analytic)*.025)

    def test_full_shape_stream_prefix_and_future_invariance(self):
        c=Config(7,64,4,256,96,4,6,output='correction');m=Forecaster(c,[0]*7,[2]*7,seed=7)
        with torch.no_grad(): m.head_weight.copy_(torch.linspace(-.02,.03,64).reshape(1,64));m.head_bias.fill_(.1)
        x=torch.randn(1,100,7,generator=torch.Generator().manual_seed(8))
        n=native.ForecastModel.from_bytes(m.export_bytes());stream=n.window_stream();session=n.session()
        with torch.no_grad():
            expected=m(x[:,:96]).numpy()[0]
            for k in ('reference','packed'):
                actual=native.ForecastModel.from_bytes(m.export_bytes(),kernel=k).session().predict(x[0,:96].numpy())
                np.testing.assert_allclose(actual,expected,atol=5e-5,rtol=2e-4)
            changed=x[:,:96].clone();changed[:,48:]+=99
            np.testing.assert_array_equal(m(changed)[:,:48].numpy(),m(x[:,:96])[:,:48].numpy())
            np.testing.assert_allclose(m(x[:,:48]).numpy(),expected[None,:48],atol=5e-5,rtol=2e-4)
        for i,row in enumerate(x[0].numpy()):
            value=stream.push(row)
            if i<95: self.assertIsNone(value)
            else: self.assertAlmostEqual(value,float(session.predict(x[0,i-95:i+1].numpy())[-1]),places=5)
        stream.reset();self.assertIsNone(stream.push(x[0,0].numpy()))

    def test_old_checkpoint_without_output_field_stays_level(self):
        model=Forecaster(replace(self.config(),output='level'),[0,0],[1,1],seed=7)
        p,_,_=read_recipe();optimizer=make_optimizer(model,p['optimizer'])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'old'
            save_checkpoint(path,model,optimizer,torch.Generator().manual_seed(9),{}, {'old':'checkpoint'})
            meta=json.loads((path/'manifest.json').read_text());meta['config'].pop('output')
            (path/'manifest.json').write_bytes(canonical(meta))
            restored,_,_,_=load_checkpoint(path,{'old':'checkpoint'})
            self.assertEqual(restored.config.output,'level')
            self.assertEqual(restored.export_bytes(),model.export_bytes())

    def test_training_resume_and_sealed_evaluation(self):
        protocol,task,_=read_recipe();protocol=copy.deepcopy(protocol)
        protocol.update(epochs=2,seeds=[7,19],batch_size=4,accumulate=2,evaluation_batch_size=8,native_parity_windows_per_split=2)
        c=Config(2,4,2,5,8,2,1,output='correction')
        task={**task,'input_columns':['a','b'],'context':8,'horizon_steps':2,'target_column':'b'}
        rows=torch.arange(100,dtype=torch.float32).reshape(50,2)/100;ends=dict(train=26,validation=38,test=50)
        binding={'experiment':'correction-fixture'}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            a=run_seed(root/'seed7',rows,ends,c,[0,0],[1,1],protocol,task,binding,7)
            b=run_seed(root/'replay',rows,ends,c,[0,0],[1,1],protocol,task,binding,7,root/'seed7/epoch-001')
            self.assertEqual(a['history'],b['history'])
            self.assertEqual((root/'seed7/model.stlf').read_bytes(),(root/'replay/model.stlf').read_bytes())
            for name in ('manifest.json','arrays.npz'):
                self.assertEqual((root/'seed7/epoch-002'/name).read_bytes(),(root/'replay/epoch-002'/name).read_bytes())
            model,_,_,_=load_checkpoint(root/'seed7/epoch-002',{**binding,'seed':7})
            self.assertEqual(model.config.output,'correction')
            with self.assertRaises(ValueError): evaluate_runs(root/'bad',[root/'seed7'],rows,ends,c,protocol,task,binding,period=8)
            run_seed(root/'seed19',rows,ends,c,[0,0],[1,1],protocol,task,binding,19)
            result=evaluate_runs(root/'evaluation',[root/'seed7',root/'seed19'],rows,ends,c,protocol,task,binding,period=8)
            self.assertEqual(set(result['models']),{'7','19'})
            self.assertTrue((root/'evaluation/complete.json').exists())

if __name__=='__main__':unittest.main()
