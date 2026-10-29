import hashlib
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from torch_forecaster import Config,Forecaster
from runtime_torch import StandardForecaster,load_export
import generate_forecast_fixture as fixture

class StandardRuntime(unittest.TestCase):
    def test_independent_equation_fixture_and_export_roundtrip(self):
        p,x,y,_,blob=fixture.fixture()
        for mode in ('level','correction'):
            c=Config(7,4,2,7,5,4,6,output=mode)
            ref=Forecaster.from_flat(c,np.array(p,np.float32))
            model=load_export(ref.export_bytes(),standard=True)
            self.assertEqual(model.export_bytes(),ref.export_bytes())
            raw=np.array(x,np.float32).reshape(1,5,7)
            expected=np.array(y) if mode=='level' else np.array(y)-p[6]+raw[0,:,6]
            with torch.inference_mode(): result=model(torch.from_numpy(raw)).numpy()[0]
            np.testing.assert_allclose(result,expected,atol=5e-5,rtol=2e-4)
    def test_full_shape_causality_batching_and_reference(self):
        torch.set_num_threads(1)
        c=Config(7,64,4,256,96,4,6,output='correction')
        r=Forecaster(c,[0]*7,[2]*7,seed=7)
        with torch.no_grad(): r.head_weight.fill_(.02);r.head_bias.fill_(.1)
        m=load_export(r.export_bytes(),standard=True)
        x=torch.randn(8,96,7,generator=torch.Generator().manual_seed(19))
        with torch.inference_mode():
            y=m(x);np.testing.assert_allclose(y.numpy(),r(x).numpy(),atol=5e-5,rtol=2e-4)
            np.testing.assert_allclose(y[:1].numpy(),m(x[:1]).numpy(),atol=5e-5,rtol=2e-4)
            z=x.clone();z[:,48:]+=100
            np.testing.assert_array_equal(y[:,:48].numpy(),m(z)[:,:48].numpy())
    def test_loader_rejects_hash_version_shape_trailing_and_nonfinite(self):
        m=Forecaster(Config(2,4,2,5,3,1,1),[0,0],[1,1]);blob=m.export_bytes()
        for bad in (b'',blob[:-1],blob+b'x',b'bad'+blob[3:]):
            with self.assertRaises(ValueError): load_export(bad)
        import struct
        for offset,value in ((8,9),(12,0),(20,999),(56,1),(60,1)):
            body=bytearray(blob[:-32]);struct.pack_into('<I',body,offset,value)
            with self.assertRaises(ValueError): load_export(bytes(body)+hashlib.sha256(body).digest())
    def test_input_contract(self):
        m=StandardForecaster(Config(2,4,2,5,3,1,1),[0,0],[1,1])
        for x in (torch.zeros(1,4,2),torch.zeros(1,3,3),torch.zeros(0,3,2),torch.zeros(1,3,2,dtype=torch.float64),torch.full((1,3,2),float('nan'))):
            with self.assertRaises(ValueError): m(x)

    def test_isolated_workers_gate_timings_and_inference_mode(self):
        import json,os,struct,subprocess,tempfile
        from runtime_common import input_bytes
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);model=Forecaster(Config(2,4,2,5,3,1,1,output='correction'),[0,0],[1,1])
            (tmp/'model').write_bytes(model.export_bytes())
            (tmp/'input').write_bytes(input_bytes(list(range(12)),2,3,2))
            (tmp/'expected').write_bytes(struct.pack('<6f',1,3,5,7,9,11))
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
            for engine in ('native_python','torch_reference','torch_standard'):
                args=[sys.executable,str(root/'tools/runtime_worker.py'),'--engine',engine,'--model',str(tmp/'model'),'--input',str(tmp/'input'),'--expected',str(tmp/'expected'),'--batch','2','--samples','2','--warmup','2']
                proc=subprocess.run(args,capture_output=True,text=True,env=env);self.assertEqual(proc.returncode,0,proc.stderr)
                result=json.loads(proc.stdout);self.assertEqual(len(result['times_ns']),2);self.assertEqual(result['max_abs_difference'],0)
                if engine!='native_python': self.assertTrue(result['versions']['inference_mode'])
                (tmp/'expected').write_bytes(struct.pack('<6f',*[99]*6))
                bad=subprocess.run(args,capture_output=True,text=True,env=env);self.assertNotEqual(bad.returncode,0);self.assertEqual(bad.stdout,'')
                (tmp/'expected').write_bytes(struct.pack('<6f',1,3,5,7,9,11))
