"""Accumulation, stochastic state, safe checkpoints and tiny learning diagnostics."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import tempfile
import unittest
import numpy as np
import torch
import stealth_cpu as native
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from torch_forecaster import Config, Forecaster
from training_state import make_optimizer, train_epoch, save_checkpoint, load_checkpoint, metrics

SETTINGS=dict(lr=.001,weight_decay=.01,betas=[.9,.999],eps=1e-8)
BINDING={'protocol_sha256':'a'*64,'data_sha256':'b'*64,'code_commit':'c'*40}


def setup():
    model=Forecaster(Config(2,4,2,6,3,1,1),[0,0],[1,1],seed=7)
    x=torch.arange(42,dtype=torch.float32).reshape(7,3,2)/40
    y=torch.linspace(-.5,.5,7)
    return model,make_optimizer(model,SETTINGS),x,y,torch.Generator().manual_seed(19)


class Training(unittest.TestCase):
    def test_accumulation_and_short_final_group(self):
        m,_,x,y,g=setup(); other=copy.deepcopy(m)
        # Check sample weighting with SGD. AdamW magnifies roundoff in the
        # analytically zero key-bias gradient; batch partitions need not be bitwise equal.
        o=torch.optim.SGD(m.parameters(),lr=.01)
        other_o=torch.optim.SGD(other.parameters(),lr=.01)
        train_epoch(m,o,x,y,g,batch_size=2,accumulate=2,clip_norm=1.)
        train_epoch(other,other_o,x,y,torch.Generator().manual_seed(19),batch_size=4,accumulate=1,clip_norm=1.)
        np.testing.assert_allclose(m.flat_parameters(),other.flat_parameters(),atol=2e-6,rtol=2e-5)
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_checkpoint_exact_resume_and_next_shuffle(self):
        m,o,x,y,g=setup()
        train_epoch(m,o,x,y,g,batch_size=2,accumulate=2,clip_norm=1.)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'epoch1'
            record={'epoch':1,'history':[{'epoch':1,'validation_mae':1.2}],'best_epoch':1}
            save_checkpoint(path,m,o,g,record,BINDING)
            restored,optimizer,rng,actual=load_checkpoint(path,BINDING)
            self.assertEqual(record,actual)
            self.assertTrue(torch.equal(g.get_state(),rng.get_state()))
            np.testing.assert_array_equal(m.flat_parameters(),restored.flat_parameters())
            a=train_epoch(m,o,x,y,g,batch_size=2,accumulate=2,clip_norm=1.)
            b=train_epoch(restored,optimizer,x,y,rng,batch_size=2,accumulate=2,clip_norm=1.)
            self.assertEqual(a,b)
            np.testing.assert_array_equal(m.flat_parameters(),restored.flat_parameters())
            for p,q in zip(m.parameters(),restored.parameters()):
                for key in ('step','exp_avg','exp_avg_sq'): self.assertTrue(torch.equal(o.state[p][key],optimizer.state[q][key]))
            with self.assertRaises(FileExistsError): save_checkpoint(path,m,o,g,record,BINDING)

    def test_corruption_binding_and_partial_checkpoint(self):
        m,o,x,y,g=setup()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state'; save_checkpoint(path,m,o,g,{'epoch':0},BINDING)
            with self.assertRaises(ValueError): load_checkpoint(path,{**BINDING,'data_sha256':'d'*64})
            blob=(path/'arrays.npz').read_bytes()
            (path/'arrays.npz').write_bytes(blob[:-1])
            with self.assertRaises(ValueError): load_checkpoint(path,BINDING)
            (path/'arrays.npz').write_bytes(blob)
            manifest=json.loads((path/'manifest.json').read_text())
            manifest['environment']['torch']='wrong'
            (path/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): load_checkpoint(path,BINDING)
            (path/'manifest.json').unlink()
            with self.assertRaises((ValueError,FileNotFoundError)): load_checkpoint(path,BINDING)

    def test_rehashed_invalid_tensor_rejected(self):
        m,o,x,y,g=setup()
        train_epoch(m,o,x,y,g,batch_size=4,accumulate=1,clip_norm=1.)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state'; save_checkpoint(path,m,o,g,{'epoch':1},BINDING)
            with np.load(path/'arrays.npz',allow_pickle=False) as data: arrays={k:data[k] for k in data.files}
            arrays['v_0'].flat[0]=-1
            with (path/'arrays.npz').open('wb') as f: np.savez(f,**arrays)
            manifest=json.loads((path/'manifest.json').read_text())
            manifest['arrays_sha256']=hashlib.sha256((path/'arrays.npz').read_bytes()).hexdigest()
            (path/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): load_checkpoint(path,BINDING)

    def test_seed_isolation(self):
        before=torch.get_rng_state().clone()
        a,_,_,_,_=setup(); b,_,_,_,_=setup()
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        np.testing.assert_array_equal(a.flat_parameters(),b.flat_parameters())

    def test_tiny_learning_and_trained_native_export(self):
        m,o,x,y,g=setup()
        with torch.no_grad(): before=float((m(x)[:,-1]-y).square().mean())
        for _ in range(40): train_epoch(m,o,x,y,g,batch_size=7,accumulate=1,clip_norm=1.)
        with torch.no_grad(): prediction=m(x).numpy()
        self.assertLess(np.mean((prediction[:,-1]-y.numpy())**2),before*.3)
        session=native.ForecastModel.from_bytes(m.export_bytes()).session()
        np.testing.assert_allclose(session.predict_batch(x.numpy()),prediction,atol=5e-5,rtol=2e-4)

    def test_metrics_and_reject_nonfinite_training(self):
        self.assertEqual(metrics(np.array([1.,3.]),np.array([2.,1.])),{'count':2,'mae':1.5,'rmse':np.sqrt(2.5)})
        for a,b in (([1],[1,2]),([],[]),([float('nan')],[0])):
            with self.assertRaises(ValueError): metrics(np.array(a),np.array(b))
        m,o,x,y,g=setup(); y[0]=float('nan')
        before=m.flat_parameters().copy()
        with self.assertRaises(ValueError): train_epoch(m,o,x,y,g,batch_size=2,accumulate=2,clip_norm=1.)
        np.testing.assert_array_equal(before,m.flat_parameters())

if __name__=='__main__': unittest.main()
