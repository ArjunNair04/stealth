"""Framework/native parity; answers are independent of the Torch implementation."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
import stealth_cpu as native
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import generate_forecast_fixture as fixture
from torch_forecaster import Config, Forecaster, layer_norm, exact_gelu, attention

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


class Reference(unittest.TestCase):
    def test_components_known_answers(self):
        x = torch.tensor([1., 3.])
        gamma, beta = torch.tensor([2., 3.]), torch.tensor([4., 5.])
        np.testing.assert_allclose(layer_norm(x, gamma, beta, 1e-5).numpy(),
                                   native.layer_norm(x.numpy(), gamma.numpy(), beta.numpy()), atol=2e-6, rtol=2e-5)
        z = torch.tensor([-8., -3., -1., 0., 1., 3., 8.])
        np.testing.assert_allclose(exact_gelu(z).numpy(), native.gelu(z.numpy()), atol=2e-6, rtol=2e-5)
        q = torch.zeros(1, 3, 4)
        v = torch.arange(1., 13.).reshape(1, 3, 4)
        np.testing.assert_allclose(attention(q,q,v,2).numpy(), [[[1,2,3,4],[3,4,5,6],[5,6,7,8]]], atol=1e-6)

    def test_compact_fixture_and_identical_export(self):
        p,x,y,_,blob = fixture.fixture()
        c = Config(7,4,2,7,5,4,6)
        model = Forecaster.from_flat(c, np.array(p,np.float32))
        actual = model(torch.tensor(x).reshape(1,5,7)).detach().numpy()[0]
        np.testing.assert_allclose(actual,y,atol=5e-5,rtol=2e-4)
        self.assertEqual(model.export_bytes(),blob)
        for kernel in ('reference','packed'):
            session=native.ForecastModel.from_bytes(model.export_bytes(),kernel=kernel).session()
            np.testing.assert_allclose(session.predict(np.array(x,np.float32).reshape(5,7)),actual,atol=5e-5,rtol=2e-4)

    def test_full_shape_independent_and_native(self):
        c=Config(7,64,4,256,96,4,6)
        sizes=(7,7,448,64,64,64,12288,192,4096,64,64,64,16384,256,16384,64,64,64,64,1)
        p=[]
        for i,size in enumerate(sizes):
            values=([float(j)-3 for j in range(size)] if i==0 else
                    [float(j)+1 for j in range(size)] if i==1 else fixture.a.pattern(size,i*7+3,128))
            if i in (4,10,16): values=[1+x for x in values]
            p.extend(values)
        model=Forecaster.from_flat(c,np.array(p,np.float32))
        x=np.array(fixture.a.pattern(96*7,17,8),np.float32).reshape(96,7)
        with torch.no_grad(): actual=model(torch.from_numpy(x)[None]).numpy()[0]
        np.testing.assert_allclose(actual,fixture.full_case(),atol=5e-5,rtol=2e-4)
        for kernel in ('reference','packed'):
            out=native.ForecastModel.from_bytes(model.export_bytes(),kernel=kernel).session().predict(x)
            np.testing.assert_allclose(out,actual,atol=5e-5,rtol=2e-4)
        self.assertEqual(sum(p.numel() for p in model.parameters()),50689)

    def test_batch_prefix_and_future_invariance(self):
        c=Config(3,8,2,11,6,2,1)
        model=Forecaster(c, np.zeros(3,np.float32),np.ones(3,np.float32),seed=11)
        x=torch.randn(3,6,3,generator=torch.Generator().manual_seed(17))
        with torch.no_grad():
            y=model(x)
            np.testing.assert_allclose(y.numpy(),torch.cat([model(row[None]) for row in x]).numpy(),atol=5e-5,rtol=2e-4)
            np.testing.assert_allclose(y[:,:3].numpy(),model(x[:,:3]).numpy(),atol=5e-5,rtol=2e-4)
            x[:,3:]+=99
            np.testing.assert_array_equal(y[:,:3].numpy(),model(x)[:,:3].numpy())

    def test_all_gradients_and_finite_difference(self):
        model=Forecaster(Config(2,4,2,5,3,1,1),np.zeros(2,np.float32),np.ones(2,np.float32),seed=7)
        x=torch.tensor([[[.1,.4],[.3,-.2],[.7,.9]]],requires_grad=True)
        loss=model(x)[0,-1]; loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for name,p in model.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
            index=np.unravel_index(int(p.grad.abs().argmax()),p.shape)
            analytic=float(p.grad[index]); original=float(p[index]); delta=.001
            with torch.no_grad():
                p[index]=original+delta; plus=float(model(x)[0,-1])
                p[index]=original-delta; minus=float(model(x)[0,-1])
                p[index]=original
            self.assertAlmostEqual(analytic,(plus-minus)/(2*delta),delta=.003+abs(analytic)*.025,msg=name)

    def test_invalid_configuration_input_and_export(self):
        for changes in ({'heads':3},{'width':0},{'context':1025},{'epsilon':0},{'features':True},{'target':2}):
            c=dict(features=2,width=4,heads=2,hidden=5,context=3,horizon=1,target=1)
            c.update(changes)
            with self.assertRaises(ValueError): Config(**c)
        c=Config(2,4,2,5,3,1,1)
        for mean,scale in (([0,0],[0,1]),([float('nan'),0],[1,1]),([0],[1,1])):
            with self.assertRaises(ValueError): Forecaster(c,mean,scale)
        model=Forecaster(c,[0,0],[1,1])
        for x in (torch.zeros(1,4,2),torch.zeros(1,3,3),torch.zeros(0,3,2),torch.zeros(1,3,2,dtype=torch.float64),torch.full((1,3,2),float('inf'))):
            with self.assertRaises(ValueError): model(x)
        bounded=Forecaster(Config(1,4,4,4,1024,1,0),[0],[1])
        with self.assertRaises(ValueError): bounded(torch.zeros(5,1024,1))
        with self.assertRaises(ValueError): Forecaster.from_flat(c,np.zeros(12,np.float32))
        with torch.no_grad(): model.head_bias.fill_(float('nan'))
        with self.assertRaises(ValueError): model.export_bytes()


if __name__=='__main__': unittest.main()
