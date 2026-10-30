"""Compiler/export boundaries; real engines are exercised in opt-in Linux CI."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from torch_forecaster import Config,Forecaster
from runtime_torch import load_export
from runtime_adapters import LatestGraph,TorchAdapter,OnnxAdapter
import generate_forecast_fixture as fixture


def setUpModule():
    torch.set_num_threads(1)


def model_and_inputs(batch=1,tokens=5,mode='correction'):
    c=Config(7,4,2,7,tokens,4,6,output=mode)
    m=Forecaster(c,[1]*7,[2]*7,seed=7)
    with torch.no_grad():
        m.head_weight.copy_(torch.tensor([[-.2,.1,.3,.4]]))
        m.head_bias.fill_(.1)
    x=np.linspace(-1,2,batch*tokens*7,dtype=np.float32).reshape(batch,tokens,7)
    return m,x


class AdapterContracts(unittest.TestCase):
    def test_pure_graph_independent_fixture(self):
        params,values,answers,_,_=fixture.fixture()
        for mode in ('level','correction'):
            c=Config(7,4,2,7,5,4,6,output=mode)
            source=Forecaster.from_flat(c,np.array(params,np.float32))
            raw=torch.tensor(values,dtype=torch.float32).reshape(1,5,7)
            graph=LatestGraph(load_export(source.export_bytes(),standard=True))
            with torch.inference_mode(): actual=graph(raw).numpy()
            expected=answers[-1] if mode=='level' else answers[-1]-params[6]+float(raw[0,-1,6])
            np.testing.assert_allclose(actual,[[expected]],atol=5e-5,rtol=2e-4)

    def test_eager_shared_graph_and_prefixes(self):
        m,x=model_and_inputs(batch=8,tokens=96)
        with torch.inference_mode():
            expected=m(torch.from_numpy(x)).numpy()
            for tokens in (1,5,96):
                adapter=TorchAdapter(m.export_bytes(),batch=8,tokens=tokens,compiled=False)
                raw=np.ascontiguousarray(x[:,:tokens])
                actual=adapter.run(adapter.prepare(raw)).numpy()
                np.testing.assert_allclose(actual,expected[:,tokens-1:tokens],atol=5e-5,rtol=2e-4)

    def test_compile_settings_and_no_eager_fallback(self):
        m,x=model_and_inputs()
        with patch('torch.compile',side_effect=lambda graph,**kwargs:graph) as compile_mock:
            adapter=TorchAdapter(m.export_bytes(),batch=1,tokens=5,compiled=True)
            self.assertEqual(compile_mock.call_args.kwargs,dict(backend='inductor',fullgraph=True,dynamic=False,mode='default'))
            with torch.inference_mode(): self.assertEqual(tuple(adapter.run(adapter.prepare(x)).shape),(1,1))
        with patch('torch.compile',side_effect=RuntimeError('compiler unavailable')):
            with self.assertRaises(RuntimeError): TorchAdapter(m.export_bytes(),batch=1,tokens=5,compiled=True)

    def test_checked_fixed_shape_finite_and_inference_boundary(self):
        m,x=model_and_inputs()
        a=TorchAdapter(m.export_bytes(),batch=1,tokens=5,compiled=False)
        for raw in (x.astype(np.float64),x[:,::2],np.repeat(x,2,axis=0),np.full_like(x,np.nan)):
            with self.assertRaises(ValueError): a.prepare(raw)
        prepared=a.prepare(x)
        with self.assertRaises(RuntimeError): a.run(prepared)
        with torch.inference_mode():
            prepared[0,0,0]=float('inf')
            with self.assertRaises(ValueError): a.run(prepared)
            prepared[0,0,0]=0
            with patch.object(a,'graph',return_value=torch.full((1,1),float('nan'))):
                with self.assertRaises(FloatingPointError): a.run(prepared)
            with patch.object(a,'graph',return_value=torch.zeros(1,5)):
                with self.assertRaises(ValueError): a.run(prepared)
        for batch,tokens in ((0,5),(True,5),(1,6),(2048,5)):
            with self.assertRaises(ValueError): TorchAdapter(m.export_bytes(),batch=batch,tokens=tokens,compiled=False)

    def test_tensor_graph_has_no_data_dependent_python_checks(self):
        # Dynamo's eager backend checks graph capture on the legacy local Torch.
        # This is not an Inductor performance/admission result.
        m,x=model_and_inputs()
        graph=LatestGraph(load_export(m.export_bytes(),standard=True))
        captured=torch.compile(graph,backend='eager',fullgraph=True,dynamic=False)
        with torch.inference_mode():
            np.testing.assert_array_equal(captured(torch.from_numpy(x)).numpy(),graph(torch.from_numpy(x)).numpy())


@unittest.skipUnless(os.environ.get('STEALTH_RUN_FRONTIER')=='1','optional Linux compiler/export dependency job')
class ActualEngines(unittest.TestCase):
    def test_compiled_and_onnx_independent_fixture_and_full_shapes(self):
        import onnx
        import onnxruntime as ort
        self.assertTrue(ort.__version__)
        params,values,answers,_,_=fixture.fixture()
        for mode in ('level','correction'):
            source=Forecaster.from_flat(Config(7,4,2,7,5,4,6,output=mode),np.array(params,np.float32))
            x=np.array(values,np.float32).reshape(1,5,7)
            expected=answers[-1] if mode=='level' else answers[-1]-params[6]+x[0,-1,6]
            self.check_engines(source,x,np.array([[expected]],np.float32))
        # Actual deployed dimensions, both workloads, nonzero correction head.
        for batch in (1,8):
            c=Config(7,64,4,256,96,4,6,output='correction')
            source=Forecaster(c,[1]*7,[2]*7,seed=19)
            with torch.no_grad(): source.head_weight.fill_(.02);source.head_bias.fill_(.1)
            rng=np.random.default_rng(7)
            x=rng.normal(size=(batch,96,7)).astype(np.float32)
            with torch.inference_mode(): expected=source(torch.from_numpy(x)).numpy()[:,-1:]
            self.check_engines(source,x,expected)

    def check_engines(self,source,x,expected):
        with tempfile.TemporaryDirectory() as tmp,torch.inference_mode():
            shape=dict(batch=x.shape[0],tokens=x.shape[1])
            compiled=TorchAdapter(source.export_bytes(),compiled=True,**shape)
            onnx=OnnxAdapter(source.export_bytes(),artifacts=Path(tmp)/'onnx',**shape)
            for engine in (compiled,onnx):
                for raw in (x, np.ascontiguousarray(x*.8)):
                    with torch.inference_mode(): target=source(torch.from_numpy(raw)).numpy()[:,-1:]
                    actual=engine.run(engine.prepare(raw))
                    if isinstance(actual,torch.Tensor):actual=actual.numpy()
                    np.testing.assert_allclose(actual,target,atol=5e-5,rtol=2e-4)
                actual=engine.run(engine.prepare(x))
                if isinstance(actual,torch.Tensor):actual=actual.numpy()
                np.testing.assert_allclose(actual,expected,atol=5e-5,rtol=2e-4)
            self.assertEqual(onnx.session.get_providers(),['CPUExecutionProvider'])
            self.assertEqual(set(onnx.metadata['graph_sha256']),{'exported','optimized'})
            for name in ('model.onnx','optimized.onnx'): self.assertTrue((Path(tmp)/'onnx'/name).is_file())
            bad=x.copy();bad.flat[0]=np.nan
            with self.assertRaises(ValueError): onnx.run(bad)
            with self.assertRaises(FileExistsError): OnnxAdapter(source.export_bytes(),artifacts=Path(tmp)/'onnx',**shape)


if __name__=='__main__':unittest.main()
