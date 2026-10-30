"""Fixed-shape latest-forecast adapters for correctness admission and CPU timing.

Validation surrounds the pure tensor graph and remains inside every run() call.
No silent fallback, dependency installation, training or dynamic shape support.
"""
import inspect
import os
from pathlib import Path
import time
import numpy as np
import torch
from runtime_common import bounded,digest
from runtime_torch import load_export


class LatestGraph(torch.nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model

    def forward(self,raw):
        return self.model.tensor_forward(raw,latest_only=True)


def fixed_shape(config,batch,tokens):
    if (type(batch) is not int or type(tokens) is not int or not 1<=batch<=1024 or
            not 1<=tokens<=config.context or batch*config.heads*tokens*tokens>16*1024*1024 or
            batch*tokens*max(config.width,config.hidden)>16*1024*1024):
        raise ValueError('Invalid fixed runtime shape')
    return batch,tokens,config.features


def check_numpy(raw,shape):
    if (not isinstance(raw,np.ndarray) or raw.dtype!=np.dtype('float32') or raw.shape!=shape or
            not raw.flags.c_contiguous or not raw.flags.aligned or not np.isfinite(raw).all()):
        raise ValueError('Expected finite aligned contiguous FP32 fixed-shape input')


class TorchAdapter:
    def __init__(self,blob,*,batch,tokens,compiled):
        self.model=load_export(blob,standard=True)
        self.shape=fixed_shape(self.model.config,batch,tokens)
        graph=LatestGraph(self.model).eval().requires_grad_(False)
        settings=dict(backend='inductor',fullgraph=True,dynamic=False,mode='default')
        if compiled:
            import torch._dynamo.config as dynamo_config
            if (dynamo_config.disable or dynamo_config.suppress_errors or
                    os.environ.get('TORCHDYNAMO_DISABLE')=='1'):
                raise RuntimeError('Compilation disablement or error suppression is forbidden')
        # Compilation is lazy; the supervisor separately bounds the first call.
        self.graph=torch.compile(graph,**settings) if compiled else graph
        self.metadata=dict(engine='torch_compiled_inductor' if compiled else 'torch_eager',
                           torch=str(torch.__version__),settings=settings if compiled else {},
                           model_sha256=digest(blob),shape=list(self.shape),output='last')

    def prepare(self,raw):
        check_numpy(raw,self.shape)
        return torch.from_numpy(raw)

    def run(self,raw):
        if not torch.is_inference_mode_enabled():
            raise RuntimeError('Runtime calls require torch.inference_mode')
        if (not isinstance(raw,torch.Tensor) or raw.device.type!='cpu' or raw.dtype!=torch.float32 or
                tuple(raw.shape)!=self.shape or not raw.is_contiguous() or not bool(torch.isfinite(raw).all())):
            raise ValueError('Expected finite contiguous CPU FP32 fixed-shape tensor')
        result=self.graph(raw)
        if (not isinstance(result,torch.Tensor) or result.device.type!='cpu' or result.dtype!=torch.float32 or
                tuple(result.shape)!=(self.shape[0],1)):
            raise ValueError('Runtime output contract mismatch')
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError('Non-finite runtime output')
        return result


class OnnxAdapter:
    def __init__(self,blob,*,batch,tokens,artifacts):
        # Optional packages are imported only for this explicitly selected engine.
        import onnx
        import onnxruntime as ort
        import onnxscript
        if 'dynamo' not in inspect.signature(torch.onnx.export).parameters:
            raise RuntimeError('The declared modern Dynamo ONNX exporter is required')
        model=load_export(blob,standard=True)
        self.shape=fixed_shape(model.config,batch,tokens)
        graph=LatestGraph(model).eval().requires_grad_(False)
        path=Path(artifacts)
        path.mkdir(parents=True,exist_ok=False)
        settings=dict(dynamo=True,opset_version=18,external_data=False,optimize=True,
                      input_names=['raw'],output_names=['forecast'],dynamic_shapes=None)
        start=time.perf_counter_ns()
        with torch.inference_mode():
            program=torch.onnx.export(graph,(torch.zeros(self.shape,dtype=torch.float32),),
                                      str(path/'model.onnx'),**settings)
        if program is None:
            raise RuntimeError('Dynamo ONNX export did not produce a program')
        exported=bounded(path/'model.onnx')
        checked=onnx.load_model_from_string(exported)
        if any(t.data_location==onnx.TensorProto.EXTERNAL for t in checked.graph.initializer):
            raise ValueError('External tensor files are not permitted')
        onnx.checker.check_model(checked,full_check=True)
        export_ns=time.perf_counter_ns()-start
        options=ort.SessionOptions()
        options.intra_op_num_threads=1
        options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        options.add_session_config_entry('session.inter_op.allow_spinning','0')
        options.optimized_model_filepath=str(path/'optimized.onnx')
        self.session=ort.InferenceSession(exported,sess_options=options,providers=['CPUExecutionProvider'])
        self.session.disable_fallback()
        if self.session.get_providers()!=['CPUExecutionProvider']:
            raise ValueError('Only the CPU execution provider is allowed')
        inputs,outputs=self.session.get_inputs(),self.session.get_outputs()
        if (len(inputs)!=1 or inputs[0].name!='raw' or inputs[0].type!='tensor(float)' or
                tuple(inputs[0].shape)!=self.shape or len(outputs)!=1 or outputs[0].name!='forecast' or
                outputs[0].type!='tensor(float)' or tuple(outputs[0].shape)!=(batch,1)):
            raise ValueError('Exported graph interface mismatch')
        self.metadata=dict(engine='onnxruntime_cpu',torch=str(torch.__version__),onnx=onnx.__version__,
                           onnxruntime=ort.__version__,onnxscript=onnxscript.__version__,
                           model_sha256=digest(blob),shape=list(self.shape),output='last',
                           export_settings=settings,export_ns=export_ns,
                           runtime_settings=dict(provider='CPUExecutionProvider',execution='sequential',
                                                 optimization='all',intra_op=1,inter_op=1,spinning=False),
                           graph_sha256=dict(exported=digest(exported),optimized=digest(bounded(path/'optimized.onnx'))))

    def prepare(self,raw):
        check_numpy(raw,self.shape)
        return raw

    def run(self,raw):
        check_numpy(raw,self.shape)
        result=self.session.run(['forecast'],{'raw':raw})[0]
        if not isinstance(result,np.ndarray) or result.dtype!=np.dtype('float32') or result.shape!=(self.shape[0],1):
            raise ValueError('Runtime output contract mismatch')
        if not np.isfinite(result).all():
            raise FloatingPointError('Non-finite runtime output')
        return result
