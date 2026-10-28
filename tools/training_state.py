"""Small CPU training loop and bounded, pickle-free epoch checkpoints."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import zipfile
import numpy as np
import torch
from torch_forecaster import Config, Forecaster

MAX_CHECKPOINT=64*1024*1024


def canonical(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def digest(data): return hashlib.sha256(data).hexdigest()


def environment():
    return dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
                system=platform.system(),machine=platform.machine(),threads=torch.get_num_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                torch_build_sha256=digest(torch.__config__.show().encode()))


def make_optimizer(model,settings):
    if set(settings)!={'lr','weight_decay','betas','eps'}: raise ValueError('Unknown AdamW settings')
    for name in ('lr','eps'):
        if not 0<float(settings[name])<float('inf'): raise ValueError('Invalid optimizer setting')
    if not 0<=float(settings['weight_decay'])<float('inf'): raise ValueError('Invalid weight decay')
    if len(settings['betas'])!=2 or any(not 0<=float(v)<1 for v in settings['betas']): raise ValueError('Invalid beta')
    return torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'],
                            betas=tuple(settings['betas']),eps=settings['eps'],foreach=False,amsgrad=False)


def train_epoch(model,optimizer,x,y,generator,*,batch_size,accumulate,clip_norm):
    if (type(batch_size) is not int or not 1<=batch_size<=1024 or type(accumulate) is not int
            or not 1<=accumulate<=64 or not 0<clip_norm<float('inf') or len(x)!=len(y)
            or len(x)==0 or y.ndim!=1 or y.dtype!=torch.float32 or y.device.type!='cpu'
            or not bool(torch.isfinite(y).all())):
        raise ValueError('Invalid training batch or target')
    model.train()
    order=torch.randperm(len(x),generator=generator)
    effective=batch_size*accumulate
    target_mean=model.mean[model.config.target].double()
    target_scale=model.scale[model.config.target].double()
    total=0.
    optimizer.zero_grad(set_to_none=True)
    for start in range(0,len(order),effective):
        group=order[start:start+effective]
        for offset in range(0,len(group),batch_size):
            indices=group[offset:offset+batch_size]
            target=((y[indices].double()-target_mean)/target_scale).float()
            prediction=model(x[indices],standardized=True)[:,-1]
            squared=(prediction-target).square().sum()
            if not bool(torch.isfinite(squared)): raise FloatingPointError('Non-finite training loss')
            (squared/len(group)).backward()
            total+=float(squared.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(),clip_norm,error_if_nonfinite=True,foreach=False)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError('Non-finite updated parameter; restore previous complete checkpoint')
    return total/len(x)


def metrics(prediction,target):
    p,t=np.asarray(prediction,dtype=np.float64),np.asarray(target,dtype=np.float64)
    if p.ndim!=1 or p.shape!=t.shape or not len(p) or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Invalid metric inputs')
    error=p-t
    result=dict(count=len(p),mae=float(np.mean(np.abs(error))),rmse=float(np.sqrt(np.mean(error**2))))
    if not all(math.isfinite(v) for v in result.values()): raise ValueError('Metric overflow')
    return result


def save_checkpoint(path,model,optimizer,generator,record,binding):
    """Create a new epoch directory; manifest written last is the completion marker."""
    if len(optimizer.param_groups)!=1 or not isinstance(optimizer,torch.optim.AdamW):
        raise ValueError('Only one AdamW parameter group is supported')
    if any(p.grad is not None for p in model.parameters()): raise ValueError('Checkpoint only between complete optimizer steps')
    group=optimizer.param_groups[0]
    settings={key:group[key] for key in ('lr','weight_decay','betas','eps')}
    if (group.get('amsgrad') or group.get('maximize') or group.get('foreach') is not False
            or group.get('capturable') or group.get('differentiable') or group.get('fused')
            or any(p is not q for p,q in zip(group['params'],model.parameters()))
            or len(group['params'])!=len(list(model.parameters()))):
        raise ValueError('Unsupported optimizer configuration')
    arrays={'flat':model.flat_parameters(),'rng':generator.get_state().numpy().copy()}
    slots=[]
    for i,p in enumerate(model.parameters()):
        slot=optimizer.state.get(p,{})
        slots.append(bool(slot))
        if slot:
            for key,field in (('m','exp_avg'),('v','exp_avg_sq'),('step','step')):
                arrays[f'{key}_{i}']=slot[field].detach().cpu().numpy().copy()
    if len(set(slots))!=1 or any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Incomplete or non-finite optimizer state')
    buffer=io.BytesIO(); np.savez(buffer,**arrays); payload=buffer.getvalue()
    if len(payload)>MAX_CHECKPOINT: raise ValueError('Checkpoint too large')
    manifest=dict(schema_version=1,config=asdict(model.config),optimizer=settings,slots=slots,
                  record=record,binding=binding,environment=environment(),arrays_sha256=digest(payload))
    encoded=canonical(manifest)
    if len(encoded)>1024*1024: raise ValueError('Checkpoint metadata too large')
    path=Path(path); path.mkdir(parents=True,exist_ok=False)
    with (path/'arrays.npz').open('xb') as f: f.write(payload)
    with (path/'manifest.json').open('xb') as f: f.write(encoded)


def bounded(path,limit):
    with Path(path).open('rb') as f: data=f.read(limit+1)
    if len(data)>limit: raise ValueError('File exceeds byte limit')
    return data


def strict_json(data):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result: raise ValueError('Duplicate metadata key')
            result[key]=value
        return result
    def invalid(value): raise ValueError('Non-finite JSON constant')
    return json.loads(data,object_pairs_hook=pairs,parse_constant=invalid)


def load_checkpoint(path,binding):
    """Return fresh candidates only. Never unpickle or modify an active model/RNG."""
    path=Path(path); meta=strict_json(bounded(path/'manifest.json',1024*1024))
    if (set(meta)!={'schema_version','config','optimizer','slots','record','binding','environment','arrays_sha256'}
            or meta['schema_version']!=1 or meta['binding']!=binding or meta['environment']!=environment()):
        raise ValueError('Checkpoint schema, experiment or environment mismatch')
    config=Config(**meta['config']); shapes=list(config.shapes().values())[2:]
    slots=meta['slots']
    if len(slots)!=len(shapes) or any(type(v) is not bool for v in slots) or len(set(slots))!=1:
        raise ValueError('Partial optimizer state')
    expected={'flat','rng'}|{f'{key}_{i}' for i,slot in enumerate(slots) if slot for key in ('m','v','step')}
    payload=bounded(path/'arrays.npz',MAX_CHECKPOINT)
    if digest(payload)!=meta['arrays_sha256']: raise ValueError('Checkpoint hash mismatch')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos=archive.infolist()
        if (len(infos)!=len(expected) or {i.filename for i in infos}!={n+'.npy' for n in expected}
                or sum(i.file_size for i in infos)>MAX_CHECKPOINT
                or any(i.compress_type!=zipfile.ZIP_STORED for i in infos)):
            raise ValueError('Unexpected checkpoint archive')
        expected_shapes={'flat':(sum(math.prod(s) for s in config.shapes().values()),),
                         'rng':tuple(torch.Generator().get_state().shape)}
        expected_shapes.update({f'{key}_{i}': (() if key=='step' else shape)
                                for i,shape in enumerate(shapes) if slots[i] for key in ('m','v','step')})
        for info in infos:
            name=info.filename[:-4]
            with archive.open(info) as stream:
                version=np.lib.format.read_magic(stream)
                if version==(1,0): reader=np.lib.format.read_array_header_1_0
                elif version==(2,0): reader=np.lib.format.read_array_header_2_0
                else: raise ValueError('Unsupported NumPy array version')
                shape,fortran,dtype=reader(stream,max_header_size=1024)
                expected_dtype=np.dtype('uint8' if name=='rng' else 'float32')
                if (shape!=expected_shapes[name] or fortran or dtype!=expected_dtype
                        or stream.tell()+math.prod(shape)*dtype.itemsize!=info.file_size):
                    raise ValueError('Invalid array header or declared payload size')
    with np.load(io.BytesIO(payload),allow_pickle=False,max_header_size=1024) as archive:
        arrays={name:archive[name] for name in expected}
    for name,array in arrays.items():
        if name=='rng':
            if array.dtype!=np.uint8 or array.shape!=tuple(torch.Generator().get_state().shape): raise ValueError('Invalid RNG state')
        elif array.dtype!=np.float32 or not np.isfinite(array).all(): raise ValueError('Invalid checkpoint tensor')
    model=Forecaster.from_flat(config,arrays['flat'])
    optimizer=make_optimizer(model,meta['optimizer'])
    steps=set()
    for i,p in enumerate(model.parameters()):
        if not slots[i]: continue
        m,v,step=(arrays[f'{key}_{i}'] for key in ('m','v','step'))
        if (m.shape!=tuple(p.shape) or v.shape!=tuple(p.shape) or (v<0).any() or step.shape!=()
                or float(step)<1 or float(step)!=int(step)):
            raise ValueError('Invalid AdamW state')
        steps.add(float(step))
        optimizer.state[p]={'exp_avg':torch.from_numpy(m.copy()),'exp_avg_sq':torch.from_numpy(v.copy()),'step':torch.from_numpy(step.copy())}
    if len(steps)>1: raise ValueError('Inconsistent optimizer steps')
    rng=torch.Generator(); rng.set_state(torch.from_numpy(arrays['rng'].copy()))
    return model,optimizer,rng,meta['record']
