"""CPU autograd reference for the fixed version-1 native forecasting contract.

Torch owns matrix kernels, gradients and optimization. Explicit FP64 reductions
match the native normalization/attention contract; this is a parity reference,
not the fastest general-purpose Torch transformer implementation.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import struct
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    features: int
    width: int
    heads: int
    hidden: int
    context: int
    horizon: int
    target: int
    epsilon: float = 1e-5
    output: str = "level"

    def __post_init__(self):
        if self.output not in ("level","correction"): raise ValueError("Invalid output mode")
        for name,maximum in (('features',64),('width',512),('heads',512),('hidden',2048),('context',1024),('horizon',1024)):
            value=getattr(self,name)
            if type(value) is not int or not 1<=value<=maximum:
                raise ValueError(f'Invalid {name}')
        if self.width%self.heads or type(self.target) is not int or not 0<=self.target<self.features:
            raise ValueError('Invalid heads or target')
        if not isinstance(self.epsilon,(int,float)) or isinstance(self.epsilon,bool) or not 0<float(self.epsilon)<=np.finfo(np.float32).max:
            raise ValueError('Invalid epsilon')
        epsilon=float(np.float32(self.epsilon))
        if epsilon<=0: raise ValueError('Epsilon underflow')
        object.__setattr__(self,'epsilon',epsilon)

    def shapes(self):
        f,w,h=self.features,self.width,self.hidden
        return dict(mean=(f,),scale=(f,),input_weight=(w,f),input_bias=(w,),
                    attn_scale=(w,),attn_shift=(w,),qkv_weights=(3*w,w),qkv_bias=(3*w,),
                    projection_weights=(w,w),projection_bias=(w,),ff_scale=(w,),ff_shift=(w,),
                    up_weights=(h,w),up_bias=(h,),down_weights=(w,h),down_bias=(w,),
                    final_scale=(w,),final_shift=(w,),head_weight=(1,w),head_bias=(1,))


def layer_norm(x,scale,shift,epsilon):
    wide=x.double()
    centered=wide-wide.mean(dim=-1,keepdim=True)
    variance=centered.square().mean(dim=-1,keepdim=True)
    return (centered*torch.rsqrt(variance+epsilon)*scale.double()+shift.double()).float()


def exact_gelu(x):
    wide=x.double()
    return (.5*wide*torch.erfc(-wide/math.sqrt(2))).float()


def attention(q,k,v,heads):
    batch,tokens,width=q.shape
    dim=width//heads
    def split(x): return x.reshape(batch,tokens,heads,dim).transpose(1,2)
    q,k,v=map(split,(q,k,v))
    scores=(torch.matmul(q.double(),k.double().transpose(-1,-2))/math.sqrt(dim)).float()
    mask=torch.ones(tokens,tokens,device=q.device,dtype=torch.bool).triu(1)
    scores=scores.masked_fill(mask,float('-inf'))
    probabilities=torch.softmax(scores.double(),dim=-1).float()
    result=torch.matmul(probabilities.double(),v.double()).float()
    return result.transpose(1,2).contiguous().reshape(batch,tokens,width)


class Forecaster(nn.Module):
    def __init__(self,config: Config,mean,scale,seed=0):
        super().__init__()
        self.config=config
        for name,values in (('mean',mean),('scale',scale)):
            array=np.asarray(values,dtype=np.float32)
            if array.shape!=(config.features,) or not np.isfinite(array).all() or (name=='scale' and not (array>0).all()):
                raise ValueError('Invalid scaler')
            self.register_buffer(name,torch.from_numpy(array.copy()))
        generator=torch.Generator(device='cpu').manual_seed(seed)
        for name,shape in list(config.shapes().items())[2:]:
            p=nn.Parameter(torch.empty(shape,dtype=torch.float32))
            if len(shape)==2: nn.init.xavier_uniform_(p,generator=generator)
            elif name.endswith('scale'): nn.init.ones_(p)
            else: nn.init.zeros_(p)
            self.register_parameter(name,p)
        if config.output=='correction':
            nn.init.zeros_(self.head_weight);nn.init.zeros_(self.head_bias)
        positions=[[ (math.sin if c%2==0 else math.cos)(t/10000**(2*(c//2)/config.width))
                    for c in range(config.width)] for t in range(config.context)]
        self.register_buffer('positions',torch.tensor(positions,dtype=torch.float32),persistent=False)

    def forward(self,raw,*,standardized=False):
        c=self.config
        if (raw.device.type!='cpu' or raw.dtype!=torch.float32 or raw.ndim!=3
                or not 1<=raw.shape[0]<=1024 or not 1<=raw.shape[1]<=c.context
                or raw.shape[2]!=c.features or raw.shape[0]*c.heads*raw.shape[1]**2>16*1024*1024
                or raw.shape[0]*raw.shape[1]*max(c.width,c.hidden)>16*1024*1024
                or not bool(torch.isfinite(raw).all())):
            raise ValueError('Expected finite CPU FP32 [batch,time,features] within bounds')
        x=((raw.double()-self.mean.double())/self.scale.double()).float()
        x=F.linear(x,self.input_weight,self.input_bias)+self.positions[:raw.shape[1]]
        q,k,v=F.linear(layer_norm(x,self.attn_scale,self.attn_shift,c.epsilon),self.qkv_weights,self.qkv_bias).chunk(3,dim=-1)
        x=x+F.linear(attention(q,k,v,c.heads),self.projection_weights,self.projection_bias)
        hidden=F.linear(layer_norm(x,self.ff_scale,self.ff_shift,c.epsilon),self.up_weights,self.up_bias)
        x=x+F.linear(exact_gelu(hidden),self.down_weights,self.down_bias)
        y=F.linear(layer_norm(x,self.final_scale,self.final_shift,c.epsilon),self.head_weight,self.head_bias).squeeze(-1)
        if c.output=='correction':
            anchor=raw[:,:,c.target].double()
            if standardized:
                y=(y.double()+(anchor-self.mean[c.target].double())/self.scale[c.target].double()).float()
            else:
                y=(y.double()*self.scale[c.target].double()+anchor).float()
        elif not standardized:
            y=(y.double()*self.scale[c.target].double()+self.mean[c.target].double()).float()
        if not bool(torch.isfinite(y).all()): raise FloatingPointError('Non-finite model output')
        return y

    def flat_parameters(self):
        values=[]
        for name,shape in self.config.shapes().items():
            tensor=getattr(self,name)
            if tensor.device.type!='cpu' or tensor.dtype!=torch.float32 or tuple(tensor.shape)!=shape:
                raise ValueError('Export requires CPU FP32 parameters with canonical shapes')
            values.append(tensor.detach().numpy().reshape(-1))
        flat=np.concatenate(values)
        if not np.isfinite(flat).all() or not (self.scale>0).all(): raise ValueError('Invalid export parameters')
        return flat

    @classmethod
    def from_flat(cls,config,flat):
        if not isinstance(flat,np.ndarray) or flat.dtype!=np.float32 or flat.ndim!=1:
            raise ValueError('Expected flat FP32 array')
        shapes=config.shapes()
        if flat.size!=sum(math.prod(s) for s in shapes.values()) or not np.isfinite(flat).all():
            raise ValueError('Invalid parameter count or values')
        f=config.features
        model=cls(config,flat[:f],flat[f:2*f])
        offset=0
        with torch.no_grad():
            for name,shape in shapes.items():
                size=math.prod(shape)
                getattr(model,name).copy_(torch.from_numpy(flat[offset:offset+size].copy()).reshape(shape))
                offset+=size
        return model

    def export_bytes(self):
        flat=self.flat_parameters(); c=self.config
        correction=c.output=='correction'
        header=(b'STLFC02\0' if correction else b'STLFC01\0')+struct.pack('<10IfIII',
                2 if correction else 1,0x01020304,64,c.features,c.width,c.heads,c.hidden,
                c.context,c.horizon,c.target,c.epsilon,flat.size,1 if correction else 0,0)
        payload=header+flat.astype('<f4',copy=False).tobytes()
        if len(payload)+32>16*1024*1024: raise ValueError('Artifact exceeds native byte limit')
        return payload+hashlib.sha256(payload).digest()
