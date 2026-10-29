"""Ordinary eager CPU kernels for benchmarking; no change to the training reference.

FP32 layer_norm, exact GELU and causal SDPA are separately parity-gated. This is
an idiomatic baseline, not a claim of the best possible Torch deployment.
"""
import hashlib
import struct
import numpy as np
import torch
from torch.nn import functional as F
from torch_forecaster import Config,Forecaster

class StandardForecaster(Forecaster):
    def forward(self,raw):
        c=self.config
        if (raw.device.type!='cpu' or raw.dtype!=torch.float32 or raw.ndim!=3
                or not 1<=raw.shape[0]<=1024 or not 1<=raw.shape[1]<=c.context
                or raw.shape[2]!=c.features or raw.shape[0]*c.heads*raw.shape[1]**2>16*1024*1024
                or raw.shape[0]*raw.shape[1]*max(c.width,c.hidden)>16*1024*1024
                or not bool(torch.isfinite(raw).all())):
            raise ValueError('Expected finite bounded CPU FP32 input')
        batch,tokens,_=raw.shape
        x=((raw.double()-self.mean.double())/self.scale.double()).float()
        x=F.linear(x,self.input_weight,self.input_bias)+self.positions[:tokens]
        z=F.layer_norm(x,(c.width,),self.attn_scale,self.attn_shift,c.epsilon)
        q,k,v=F.linear(z,self.qkv_weights,self.qkv_bias).chunk(3,dim=-1)
        def split(a): return a.reshape(batch,tokens,c.heads,c.width//c.heads).transpose(1,2)
        a=F.scaled_dot_product_attention(split(q),split(k),split(v),dropout_p=0.,is_causal=True)
        a=a.transpose(1,2).contiguous().reshape(batch,tokens,c.width)
        x=x+F.linear(a,self.projection_weights,self.projection_bias)
        z=F.layer_norm(x,(c.width,),self.ff_scale,self.ff_shift,c.epsilon)
        z=F.gelu(F.linear(z,self.up_weights,self.up_bias),approximate='none')
        x=x+F.linear(z,self.down_weights,self.down_bias)
        z=F.layer_norm(x,(c.width,),self.final_scale,self.final_shift,c.epsilon)
        y=F.linear(z,self.head_weight,self.head_bias).squeeze(-1)
        anchor=raw[:,:,c.target].double() if c.output=='correction' else self.mean[c.target].double()
        y=(y.double()*self.scale[c.target].double()+anchor).float()
        if not bool(torch.isfinite(y).all()): raise FloatingPointError('Nonfinite model output')
        return y

def load_export(blob,*,standard=False):
    if not 96<=len(blob)<=16*1024*1024 or hashlib.sha256(blob[:-32]).digest()!=blob[-32:]:
        raise ValueError('Invalid export length or digest')
    v,sentinel,header,f,w,heads,h,context,horizon,target,eps,count,mode,reserved=struct.unpack_from('<10IfIII',blob,8)
    if (sentinel!=0x01020304 or header!=64 or reserved!=0 or len(blob)!=96+4*count
            or (blob[:8],v,mode) not in ((b'STLFC01\0',1,0),(b'STLFC02\0',2,1))):
        raise ValueError('Invalid export contract')
    c=Config(f,w,heads,h,context,horizon,target,eps,'correction' if mode else 'level')
    model=(StandardForecaster if standard else Forecaster).from_flat(c,np.frombuffer(blob[64:-32],dtype='<f4').astype(np.float32))
    if model.export_bytes()!=blob: raise ValueError('Noncanonical model artifact')
    return model.eval().requires_grad_(False)
