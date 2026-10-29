"""Small dependency-free contracts shared by the runtime experiment."""
import hashlib
import json
import math
import struct
import sys

LIMIT=16*1024*1024
MAGIC=b'STLBN01\0'

def canonical(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

def digest(value): return hashlib.sha256(value).hexdigest()

def bounded(path):
    with open(path,'rb') as f: data=f.read(LIMIT+1)
    if len(data)>LIMIT: raise ValueError('Benchmark artifact exceeds limit')
    return data

def summarize(values):
    values=list(values)
    if not values or len(values)>1000000 or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in values):
        raise ValueError('Expected bounded finite nonnegative timings')
    values.sort();n=len(values)
    return dict(count=n,min_ns=values[0],max_ns=values[-1],p50_ns=values[math.ceil(.5*n)-1],
                p95_ns=values[math.ceil(.95*n)-1],p99_ns=values[math.ceil(.99*n)-1],mean_ns=math.fsum(values)/n)

def parity(expected,actual,atol,rtol):
    if len(expected)!=len(actual) or len(expected)==0: raise ValueError('Parity shape mismatch')
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol<0 or rtol<0: raise ValueError('Bad parity tolerance')
    largest=0.
    for e,a in zip(expected,actual):
        if not math.isfinite(e) or not math.isfinite(a) or abs(a-e)>atol+rtol*abs(e):
            raise ValueError('Runtime parity failed')
        largest=max(largest,abs(a-e))
    return largest

def dimensions(windows,tokens,features):
    if any(type(x) is not int for x in (windows,tokens,features)) or not (1<=windows<=1024 and 1<=tokens<=1024 and 1<=features<=64) or windows*tokens*features*4+20>LIMIT:
        raise ValueError('Invalid benchmark dimensions')

def input_bytes(values,windows,tokens,features):
    dimensions(windows,tokens,features)
    if len(values)!=windows*tokens*features or not all(math.isfinite(x) for x in values):
        raise ValueError('Invalid benchmark input')
    return MAGIC+struct.pack('<III',windows,tokens,features)+struct.pack('<'+'f'*len(values),*values)

def read_input(blob):
    if len(blob)<20 or len(blob)>LIMIT or blob[:8]!=MAGIC: raise ValueError('Invalid input header')
    windows,tokens,features=struct.unpack_from('<III',blob,8);dimensions(windows,tokens,features)
    count=windows*tokens*features
    if len(blob)!=20+4*count: raise ValueError('Input length mismatch')
    values=list(struct.unpack_from('<'+'f'*count,blob,20))
    if not all(math.isfinite(x) for x in values): raise ValueError('Nonfinite input')
    return windows,tokens,features,values

def peak_rss_bytes():
    # ru_maxrss is a lifetime high-water mark. Windows deliberately returns null.
    if sys.platform not in ('darwin','linux'): return None
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
