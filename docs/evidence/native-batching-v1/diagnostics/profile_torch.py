"""Operator attribution only; profiled durations are not benchmark latencies."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent/'source'
sys.path.insert(0,str(ROOT/'tools'))
from runtime_torch import load_export
from runtime_common import read_input
base=Path(__file__).resolve().parent
_,tokens,features,values=read_input((base/'artifacts/input.bin').read_bytes())
x=np.array(values,np.float32).reshape(-1,tokens,features)
model=load_export((base/'artifacts/model-7.stlf').read_bytes(),standard=True)
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
records=[]
with torch.inference_mode():
 for batch in (1,8):
  value=torch.from_numpy(x[:batch])
  for _ in range(64): model(value,latest_only=True)
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as profile:
   for _ in range(100): model(value,latest_only=True)
  rows=[dict(key=r.key,count=r.count,self_cpu_time_total_us=r.self_cpu_time_total,cpu_time_total_us=r.cpu_time_total,input_shapes=r.input_shapes) for r in profile.key_averages(group_by_input_shape=True)]
  records.append(dict(batch=batch,iterations=100,operators=sorted(rows,key=lambda r:-r['self_cpu_time_total_us'])))
print(json.dumps(dict(torch=torch.__version__,threads=1,scope='diagnostic profiler attribution only',records=records)))
