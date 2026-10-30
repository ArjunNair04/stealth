"""Change only dense accumulation order in the Torch reference on the failed window."""
import json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path.cwd()/'tools'))
import bike_size as study
import bike_demand as data
from torch_forecaster import Config,Forecaster
from train_ettm1 import file_hash,write

torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
p,base=study.read_recipe();frame=study.bike.load_frame(Path('local/datasets/bike-v1/hour.csv'),base)
samples=data.split(frame,48,data.hour(base['split_ends']['validation']),data.hour(base['split_ends']['test']))
blob=Path('local/bike-size-v1/width-256/seed-7/model.stlf').read_bytes()
model=Forecaster.from_flat(Config(**study.child_recipe(base,256)['model']),np.frombuffer(blob[64:-32],dtype='<f4').copy())
raw=np.array(samples.x[2974:2975],dtype=np.float32,copy=True,order='C');x=torch.from_numpy(raw)
def serial_dense(inputs,weights,bias):
 output=bias.expand(*inputs.shape[:-1],weights.shape[0]).clone()
 for i in range(weights.shape[1]):output=output+inputs[...,i,None]*weights[:,i]
 return output
with torch.inference_mode():
 original=float(model(x)[0,-1])
 with patch('torch_forecaster.F.linear',side_effect=serial_dense):ordered=float(model(x)[0,-1])
cpp={kernel:float(study.bike.native.ForecastModel.from_bytes(blob,kernel=kernel).session().predict_last(raw[0]))
     for kernel in ('reference','packed')}
result=dict(source_sha256=file_hash(Path(__file__)),model_sha256=study.digest(blob),
            input_sha256=study.digest(raw.astype('<f4').tobytes()),index=2974,
            dense_change='FP32 bias-first ordered multiply then add, matching the native dense contract; every other Torch operation unchanged',
            torch_standard=original,torch_ordered_dense=ordered,native=cpp,
            ordered_dense_equals_native=ordered==cpp['reference']==cpp['packed'])
write(Path('local/bike-size-v1-postmortem/rounding-probe.json'),result)
print(json.dumps(result))
