import json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path.cwd()/'tools'))
import bike_size as sizing
import train_bike_demand as bike
import bike_demand as data
from train_btc_returns import tensors,native_predictions
from train_ettm1 import predictions,write,file_hash
from torch_forecaster import Forecaster,Config
from training_state import bounded
import stealth_cpu as native

torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
protocol,base=sizing.read_recipe();frame=bike.load_frame(Path('local/datasets/bike-v1/hour.csv'),base)
samples=data.split(frame,48,data.hour(base['split_ends']['validation']),data.hour(base['split_ends']['test']))
x,_=tensors(samples)
output=Path('local/bike-size-v1-diagnostic');output.mkdir(exist_ok=False)
reports={}
for seed in base['seeds']:
 blob=bounded(Path(f'local/bike-size-v1/width-256/seed-{seed}/model.stlf'),16*1024*1024)
 c=Config(**sizing.child_recipe(base,256)['model'])
 model=Forecaster.from_flat(c,np.frombuffer(blob[64:-32],dtype='<f4').copy())
 expected=predictions(model,x,32);actual=native_predictions(blob,samples,32)
 diff=np.abs(actual.astype(np.float64)-expected.astype(np.float64));limit=5e-5+2e-4*np.abs(expected.astype(np.float64))
 bad=np.flatnonzero(diff>limit)
 np.savez(output/f'seed-{seed}.npz',expected=expected,actual=actual,actual_rentals=samples.actual,target_hours=samples.target_hours)
 errors=[]
 for i in bad:
  raw=np.array(samples.x[i],dtype=np.float32,copy=True,order='C')
  scalar=native.ForecastModel.from_bytes(blob,kernel='reference').session().predict_last(raw)
  packed=native.ForecastModel.from_bytes(blob,kernel='packed').session().predict_last(raw)
  with torch.inference_mode(): torch_single=float(model(torch.from_numpy(raw[None]))[0,-1])
  errors.append(dict(index=int(i),hour=data.label(samples.target_hours[i]),torch_batch32=float(expected[i]),
                     torch_batch1=torch_single,native_packed=float(packed),native_reference=float(scalar),
                     difference=float(diff[i]),tolerance=float(limit[i]),actual_rentals=float(samples.actual[i])))
 row=dict(seed=seed,model_sha256=sizing.digest(blob),violations=len(bad),windows=len(actual),
          max_absolute_error=float(diff.max()),max_tolerance_ratio=float((diff/limit).max()),errors=errors,
          native_scores=data.scores(actual,samples.actual),torch_scores=data.scores(expected,samples.actual),
          negative_predictions=int((actual<0).sum()))
 reports[str(seed)]=row;write(output/f'seed-{seed}.json',row)
 print(json.dumps(row),flush=True)
write(output/'summary.json',reports)
