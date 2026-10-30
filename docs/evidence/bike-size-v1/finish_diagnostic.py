"""Postmortem of frozen bike_size_v1; preserves rejection and all original files."""
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path.cwd()/'tools'))
import bike_size as study
import bike_demand as data
import train_bike_demand as bike
from training_state import bounded,canonical,digest,load_checkpoint,strict_json
from train_ettm1 import file_hash,write
from train_btc_returns import native_predictions
from runtime_common import input_bytes,summarize
from torch_forecaster import Config,Forecaster
from build_identity import extension_identity

torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
p,base=study.read_recipe()
root=Path('local/bike-size-v1');output=Path('local/bike-size-v1-postmortem')
output.mkdir(exist_ok=False)
frame=bike.load_frame(Path('local/datasets/bike-v1/hour.csv'),base)
test=data.split(frame,48,data.hour(base['split_ends']['validation']),data.hour(base['split_ends']['test']))
validation=data.split(frame,48,data.hour(base['split_ends']['train']),data.hour(base['split_ends']['validation']))
assert (root/'failure.json').exists() and not (root/'complete.json').exists()
assert json.loads((root/'protocol.json').read_bytes())==p
identity=extension_identity();binary=Path('build/release/stealth_runtime_bench').resolve()
build=study.native_build(binary,identity)
inputs={str(q.relative_to(root)):file_hash(q) for q in sorted(root.rglob('*')) if q.is_file()}
write(output/'source-inventory.json',inputs)
models={};results={};ledgers={};complete_controls={}
for width in p['widths']:
 child=root/f'width-{width}';recipe=study.child_recipe(base,width)
 assert json.loads((child/'protocol.json').read_bytes())==recipe
 if (child/'complete.json').exists():bike.verify_run(child)
 else:assert width==256 and (child/'failure.json').exists()
 selected=json.loads((child/'selection.json').read_bytes())
 assert selected['evaluation_started'] is False and set(selected['seeds'])=={'7','19','43'}
 binding=json.loads((child/'started.json').read_bytes())['binding']
 assert binding['width']==width and binding['child_protocol_sha256']==digest(canonical(recipe))
 seeds={};preds={}
 for seed in base['seeds']:
  record=selected['seeds'][str(seed)];history=json.loads((child/f'seed-{seed}/history.json').read_bytes())
  assert history==record['validation_history'] and [r['epoch'] for r in history]==list(range(9))
  epoch=min(history,key=lambda r:r['validation']['mae'])['epoch']
  assert epoch==record['selected_epoch']
  model,_,_,checkpoint=load_checkpoint(child/f'seed-{seed}/epoch-{epoch:03d}',{**binding,'seed':seed})
  blob=bounded(child/f'seed-{seed}/model.stlf',16*1024*1024)
  assert model.export_bytes()==blob and digest(blob)==record['model_sha256']
  assert checkpoint['history']==history[:epoch+1]
  models[width,seed]=model
  if width==256:
   diagnostics=json.loads(Path(f'local/bike-size-v1-diagnostic/seed-{seed}.json').read_bytes())
   assert diagnostics['model_sha256']==digest(blob)
   with np.load(f'local/bike-size-v1-diagnostic/seed-{seed}.npz',allow_pickle=False) as arrays:
    actual=arrays['actual'].copy();expected=arrays['expected'].copy()
    assert np.array_equal(arrays['target_hours'],test.target_hours)
   assert len(actual)==len(test.y) and np.isfinite(actual).all() and np.isfinite(expected).all()
   delta=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
   allowance=base['parity_absolute_tolerance']+base['parity_relative_tolerance']*np.abs(expected.astype(np.float64))
   assert int((delta>allowance).sum())==diagnostics['violations']
   indices=np.unique(np.linspace(0,len(test.y)-1,base['full_trace_parity_windows'],dtype=int))
   raw=np.array(test.x[indices],dtype=np.float32,copy=True,order='C')
   with torch.inference_mode():full=model(torch.from_numpy(raw)).numpy()
   traces={}
   for kernel in ('reference','packed'):
    trace=study.bike.native.ForecastModel.from_bytes(blob,kernel=kernel).session().predict_batch(raw)
    difference=np.abs(trace.astype(np.float64)-full.astype(np.float64))
    allowed=5e-5+2e-4*np.abs(full.astype(np.float64))
    traces[kernel]=dict(passed=bool(np.all(difference<=allowed)),max_absolute_error=float(difference.max()))
   detail={**diagnostics,'full_trace_windows':len(indices),'full_trace_parity':traces}
   write(output/f'diagnostic-seed-{seed}.json',detail)
   seeds[str(seed)]={**record,'test':diagnostics['native_scores'],'negative_predictions':diagnostics['negative_predictions'],
                     'test_parity':detail,'accepted':diagnostics['violations']==0 and all(v['passed'] for v in traces.values())}
  else:
   actual=native_predictions(blob,test,base['evaluation_batch_size'])
   original=json.loads((child/'scores.json').read_bytes())
   seeds[str(seed)]={**original['models'][str(seed)],'accepted':True}
  preds[str(seed)]=actual
 baselines=json.loads((child/'baselines.json').read_bytes())
 ledger=bike.ledger(test,preds,bike.baseline_values(baselines,test))
 (output/f'width-{width}.jsonl').write_bytes(ledger)
 if width!=256:
  assert ledger==bounded(child/'predictions.jsonl',64*1024*1024)
  complete_controls[str(width)]=True
 results[str(width)]=dict(parameters=study.parameter_counts(recipe),models=seeds,
                         accepted=all(s['accepted'] for s in seeds.values()),
                         baselines={k:data.scores(v,test.actual) for k,v in bike.baseline_values(baselines,test).items()},
                         train_samples=6595,validation_samples=len(validation.y),evaluation_samples=len(test.y))
write(output/'accuracy.json',dict(evidence=p['evidence'],original_study_status='failed_numerical_gate',widths=results))
# Prepare all independent Torch references before any native worker starts.
raw=np.array(validation.x[:32],dtype=np.float32,copy=True,order='C')
(output/'input.bin').write_bytes(input_bytes(raw.ravel().tolist(),*raw.shape))
for (width,seed),model in models.items():
 with torch.inference_mode():reference=model(torch.from_numpy(raw)).numpy().astype('<f4').tobytes()
 (output/f'expected-{width}-{seed}.bin').write_bytes(reference)
write(output/'started.json',dict(status='diagnostic_only',original_study_status='failed_numerical_gate',
                                 code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                                 source_sha256=file_hash(Path(__file__)),source_inventory_sha256=file_hash(output/'source-inventory.json'),
                                 protocol_sha256=file_hash(study.PROTOCOL),native=identity,build=build,
                                 binary_sha256=file_hash(binary),input_sha256=file_hash(output/'input.bin'),
                                 completed_width_replays=complete_controls))
records=[]
for trial,seed,width in study.worker_order(p,base['seeds']):
 name=f'trial-{trial}-seed-{seed}-width-{width}'
 command=[str(binary),'--model',str(root/f'width-{width}/seed-{seed}/model.stlf'),
          '--input',str(output/'input.bin'),'--expected',str(output/f'expected-{width}-{seed}.bin'),
          '--batch','1','--output','last','--samples','1000','--warmup','64']
 process=subprocess.run(command,capture_output=True,text=True,timeout=120)
 (output/f'{name}.stderr').write_text(process.stderr)
 process.check_returncode();result=strict_json(process.stdout.encode())
 assert result['engine']=='native_core' and result['batch']==1 and result['output_mode']=='last' and len(result['times_ns'])==1000
 write(output/f'{name}.json',result)
 row={k:v for k,v in result.items() if k not in ('times_ns','clock_times_ns')}
 row.update(width=width,seed=seed,trial=trial,latency=summarize(result['times_ns']),clock=summarize(result['clock_times_ns']),
            raw_sha256=file_hash(output/f'{name}.json'),model_accepted=results[str(width)]['models'][str(seed)]['accepted'])
 records.append(row);print(json.dumps(dict(worker=name,p50_us=row['latency']['p50_ns']/1000)),flush=True)
write(output/'latency.json',dict(status='diagnostic_only',widths=study.aggregate_latency(records,p,base['seeds']),records=records))
# This is an analysis inventory, never a successful training/admission seal.
assert inputs=={str(q.relative_to(root)):file_hash(q) for q in sorted(root.rglob('*')) if q.is_file()}
write(output/'analysis.json',dict(status='diagnostic_finished_original_gate_failed',
                                files={q.name:file_hash(q) for q in sorted(output.iterdir()) if q.is_file()}))
print('Diagnostic finished; original numerical gate remains FAILED',flush=True)
