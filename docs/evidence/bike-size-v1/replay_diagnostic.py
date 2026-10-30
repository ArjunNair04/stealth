"""Fresh-process native replay of every diagnostic evaluation ledger."""
import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path.cwd()/'tools'))
import bike_size as study
import bike_demand as data
from training_state import bounded
from train_btc_returns import native_predictions
from train_ettm1 import write,file_hash

torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
p,base=study.read_recipe();root=Path('local/bike-size-v1');analysis=Path('local/bike-size-v1-postmortem')
output=Path('local/bike-size-v1-postmortem-replay');output.mkdir(exist_ok=False)
manifest=json.loads((analysis/'analysis.json').read_bytes())
if manifest['status']!='diagnostic_finished_original_gate_failed':raise ValueError('Wrong analysis status')
for name,sha in manifest['files'].items():
 if file_hash(analysis/name)!=sha:raise ValueError('Analysis evidence changed')
for name,sha in json.loads((analysis/'source-inventory.json').read_bytes()).items():
 if file_hash(root/name)!=sha:raise ValueError('Training artifact changed')
frame=study.bike.load_frame(Path('local/datasets/bike-v1/hour.csv'),base)
test=data.split(frame,48,data.hour(base['split_ends']['validation']),data.hour(base['split_ends']['test']))
records={}
for width in p['widths']:
 child=root/f'width-{width}'
 preds={str(seed):native_predictions(bounded(child/f'seed-{seed}/model.stlf',16*1024*1024),test,32) for seed in base['seeds']}
 baseline=json.loads((child/'baselines.json').read_bytes())
 ledger=study.bike.ledger(test,preds,study.bike.baseline_values(baseline,test))
 if ledger!=bounded(analysis/f'width-{width}.jsonl',64*1024*1024):raise ValueError('Native replay changed')
 (output/f'width-{width}.jsonl').write_bytes(ledger)
 records[str(width)]=dict(byte_identical=True,sha256=study.digest(ledger),predictions=len(test.y)*3)
write(output/'result.json',dict(status='diagnostic_replay_identical_original_gate_failed',widths=records,
                               source_sha256=file_hash(Path(__file__)),analysis_sha256=file_hash(analysis/'analysis.json'),
                               native=study.extension_identity()))
print(json.dumps(records))
