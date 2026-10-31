"""Independently reconcile the sealed Linux batching result tree."""
import hashlib,itertools,json,math,statistics,sys
from pathlib import Path
base=Path(sys.argv[1]);out=Path(sys.argv[2])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
seal=json.loads((base/'complete.json').read_text())['files']
assert set(seal)=={p.name for p in base.iterdir() if p.is_file() and p.name!='complete.json'}
for name,digest in seal.items():assert sha(base/name)==digest,('Hash mismatch',name)
validation=json.loads((base/'validation.json').read_text());assert validation['status']=='passed'
assert all(r['returncode']==0 for r in validation['steps'])
rows=json.loads((base/'timings.json').read_text())
expected=set(itertools.product((1,2,3),(7,19,43),(1,2,4,8,16),('baseline','candidate','native_python','torch_standard')))
keys=[(r['trial'],r['seed'],r['batch'],r['engine']) for r in rows]
assert len(rows)==180 and len(set(keys))==180 and set(keys)==expected
samples={1:1000,2:500,4:300,8:200,16:100}
raw={};maximum=0;count=0
for row,key in zip(rows,keys):
 file=base/(row['name']+'.json');assert sha(file)==row['sha256']
 r=json.loads(file.read_text());raw[key]=r
 a=sorted(r['times_ns']);assert len(a)==samples[row['batch']]
 assert all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0 for v in a)
 assert r['batch']==row['batch'] and r['output_mode']=='last'
 assert r['max_abs_difference']==row['max_abs_difference'] and math.isfinite(r['checksum'])
 maximum=max(maximum,r['max_abs_difference']);count+=len(a)
 for q,name in [(.5,'p50_ns'),(.95,'p95_ns'),(.99,'p99_ns')]:assert a[math.ceil(q*len(a))-1]==row['latency'][name]
 assert row['latency']['min_ns']==a[0] and row['latency']['max_ns']==a[-1]
 if row['engine']=='torch_standard':
  v=r['versions'];assert v['threads']==v['interop_threads']==1 and v['inference_mode'] and v['deterministic']
 if row['engine']=='native_python' and row['batch']>1:
  assert r['workspace_bytes']>r['workspace_bytes_before_first_call']
bykey=dict(zip(keys,rows));comparisons=[]
for batch in (1,2,4,8,16):
 reductions=[];tails=[];seeds={}
 for seed in (7,19,43):
  seed_reductions=[]
  for trial in (1,2,3):
   b=bykey[trial,seed,batch,'baseline']['latency'];c=bykey[trial,seed,batch,'candidate']['latency']
   reduction=1-c['p50_ns']/b['p50_ns'];seed_reductions.append(reduction);reductions.append(reduction)
   tails.append(c['p99_ns']/b['p99_ns']-1)
   assert raw[trial,seed,batch,'baseline']['checksum']==raw[trial,seed,batch,'candidate']['checksum']
  seeds[str(seed)]=statistics.median(seed_reductions)
 comparisons.append(dict(batch=batch,median_paired_p50_reduction=statistics.median(reductions),median_paired_p99_increase=statistics.median(tails),per_seed_median_reduction=seeds))
summary=[]
for batch in (1,2,4,8,16):
 for engine in ('baseline','candidate','native_python','torch_standard'):
  cells=[r for r in rows if r['batch']==batch and r['engine']==engine]
  summary.append(dict(batch=batch,engine=engine,p50_us=statistics.median(r['latency']['p50_ns'] for r in cells)/1000,p99_us=statistics.median(r['latency']['p99_ns'] for r in cells)/1000))
assert summary==json.loads((base/'summary.json').read_text())['summary']
full=[]
for seed,batch,engine in itertools.product((7,19,43),(1,8),('baseline','candidate','portable')):
 file=base/f'full-seed{seed}-batch{batch}-{engine}.json';r=json.loads(file.read_text());a=sorted(r['times_ns'])
 assert r['output_mode']=='all' and r['batch']==batch and len(a)==50
 full.append(dict(seed=seed,batch=batch,engine=engine,max_abs_difference=r['max_abs_difference'],p50_us=a[24]/1000,p99_us=a[49]/1000))
for seed,batch in itertools.product((7,19,43),(1,8)):
 a=json.loads((base/f'full-seed{seed}-batch{batch}-baseline.json').read_text());b=json.loads((base/f'full-seed{seed}-batch{batch}-candidate.json').read_text());assert a['checksum']==b['checksum']
primary=next(r for r in comparisons if r['batch']==8);single=comparisons[0]
gates=dict(batch8_improves_at_least_5_percent=primary['median_paired_p50_reduction']>=.05,
           batch1_regresses_at_most_5_percent=single['median_paired_p50_reduction']>=-.05,
           primary_and_single_median_p99_regress_at_most_10_percent=max(primary['median_paired_p99_increase'],single['median_paired_p99_increase'])<=.10,
           every_seed_batch8_improves=all(v>0 for v in primary['per_seed_median_reduction'].values()))
result=dict(status='verified',files_verified=len(seal),completion_sha256=sha(base/'complete.json'),cells=180,timed_calls=count,max_latest_reference_error=maximum,native_checksums_identical=True,summary=summary,paired=comparisons,acceptance=gates,accepted=all(gates.values()),full_output_diagnostics=full)
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
