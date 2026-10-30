"""Independent local verification of the complete Beaker result grid."""
import hashlib,json,math,re,statistics,sys
from pathlib import Path
p=Path(sys.argv[1]);complete=json.loads((p/'complete.json').read_text())
for n,h in complete['files'].items():
 if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h: raise ValueError('Hash mismatch '+n)
validation=json.loads((p/'validation.json').read_text())
if validation['status']!='passed' or not all(validation['configurations'].values()): raise ValueError('Failed validation')
if any(r['returncode'] for r in validation['steps']): raise ValueError('Failed step')
rows=json.loads((p/'timings.json').read_text());engines=['native_avx','native_portable','native_python','torch_standard']
expected={(trial,seed,batch,engine) for trial in [1,2,3] for seed in [7,19,43] for batch in [1,8] for engine in engines}
actual={(r['trial'],r['seed'],r['batch'],r['engine']) for r in rows}
if len(rows)!=72 or actual!=expected: raise ValueError('Incomplete or duplicated grid')
for r in rows:
 f=p/(r['name']+'.json');value=json.loads(f.read_text())
 if hashlib.sha256(f.read_bytes()).hexdigest()!=r['sha256']: raise ValueError('Cell hash')
 values=sorted(value['times_ns']);n=1000 if r['batch']==1 else 200
 if len(values)!=n or any(type(x)!=int or x<0 for x in values): raise ValueError('Timings')
 for quantile,key in [(.5,'p50_ns'),(.95,'p95_ns'),(.99,'p99_ns')]:
  if values[math.ceil(n*quantile)-1]!=r['latency'][key]: raise ValueError('Quantile')
 if not math.isfinite(value['checksum']) or value['max_abs_difference']!=r['max_abs_difference']: raise ValueError('Output contract')
summary=[]
for batch in [1,8]:
 for engine in engines:
  group=[r for r in rows if r['batch']==batch and r['engine']==engine]
  summary.append(dict(batch=batch,engine=engine,p50_us=statistics.median(r['latency']['p50_ns'] for r in group)/1000,p99_us=statistics.median(r['latency']['p99_ns'] for r in group)/1000,p50_range_us=[min(r['latency']['p50_ns'] for r in group)/1000,max(r['latency']['p50_ns'] for r in group)/1000],max_abs_difference=max(r['max_abs_difference'] for r in group)))
recorded=json.loads((p/'summary.json').read_text())['summary']
for r,s in zip(recorded,summary):
 if any(r[k]!=s[k] for k in r): raise ValueError('Summary mismatch')
print(json.dumps(dict(status='verified',files=len(complete['files']),cells=len(rows),summary=summary),indent=2))
for name in ['dev','release','portable','sanitize']:
 s=(p/(name+'-test.log')).read_text();print(name, re.findall(r'\d+% tests passed, \d+ tests failed out of \d+',s))
for name in ['offline-tests','wrapper-tests','training-tests']:
 s=(p/(name+'.log')).read_text();print(name,re.findall(r'Ran \d+ tests in [^\n]+',s),re.findall(r'^OK[^\n]*',s,re.MULTILINE))
