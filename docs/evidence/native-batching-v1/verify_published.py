"""Verify published raw timings and independently recompute acceptance (stdlib only).

Run: python3 docs/evidence/native-batching-v1/verify_published.py
The complete source logs stay private. Each worker payload is retained verbatim
and checked against both its original completion seal and the public manifest.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def read(path):
    return json.loads(path.read_text())

def unpack(path, seal):
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        name, payload = row['name'], row['payload']
        require(name not in result, 'Duplicate worker')
        require(digest(payload.encode()) == row['sha256'] == seal[name], 'Worker hash mismatch')
        result[name] = json.loads(payload)
    return result

def quantile(values, fraction):
    return sorted(values)[math.ceil(len(values) * fraction) - 1]

def verify_run(label):
    base = ROOT / label
    seal = read(base/'complete.json')['files']
    workers = unpack(base/'workers.jsonl', seal)
    published = read(base/'verification.json')
    require(digest((base/'complete.json').read_bytes()) == published['completion_sha256'], 'Completion hash mismatch')
    require(len(workers) == 198, 'Missing raw workers')
    primary = {}
    calls = 0
    maximum = 0
    for trial, seed, batch, engine in itertools.product((1,2,3),(7,19,43),(1,2,4,8,16),('baseline','candidate','native_python','torch_standard')):
        name = f'trial{trial}-seed{seed}-batch{batch}-{engine}.json'
        row = workers[name]
        values = row['times_ns']
        require(row['batch'] == batch and row['output_mode'] == 'last', 'Worker shape mismatch')
        require(len(values) == {1:1000,2:500,4:300,8:200,16:100}[batch], 'Sample count mismatch')
        require(all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in values), 'Invalid timing')
        maximum = max(maximum, row['max_abs_difference'])
        calls += len(values)
        primary[trial,seed,batch,engine] = row
        if engine == 'torch_standard':
            v = row['versions']
            require(v['threads'] == v['interop_threads'] == 1 and v['inference_mode'] and v['deterministic'], 'Torch settings changed')
    summary = []
    paired = []
    for batch in (1,2,4,8,16):
        reductions, tails, per_seed = [], [], {}
        for seed in (7,19,43):
            s = []
            for trial in (1,2,3):
                old, new = (primary[trial,seed,batch,e] for e in ('baseline','candidate'))
                require(old['checksum'] == new['checksum'], 'Native checksum changed')
                reduction = 1 - quantile(new['times_ns'],.5) / quantile(old['times_ns'],.5)
                s.append(reduction)
                reductions.append(reduction)
                tails.append(quantile(new['times_ns'],.99) / quantile(old['times_ns'],.99) - 1)
            per_seed[str(seed)] = statistics.median(s)
        paired.append(dict(batch=batch,median_paired_p50_reduction=statistics.median(reductions),median_paired_p99_increase=statistics.median(tails),per_seed_median_reduction=per_seed))
        for engine in ('baseline','candidate','native_python','torch_standard'):
            values = [primary[t,s,batch,engine]['times_ns'] for t,s in itertools.product((1,2,3),(7,19,43))]
            summary.append(dict(batch=batch,engine=engine,p50_us=statistics.median(quantile(v,.5) for v in values)/1000,p99_us=statistics.median(quantile(v,.99) for v in values)/1000))
    full = []
    for seed,batch,engine in itertools.product((7,19,43),(1,8),('baseline','candidate','portable')):
        row = workers[f'full-seed{seed}-batch{batch}-{engine}.json']
        require(row['batch'] == batch and row['output_mode'] == 'all' and len(row['times_ns']) == 50, 'Full-output contract mismatch')
        full.append(dict(seed=seed,batch=batch,engine=engine,max_abs_difference=row['max_abs_difference'],p50_us=quantile(row['times_ns'],.5)/1000,p99_us=quantile(row['times_ns'],.99)/1000))
    for seed,batch in itertools.product((7,19,43),(1,8)):
        require(workers[f'full-seed{seed}-batch{batch}-baseline.json']['checksum'] == workers[f'full-seed{seed}-batch{batch}-candidate.json']['checksum'], 'Full-output checksum mismatch')
    single, eight = paired[0], paired[3]
    gates = dict(batch8_improves_at_least_5_percent=eight['median_paired_p50_reduction']>=.05,batch1_regresses_at_most_5_percent=single['median_paired_p50_reduction']>=-.05,primary_and_single_median_p99_regress_at_most_10_percent=max(single['median_paired_p99_increase'],eight['median_paired_p99_increase'])<=.10,every_seed_batch8_improves=all(v>0 for v in eight['per_seed_median_reduction'].values()))
    for key,value in dict(summary=summary,paired=paired,full_output_diagnostics=full,timed_calls=calls,max_latest_reference_error=maximum,acceptance=gates,accepted=all(gates.values())).items():
        require(published[key] == value, 'Published result differs: '+key)
    require(published['accepted'] == (label == 'accepted'), 'Decision differs from label')
    print(label, '198 workers verified;', calls, 'primary calls; accepted =', all(gates.values()))

if __name__ == '__main__':
    manifest = read(ROOT/'manifest.json')
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
    require(set(manifest) == actual, 'Public manifest file set differs')
    for name,expected in manifest.items():
        require(digest((ROOT/name).read_bytes()) == expected, 'Public hash mismatch: '+name)
    for label in ('rejected','accepted'):
        verify_run(label)
    d = ROOT/'diagnostics'
    ablations = unpack(d/'ablations-workers.jsonl',read(d/'ablations-complete.json'))
    require(len(ablations) == 36, 'Ablation worker count changed')
    print('Public manifest and all 36 ablation workers verified.')
