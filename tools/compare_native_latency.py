"""Compare two preserved native binaries on the frozen local latency protocol."""
import argparse
import itertools
import json
import math
from pathlib import Path
import subprocess
from runtime_common import bounded, canonical, digest, summarize
from native_profile import accept_comparison

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'benchmarks/ettm1/local_latency_v1.json'


def aggregate(records):
    keys = [(r['seed'],r['trial'],r['batch'],r['mode'],r['implementation']) for r in records]
    expected = set(itertools.product((7,19,43),(1,2,3),(1,8),('last','all'),('baseline','candidate')))
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Incomplete or duplicated latency comparison')
    import statistics
    by_key = dict(zip(keys,records))
    for record in records:
        timing = record['latency']
        if (not all(math.isfinite(timing[k]) and timing[k] > 0 for k in ('p50_ns','p99_ns'))
                or timing['p99_ns'] < timing['p50_ns'] or not math.isfinite(record['checksum'])):
            raise ValueError('Invalid latency or output checksum')
    summary = {}
    for mode,batch in itertools.product(('last','all'),(1,8)):
        ratios, tails = [], []
        for seed,trial in itertools.product((7,19,43),(1,2,3)):
            a = by_key[seed,trial,batch,mode,'baseline']
            b = by_key[seed,trial,batch,mode,'candidate']
            if a['checksum'] != b['checksum']:
                raise ValueError('Native output checksum changed')
            ratios.append(b['latency']['p50_ns']/a['latency']['p50_ns'])
            tails.append(b['latency']['p99_ns']/a['latency']['p99_ns'])
        summary[f'{mode}_batch{batch}'] = dict(median_p50_ratio=statistics.median(ratios),
            median_p99_ratio=statistics.median(tails),p50_ratios=ratios,p99_ratios=tails)
    gate = accept_comparison([r for r in records if r['mode']=='last'])
    gate['accepted'] = gate['accepted'] and all(summary[key]['median_p50_ratio'] <= 1.05
        for key in ('last_batch8','all_batch1','all_batch8'))
    return dict(gate=gate,workloads=summary)


def run(baseline,candidate,inputs,models,output,protocol=PROTOCOL):
    paths = {k:Path(v).resolve(strict=True) for k,v in
             dict(baseline=baseline,candidate=candidate,inputs=inputs,models=models).items()}
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'local'):
        raise ValueError('Results belong under ignored local/')
    protocol_path = Path(protocol).resolve(strict=True)
    protocol = json.loads(protocol_path.read_bytes())
    # Do not silently accept a different model/data snapshot under the same protocol.
    previous = json.loads((ROOT/'docs/evidence/latest-v1/runtime.json').read_bytes())
    payload = bounded(paths['inputs']/'input.bin')
    if digest(payload) != previous['input_sha256']:
        raise ValueError('Frozen input hash mismatch')
    files = {str(paths[k]):digest(bounded(paths[k])) for k in ('baseline','candidate')}
    for seed in protocol['seeds']:
        model=paths['models']/f'seed{seed}'/'model.stlf'
        if digest(bounded(model)) != previous['model_sha256'][str(seed)]:
            raise ValueError('Frozen model hash mismatch')
        files[str(model)] = digest(bounded(model))
        expected=paths['inputs']/f'expected-{seed}.bin'
        files[str(expected)] = digest(bounded(expected))
    output.mkdir(parents=True,exist_ok=False)
    started = dict(protocol=protocol,protocol_sha256=digest(protocol_path.read_bytes()),
                   source_sha256=digest(Path(__file__).read_bytes()),files=files,
                   input_sha256=digest(payload),
                   commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()))
    (output/'started.json').write_bytes(canonical(started))
    records=[]
    try:
        for trial,si,bi,mi in itertools.product(range(3),range(3),range(2),range(2)):
            seed,batch,mode=protocol['seeds'][si],protocol['batches'][bi],protocol['modes'][mi]
            order=('baseline','candidate') if (trial+si+bi+mi)%2==0 else ('candidate','baseline')
            for implementation in order:
                key=f'trial{trial+1}-seed{seed}-batch{batch}-{mode}-{implementation}'
                args=[paths[implementation],'--model',paths['models']/f'seed{seed}'/'model.stlf',
                      '--input',paths['inputs']/'input.bin','--expected',paths['inputs']/f'expected-{seed}.bin',
                      '--batch',str(batch),'--output',mode,'--warmup',str(protocol['warmup']),
                      '--samples',str(protocol['samples'][mode][str(batch)])]
                result=subprocess.run([str(x) for x in args],cwd=ROOT,capture_output=True,text=True,timeout=120)
                (output/f'{key}.stderr').write_text(result.stderr)
                if result.returncode:
                    raise RuntimeError(f'{key} failed: '+result.stderr)
                raw=json.loads(result.stdout)
                if (raw['engine']!='native_core' or raw['batch']!=batch or raw['output_mode']!=mode or
                        len(raw['times_ns']) != protocol['samples'][mode][str(batch)]):
                    raise ValueError('Worker result mismatch')
                (output/f'{key}.json').write_bytes(canonical(raw))
                compact={k:v for k,v in raw.items() if k not in ('times_ns','clock_times_ns')}
                compact.update(seed=seed,trial=trial+1,mode=mode,implementation=implementation,
                               latency=summarize(raw['times_ns']),clock=summarize(raw['clock_times_ns']),
                               record_sha256=digest(canonical(raw)))
                records.append(compact)
                print(json.dumps(dict(cell=key,p50_us=compact['latency']['p50_ns']/1000)),flush=True)
        result={**started,**aggregate(records), 'records':records}
        (output/'results.json').write_bytes(canonical(result))
        (output/'complete.json').write_bytes(canonical(dict(files={
            p.name:digest(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()})))
        return result
    except Exception as exc:
        (output/'failed.json').write_bytes(canonical(dict(error=type(exc).__name__,message=str(exc))))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidate','inputs','models','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=PROTOCOL)
    args=p.parse_args()
    result=run(**vars(args))
    print(json.dumps(result['gate']))

if __name__=='__main__':
    main()
