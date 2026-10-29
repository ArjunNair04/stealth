#!/usr/bin/env python3
"""Compare the preserved native binary and one candidate, without sampling."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
from runtime_common import bounded,canonical,digest,summarize
from native_profile import accept_comparison
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'benchmarks/ettm1/attention_tiling_v1.json')
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    for name in ('baseline','inputs','models','output'): setattr(args,name,getattr(args,name).resolve())
    if not args.output.is_relative_to(ROOT/'local'): raise ValueError('Use ignored local/ output')
    env={**os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}}
    def run(command):return subprocess.run([str(x) for x in command],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    if run(['git','status','--porcelain']).stdout.strip():raise ValueError('Commit the candidate and protocol before timing')
    protocol_path=args.protocol.resolve()
    protocol=json.loads(bounded(protocol_path))
    original=json.loads(bounded(ROOT/protocol.get('baseline_results','docs/evidence/runtime-v1/results.json')))
    if digest(bounded(args.baseline))!=protocol['baseline_binary_sha256']:raise ValueError('Wrong baseline binary')
    if digest(bounded(args.inputs/'input.bin'))!=original['input_sha256']:raise ValueError('Wrong inputs')
    for seed in protocol['seeds']:
        if digest(bounded(args.models/f'seed{seed}'/'model.stlf'))!=original['model_sha256'][str(seed)]:raise ValueError('Wrong weights')
        if digest(bounded(args.inputs/f'expected-{seed}.bin'))!=original['preparation'][str(seed)]['expected_sha256']:raise ValueError('Wrong expected outputs')
    args.output.mkdir(parents=True,exist_ok=False)
    configure=run(['cmake','--preset','release','-DSTEALTH_BUILD_BENCHMARKS=ON','-DSTEALTH_ENABLE_SANITIZERS=OFF','-DSTEALTH_WARNINGS_AS_ERRORS=ON'])
    build=run(['cmake','--build','--preset','release'])
    (args.output/'build.log').write_text(configure.stdout+configure.stderr+build.stdout+build.stderr)
    candidate=ROOT/'build/release'/('stealth_runtime_bench.exe' if os.name=='nt' else 'stealth_runtime_bench')
    cli=candidate.parent/('stealth.exe' if os.name=='nt' else 'stealth')
    record=dict(schema_version=1,experiment_id=protocol['experiment_id'],code_commit=run(['git','rev-parse','HEAD']).stdout.strip(),
                started_at=datetime.now(timezone.utc).isoformat(),protocol=protocol,protocol_sha256=digest(bounded(protocol_path)),
                baseline_binary_sha256=digest(bounded(args.baseline)),candidate_binary_sha256=digest(bounded(candidate)),
                input_sha256=original['input_sha256'],model_sha256=original['model_sha256'],
                build_info=run([cli,'--build-info']).stdout,thread_env={k:v for k,v in env.items() if k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
    (args.output/'started.json').write_bytes(canonical(record));records=[]
    try:
        for trial in range(1,4):
            for si,seed in enumerate(protocol['seeds']):
                for wi,work in enumerate(protocol['workloads']):
                    order=['baseline','candidate']
                    if (trial-1+si+wi)%2:order.reverse()
                    for implementation in order:
                        binary=args.baseline if implementation=='baseline' else candidate
                        result=run([binary,'--model',args.models/f'seed{seed}'/'model.stlf','--input',args.inputs/'input.bin',
                            '--expected',args.inputs/f'expected-{seed}.bin','--batch',work['batch'],'--samples',work['samples'],'--warmup',protocol['warmup_calls']])
                        raw=json.loads(result.stdout)
                        if raw['batch']!=work['batch'] or len(raw['times_ns'])!=work['samples']:raise ValueError('Wrong worker result')
                        key=f'trial{trial}-seed{seed}-batch{work["batch"]}-{implementation}'
                        (args.output/f'{key}.json').write_bytes(canonical(raw))
                        compact={k:v for k,v in raw.items() if k not in ('times_ns','clock_times_ns')}
                        compact.update(implementation=implementation,seed=seed,trial=trial,latency=summarize(raw['times_ns']),clock=summarize(raw['clock_times_ns']),record_sha256=digest(canonical(raw)))
                        records.append(compact);print(key,round(compact['latency']['p50_ns']/1e6,4),'ms',flush=True)
        final={**record,'records':records,'performance_gate':accept_comparison(records),'completed_at':datetime.now(timezone.utc).isoformat()}
        (args.output/'results.json').write_bytes(canonical(final))
        (args.output/'complete.json').write_bytes(canonical(dict(files={p.name:digest(bounded(p)) for p in sorted(args.output.iterdir()) if p.is_file()})))
    except Exception as error:
        (args.output/'failed.json').write_bytes(canonical(dict(error=str(error))))
        if isinstance(error,subprocess.CalledProcessError):(args.output/'worker-failure.log').write_text((error.stdout or '')+(error.stderr or ''))
        raise

if __name__=='__main__':main()
