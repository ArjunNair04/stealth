#!/usr/bin/env python3
"""Frozen all-versus-latest benchmark. Immutable outputs; no dataset download."""
import argparse
import csv
from datetime import datetime,timezone
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from runtime_common import bounded,canonical,digest,input_bytes,summarize
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'benchmarks/ettm1/latest_v1.json'
THREADS={k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}

def command(args,env):
    return subprocess.run([str(x) for x in args],cwd=ROOT,env=env,check=True,capture_output=True,text=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.runs=a.runs.resolve()
    a.native=ROOT/'build/release'/('stealth_runtime_bench.exe' if os.name=='nt' else 'stealth_runtime_bench')
    if not a.output.is_relative_to(ROOT/'local'): raise ValueError('Outputs belong under ignored local/')
    env={**os.environ,**THREADS}
    if command(['git','status','--porcelain'],env).stdout.strip(): raise ValueError('Commit the protocol and code before measuring')
    commit=command(['git','rev-parse','HEAD'],env).stdout.strip()
    protocol=json.loads(bounded(PROTOCOL));prior=json.loads(bounded(ROOT/'docs/evidence/correction-v1/results.json'))
    raw=bounded(a.raw/'ETTm1.csv')
    if digest(raw)!=prior['binding']['data_sha256']: raise ValueError('Wrong data snapshot')
    # Source hash covers the file; numeric features are parsed only through validation.
    reader=csv.reader(raw.decode().splitlines());header=next(reader)
    if header!=['date','HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']: raise ValueError('Wrong feature order')
    ends=protocol['window_ends'];context=protocol['context'];features=protocol['features']
    if len(ends)!=protocol['windows'] or not all(context<=end<=46080 for end in ends): raise ValueError('Invalid frozen validation windows')
    rows=[list(map(float,row[1:])) for row in itertools.islice(reader,max(ends))]
    if len(rows)!=max(ends) or any(len(row)!=features for row in rows): raise ValueError('Missing input rows')
    payload=input_bytes([x for end in ends for row in rows[end-context:end] for x in row],len(ends),context,features)
    model_hashes={}
    for seed in protocol['seeds']:
        sha=digest(bounded(a.runs/f'seed{seed}'/'model.stlf'))
        if sha!=prior['models'][str(seed)]['model_sha256']: raise ValueError('Saved selected model changed')
        model_hashes[str(seed)]=sha
    extension=json.loads(command([sys.executable,ROOT/'tools/build_identity.py','--check-extension'],env).stdout)
    a.output.mkdir(parents=True,exist_ok=False)
    # Rebuild from the clean recorded tree before any measurement.
    configured=command(['cmake','--preset','release','-DSTEALTH_BUILD_BENCHMARKS=ON','-DSTEALTH_WARNINGS_AS_ERRORS=ON'],env)
    built=command(['cmake','--build','--preset','release'],env)
    (a.output/'build.log').write_text(configured.stdout+configured.stderr+built.stdout+built.stderr)
    cli=a.native.parent/('stealth.exe' if os.name=='nt' else 'stealth')
    build_info=command([cli,'--build-info'],env).stdout
    if 'configuration: Release' not in build_info: raise ValueError('Require Release build')
    if 'source_fingerprint: '+extension['build_info']['source_fingerprint'] not in build_info:
        raise ValueError('Native and Python source identities differ')
    compile_entries=json.loads((a.native.parent/'compile_commands.json').read_text())
    flags={}
    for item in compile_entries:
        if Path(item['file']).name in ('numeric.cpp','runtime_benchmark.cpp'):
            flags[Path(item['file']).name]=item['command'].replace(str(ROOT),'<repo>')
    (a.output/'input.bin').write_bytes(payload)
    cpu=platform.processor()
    if sys.platform=='darwin': cpu=command(['sysctl','-n','machdep.cpu.brand_string'],env).stdout.strip()
    elif sys.platform.startswith('linux'):
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'): cpu=line.split(':',1)[1].strip();break
    source_paths=['bench/runtime_benchmark.cpp','tools/runtime_common.py','tools/build_identity.py','tools/runtime_torch.py','tools/runtime_worker.py','tools/benchmark_latest.py','CMakeLists.txt']+sorted(str(x.relative_to(ROOT)) for x in (ROOT/'src').glob('*.cpp'))+sorted(str(x.relative_to(ROOT)) for x in (ROOT/'include').rglob('*.hpp'))+['bindings/module.cpp']
    started=dict(schema_version=1,experiment_id=protocol['experiment_id'],code_commit=commit,started_at=datetime.now(timezone.utc).isoformat(),
        extension_identity=extension,protocol_sha256=digest(bounded(PROTOCOL)),data_sha256=digest(raw),input_sha256=digest(payload),model_sha256=model_hashes,
        native_binary_sha256=digest(bounded(a.native)),source_sha256={p:digest(bounded(ROOT/p)) for p in source_paths},
        build=dict(info=build_info,compile_commands=flags),
        environment=dict(cpu=cpu,system=platform.system(),release=platform.release(),machine=platform.machine(),python=platform.python_version(),thread_env=THREADS),
        protocol=protocol,reserved_rows_used=0)
    (a.output/'started.json').write_bytes(canonical(started))
    try:
        for seed in protocol['seeds']:
            result=command([sys.executable,ROOT/'tools/runtime_worker.py','--engine','torch_reference','--model',a.runs/f'seed{seed}'/'model.stlf','--input',a.output/'input.bin','--expected',a.output/f'expected-{seed}.bin','--prepare'],env)
            (a.output/f'prepare-{seed}.json').write_bytes(canonical(json.loads(result.stdout)))
        records=[]
        for trial in range(protocol['trials']):
            for si,seed in enumerate(protocol['seeds']):
                for wi,workload in enumerate(protocol['workloads']):
                    cells=list(itertools.product(protocol['engines'],protocol['output_modes']))
                    rotation=(trial+si+wi)%len(cells)
                    for engine,mode in cells[rotation:]+cells[:rotation]:
                        key=f'trial{trial+1}-seed{seed}-batch{workload["batch"]}-{engine}-{mode}'
                        base=([a.native] if engine=='native_core' else [sys.executable,ROOT/'tools/runtime_worker.py','--engine',engine])
                        args=base+['--model',a.runs/f'seed{seed}'/'model.stlf','--input',a.output/'input.bin','--expected',a.output/f'expected-{seed}.bin','--batch',workload['batch'],'--samples',workload['samples'],'--warmup',protocol['warmup_calls'],'--output',mode]
                        wall=time.perf_counter();result=command(args,env);wall=time.perf_counter()-wall
                        r=json.loads(result.stdout)
                        if r['engine']!=engine or r['output_mode']!=mode or r['batch']!=workload['batch'] or len(r['times_ns'])!=workload['samples']: raise ValueError('Worker result mismatch')
                        (a.output/f'{key}.json').write_bytes(canonical(r))
                        compact={k:v for k,v in r.items() if k not in ('times_ns','clock_times_ns')}
                        compact.update(seed=seed,trial=trial+1,latency=summarize(r['times_ns']),clock=summarize(r['clock_times_ns']),record_sha256=digest(canonical(r)))
                        records.append(compact)
                        print(json.dumps(dict(cell=key,p50_us=compact['latency']['p50_ns']/1000,worker_wall_seconds=round(wall,2))),flush=True)
        results={**started,'completed_at':datetime.now(timezone.utc).isoformat(),'records':records,'preparation':{str(s):json.loads(bounded(a.output/f'prepare-{s}.json')) for s in protocol['seeds']}}
        (a.output/'results.json').write_bytes(canonical(results))
        hashes={p.name:digest(bounded(p)) for p in sorted(a.output.iterdir()) if p.is_file()}
        (a.output/'complete.json').write_bytes(canonical(dict(schema_version=1,files=hashes)))
    except Exception as error:
        if isinstance(error,subprocess.CalledProcessError):
            (a.output/'worker-failure.log').write_text((error.stdout or '')+(error.stderr or ''))
        (a.output/'failed.json').write_bytes(canonical(dict(error_type=type(error).__name__,message=str(error))))
        raise

if __name__=='__main__': main()
