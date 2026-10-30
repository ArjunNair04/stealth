"""One frozen Beaker portability job. No training data download or package changes."""
import hashlib,json,os,platform,shutil,statistics,subprocess,sys,tarfile,time,traceback,zipfile
from pathlib import Path
BASE=Path(__file__).resolve().parent
OUT=BASE/'results'
OUT.mkdir(exist_ok=False)
ROOT=BASE.parent/'linux-v3/source'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
records=[]
def run(name,args,cwd=ROOT,timeout=900,env=None):
    start=time.monotonic()
    with (OUT/(name+'.log')).open('x') as out:
        try:
            p=subprocess.run([str(x) for x in args],cwd=cwd,stdout=out,stderr=subprocess.STDOUT,timeout=timeout,env=env)
            code=p.returncode
        except subprocess.TimeoutExpired: code=124
    r=dict(name=name,args=[str(x) for x in args],returncode=code,wall_seconds=time.monotonic()-start)
    records.append(r);save('steps.json',records);print(json.dumps(r),flush=True)
    return code==0
try:
    manifest=json.loads((BASE/'artifact-manifest.json').read_text())
    for name,h in manifest.items():
        if sha(BASE/name)!=h: raise ValueError('Transferred artifact hash mismatch: '+name)
    # Resume only generated builds after the previous job terminates; every source
    # byte must still match the original immutable commit archive.
    if not (BASE.parent/'linux-v3/results/failed.json').is_file():
        raise ValueError('Prior validation must have terminated with a retained failure')
    with tarfile.open(BASE/'source.tar.gz') as t:
        for m in t.getmembers():
            if not (ROOT/m.name).resolve().is_relative_to(ROOT) or m.issym() or m.islnk(): raise ValueError('Unsafe archive member')
            if m.isfile() and t.extractfile(m).read()!=(ROOT/m.name).read_bytes():
                raise ValueError('Changed source archive member: '+m.name)
    for line in (BASE/'toolchain-files.sha256').read_text().splitlines():
        digest,name=line.split(None,1)
        if sha(name.strip())!=digest: raise ValueError('Changed toolchain file')
    with zipfile.ZipFile(next((BASE/'wheels').glob('pybind11-*.whl'))) as z: z.extractall(BASE/'deps')
    affinity=sorted(os.sched_getaffinity(0))
    save('environment.json',dict(platform=platform.platform(),python=sys.version,cpuinfo=Path('/proc/cpuinfo').read_text(),affinity=affinity,job_id=os.environ.get('JOB_ID'),threads={k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},source_manifest=manifest,protocol=json.loads((BASE/'protocol.json').read_text())))
    run('toolchain',['bash','-c','g++ --version; cmake --version; ninja --version; uname -srmo; lscpu; cat /proc/loadavg; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true'])
    run('python-environment',[sys.executable,'-m','pip','freeze','--all'])
    configurations={}
    for name in ('dev','release','portable','sanitize'):
        args=['cmake','-S','.', '-B','build/'+name,'-G','Ninja','-DCMAKE_BUILD_TYPE='+('Debug' if name in ('dev','sanitize') else 'Release'),'-DSTEALTH_WARNINGS_AS_ERRORS=ON','-DSTEALTH_BUILD_BENCHMARKS=ON']
        if name=='portable': args+=['-DSTEALTH_ENABLE_AVX=OFF']
        if name=='sanitize': args+=['-DSTEALTH_ENABLE_SANITIZERS=ON']
        if name=='release': args+=['-DSTEALTH_BUILD_PYTHON=ON','-DPython_EXECUTABLE='+sys.executable,'-Dpybind11_DIR='+str(BASE/'deps/pybind11/share/cmake/pybind11')]
        ok=run(name+'-configure',args)
        if ok: ok=run(name+'-build',['cmake','--build','build/'+name,'--parallel','1'],timeout=1500)
        if ok: ok=run(name+'-test',['ctest','--test-dir','build/'+name,'--output-on-failure'],timeout=900)
        configurations[name]=ok
    if configurations['release']:
        site=BASE/'python-site';shutil.copytree(ROOT/'python/stealth_cpu',site/'stealth_cpu')
        install=run('wrapper-install',['cmake','--install','build/release','--prefix',site,'--component','python'])
        os.environ['PYTHONPATH']=str(site)
        run('identity',[sys.executable,'tools/build_identity.py','--check-extension'])
        run('offline-tests',[sys.executable,'-m','unittest','discover','-s','tests','-p','test_*.py','-v'])
        run('wrapper-tests',[sys.executable,'-m','unittest','discover','-s','tests/python','-v'])
        run('training-tests',[sys.executable,'-m','unittest','discover','-s','tests/training','-v'])
    gates=all(configurations.values()) and all(r['returncode']==0 for r in records)
    save('validation.json',dict(status='passed' if gates else 'failed',configurations=configurations,steps=records))
    if not gates: raise RuntimeError('Validation gate failed; no acceptance timing collected')
    # Freeze binary identities and host details before the first timed worker.
    binaries={name:sha(ROOT/'build'/build/'stealth_runtime_bench') for name,build in [('native_avx','release'),('native_portable','portable')]}
    save('timing-started.json',dict(binaries=binaries,affinity_before=affinity,pinned_cpu=affinity[0],load=Path('/proc/loadavg').read_text()))
    sys.path.insert(0,str(ROOT/'tools'));from runtime_common import summarize
    protocol=json.loads((BASE/'protocol.json').read_text());timings=[]
    for trial in range(3):
        for si,seed in enumerate(protocol['seeds']):
            for bi,batch in enumerate(protocol['batches']):
                engines=protocol['engines'];rotation=(trial+si+bi)%len(engines)
                for engine in engines[rotation:]+engines[:rotation]:
                    name=f'trial{trial+1}-seed{seed}-batch{batch}-{engine}'
                    if engine in ('native_avx','native_portable'):
                        cmd=[ROOT/'build'/('release' if engine=='native_avx' else 'portable')/'stealth_runtime_bench']
                    else: cmd=[sys.executable,ROOT/'tools/runtime_worker.py','--engine',engine]
                    cmd=['taskset','-c',str(affinity[0])]+cmd+['--model',BASE/f'artifacts/model-{seed}.stlf','--input',BASE/'artifacts/input.bin','--expected',BASE/f'artifacts/expected-{seed}.bin','--batch',str(batch),'--samples',str(protocol['samples'][str(batch)]),'--warmup','64','--output','last']
                    with (OUT/(name+'.json')).open('x') as output,(OUT/(name+'.stderr')).open('x') as err:
                        p=subprocess.run([str(x) for x in cmd],cwd=ROOT,stdout=output,stderr=err,timeout=180)
                    if p.returncode: raise RuntimeError('Worker failed: '+name)
                    result=json.loads((OUT/(name+'.json')).read_text())
                    if len(result['times_ns'])!=protocol['samples'][str(batch)] or result['batch']!=batch or result['output_mode']!='last': raise ValueError('Worker mismatch')
                    timings.append(dict(name=name,trial=trial+1,seed=seed,batch=batch,engine=engine,latency=summarize(result['times_ns']),max_abs_difference=result['max_abs_difference'],sha256=sha(OUT/(name+'.json'))))
                    save('timings.json',timings);print(name,timings[-1]['latency']['p50_ns']/1000,flush=True)
    if len(timings)!=72: raise ValueError('Incomplete grid')
    summary=[]
    for batch in (1,8):
        for engine in protocol['engines']:
            cells=[r for r in timings if r['batch']==batch and r['engine']==engine]
            if len(cells)!=9: raise ValueError('Incomplete engine grid')
            summary.append(dict(engine=engine,batch=batch,p50_us=statistics.median(r['latency']['p50_ns'] for r in cells)/1000,p99_us=statistics.median(r['latency']['p99_ns'] for r in cells)/1000))
    save('summary.json',dict(status='compatibility_measurement_complete',summary=summary,load_after=Path('/proc/loadavg').read_text(),limitations=protocol['scope']))
    save('complete.json',dict(files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
except Exception as e:
    save('failed.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc()))
    raise
