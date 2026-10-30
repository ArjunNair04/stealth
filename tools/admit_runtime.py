#!/usr/bin/env python3
"""Bounded correctness admission for one saved model/input/engine cell.

No latency acceptance or tuning. POSIX supervisor kills the process group on
expiry, keeps stdout/stderr and immutable input copies, and never writes a success
seal after a mismatch, export error, compiler failure or timeout.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from runtime_common import bounded,canonical,digest,read_input

ROOT=Path(__file__).resolve().parents[1]
ENGINES=('torch_eager','torch_compiled_inductor','onnxruntime_cpu')
THREADS={k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS')}
SOURCES=('admit_runtime.py','runtime_admission_worker.py','runtime_adapters.py',
         'runtime_torch.py','torch_forecaster.py','runtime_common.py')


class AdmissionFailure(RuntimeError):
    pass


def write(path,value):
    with Path(path).open('xb') as target:target.write(canonical(value))


def supervise(command,output,*,timeout,env):
    if os.name!='posix':raise RuntimeError('Compiler process-group supervision requires a POSIX host')
    started=time.perf_counter()
    with (output/'stdout.log').open('xb') as out,(output/'stderr.log').open('xb') as err:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
        timed_out=False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
        except BaseException:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
            raise
    return dict(returncode=process.returncode,timed_out=timed_out,wall_seconds=time.perf_counter()-started)


def run_cell(output,*,engine,model,inputs,expected,batch,timeout=180):
    if (engine not in ENGINES or type(batch) is not int or not 1<=batch<=1024 or
            type(timeout) not in (int,float) or not math.isfinite(timeout) or not 0<timeout<=180):
        raise ValueError('Invalid engine, fixed batch or bounded admission deadline')
    if os.name!='posix':raise RuntimeError('Admission supervision requires a POSIX host')
    copies={name:bounded(path) for name,path in
            (('model.stlf',model),('input.bin',inputs),('expected.bin',expected))}
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    binding={name:digest(blob) for name,blob in copies.items()}
    started=dict(schema_version=1,purpose='correctness_admission_only',engine=engine,batch=batch,
                 timeout_seconds=timeout,inputs_sha256=binding,
                 source_sha256={name:digest(bounded(ROOT/'tools'/name)) for name in SOURCES},
                 thread_environment=THREADS,cache_policy='fresh_cell_directory')
    write(output/'started.json',started)
    try:
        for name,blob in copies.items():
            with (output/name).open('xb') as target:target.write(blob)
        env={**os.environ,**THREADS,'TORCHINDUCTOR_CACHE_DIR':str(output/'inductor-cache'),
             'TORCHINDUCTOR_FORCE_DISABLE_CACHES':'1'}
        for name in ('TORCH_COMPILE_DISABLE','TORCHDYNAMO_DISABLE','TORCHDYNAMO_SUPPRESS_ERRORS'):
            env.pop(name,None)
        command=[sys.executable,str(ROOT/'tools/runtime_admission_worker.py'),'--engine',engine,
                 '--cell',str(output),'--batch',str(batch)]
        process=supervise(command,output,timeout=timeout,env=env)
        write(output/'process.json',process)
        if process['timed_out']:raise AdmissionFailure('admission timeout')
        if process['returncode']!=0:raise AdmissionFailure('worker failure; see stderr.log')
        result=json.loads(bounded(output/'stdout.log'))
        if (result.get('status')!='admitted' or result.get('engine')!=engine or result.get('batch')!=batch or
                result.get('inputs_sha256')!=binding or type(result.get('windows_checked')) is not int or
                result['windows_checked']!=read_input(copies['input.bin'])[0] or
                type(result.get('max_abs_difference')) not in (int,float) or
                not math.isfinite(result['max_abs_difference']) or result['max_abs_difference']<0):
            raise AdmissionFailure('worker result contract mismatch')
        write(output/'result.json',result)
        names=['started.json','process.json','result.json','stdout.log','stderr.log',*copies]
        if engine=='onnxruntime_cpu':names+=['onnx/model.onnx','onnx/optimized.onnx']
        files={name:digest(bounded(output/name)) for name in names}
        write(output/'complete.json',dict(schema_version=1,status='admitted',files=files))
        return result
    except Exception as exc:
        write(output/'failed.json',dict(status='failed',error_type=type(exc).__name__,message=str(exc)))
        if isinstance(exc,AdmissionFailure):raise
        raise AdmissionFailure('admission failed; see failed.json') from exc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',choices=ENGINES,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--expected',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,required=True)
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT/'local'):
        raise ValueError('Keep admission outputs under ignored local/')
    print(json.dumps(run_cell(a.output,engine=a.engine,model=a.model,inputs=a.input,
                              expected=a.expected,batch=a.batch),indent=2))


if __name__=='__main__':main()
