"""Build sdist -> wheel, audit contents, and test an isolated runtime installation.

Run with the declared build dependencies already installed. Runtime installation
uses only the supplied local NumPy wheel and newly built package, with no index.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import venv
import zipfile


def run(args, *, cwd, log):
    env = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTHONHOME'): env.pop(name, None)
    with log.open('w', encoding='utf-8') as f:
        result = subprocess.run([str(x) for x in args], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{log.name} failed with exit code {result.returncode}; inspect the retained log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--numpy-wheel', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    numpy_wheel = args.numpy_wheel.resolve(strict=True)
    if not numpy_wheel.name.startswith('numpy-2.2.6-') or numpy_wheel.suffix != '.whl':
        raise ValueError('Supply the pinned NumPy 2.2.6 wheel for this interpreter/platform')
    output.mkdir(parents=True, exist_ok=False)
    run([sys.executable, '-m', 'build', '--no-isolation', '--outdir', output/'dist', root],
        cwd=output, log=output/'build.log')
    wheels = list((output/'dist').glob('stealth_cpu-*.whl'))
    sdists = list((output/'dist').glob('stealth_cpu-*.tar.gz'))
    if len(wheels)!=1 or len(sdists)!=1: raise RuntimeError('Expected exactly one source and wheel distribution')
    wheel, sdist = wheels[0], sdists[0]
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or not (name.startswith('stealth_cpu/') or name.startswith('stealth_cpu-0.1.0.dist-info/')):
                raise RuntimeError(f'Unexpected wheel member: {name}')
        extensions = [n for n in names if n.startswith('stealth_cpu/_core.') and n.endswith(('.so','.pyd'))]
        if len(extensions)!=1 or 'stealth_cpu/__init__.py' not in names:
            raise RuntimeError('Wheel must contain the native extension and Python interface')
    with tarfile.open(sdist) as archive:
        source_names = [PurePosixPath(member.name) for member in archive.getmembers()]
        for name in source_names:
            if name.is_absolute() or '..' in name.parts or any(part in ('local','build','install','.git','.venv','CMakeUserPresets.json','.env') for part in name.parts):
                raise RuntimeError(f'Unexpected source member: {name}')
        relative = {str(PurePosixPath(*n.parts[1:])) for n in source_names}
        for required in ('CMakeLists.txt','bindings/module.cpp','bindings/signal_index.cpp','src/signal_index.cpp','include/stealth/signal_index.hpp','src/window_stream.cpp','include/stealth/window_stream.hpp','python/stealth_cpu/__init__.py','tools/build_identity.py'):
            if required not in relative: raise RuntimeError(f'Missing source file: {required}')
    runtime = output/'runtime'
    venv.EnvBuilder(with_pip=True).create(runtime)
    python = runtime/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    run([python,'-m','pip','install','--no-index','--no-deps',numpy_wheel,wheel],cwd=output,log=output/'install.log')
    run([python,'-c',"import pathlib,sys,importlib.util,stealth_cpu; p=pathlib.Path(stealth_cpu.__file__).resolve(); assert p.is_relative_to(pathlib.Path(sys.prefix).resolve()); assert all(importlib.util.find_spec(m) is None for m in ('pybind11','scikit_build_core','torch')); print('Runtime-only installation verified')"],cwd=output,log=output/'isolation.log')
    run([python,'-m','unittest','discover','-s',root/'tests/python','-v'],cwd=output,log=output/'tests.log')
    run([python,root/'examples/python_components.py'],cwd=output,log=output/'example.log')
    receipt = {'schema_version':1,'status':'passed','built_wheel_from_sdist':True,
               'runtime_install_network':False,'runtime_dependencies':['numpy==2.2.6'],
               'python':sys.version.split()[0], 'wheel_members':names,
               'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (wheel,sdist,numpy_wheel)}}
    (output/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__': main()
