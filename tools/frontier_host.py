"""Inspect the measurement host and bind its installed packages to wheel files.

This module never installs software or changes CPU/thread settings. The caller
must already be inside the intended environment and CPU allocation.
"""
import email
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import zipfile

from admit_runtime import THREADS
from build_identity import extension_identity, source_fingerprint
from runtime_common import bounded, digest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('benchmark_frontier.py', 'frontier_worker.py', 'frontier_host.py', 'admit_runtime.py',
           'runtime_worker.py', 'runtime_common.py', 'runtime_adapters.py', 'runtime_torch.py',
           'torch_forecaster.py', 'build_identity.py')


def package_name(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def wheel_lock(directory, packages):
    found = {}
    for path in sorted(Path(directory).glob('*.whl')):
        if path.is_symlink():
            raise ValueError('Keep actual wheel files in the wheel archive')
        with zipfile.ZipFile(path) as archive:
            metadata = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            if len(metadata) != 1 or archive.getinfo(metadata[0]).file_size > 1024 * 1024:
                raise ValueError('Invalid wheel metadata')
            message = email.message_from_bytes(archive.read(metadata[0]))
            name, version = message['Name'], message['Version']
            if not name or not version:
                raise ValueError('Wheel lacks package identity')
            name = package_name(name)
        if name not in packages:
            continue
        if name in found or packages[name] != version:
            raise ValueError('Duplicate or wrong-version wheel: ' + name)
        found[name] = dict(version=version, file=path.name, sha256=file_hash(path))
    if set(found) != set(packages):
        raise ValueError('Missing installed-package wheels: ' + ', '.join(sorted(set(packages) - set(found))))
    return found


def command(*args):
    return subprocess.run([str(a) for a in args], check=True, capture_output=True, text=True, timeout=60).stdout.strip()


def observe(native):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3, 12):
        raise ValueError('The frozen comparison requires Linux x86_64 / Python 3.12')
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise ValueError('Pin the process to one allocated CPU before admission or timing')
    packages = {}
    for dist in importlib.metadata.distributions():
        name = package_name(dist.metadata['Name'])
        if name in packages:
            raise ValueError('Duplicate installed distribution: ' + name)
        packages[name] = dist.version
    pins = dict(torch='2.15.0+cpu', onnxruntime='1.31.1', onnx='1.23.2', numpy='1.26.4', onnxscript='0.7.2')
    if any(packages.get(name) != version for name, version in pins.items()):
        raise ValueError('Installed packages do not match the frozen candidate environment')
    command(sys.executable, '-m', 'pip', 'check')
    extension = extension_identity()
    build_info = command(native.parent / 'stealth', '--build-info')
    if 'configuration: Release' not in build_info or 'source_fingerprint: ' + source_fingerprint() not in build_info:
        raise ValueError('Native binary must be rebuilt in Release from the current source')
    cpu = {}
    for block in Path('/proc/cpuinfo').read_text().split('\n\n'):
        fields = dict(line.split(':', 1) for line in block.splitlines() if ':' in line)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        if fields.get('processor') == str(affinity[0]):
            cpu = {k: fields[k] for k in ('vendor_id', 'cpu family', 'model', 'model name', 'stepping', 'flags', 'microcode') if k in fields}
    if 'flags' not in cpu or 'model name' not in cpu:
        raise ValueError('Cannot identify the pinned CPU')
    policy_path = Path(f'/sys/devices/system/cpu/cpu{affinity[0]}/cpufreq/scaling_governor')
    try:
        power = dict(governor=policy_path.read_text().strip())
    except OSError as error:
        power = dict(unavailable_reason=type(error).__name__)
    cache = (native.parent / 'CMakeCache.txt').read_text()
    compiler = re.search(r'^CMAKE_CXX_COMPILER:(?:FILEPATH|STRING)=(.+)$', cache, re.MULTILINE)
    if compiler is None:
        raise ValueError('Cannot resolve the native C++ compiler')
    entries = json.loads(bounded(native.parent / 'compile_commands.json'))
    flags = [dict(file=Path(e['file']).relative_to(ROOT).as_posix(),
                  command=e['command'].replace(str(ROOT), '<repo>')) for e in entries]
    # Record inherited runtime controls as well as our explicit single-thread settings.
    inherited = {k: v for k, v in os.environ.items()
                 if k.startswith(('OMP_', 'MKL_', 'OPENBLAS_', 'DNNL_', 'ONEDNN_', 'KMP_', 'TORCH_', 'TORCHINDUCTOR_'))
                 or k in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH')}
    inherited = {k: v.replace(str(ROOT), '<repo>').replace(str(Path.home()), '<home>')
                 for k, v in inherited.items()}
    return dict(cpu_model_and_isa=cpu, os_and_kernel=dict(os_release=Path('/etc/os-release').read_text(),
                kernel=platform.release()), glibc=list(platform.libc_ver()), python_patch=platform.python_version(),
                compiler_and_flags=dict(version=command(compiler.group(1), '--version'), commands=flags),
                cmake_and_ninja=dict(cmake=command('cmake', '--version'), ninja=command('ninja', '--version')),
                packages=dict(sorted(packages.items())), native_binary_sha256=file_hash(native),
                extension_identity=extension, source_fingerprint=source_fingerprint(), cpu_affinity=affinity,
                power_governor_or_unavailable_reason=power,
                thread_and_library_environment=dict(inherited=inherited, worker_overrides=THREADS),
                source_sha256={name: digest(bounded(ROOT / 'tools' / name)) for name in SOURCES})
