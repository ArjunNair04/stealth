#!/usr/bin/env python3
"""Record fresh, bounded native benchmark trials; Python is not a core dependency."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New run directory, normally under local/')
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--trials', type=int, default=3)
    args = parser.parse_args()
    if not 100 <= args.samples <= 1000000 or not 1 <= args.trials <= 20:
        parser.error('Samples must be 100..1000000 and trials 1..20')
    if run('git', 'status', '--porcelain'):
        parser.error('Commit reviewed source changes before recording a benchmark')
    if args.output.exists():
        parser.error('The run directory already exists; choose a fresh path')
    subprocess.run(['cmake', '--preset', 'release', '-DSTEALTH_WARNINGS_AS_ERRORS=ON',
                    '-DSTEALTH_BUILD_BENCHMARKS=ON'], cwd=ROOT, check=True)
    subprocess.run(['cmake', '--build', '--preset', 'release'], cwd=ROOT, check=True)
    binary = ROOT / 'build/release' / ('stealth_bench.exe' if os.name == 'nt' else 'stealth_bench')
    commands = json.loads((ROOT / 'build/release/compile_commands.json').read_text())
    numeric_command = next(item['command'] for item in commands if item['file'].replace('\\', '/').endswith('/src/numeric.cpp'))
    numeric_command = numeric_command.replace(str(ROOT), '<source>').replace(ROOT.as_posix(), '<source>')
    # Deliberately fail if a run directory already exists. Preserve partial failures too.
    args.output.mkdir(parents=True, exist_ok=False)
    tracked = run('git', 'ls-files').splitlines()
    source_files = [p for p in tracked if p.startswith(('src/', 'include/', 'bench/', 'cmake/')) or p in ('CMakeLists.txt', 'CMakePresets.json', 'tools/record_benchmarks.py')]
    cpu = platform.processor() or platform.machine()
    if platform.system() == 'Darwin':
        cpu = run('sysctl', '-n', 'machdep.cpu.brand_string')
    manifest = {
        'schema_version': 1,
        'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'source_commit': run('git', 'rev-parse', 'HEAD'),
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_files},
        'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'numeric_compile_command': numeric_command,
        'cpu': cpu,
        'architecture': platform.machine(),
        'os': platform.system(),
        'os_version': platform.mac_ver()[0] if platform.system() == 'Darwin' else platform.release(),
        'requested_trials': args.trials,
        'samples_per_kernel': args.samples,
        'protocol': 'Sequential processes; alternating scalar/packed call order; warm cache; one inference thread; no affinity or frequency control; ordinary desktop load; no clock-overhead subtraction.',
        'trials': [],
    }
    manifest_path = args.output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    for trial in range(1, args.trials + 1):
        completed = subprocess.run([str(binary), '--samples', str(args.samples)], text=True, capture_output=True)
        name = f'trial-{trial:02}.json'
        (args.output / name).write_text(completed.stdout)
        (args.output / f'trial-{trial:02}.stderr.txt').write_text(completed.stderr)
        completed.check_returncode()
        result = json.loads(completed.stdout)
        if result['configuration'] != 'Release' or result['samples_per_kernel'] != args.samples:
            raise RuntimeError('Unexpected benchmark metadata')
        manifest['trials'].append({'file': name, 'sha256': hashlib.sha256((args.output / name).read_bytes()).hexdigest()})
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'Trial {trial}/{args.trials} recorded', flush=True)
    print(f'Preserved benchmark receipt: {manifest_path}')


if __name__ == '__main__':
    main()
