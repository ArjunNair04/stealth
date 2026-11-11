#!/usr/bin/env python3
"""Admit, lock and compare the five CPU engines under the frozen frontier protocol."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from admit_runtime import THREADS, supervise, write
from runtime_common import bounded, canonical, digest, read_input, summarize

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'benchmarks/ettm1/frontier_v1.json'
PROTOCOL_SHA = '606a25000f3b7b853023e46e0a01a433f62e009b44f5e185fafd6716d011f649'
ENGINES = ('native_core', 'native_python', 'torch_eager', 'torch_compiled_inductor', 'onnxruntime_cpu')
BASELINES = ENGINES[2:]
CACHE_POLICY = 'fresh_per_cell; TORCHINDUCTOR_FORCE_DISABLE_CACHES=1'


def validate_protocol(protocol):
    original = bounded(PROTOCOL)
    if digest(original) != PROTOCOL_SHA or protocol != json.loads(original):
        raise ValueError('Frozen frontier_v1 protocol changed; register a new experiment')


def matrix(protocol, *, admission=False):
    cells = []
    for trial in range(1 if admission else protocol['trials']):
        for si, seed in enumerate(protocol['seeds']):
            for wi, workload in enumerate(protocol['workloads']):
                engines = list(protocol['engines'])
                rotation = (trial + si + wi) % len(engines)
                for engine in engines[rotation:] + engines[:rotation]:
                    cells.append(dict(trial=trial + 1, seed=seed, batch=workload['batch'], engine=engine,
                                      samples=1 if admission else workload['samples']))
    return cells


def cell_id(cell):
    return f'trial{cell["trial"]}-seed{cell["seed"]}-batch{cell["batch"]}-{cell["engine"]}'


def finite(value, *, nonnegative=False):
    return (type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0))


def validate_result(cell, result):
    if (result.get('engine') != cell['engine'] or result.get('batch') != cell['batch'] or
            result.get('output_mode') != 'last' or len(result.get('times_ns', [])) != cell['samples'] or
            len(result.get('clock_times_ns', [])) != 1000 or
            not finite(result.get('checksum')) or
            any(not finite(result.get(k), nonnegative=True)
                for k in ('setup_ns', 'first_call_ns', 'max_abs_difference'))):
        raise ValueError('Worker result contract mismatch')
    summarize(result['times_ns'])
    summarize(result['clock_times_ns'])
    if min(result['times_ns']) <= 0:
        raise ValueError('Nonpositive elapsed time')


def aggregate(protocol, records):
    expected = {cell_id(c): c for c in matrix(protocol)}
    found = {}
    for record in records:
        cell = record['cell']
        key = cell_id(cell)
        if key in found or expected.get(key) != cell:
            raise ValueError('Duplicate or unplanned comparison cell')
        if record['status'] not in ('complete', 'failed', 'not_run'):
            raise ValueError('Unknown comparison status')
        if record['status'] == 'complete':
            validate_result(cell, record['result'])
        found[key] = record
    missing = sorted(set(expected) - set(found))
    failed = sorted(k for k, r in found.items() if r['status'] != 'complete')
    report = dict(status='incomplete', expected_cells=len(expected), completed_cells=len(found) - len(failed),
                  missing=missing, failed=failed, batches={},
                  timed_calls=sum(len(r['result']['times_ns']) for r in records if r['status'] == 'complete'))
    if missing or failed:
        return report
    report['status'] = 'complete'
    for workload in protocol['workloads']:
        by_engine = {}
        for engine in ENGINES:
            results = [r['result'] for r in records if r['cell']['batch'] == workload['batch'] and r['cell']['engine'] == engine]
            timings = [summarize(r['times_ns']) for r in results]
            by_engine[engine] = dict(cells=len(results),
                                    p50_ns=statistics.median(r['p50_ns'] for r in timings),
                                    p99_ns=statistics.median(r['p99_ns'] for r in timings),
                                    worst_ns=max(r['max_ns'] for r in timings))
        ratios = {p: by_engine['native_core'][p + '_ns'] /
                     min(by_engine[e][p + '_ns'] for e in BASELINES) for p in ('p50', 'p99')}
        report['batches'][str(workload['batch'])] = dict(engines=by_engine, p50_ratio=ratios['p50'],
                                                       p99_ratio=ratios['p99'],
                                                       passed=all(r <= 1.20 for r in ratios.values()))
    return report


def files_in(root):
    result = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if 'inductor-cache' in relative.parts or relative.as_posix() == 'seal.json':
            continue
        if path.is_symlink():
            raise ValueError('Symlinks are not experiment artifacts')
        if path.is_file():
            result[relative.as_posix()] = digest(bounded(path))
    return result


def seal(root, status):
    write(root / 'seal.json', dict(schema_version=1, status=status, files=files_in(root),
                                  excluded='per-cell inductor-cache directories'))


def verify_seal(root):
    root = Path(root)
    receipt = json.loads(bounded(root / 'seal.json'))
    if receipt['files'] != files_in(root):
        raise ValueError('Experiment files differ from their receipt')
    return receipt


def execute_cells(cells, output, runner, *, budget):
    deadline = time.monotonic() + budget
    records = []
    for cell in cells:
        path = output / cell_id(cell)
        if path.exists():
            raise FileExistsError(path)
        remaining = deadline - time.monotonic()
        record = dict(cell=cell)
        if remaining <= 0:
            path.mkdir()
            record.update(status='not_run', error='total run deadline exhausted')
        else:
            try:
                result = runner(cell, path, min(180, remaining))
                record.update(status='complete', result=result)
            except Exception as error:
                path.mkdir(exist_ok=True)
                record.update(status='failed', error_type=type(error).__name__, error=str(error))
        write(path / 'record.json', record)
        records.append(record)
        print(json.dumps(dict(cell=cell_id(cell), status=record['status'])), flush=True)
    return records


def environment(path):
    env = {**os.environ, **THREADS, 'TORCHINDUCTOR_CACHE_DIR': str(path / 'inductor-cache'),
           'TORCHINDUCTOR_FORCE_DISABLE_CACHES': '1'}
    for name in ('TORCH_COMPILE_DISABLE', 'TORCHDYNAMO_DISABLE', 'TORCHDYNAMO_SUPPRESS_ERRORS'):
        env.pop(name, None)
    return env


def run_cell(cell, output, *, artifacts, native, warmup, timeout, expected_graphs=None):
    output.mkdir(parents=True, exist_ok=False)
    paths = dict(model=artifacts / f'model-{cell["seed"]}.stlf', input=artifacts / 'input.bin',
                 expected=artifacts / f'expected-{cell["seed"]}.bin')
    binding = {k: digest(bounded(p)) for k, p in paths.items()}
    engine = cell['engine']
    if engine == 'native_core':
        command = [str(native), '--output', 'last', '--batch-execution', 'tiled']
    elif engine == 'native_python':
        command = [sys.executable, str(ROOT / 'tools/runtime_worker.py'), '--engine', engine, '--output', 'last']
    else:
        command = [sys.executable, str(ROOT / 'tools/frontier_worker.py'), '--engine', engine,
                   '--artifacts', str(output)]
    for name, value in {**paths, 'batch': cell['batch'], 'samples': cell['samples'], 'warmup': warmup}.items():
        command += ['--' + name, str(value)]
    write(output / 'started.json', dict(cell=cell, inputs_sha256=binding, timeout_seconds=timeout))
    process = supervise(command, output, timeout=timeout, env=environment(output))
    write(output / 'process.json', process)
    if process['timed_out']:
        raise RuntimeError('worker timeout (whole cell, including compilation)')
    if process['returncode'] != 0:
        raise RuntimeError('worker failed; see stderr.log')
    result = json.loads(bounded(output / 'stdout.log'))
    validate_result(cell, result)
    if binding != {k: digest(bounded(p)) for k, p in paths.items()}:
        raise ValueError('Inputs changed during worker execution')
    if engine in BASELINES:
        windows = read_input(bounded(paths['input']))[0]
        if (result.get('windows_checked') != windows or result.get('inputs_sha256') != binding or
                result.get('absolute_tolerance') != 5e-5 or result.get('relative_tolerance') != 2e-4):
            raise ValueError('Adapter parity contract mismatch')
    if engine == 'torch_compiled_inductor' and result['adapter'].get('generated_kernel_count', 0) <= 0:
        raise ValueError('Compilation not established')
    if engine == 'onnxruntime_cpu':
        hashes = {name: digest(bounded(output / 'onnx' / filename))
                  for name, filename in (('exported', 'model.onnx'), ('optimized', 'optimized.onnx'))}
        if hashes != result['adapter']['graph_sha256'] or (expected_graphs is not None and hashes != expected_graphs):
            raise ValueError('ONNX graph differs from admitted graph')
    return result


def snapshot_inputs(protocol, runs, references, output):
    payload = bounded(references / 'input.bin')
    if digest(payload) != protocol['input_sha256'] or read_input(payload)[:3] != (32, 96, 7):
        raise ValueError('Wrong frozen input bundle')
    prior = json.loads(bounded(ROOT / 'docs/evidence/latest-v1/runtime.json'))['preparation']
    blobs = {'input.bin': payload}
    for seed in protocol['seeds']:
        model = bounded(runs / f'seed{seed}' / 'model.stlf')
        expected = bounded(references / f'expected-{seed}.bin')
        if digest(model) != protocol['model_sha256'][str(seed)] or digest(expected) != prior[str(seed)]['expected_sha256']:
            raise ValueError('Frozen model or independent reference changed')
        blobs[f'model-{seed}.stlf'] = model
        blobs[f'expected-{seed}.bin'] = expected
    output.mkdir()
    for name, blob in blobs.items():
        with (output / name).open('xb') as target:
            target.write(blob)


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, check=True, text=True, capture_output=True, timeout=60).stdout.strip()


def clean_commit():
    if git('status', '--porcelain'):
        raise ValueError('Commit code and the host lock before running the study')
    return git('rev-parse', 'HEAD')


def local_path(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT / 'local'):
        raise ValueError('Keep experiment output under ignored local/')
    return path


def observed(native):
    from frontier_host import observe
    return observe(native)


def admit(args, protocol):
    code = clean_commit()
    host = observed(args.native)
    output = local_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'started.json', dict(purpose='correctness_admission', code_commit=code, host=host,
                                       protocol_sha256=PROTOCOL_SHA, started_at=datetime.now(timezone.utc).isoformat()))
    try:
        snapshot_inputs(protocol, args.runs, args.references, output / 'artifacts')
        def runner(cell, path, timeout):
            return run_cell(cell, path, artifacts=output / 'artifacts', native=args.native,
                            warmup=1, timeout=timeout)
        records = execute_cells(matrix(protocol, admission=True), output, runner, budget=2700)
        passed = all(r['status'] == 'complete' for r in records)
        if observed(args.native) != host:
            raise ValueError('Host or software changed during admission')
        write(output / 'admission.json', dict(status='admitted' if passed else 'incomplete', records=records,
                                             timings_are_acceptance_measurements=False))
        seal(output, 'admitted' if passed else 'incomplete')
        return 0 if passed else 1
    except BaseException as error:
        write(output / 'failed.json', dict(error_type=type(error).__name__, message=str(error)))
        seal(output, 'incomplete')
        raise


def admitted_records(directory, protocol):
    receipt = verify_seal(directory)
    result = json.loads(bounded(directory / 'admission.json'))
    planned = {cell_id(c): c for c in matrix(protocol, admission=True)}
    records = result['records']
    seen = set()
    if receipt['status'] != 'admitted' or result['status'] != 'admitted' or len(records) != len(planned):
        raise ValueError('All 30 admission cells must succeed first')
    for record in records:
        cell = record['cell']
        key = cell_id(cell)
        if key in seen or planned.get(key) != cell or record['status'] != 'complete':
            raise ValueError('Incomplete or duplicate admission matrix')
        seen.add(key)
        validate_result(cell, record['result'])
        if record != json.loads(bounded(directory / key / 'record.json')):
            raise ValueError('Admission summary differs from original worker record')
    return records


def lock(args, protocol):
    from frontier_host import wheel_lock
    records = admitted_records(args.admission, protocol)
    start = json.loads(bounded(args.admission / 'started.json'))
    host = observed(args.native)
    if host != start['host'] or start['protocol_sha256'] != PROTOCOL_SHA:
        raise ValueError('Admission no longer matches this host/software/protocol')
    graphs = {f'{r["cell"]["seed"]}-{r["cell"]["batch"]}': r['result']['adapter']['graph_sha256']
              for r in records if r['cell']['engine'] == 'onnxruntime_cpu'}
    value = dict(schema_version=1, protocol_sha256=PROTOCOL_SHA, host=host,
                 admission_seal_sha256=digest(bounded(args.admission / 'seal.json')),
                 wheels=wheel_lock(args.wheels, host['packages']), graph_sha256=graphs,
                 allocation=args.allocation, cache_policy=CACHE_POLICY)
    # Public host locks are committed separately, after inspection for private paths.
    write(args.output, value)
    print('Host lock written. Inspect it and commit it before the acceptance run.')
    return 0


def measure(args, protocol):
    code = clean_commit()
    lock_path = args.host_lock.resolve()
    if not lock_path.is_relative_to(ROOT) or not git('ls-files', '--', str(lock_path.relative_to(ROOT))):
        raise ValueError('The host lock must be committed in this checkout')
    locked = json.loads(bounded(lock_path))
    records = admitted_records(args.admission, protocol)
    graphs = {f'{r["cell"]["seed"]}-{r["cell"]["batch"]}': r['result']['adapter']['graph_sha256']
              for r in records if r['cell']['engine'] == 'onnxruntime_cpu'}
    if (locked['graph_sha256'] != graphs or locked['protocol_sha256'] != PROTOCOL_SHA or
            locked['admission_seal_sha256'] != digest(bounded(args.admission / 'seal.json')) or
            locked['host'] != observed(args.native) or locked['cache_policy'] != CACHE_POLICY):
        raise ValueError('Host lock or admission no longer matches')
    output = local_path(args.output)
    if output.exists():
        raise FileExistsError(output)
    # One attempt for this admitted experiment, even if a different output is requested.
    claim = args.admission.with_name(args.admission.name + '-acceptance-claim.json')
    write(claim, dict(output=str(output), host_lock_sha256=digest(bounded(lock_path)), code_commit=code))
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'started.json', dict(purpose='acceptance_measurement', code_commit=code,
                                       protocol_sha256=PROTOCOL_SHA, host_lock=locked,
                                       started_at=datetime.now(timezone.utc).isoformat()))
    try:
        artifacts = output / 'artifacts'
        artifacts.mkdir()
        for source in (args.admission / 'artifacts').iterdir():
            with (artifacts / source.name).open('xb') as target:
                target.write(bounded(source))
        def runner(cell, path, timeout):
            graph = locked['graph_sha256'].get(f'{cell["seed"]}-{cell["batch"]}')
            return run_cell(cell, path, artifacts=artifacts, native=args.native, warmup=protocol['warmup_calls'],
                            timeout=timeout, expected_graphs=graph)
        records = execute_cells(matrix(protocol), output, runner, budget=protocol['budget']['max_measurement_run_seconds'])
        report = aggregate(protocol, records)
        if observed(args.native) != locked['host']:
            raise ValueError('Host or software changed during measurement')
        write(output / 'results.json', dict(report=report, records=records))
        seal(output, report['status'])
        print(json.dumps(report, indent=2))
        return 0 if report['status'] == 'complete' else 1
    except BaseException as error:
        write(output / 'failed.json', dict(error_type=type(error).__name__, message=str(error)))
        seal(output, 'incomplete')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    for name in ('admit', 'lock', 'run'):
        sub = subs.add_parser(name)
        sub.add_argument('--native', type=Path, default=ROOT / 'build/release/stealth_runtime_bench')
        sub.add_argument('--output', type=Path, required=True)
        if name == 'admit':
            sub.add_argument('--runs', type=Path, required=True)
            sub.add_argument('--references', type=Path, required=True)
        else:
            sub.add_argument('--admission', type=Path, required=True)
        if name == 'lock':
            sub.add_argument('--wheels', type=Path, required=True)
            sub.add_argument('--allocation', required=True, help='Description of the quiet CPU allocation; no account/host identifiers')
        if name == 'run':
            sub.add_argument('--host-lock', type=Path, required=True)
    sub = subs.add_parser('verify')
    sub.add_argument('directory', type=Path)
    args = parser.parse_args()
    protocol = json.loads(bounded(PROTOCOL))
    validate_protocol(protocol)
    if args.command == 'verify':
        receipt = verify_seal(args.directory)
        if (args.directory / 'results.json').exists():
            result = json.loads(bounded(args.directory / 'results.json'))
            for record in result['records']:
                original = json.loads(bounded(args.directory / cell_id(record['cell']) / 'record.json'))
                if record != original:
                    raise ValueError('Summary differs from per-cell record')
            if receipt['status'] != result['report']['status'] or aggregate(protocol, result['records']) != result['report']:
                raise ValueError('Stored summary differs from recomputed gate')
        print(json.dumps(dict(status=receipt['status'], files=len(receipt['files']))))
        return 0
    args.native = args.native.resolve()
    if hasattr(args, 'admission'):
        args.admission = local_path(args.admission)
    return {'admit': admit, 'lock': lock, 'run': measure}[args.command](args, protocol)


if __name__ == '__main__':
    sys.exit(main())
