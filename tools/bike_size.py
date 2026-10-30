#!/usr/bin/env python3
"""Frozen bike model-capacity study; training, replay and serial native timings."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch

import bike_demand as data
import train_bike_demand as bike
from build_identity import extension_identity, source_fingerprint
from runtime_common import input_bytes, summarize
from torch_forecaster import Config, Forecaster
from training_state import bounded, canonical, digest, environment, strict_json
from train_ettm1 import ROOT, clean_commit, file_hash, write

PROTOCOL = ROOT / 'benchmarks/bike/size_v1.json'
# A changed recipe must get a new version, rather than silently change this study.
PROTOCOL_SHA256 = 'bc1adc0168382f2ecead3d2dedc8e01e3ac0383542cd57b7b51623b97c88857a'


def validate_protocol(protocol):
    if digest(canonical(protocol)) != PROTOCOL_SHA256:
        raise ValueError('Changed frozen size protocol; register a new version')
    if file_hash(bike.PROTOCOL) != protocol['base_protocol_sha256']:
        raise ValueError('Changed frozen base protocol')


def read_recipe():
    protocol = strict_json(bounded(PROTOCOL, 65536))
    validate_protocol(protocol)
    return protocol, bike.read_recipe()


def child_recipe(base, width):
    recipe = copy.deepcopy(base)
    recipe['experiment_id'] = f'bike_size_v1_width_{width}'
    recipe['model'].update(width=width, hidden=4 * width)
    Config(**recipe['model'])
    return recipe


def parameter_counts(recipe):
    config = Config(**recipe['model'])
    stored = sum(math.prod(shape) for shape in config.shapes().values())
    return dict(stored=stored, trainable=stored - 2 * config.features)


def failure(output, exc):
    write(output / 'failure.json', dict(error_type=type(exc).__name__))


def seal(output, names):
    write(output / 'complete.json', dict(status='complete', schema_version=1,
                                        files={name: file_hash(output / name) for name in names}))


def train(output, frame, protocol, base, binding):
    """Small alternate settings are allowed only for synthetic unit tests."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        write(output / 'protocol.json', protocol)
        write(output / 'started.json', dict(binding=binding, environment=environment(), native=extension_identity()))
        rows = {}
        for width in protocol['widths']:
            recipe = child_recipe(base, width)
            print(json.dumps(dict(starting_width=width, **parameter_counts(recipe))), flush=True)
            begin = time.perf_counter()
            result = bike.run_study(output / f'width-{width}', frame, recipe,
                                   {**binding, 'width': width, 'child_protocol_sha256': digest(canonical(recipe))})
            rows[str(width)] = dict(parameters=parameter_counts(recipe),
                                   study_wall_seconds=time.perf_counter() - begin,
                                   models=result['models'], baselines=result['baselines'],
                                   train_samples=result['train_samples'],
                                   validation_samples=result['validation_samples'],
                                   evaluation_samples=result['test_samples'])
        write(output / 'scores.json', dict(evidence=protocol['evidence'], widths=rows))
        seal(output, ['protocol.json', 'started.json', 'scores.json'] +
             [f'width-{width}/complete.json' for width in protocol['widths']])
    except Exception as exc:
        failure(output, exc)
        raise


def verify(root, protocol, base):
    root = Path(root)
    complete = strict_json(bounded(root / 'complete.json', 65536))
    names = {'protocol.json', 'started.json', 'scores.json'} | {
        f'width-{width}/complete.json' for width in protocol['widths']}
    if (complete.get('schema_version') != 1 or complete.get('status') != 'complete'
            or set(complete.get('files', {})) != names):
        raise ValueError('Incomplete size study')
    for name, sha in complete['files'].items():
        if file_hash(root / name) != sha:
            raise ValueError('Size study artifact mismatch')
    if strict_json(bounded(root / 'protocol.json', 65536)) != protocol:
        raise ValueError('Size study protocol mismatch')
    started = strict_json(bounded(root / 'started.json', 65536))
    scores = strict_json(bounded(root / 'scores.json', 4 * 1024 * 1024))
    if set(scores['widths']) != {str(w) for w in protocol['widths']} or scores['evidence'] != protocol['evidence']:
        raise ValueError('Incomplete score grid')
    for width in protocol['widths']:
        child = root / f'width-{width}'
        recipe = child_recipe(base, width)
        complete_child = bike.verify_run(child)
        expected = {**started['binding'], 'width': width, 'child_protocol_sha256': digest(canonical(recipe))}
        if (strict_json(bounded(child / 'protocol.json', 65536)) != recipe
                or any(complete_child['binding'].get(k) != v for k, v in expected.items())):
            raise ValueError('Child protocol or binding mismatch')
        result = strict_json(bounded(child / 'scores.json', 1024 * 1024))
        row = scores['widths'][str(width)]
        if (row['models'] != result['models'] or row['baselines'] != result['baselines']
                or row['parameters'] != parameter_counts(recipe)
                or row['train_samples'] != result['train_samples']
                or row['validation_samples'] != result['validation_samples']
                or row['evaluation_samples'] != result['test_samples']
                or not math.isfinite(row['study_wall_seconds']) or row['study_wall_seconds'] <= 0):
            raise ValueError('Inconsistent size summary')
    return complete


def replay(root, output, frame, protocol, base):
    root, output = Path(root), Path(output)
    verify(root, protocol, base)
    output.mkdir(parents=True, exist_ok=False)
    try:
        for width in protocol['widths']:
            child = root / f'width-{width}'
            raw = bike.replay(child, frame)
            if raw != bounded(child / 'predictions.jsonl', 64 * 1024 * 1024):
                raise ValueError('Native replay differs')
            (output / f'width-{width}.jsonl').write_bytes(raw)
        write(output / 'result.json', dict(byte_identical=True, widths=protocol['widths'],
                                          run_seal_sha256=file_hash(root / 'complete.json'),
                                          native=extension_identity()))
        seal(output, ['result.json'] + [f'width-{w}.jsonl' for w in protocol['widths']])
    except Exception as exc:
        failure(output, exc)
        raise


def worker_order(protocol, seeds):
    widths = protocol['widths']
    return [(trial + 1, seed, widths[(trial + si + offset) % len(widths)])
            for trial in range(protocol['latency']['trials']) for si, seed in enumerate(seeds)
            for offset in range(len(widths))]


def aggregate_latency(records, protocol, seeds):
    keys = [(r['trial'], r['seed'], r['width']) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != set(worker_order(protocol, seeds)):
        raise ValueError('Incomplete or duplicated latency grid')
    for width, seed in itertools.product(protocol['widths'], seeds):
        checksums = [r['checksum'] for r in records if r['width'] == width and r['seed'] == seed]
        if any(not math.isfinite(v) for v in checksums) or len(set(checksums)) != 1:
            raise ValueError('Prediction checksum changed between timing trials')
    result = {}
    for width in protocol['widths']:
        cells = [r for r in records if r['width'] == width]
        pairs = [(r['latency']['p50_ns'], r['latency']['p99_ns']) for r in cells]
        if any(not (math.isfinite(p50) and math.isfinite(p99) and 0 < p50 <= p99) for p50, p99 in pairs):
            raise ValueError('Invalid worker latency')
        memory = {(r['parameter_bytes'], r['position_bytes'], r['workspace_bytes']) for r in cells}
        if len(memory) != 1 or any(type(n) is not int or n <= 0 for n in next(iter(memory))):
            raise ValueError('Inconsistent native memory')
        p50, p99 = zip(*pairs)
        result[str(width)] = dict(p50_us=statistics.median(p50)/1000, p99_us=statistics.median(p99)/1000,
                                 p50_range_us=[min(p50)/1000, max(p50)/1000],
                                 p99_range_us=[min(p99)/1000, max(p99)/1000],
                                 owned_bytes=sum(next(iter(memory))), workers=len(cells))
    return result


def native_build(binary, identity):
    cli = binary.parent / ('stealth.exe' if os.name == 'nt' else 'stealth')
    info = subprocess.run([str(cli), '--build-info'], capture_output=True, text=True, check=True).stdout
    if ('configuration: Release' not in info or
            'source_fingerprint: ' + identity['build_info']['source_fingerprint'] not in info):
        raise ValueError('Rebuild the native Release worker from the current source')
    return dict(info=info, cli_sha256=file_hash(cli))


def benchmark(root, output, frame, protocol, base, binary):
    root, output, binary = Path(root), Path(output), Path(binary).resolve(strict=True)
    verify(root, protocol, base)
    identity = extension_identity()
    commit = clean_commit()
    build = native_build(binary, identity)
    output.mkdir(parents=True, exist_ok=False)
    try:
        settings = protocol['latency']
        samples = data.split(frame, base['model']['context'], data.hour(base['split_ends']['train']),
                             data.hour(base['split_ends']['validation']))
        raw = np.array(samples.x[:settings['validation_windows']], dtype=np.float32, order='C', copy=True)
        if len(raw) != settings['validation_windows']:
            raise ValueError('Insufficient timing windows')
        (output / 'input.bin').write_bytes(input_bytes(raw.ravel().tolist(), *raw.shape))
        # Prepare every Torch reference before any timing worker starts.
        references = {}
        for width, seed in itertools.product(protocol['widths'], base['seeds']):
            blob = bounded(root / f'width-{width}/seed-{seed}/model.stlf', 16 * 1024 * 1024)
            config = Config(**child_recipe(base, width)['model'])
            model = Forecaster.from_flat(config, np.frombuffer(blob[64:-32], dtype='<f4').copy())
            if model.export_bytes() != blob:
                raise ValueError('Model differs from declared configuration')
            with torch.inference_mode():
                reference = model(torch.from_numpy(raw)).numpy().astype('<f4').tobytes()
            name = f'expected-{width}-{seed}.bin'
            (output / name).write_bytes(reference)
            references[name] = digest(reference)
        binary_hash = file_hash(binary)
        write(output / 'started.json', dict(code_commit=commit, source_fingerprint=source_fingerprint(),
                                            binary_sha256=binary_hash, native=identity, build=build,
                                            protocol_sha256=digest(canonical(protocol)),
                                            study_seal_sha256=file_hash(root / 'complete.json'),
                                            input_sha256=file_hash(output / 'input.bin'), references=references,
                                            validation_target_hours=samples.target_hours[:len(raw)].tolist()))
        records = []
        for trial, seed, width in worker_order(protocol, base['seeds']):
            name = f'trial-{trial}-seed-{seed}-width-{width}'
            command = [binary, '--model', root / f'width-{width}/seed-{seed}/model.stlf',
                       '--input', output / 'input.bin', '--expected', output / f'expected-{width}-{seed}.bin',
                       '--batch', str(settings['batch']), '--output', settings['output'],
                       '--samples', str(settings['samples']), '--warmup', str(settings['warmup'])]
            process = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=120)
            (output / f'{name}.stderr').write_text(process.stderr)
            if process.returncode:
                raise RuntimeError('Native timing worker failed: ' + process.stderr)
            result = strict_json(process.stdout.encode())
            if (result.get('engine') != 'native_core' or result.get('batch') != settings['batch']
                    or result.get('output_mode') != settings['output']
                    or len(result['times_ns']) != settings['samples']
                    or not math.isfinite(result['checksum']) or not math.isfinite(result['max_abs_difference'])):
                raise ValueError('Unexpected timing worker output')
            write(output / f'{name}.json', result)
            row = {k: v for k, v in result.items() if k not in ('times_ns', 'clock_times_ns')}
            row.update(width=width, seed=seed, trial=trial, latency=summarize(result['times_ns']),
                       clock=summarize(result['clock_times_ns']), raw_sha256=file_hash(output / f'{name}.json'))
            records.append(row)
            print(json.dumps(dict(worker=name, p50_us=row['latency']['p50_ns']/1000)), flush=True)
        if file_hash(binary) != binary_hash:
            raise ValueError('Native binary changed during measurement')
        write(output / 'results.json', dict(widths=aggregate_latency(records, protocol, base['seeds']), records=records))
        seal(output, sorted(p.name for p in output.iterdir() if p.is_file()))
    except Exception as exc:
        failure(output, exc)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'replay', 'benchmark'))
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--binary', type=Path)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / 'local'):
        raise ValueError('Use ignored local/ for study outputs')
    if args.output.exists():
        raise FileExistsError('Output already exists')
    if (args.command == 'train') != (args.run is None) or (args.command == 'benchmark') != (args.binary is not None):
        parser.error('--run is required for replay/benchmark; --binary is required only for benchmark')
    protocol, base = read_recipe()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    commit = clean_commit()
    extension_identity()
    frame = bike.load_frame(args.csv, base)
    if args.command == 'train':
        train(args.output, frame, protocol, base,
              dict(code_commit=commit, protocol_sha256=digest(canonical(protocol)), data_sha256=base['data']['csv_sha256']))
    elif args.command == 'replay':
        replay(args.run, args.output, frame, protocol, base)
    else:
        benchmark(args.run, args.output, frame, protocol, base, args.binary)
    print(json.dumps(dict(status='complete', command=args.command)), flush=True)


if __name__ == '__main__':
    main()
