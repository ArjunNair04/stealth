"""Local paired Python-adapter probe; both arms use the same native extension.

The baseline adapter is a preserved source file pinned by the protocol. All
workers run serially in fresh processes. Synthetic observations test execution,
not forecast accuracy. Model/tape construction is outside the timed region.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from runtime_common import canonical, summarize

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'benchmarks/temporal/bridge_latency_v1.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def aggregate(records):
    keys = [(r['trial'], r['implementation']) for r in records]
    if len(keys) != 6 or set(keys) != set(itertools.product((1, 2, 3), ('baseline', 'candidate'))):
        raise ValueError('Incomplete or duplicated bridge comparison')
    for r in records:
        timing = r['latency']
        if not (0 < timing['p50_ns'] <= timing['p99_ns'] < math.inf):
            raise ValueError('Invalid bridge timing')
    paired = dict(zip(keys, records))
    p50, p99 = [], []
    for trial in (1, 2, 3):
        a, b = (paired[trial, arm] for arm in ('baseline', 'candidate'))
        if (a['output_sha256'] != b['output_sha256'] or a['build'] != b['build'] or
                a['model_sha256'] != b['model_sha256']):
            raise ValueError('Bridge predictions or common native build changed')
        p50.append(b['latency']['p50_ns']/a['latency']['p50_ns'])
        p99.append(b['latency']['p99_ns']/a['latency']['p99_ns'])
    return dict(accepted=all(x < 1 for x in p50) and statistics.median(p50) <= .95
                and statistics.median(p99) <= 1.1, p50_ratios=p50, p99_ratios=p99,
                median_p50_reduction=1-statistics.median(p50),
                median_p99_increase=statistics.median(p99)-1)


def worker(arm, baseline, model, output):
    import time
    import numpy as np
    from build_identity import extension_identity
    from feature_inference import FeatureReplay
    from model_bundle import create_bundle, load_bundle
    from native_signal_timeline import NativeSignalTimeline
    from signal_timeline import SignalSpec, Observation
    protocol = json.loads(PROTOCOL.read_bytes())
    build = extension_identity()
    blob = model.read_bytes()
    if sha(blob) != protocol['model_sha256'] or sha(baseline.read_bytes()) != protocol['baseline_adapter_sha256']:
        raise ValueError('Frozen bridge input/source mismatch')
    adapter = NativeSignalTimeline
    if arm == 'baseline':
        spec = importlib.util.spec_from_file_location('preserved_bridge_adapter', baseline)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        adapter = module.NativeSignalTimeline
    scaler = np.frombuffer(blob, dtype='<f4', count=14, offset=64)
    specs = tuple(SignalSpec(f'feature{i}', 'diagnostic', 'sensor', str(i), 20, 20)
                  for i in range(7))
    records = tuple(Observation(*spec.key, str(t), 0, t*10, t*10, t*10,
                    float(scaler[i]+scaler[7+i]*math.sin(t/37+i)))
                    for t in range(protocol['records_per_channel']) for i, spec in enumerate(specs))
    timeline = adapter(specs, records)
    lookup = getattr(timeline, '_observations', None)
    # Container/key storage only: record objects already belong to the timeline.
    lookup_bytes = (sys.getsizeof(dict(lookup)) + sum(sys.getsizeof(k) for k in lookup)) if lookup else 0
    output.mkdir(parents=True, exist_ok=False)
    bundle_id = create_bundle(output/'bundle', blob, specs, interval_ns=10,
                              grid_origin_ns=1, clock_domain='synthetic')
    replay = FeatureReplay(load_bundle(output/'bundle', expected_id=bundle_id))
    warmup, samples = protocol['warmup'], protocol['samples']
    for t in range(warmup):
        replay.advance(timeline, t*10+1)
    times, values = [], []
    for t in range(warmup, warmup+samples):
        begin = time.perf_counter_ns()
        result = replay.advance(timeline, t*10+1)
        times.append(time.perf_counter_ns()-begin)
        if result.state != 'forecast' or not math.isfinite(result.value):
            raise ValueError('Expected a finite warmed forecast')
        values.append(result.value)
    source = baseline if arm == 'baseline' else ROOT/'tools/native_signal_timeline.py'
    return dict(implementation=arm, build=build, times_ns=times, latency=summarize(times),
                output_sha256=sha(np.asarray(values, dtype='<f4').tobytes()),
                model_sha256=sha(blob), adapter_sha256=sha(source.read_bytes()),
                lookup_container_bytes=lookup_bytes, canonical_records=timeline.record_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-adapter', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=('baseline', 'candidate'))
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'local'):
        raise ValueError('Probe outputs belong under ignored local/')
    if args.worker:
        result = worker(args.worker, args.baseline_adapter.resolve(), args.model.resolve(), output)
        print(json.dumps(result, allow_nan=False))
        return
    output.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for trial in (1, 2, 3):
            order = ('baseline', 'candidate') if trial % 2 else ('candidate', 'baseline')
            for arm in order:
                cell = f'trial{trial}-{arm}'
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    '--baseline-adapter', str(args.baseline_adapter.resolve()), '--model', str(args.model.resolve()),
                    '--output', str(output/cell), '--worker', arm], capture_output=True, text=True, timeout=120)
                (output/(cell+'.stderr')).write_text(result.stderr)
                if result.returncode:
                    raise RuntimeError(f'{cell} failed; inspect retained stderr')
                record = json.loads(result.stdout)
                record['trial'] = trial
                records.append(record)
        report = dict(protocol=json.loads(PROTOCOL.read_bytes()), protocol_sha256=sha(PROTOCOL.read_bytes()),
                      script_sha256=sha(Path(__file__).read_bytes()), records=records, gate=aggregate(records))
        (output/'results.json').write_bytes(canonical(report))
        print(json.dumps(report['gate']))
    except Exception as exc:
        (output/'failed.json').write_bytes(canonical(dict(error=type(exc).__name__, message=str(exc))))
        raise


if __name__ == '__main__':
    main()
