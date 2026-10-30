"""Replay one fixed adversarial tape through checked features and CPU forecasts.

Outputs are new directories only. This is correctness admission, not a latency
benchmark, live feed or predictive study. A rerun reconstructs the frozen inputs
and models from scratch; it does not demonstrate crash-durable recovery.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from feature_inference import FeatureReplay
from model_bundle import create_bundle, load_bundle
from signal_timeline import SignalSpec, Observation, SignalTimeline
from native_signal_timeline import NativeSignalTimeline

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'benchmarks/temporal/conformance_v1.json'
ATOL, RTOL = 5e-5, 2e-4
ENGINES = ('torch_eager', 'torch_compiled_inductor', 'onnxruntime_cpu')


def encoded(obj):
    return (json.dumps(obj, sort_keys=True, allow_nan=False, separators=(',', ':'))+'\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def tape():
    # Invented integer clocks, not exchange calendars or claimed source precision.
    periods = (10, 20, 40, 60, 80, 100, 200)
    specs = tuple(SignalSpec('feature_'+str(i), 'diagnostic', 'sensor', str(i),
                             p*2, p*2) for i,p in enumerate(periods))
    records = []
    for i, (spec, period) in enumerate(zip(specs, periods)):
        for t in range(0, 1000, period):
            if i == 4 and t in (160,240,320):
                continue
            received = 15 if i == 6 and t == 0 else t
            if i == 0 and t in (110,120):
                received = 130
            value = (t//period % 13)*0.1 + i*0.2
            r = Observation(*spec.key, str(t), 0, t, t, received, value)
            records.append(r)
            if t % 100 == 0:
                records.append(replace(r, received_at_ns=received+17))
    def revision(channel, event, number, published, received, value):
        return Observation(*specs[channel].key, str(event), number, event, published,
                           received, value)
    records += [revision(0,100,1,115,121,2.1), revision(0,100,2,118,120,2.2),
                revision(0,0,1,140,150,9.0), # old event cannot replace the current event
                revision(3,120,1,130,140,None),
                revision(3,120,2,150,160,2.5)]
    # Reverse delivery list proves list order does not define historical knowledge.
    return specs, tuple(reversed(records)), tuple(t for t in range(1,1000,10) if t not in (501,511))


def diagnostic_blob():
    import torch
    from torch_forecaster import Config, Forecaster
    from generate_forecast_fixture import fixture
    torch.set_num_threads(1)
    params, _, _, _, _ = fixture()
    return Forecaster.from_flat(Config(7,4,2,7,5,4,6,output='level'),
                                np.asarray(params,dtype=np.float32)).export_bytes()


class CapturingReplay(FeatureReplay):
    """Diagnostic capture; the unchanged bridge still validates and commits."""
    last_input = None

    def _predict(self, rows):
        value = super()._predict(rows)
        self.last_input = np.asarray(rows, dtype=np.float32)
        return value


def external_predictions(blob, windows, engine, output):
    import torch
    from runtime_adapters import TorchAdapter, OnnxAdapter
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if engine == 'torch_compiled_inductor':
        import torch._dynamo as dynamo
        import torch._inductor.metrics as metrics
        dynamo.reset()
        metrics.reset()
    if engine == 'onnxruntime_cpu':
        adapter = OnnxAdapter(blob, batch=1, tokens=5, artifacts=output/'onnx')
    else:
        adapter = TorchAdapter(blob, batch=1, tokens=5,
                               compiled=engine == 'torch_compiled_inductor')
    predictions = []
    with torch.inference_mode():
        for window in windows:
            value = adapter.run(adapter.prepare(np.ascontiguousarray(window[None])))
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            predictions.append(float(value[0,0]))
    if engine == 'torch_compiled_inductor' and metrics.generated_kernel_count <= 0:
        raise RuntimeError('Inductor did not generate a kernel; compilation is not admitted')
    if engine == 'torch_compiled_inductor':
        adapter.metadata['generated_kernel_count'] = metrics.generated_kernel_count
    (output/'engine.json').write_bytes(encoded(adapter.metadata))
    return predictions


def run(output, *, engine='torch_eager'):
    output = Path(output)
    if engine not in ENGINES:
        raise ValueError('Unsupported conformance engine')
    output.mkdir(parents=True, exist_ok=False)
    stage = 'setup'
    try:
        from build_identity import extension_identity
        identity = extension_identity()
        specs, records, clocks = tape()
        blob = diagnostic_blob()
        raw_tape = encoded(dict(specs=[asdict(s) for s in specs],
                               records=[asdict(r) for r in records], decisions=clocks))
        (output/'tape.json').write_bytes(raw_tape)
        (output/'model.stlf').write_bytes(blob)
        (output/'protocol.json').write_bytes(PROTOCOL.read_bytes())
        ident = create_bundle(output/'bundle',blob,specs,interval_ns=10,
                              grid_origin_ns=1,clock_domain='synthetic')
        bundle = load_bundle(output/'bundle',expected_id=ident)
        oracle, native = SignalTimeline(specs,records), NativeSignalTimeline(specs,records)
        reference, actual = CapturingReplay(bundle), CapturingReplay(bundle)
        windows, expected, forecast_clocks, states = [], [], [], Counter()
        stage = 'snapshots_and_native_forecasts'
        with (output/'oracle.jsonl').open('wb') as left, (output/'native.jsonl').open('wb') as right:
            for clock in clocks:
                if oracle.snapshot(clock) != native.snapshot(clock):
                    raise AssertionError('Native snapshot disagrees with independent scan')
                a, b = reference.advance(oracle,clock), actual.advance(native,clock)
                left.write(encoded(asdict(a)))
                right.write(encoded(asdict(b)))
                if a != b:
                    raise AssertionError('Native decision disagrees with oracle replay')
                states[b.state] += 1
                if b.state == 'forecast':
                    np.testing.assert_array_equal(reference.last_input,actual.last_input)
                    windows.append(actual.last_input.copy())
                    expected.append(b.value)
                    forecast_clocks.append(clock)
        stage = 'external_runtime'
        predictions = external_predictions(blob,windows,engine,output)
        with (output/'external.jsonl').open('wb') as stream:
            for clock, wanted, value in zip(forecast_clocks,expected,predictions):
                stream.write(encoded(dict(decision_at_ns=clock, expected=wanted, actual=value)))
        if len(predictions) != len(expected):
            raise AssertionError('External prediction count mismatch')
        max_error = 0.
        for wanted, value in zip(expected,predictions):
            if not math.isfinite(value) or abs(value-wanted) > ATOL+RTOL*abs(wanted):
                raise AssertionError('External runtime forecast exceeds frozen tolerance')
            max_error = max(max_error,abs(value-wanted))
        if not expected:
            raise AssertionError('No forecasts exercised')
        hashes = {p.relative_to(output).as_posix():digest(p.read_bytes())
                  for p in sorted(output.rglob('*')) if p.is_file()}
        source_files = ('tools/temporal_conformance.py','tools/native_signal_timeline.py',
                        'tools/signal_timeline.py','tools/feature_inference.py',
                        'tools/model_bundle.py','tools/runtime_adapters.py',
                        'tools/runtime_torch.py','tools/torch_forecaster.py',
                        'tools/generate_forecast_fixture.py')
        report = dict(schema_version=1,passed=True,timing_is_benchmark=False,
                      engine=engine,decisions=len(clocks),forecasts=len(expected),
                      decision_states=dict(states),deliveries=len(records),
                      unique_records=native.record_count,index_entries=native.index_entry_count,
                      max_absolute_error=max_error,absolute_tolerance=ATOL,relative_tolerance=RTOL,
                      artifacts=hashes,native_build=identity,
                      source_hashes={p:digest((ROOT/p).read_bytes()) for p in source_files})
        (output/'report.json').write_bytes(encoded(report))
        return report
    except Exception as exc:
        (output/'failure.json').write_bytes(encoded(dict(stage=stage,type=type(exc).__name__,
                                                        message=str(exc))))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--engine',choices=ENGINES,default='torch_eager')
    args = parser.parse_args()
    report = run(args.output,engine=args.engine)
    print(json.dumps({key:report[key] for key in ('passed','decisions','forecasts',
                                                'decision_states','max_absolute_error')}))


if __name__ == '__main__':
    main()
