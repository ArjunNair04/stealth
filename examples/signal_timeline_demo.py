"""Invented mixed-frequency feed replay. Run from any directory; no dependencies."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from signal_timeline import Observation, SignalSpec, SignalTimeline


def main():
    # Small integer clocks keep the example legible. Production adapters supply
    # UTC epoch nanoseconds and source-specific, explicitly chosen age limits.
    specs = [SignalSpec("minute", "test", "A", "value", 10, 10),
             SignalSpec("month", "test", "M", "value", 100, 100)]
    records = [Observation("test", "A", "value", "m1", 0, 10, 11, 12, 10.),
               Observation("test", "A", "value", "m2", 0, 20, 21, 22, 20.),
               Observation("test", "M", "value", "jan", 0, 1, 15, 16, 100.),
               Observation("test", "M", "value", "jan", 1, 1, 30, 31, 101.)]
    timeline = SignalTimeline(specs, records)
    rows = []
    for decision in (14, 17, 23, 32):
        snapshot = timeline.snapshot(decision)
        rows.append({"decision": decision, "values": snapshot.values,
                     "states": [f.state for f in snapshot.features], "ready": snapshot.ready})
    print(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
