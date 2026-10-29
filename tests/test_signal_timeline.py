"""Invented observations and independent timing/selection oracles; no feeds."""
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from signal_timeline import InputsNotReady, Observation, SignalSpec, SignalTimeline


def spec(name="price", entity="A", observation_age=1000, publication_age=1000):
    return SignalSpec(name, "test", entity, "value", observation_age, publication_age)


def obs(event="a", effective=10, available=11, received=12, value=1., revision=0,
        entity="A", source="test", field="value"):
    return Observation(source, entity, field, event, revision, effective, available,
                       received, value)


def values(timeline, decision):
    return timeline.snapshot(decision).require_values(schema_id=timeline.schema_id)


class TimelineTests(unittest.TestCase):
    def test_mixed_frequency_and_release_clock(self):
        # Units are invented nanoseconds; no claim about real release schedules.
        timeline = SignalTimeline([spec("minute", "A"), spec("month", "M")], [
            obs("m1", 10, 11, 12, 10.), obs("m2", 20, 21, 22, 20.),
            obs("jan", 1, 15, 16, 100., entity="M"),
            obs("jan", 1, 30, 31, 101., revision=1, entity="M")])
        self.assertEqual(timeline.snapshot(14).values, (10., None))
        self.assertEqual(values(timeline, 17), (10., 100.))
        self.assertEqual(values(timeline, 23), (20., 100.))
        self.assertEqual(values(timeline, 32), (20., 101.))
        self.assertEqual(values(timeline, 17), (10., 100.))

    def test_minute_daily_weekly_monthly_share_one_decision_clock(self):
        minute = 60 * 10**9
        day = 1440 * minute
        origin = 120 * day
        specs = [spec("minute", "A", 2*minute, 2*minute),
                 spec("daily", "D", 3*day, 3*day),
                 spec("weekly", "W", 14*day, 10*day),
                 spec("monthly", "M", 60*day, 40*day)]
        records = [obs("min", origin, origin+1, origin+2, 10.),
                   obs("day", 119*day, 119*day+1, 119*day+2, 20., entity="D"),
                   obs("week", 112*day, 115*day, 115*day+1, 30., entity="W"),
                   obs("month", 90*day, 105*day, 105*day+1, 40., entity="M")]
        timeline = SignalTimeline(specs, records)
        self.assertEqual(values(timeline, origin+minute), (10., 20., 30., 40.))
        later = timeline.snapshot(origin+3*minute)
        self.assertEqual(later.values, (None, 20., 30., 40.))
        self.assertFalse(later.ready)

    def test_strict_receipt_boundary_and_age_metadata(self):
        timeline = SignalTimeline([spec()], [obs(received=20)])
        self.assertFalse(timeline.snapshot(20).ready)
        selected = timeline.snapshot(21).features[0]
        self.assertEqual((selected.observation_age_ns, selected.publication_age_ns,
                          selected.receipt_age_ns), (11, 10, 1))
        self.assertEqual(selected.observation.event_id, "a")
        self.assertEqual(values(timeline, 21), (1.,))

    def test_older_revision_arrives_later(self):
        timeline = SignalTimeline([spec()], [obs(received=40),
            obs(available=20, received=21, revision=2, value=2.)])
        self.assertEqual(values(timeline, 22), (2.,))
        self.assertEqual(values(timeline, 41), (2.,))

    def test_old_period_revision_does_not_displace_new_period(self):
        timeline = SignalTimeline([spec()], [obs(), obs("b", 20, 21, 22, 2.),
            obs(available=30, received=31, value=99., revision=1)])
        self.assertEqual(values(timeline, 32), (2.,))

    def test_withdrawal_blocks_fallback_and_can_be_corrected(self):
        timeline = SignalTimeline([spec()], [obs(), obs("b", 20, 21, 22, 2.),
            obs("b", 20, 30, 31, None, 1), obs("b", 20, 40, 41, 3., 2)])
        snap = timeline.snapshot(32)
        self.assertEqual(snap.features[0].state, "withdrawn")
        self.assertEqual(snap.values, (None,))
        with self.assertRaises(InputsNotReady): snap.require_values(schema_id=timeline.schema_id)
        self.assertEqual(values(timeline, 42), (3.,))

    def test_duplicates_retain_earliest_receipt_and_are_order_independent(self):
        records = [obs(received=50), obs(received=12), obs(received=12)]
        first = SignalTimeline([spec()], records)
        second = SignalTimeline([spec()], reversed(records))
        self.assertEqual(first.snapshot(13), second.snapshot(13))
        self.assertEqual(values(first, 13), (1.,))
        self.assertEqual(first.record_count, 1)

    def test_conflicting_duplicates_fail(self):
        for record in [obs(value=2.), obs(available=12), obs(value=None)]:
            with self.subTest(record=record), self.assertRaises(ValueError):
                SignalTimeline([spec()], [obs(), record])

    def test_signed_zero_conflict_is_not_order_dependent(self):
        positive, negative = obs(value=0.), obs(value=-0.)
        for records in ([positive, negative], [negative, positive]):
            with self.assertRaises(ValueError): SignalTimeline([spec()], records)
        self.assertEqual(values(SignalTimeline([spec()], [positive, positive]), 13), (0.,))

    def test_unicode_control_identifiers_are_rejected(self):
        for name in ("a\u0085b", "a\u202eb", "a\nb"):
            with self.assertRaises(ValueError): dataclasses.replace(spec(), name=name)

    def test_ambiguous_lineage_and_event_clocks_fail(self):
        cases = [[obs(), obs(effective=9, revision=1)],
                 [obs(), obs(event="different")],
                 [obs(available=20, received=21),
                  obs(available=15, received=22, revision=1)]]
        for records in cases:
            with self.subTest(records=records), self.assertRaises(ValueError):
                SignalTimeline([spec()], records)

    def test_age_limits_are_per_feature_and_inclusive(self):
        timeline = SignalTimeline([spec(observation_age=10, publication_age=100),
                                   spec("slow", "M", observation_age=100)],
                                  [obs(), obs(entity="M")])
        self.assertTrue(timeline.snapshot(20).ready)
        snap = timeline.snapshot(21)
        self.assertEqual(snap.values, (None, 1.))
        self.assertEqual([f.state for f in snap.features], ["stale", "valid"])
        self.assertIsNotNone(snap.features[0].observation)

    def test_publication_limit_is_independent_and_late_receipt_does_not_refresh(self):
        timeline = SignalTimeline([spec(observation_age=100, publication_age=10)],
                                  [obs(received=20)])
        self.assertTrue(timeline.snapshot(21).ready)
        self.assertEqual(timeline.snapshot(22).features[0].state, "stale")
        # A new revision has a new publication age but the same observation age.
        old = SignalTimeline([spec(observation_age=20)],
                             [obs(available=90, received=91, revision=5)])
        self.assertEqual(old.snapshot(92).features[0].state, "stale")

    def test_feature_order_and_schema_are_bound(self):
        a, b = spec(), spec("second", "B")
        records = [obs(value=2.), obs(value=3., entity="B")]
        ab, ba = SignalTimeline([a, b], records), SignalTimeline([b, a], records)
        self.assertEqual(values(ab, 13), (2., 3.))
        self.assertEqual(values(ba, 13), (3., 2.))
        self.assertNotEqual(ab.schema_id, ba.schema_id)
        self.assertNotEqual(ab.schema_id, SignalTimeline([dataclasses.replace(a,
            max_publication_age_ns=999), b], records).schema_id)
        with self.assertRaises(ValueError):
            ab.snapshot(13).require_values(schema_id=ba.schema_id)
        canonical = {"version": 1, "features": [dataclasses.asdict(a), dataclasses.asdict(b)]}
        expected = hashlib.sha256(json.dumps(canonical, sort_keys=True,
            separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
        self.assertEqual(ab.schema_id, expected)

    def test_missing_input_never_becomes_zero(self):
        timeline = SignalTimeline([spec()], [])
        snap = timeline.snapshot(100)
        self.assertEqual(snap.values, (None,))
        self.assertEqual(snap.features[0].state, "missing")
        self.assertIsNone(snap.features[0].observation_age_ns)
        with self.assertRaises(InputsNotReady): snap.require_values(schema_id=timeline.schema_id)

    def test_schema_and_channel_errors(self):
        for specs, records in [([], []), ([spec(), spec()], []),
            ([spec(), spec("other")], []), ([spec()], [obs(entity="unknown")]),
            (["bad"], []), ([spec()], ["bad"])]:
            with self.subTest(specs=specs), self.assertRaises((TypeError, ValueError)):
                SignalTimeline(specs, records)
        with self.assertRaises(ValueError):
            SignalTimeline([spec(str(i), str(i)) for i in range(257)], [])

    def test_input_validation(self):
        for value in [math.nan, math.inf, -math.inf, True, "1", 10**1000]:
            with self.subTest(value=str(value)[:20]), self.assertRaises((TypeError, ValueError)):
                obs(value=value)
        for bad in [-1, 2**63, 1., True, "2026-10-29"]:
            for field in ["effective_at_ns", "available_at_ns", "received_at_ns", "revision"]:
                with self.subTest(field=field, bad=bad), self.assertRaises((TypeError, ValueError)):
                    dataclasses.replace(obs(), **{field: bad})
            with self.assertRaises((TypeError, ValueError)):
                SignalTimeline([spec()], []).snapshot(bad)
        for changes in [{"effective_at_ns":12}, {"received_at_ns":10},
                        {"source":""}, {"event_id":" bad"}, {"field":"x"*129}]:
            with self.assertRaises((TypeError, ValueError)): dataclasses.replace(obs(), **changes)
        for changes in [{"name":""}, {"entity":" A "},
                        {"max_observation_age_ns":-1}, {"max_publication_age_ns":True}]:
            with self.assertRaises((TypeError, ValueError)): dataclasses.replace(spec(), **changes)

    def test_bounded_consumption_including_duplicates(self):
        seen = []
        def endless():
            while True:
                seen.append(1)
                yield obs()
        with self.assertRaises(ValueError): SignalTimeline([spec()], endless(), max_records=2)
        self.assertEqual(len(seen), 3)
        for bad in [0, -1, 100001, True, 1.5]:
            with self.assertRaises((TypeError, ValueError)):
                SignalTimeline([spec()], [], max_records=bad)

    def test_input_and_output_ownership(self):
        specs, records = [spec()], [obs()]
        timeline = SignalTimeline(specs, records)
        before = timeline.snapshot(13)
        specs.clear(); records.clear()
        self.assertEqual(timeline.snapshot(13), before)
        with self.assertRaises(dataclasses.FrozenInstanceError): before.features[0].value = 4.
        with self.assertRaises(dataclasses.FrozenInstanceError): before.decision_at_ns = 3

    def test_future_receipts_cannot_change_past_snapshot(self):
        records = [obs()]
        before = SignalTimeline([spec()], records).snapshot(20)
        records += [obs(available=30, received=31, value=99., revision=1),
                    obs("b", 15, 16, 35, 8.)]
        self.assertEqual(before, SignalTimeline([spec()], records).snapshot(20))

    def test_generated_replay_matches_independent_brute_force(self):
        rng = random.Random(8123)
        specs = [spec("x", "A", 37, 23), spec("y", "B", 51, 31)]
        records = []
        for entity in ("A", "B"):
            for n in range(8):
                effective = 10*n
                for revision in range(3):
                    available = effective + 2 + 4*revision
                    records.append(obs(str(n), effective, available,
                        available+rng.randrange(15), rng.randrange(-10, 11), revision, entity))
        records += [dataclasses.replace(records[0], received_at_ns=100)]
        rng.shuffle(records)
        timeline = SignalTimeline(specs, records)
        for decision in range(110):
            expected = []
            for s in specs:
                visible = [r for r in records if r.entity == s.entity and r.received_at_ns < decision]
                if not visible:
                    expected.append(None); continue
                event_time = max(r.effective_at_ns for r in visible)
                revision = max(r.revision for r in visible if r.effective_at_ns == event_time)
                chosen = min((r for r in visible if r.effective_at_ns == event_time and
                              r.revision == revision), key=lambda r:r.received_at_ns)
                fresh = decision-event_time <= s.max_observation_age_ns and                         decision-chosen.available_at_ns <= s.max_publication_age_ns
                expected.append(chosen.value if fresh else None)
            self.assertEqual(timeline.snapshot(decision).values, tuple(expected))

    def test_demo_exercises_component_composition(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root / "examples/signal_timeline_demo.py")],
                                text=True, capture_output=True, check=True, cwd=root)
        rows = json.loads(result.stdout)
        self.assertEqual([row["values"] for row in rows],
                         [[10., None], [10., 100.], [20., 100.], [None, 101.]])
        self.assertEqual([row["ready"] for row in rows], [False, True, True, False])
        self.assertEqual(rows[-1]["states"], ["stale", "valid"])


if __name__ == "__main__":
    unittest.main()
