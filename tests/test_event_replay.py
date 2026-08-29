"""A captured trace replays in order and at the right pace (Task 12, Part B).

The timing assertions inject a fake `sleep`, so the suite records what a replay *would*
wait without spending it. A test that actually slept through `sample_run.jsonl` would
add 13.5 seconds to every CI run to prove arithmetic.

The committed fixture is exercised here as well as the synthetic ones: it is the
fallback demo, so a change that makes it unreadable should fail a test rather than
surface during a recording.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelsmith.ui.event_replay import (
    MAX_GAP_S,
    gap_seconds,
    list_traces,
    load_events,
    pace_events,
    replay_events,
    total_duration_s,
)

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "traces" / "sample_run.jsonl"


class FakeSleep:
    """Records what it was asked to wait for, and waits for none of it."""

    def __init__(self):
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def write_trace(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def trace(*times: float) -> list[dict]:
    return [{"elapsed_s": t, "author": "Supervisor", "event_type": "text"} for t in times]


# --------------------------------------------------------------------------- #
# load_events
# --------------------------------------------------------------------------- #


def test_load_events_reads_every_line_in_order(tmp_path: Path):
    path = write_trace(tmp_path / "t.jsonl", trace(0.0, 1.0, 2.5))
    assert [e["elapsed_s"] for e in load_events(path)] == [0.0, 1.0, 2.5]


def test_load_events_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"elapsed_s": 0.0}\n\n\n{"elapsed_s": 1.0}\n', encoding="utf-8")
    assert len(load_events(path)) == 2


def test_load_events_skips_a_malformed_line_rather_than_raising(tmp_path: Path):
    """Half a demo beats a traceback in front of an audience."""
    path = tmp_path / "t.jsonl"
    path.write_text('{"elapsed_s": 0.0}\nnot json at all\n{"elapsed_s": 1.0}\n', encoding="utf-8")
    assert [e["elapsed_s"] for e in load_events(path)] == [0.0, 1.0]


def test_load_events_skips_a_line_that_is_not_an_object(tmp_path: Path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"elapsed_s": 0.0}\n[1, 2, 3]\n', encoding="utf-8")
    assert len(load_events(path)) == 1


def test_load_events_on_an_empty_file_returns_an_empty_list(tmp_path: Path):
    path = tmp_path / "t.jsonl"
    path.write_text("", encoding="utf-8")
    assert load_events(path) == []


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def test_replay_yields_events_in_file_order(tmp_path: Path):
    records = [
        {"elapsed_s": 0.0, "author": "Supervisor", "event_type": "text"},
        {"elapsed_s": 0.8, "author": "Profiler", "event_type": "transfer"},
        {"elapsed_s": 2.1, "author": "Judge", "event_type": "function_response"},
    ]
    path = write_trace(tmp_path / "t.jsonl", records)
    replayed = list(replay_events(path, speed=0, sleep=FakeSleep()))
    assert [e["author"] for e in replayed] == ["Supervisor", "Profiler", "Judge"]


def test_replay_does_not_rewrite_the_recorded_timestamps(tmp_path: Path):
    """Playing at 2x must still be able to show what the original run took."""
    path = write_trace(tmp_path / "t.jsonl", trace(0.0, 1.0, 4.0))
    replayed = list(replay_events(path, speed=2.0, sleep=FakeSleep()))
    assert [e["elapsed_s"] for e in replayed] == [0.0, 1.0, 4.0]


# --------------------------------------------------------------------------- #
# Pacing
# --------------------------------------------------------------------------- #


def test_the_first_event_never_waits(tmp_path: Path):
    """`elapsed_s` is an offset from the run's start, not a delay before the first frame."""
    path = write_trace(tmp_path / "t.jsonl", trace(5.0, 6.0))
    sleep = FakeSleep()
    list(replay_events(path, speed=1.0, sleep=sleep))
    assert sleep.delays == [1.0]


def test_gaps_are_the_differences_between_consecutive_events(tmp_path: Path):
    path = write_trace(tmp_path / "t.jsonl", trace(0.0, 0.8, 1.2, 2.1))
    sleep = FakeSleep()
    list(replay_events(path, speed=1.0, sleep=sleep))
    assert sleep.delays == pytest.approx([0.8, 0.4, 0.9])


def test_double_speed_halves_every_gap(tmp_path: Path):
    path = write_trace(tmp_path / "t.jsonl", trace(0.0, 1.0, 3.0))
    sleep = FakeSleep()
    list(replay_events(path, speed=2.0, sleep=sleep))
    assert sleep.delays == pytest.approx([0.5, 1.0])


def test_half_speed_doubles_every_gap(tmp_path: Path):
    path = write_trace(tmp_path / "t.jsonl", trace(0.0, 1.0))
    sleep = FakeSleep()
    list(replay_events(path, speed=0.5, sleep=sleep))
    assert sleep.delays == pytest.approx([2.0])


def test_speed_zero_never_sleeps_at_all(tmp_path: Path):
    path = write_trace(tmp_path / "t.jsonl", trace(0.0, 1.0, 9.0))
    sleep = FakeSleep()
    events = list(replay_events(path, speed=0, sleep=sleep))
    assert sleep.delays == []
    assert len(events) == 3


def test_a_negative_speed_is_treated_as_instant_not_as_a_rewind():
    assert gap_seconds(0.0, 5.0, -1.0) == 0.0


def test_an_out_of_order_trace_never_sleeps_a_negative_duration():
    assert gap_seconds(5.0, 1.0, 1.0) == 0.0


def test_a_long_pause_in_the_original_run_is_clamped():
    """Replaying a coffee break faithfully is a dashboard that looks hung."""
    assert gap_seconds(0.0, 600.0, 1.0) == MAX_GAP_S


def test_an_event_with_no_timestamp_inherits_its_predecessors(tmp_path: Path):
    records = [
        {"elapsed_s": 1.0, "author": "A", "event_type": "text"},
        {"author": "B", "event_type": "text"},  # no elapsed_s at all
        {"elapsed_s": 2.0, "author": "C", "event_type": "text"},
    ]
    path = write_trace(tmp_path / "t.jsonl", records)
    sleep = FakeSleep()
    assert len(list(replay_events(path, speed=1.0, sleep=sleep))) == 3
    assert sleep.delays == pytest.approx([1.0])  # 0 for B (inherited), 1.0 for C


def test_pace_events_works_on_an_in_memory_list():
    sleep = FakeSleep()
    events = list(pace_events(trace(0.0, 2.0), speed=1.0, sleep=sleep))
    assert len(events) == 2
    assert sleep.delays == pytest.approx([2.0])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def test_total_duration_is_the_last_events_timestamp():
    assert total_duration_s(trace(0.0, 1.0, 13.5)) == 13.5


def test_total_duration_of_an_empty_trace_is_zero():
    assert total_duration_s([]) == 0.0


def test_list_traces_returns_newest_first(tmp_path: Path):
    import os
    import time

    old = write_trace(tmp_path / "old.jsonl", trace(0.0))
    write_trace(tmp_path / "new.jsonl", trace(0.0))
    os.utime(old, (time.time() - 100, time.time() - 100))
    assert [p.name for p in list_traces(tmp_path)] == ["new.jsonl", "old.jsonl"]


def test_list_traces_on_a_missing_directory_is_empty_not_an_error(tmp_path: Path):
    assert list_traces(tmp_path / "nope") == []


# --------------------------------------------------------------------------- #
# The committed fixture — the fallback demo
# --------------------------------------------------------------------------- #


def test_the_sample_trace_exists_and_has_twelve_events():
    assert SAMPLE.exists(), f"{SAMPLE} is the recorded fallback demo and must be committed"
    assert len(load_events(SAMPLE)) == 12


def test_the_sample_trace_exercises_every_event_type():
    from kernelsmith.ui.event_capture import EVENT_TYPES

    kinds = {event["event_type"] for event in load_events(SAMPLE)}
    assert kinds == set(EVENT_TYPES)


def test_the_sample_trace_has_the_recorded_timings():
    expected = [0.0, 0.8, 1.2, 2.1, 2.3, 2.5, 4.0, 8.5, 12.0, 12.2, 13.0, 13.5]
    assert [e["elapsed_s"] for e in load_events(SAMPLE)] == expected


def test_every_line_of_the_sample_trace_carries_the_full_record_shape():
    """A field the dashboard reads but the fixture omits is a KeyError mid-recording."""
    fields = {
        "elapsed_s",
        "author",
        "event_type",
        "content_text",
        "function_calls",
        "function_responses",
        "state_delta",
        "transfer_to",
        "escalate",
        "partial",
        "is_final",
    }
    for event in load_events(SAMPLE):
        assert fields <= set(event), f"missing {fields - set(event)}"


def test_the_sample_trace_replays_in_under_a_second_at_instant_speed():
    sleep = FakeSleep()
    assert len(list(replay_events(SAMPLE, speed=0, sleep=sleep))) == 12
    assert sleep.delays == []
