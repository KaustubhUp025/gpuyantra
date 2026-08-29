"""Replay a captured JSONL trace with its original timing (Task 12).

`load_events` is the whole file at once; `replay_events` is the same events paced the
way they arrived. The demo dashboard's rendering code does not know which one it is
reading from — that is the point of capturing dicts rather than ADK objects.

On the pacing: the gap before an event is `elapsed_s[n] - elapsed_s[n-1]`, divided by
`speed`. `speed=1.0` is the original run, `2.0` is twice as fast, `0` is instant. The
first event fires immediately; its own `elapsed_s` is an offset from the start of the
run, not a delay anyone wants to sit through before the screen shows anything.

A malformed line is skipped rather than fatal. A trace is a recording of something that
already happened, and half a demo beats a traceback in front of an audience.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Longest gap `replay_events` will honour between two events. A trace recorded across
#: a coffee break has a two-minute hole in it; replaying that hole faithfully is a
#: dashboard that looks hung. Clamped, and the clamp is visible in the returned events'
#: own `elapsed_s`, which is never rewritten.
MAX_GAP_S = 10.0


def load_events(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Read a whole trace into memory, in file order.

    Blank lines and lines that are not JSON objects are skipped with a warning. The
    returned dicts are exactly what `EventLogger` wrote — no normalization, so a
    consumer that reads a field the capture did not write gets a `KeyError` here rather
    than a plausible-looking default that hides a schema drift.
    """
    path = Path(jsonl_path)
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("%s:%d is not valid JSON (%s) — skipped", path, number, exc)
                continue
            if not isinstance(record, dict):
                logger.warning(
                    "%s:%d is a %s, not an object — skipped", path, number, type(record).__name__
                )
                continue
            events.append(record)
    return events


def replay_events(
    jsonl_path: str | Path,
    speed: float = 1.0,
    *,
    sleep: Any = time.sleep,
) -> Generator[dict[str, Any], None, None]:
    """Yield a trace's events, paced by the gaps in the original run.

    Args:
        jsonl_path: The trace to replay.
        speed: Playback rate. 1.0 is real time, 2.0 twice as fast, 0 (or anything
            non-positive) is instant — every event yields with no sleep at all.
        sleep: Injected for tests, which assert on the delays without spending them.

    Yields:
        The event dicts, in file order. `elapsed_s` is left as recorded, so a consumer
        can still show the original timings even when replaying at 2x or instantly.
    """
    yield from pace_events(load_events(jsonl_path), speed, sleep=sleep)


def pace_events(
    events: Iterable[dict[str, Any]],
    speed: float = 1.0,
    *,
    sleep: Any = time.sleep,
) -> Generator[dict[str, Any], None, None]:
    """The pacing half of `replay_events`, over events already in memory.

    Split out because the dashboard loads a trace once (to build its turn structure)
    and then plays the same list; re-reading the file to replay it would be a second
    source of truth for what is on screen.
    """
    previous: float | None = None
    for event in events:
        current = _elapsed(event, fallback=previous)
        delay = gap_seconds(previous, current, speed)
        if delay > 0:
            sleep(delay)
        previous = current
        yield event


def gap_seconds(previous: float | None, current: float, speed: float) -> float:
    """How long to wait before an event whose predecessor was at `previous`.

    Total and defensive, because every input here comes off disk:

    - the first event (``previous is None``) never waits;
    - a non-positive `speed` means instant, not a division by zero;
    - a *negative* gap — an out-of-order or clock-skewed trace — is zero, not a
      negative sleep;
    - a gap longer than `MAX_GAP_S` is clamped, so a pause in the original run does
      not look like a hung dashboard in the replay.
    """
    if previous is None or speed <= 0:
        return 0.0
    gap = (current - previous) / speed
    if gap <= 0:
        return 0.0
    return min(gap, MAX_GAP_S)


def total_duration_s(events: Iterable[dict[str, Any]]) -> float:
    """Wall-clock length of a trace: the last event's `elapsed_s`, or 0 for an empty one."""
    last = 0.0
    for event in events:
        last = _elapsed(event, fallback=last)
    return last


def list_traces(trace_dir: str | Path) -> list[Path]:
    """Every `.jsonl` in `trace_dir`, newest first, or [] if the directory is absent.

    Newest first because the file the operator wants during a recording session is
    almost always the run that just finished.
    """
    directory = Path(trace_dir)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def _elapsed(event: dict[str, Any], fallback: float | None) -> float:
    """`event["elapsed_s"]` as a float, falling back to the previous event's time.

    An event with a missing or unparseable timestamp inherits its predecessor's, which
    makes it fire immediately rather than either crashing the replay or jumping the
    clock to zero and replaying the rest of the trace at once.
    """
    try:
        return float(event.get("elapsed_s"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(fallback or 0.0)
