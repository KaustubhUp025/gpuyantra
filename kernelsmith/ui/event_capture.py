"""Capture ADK events to JSONL so a run can be replayed without a GPU (Task 12).

The demo is recorded once, but rehearsed a dozen times. A live run needs Vertex AI, a
Firestore index, an L4 and about ninety seconds of luck; a recorded trace needs a file.
This module writes the trace.

Three properties are load-bearing:

1. **Every line is flushed.** A run that dies halfway still leaves a usable trace, which
   is exactly the run whose trace you most want. Buffering would throw away the tail of
   the only recording that mattered.

2. **Logging never raises into the agent loop.** `log_event` is called from
   `EventStreamConsumer`'s background thread, inside the `async for` that is draining
   `Runner.run_async`. An exception there would kill the run to protect a log file.
   Every failure degrades to a written line with an `error` field, or to nothing.

3. **Non-serializable fields degrade per value, not per event.** ADK `Event` objects
   carry `types.Part`, Pydantic models and occasionally raw protos. `_jsonable` walks
   containers and falls back to `repr()` only at the leaf that actually failed, so one
   awkward tool argument does not reduce a whole `state_delta` to a string.

Traces live under `data/traces/`, never `/tmp` — CLAUDE.md rule 15. The VM is
preemptible and `/tmp` does not survive a restart, which is the one moment you would
reach for a recorded fallback.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Any

logger = logging.getLogger(__name__)

#: Where traces are written. Relative to the working directory, which for `make demo`
#: and `make serve-demo` is the repo root.
DEFAULT_TRACE_DIR = "data/traces"

#: The five values `event_type` can take. Ordered as they are tested in
#: `classify_event`; the order is the classification, so it is not cosmetic.
EVENT_TYPES = ("function_call", "function_response", "transfer", "escalate", "text")

#: How deep `_jsonable` will walk a nested container before it gives up and reprs.
#: A `state_delta` carrying a whole KernelDraft is about four levels; ten is slack.
MAX_JSON_DEPTH = 10


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def classify_event(event: Any) -> str:
    """Name what an ADK `Event` *did*, in one word.

    The order matters and is the spec's: an event that both calls a tool and carries
    text is a `function_call`, because the call is the decision and the text is
    commentary on it. Duck-typed, and never raises — an event nobody can classify is
    "text", which renders as a plain line rather than as an exception.
    """
    if _function_calls(event):
        return "function_call"
    if _function_responses(event):
        return "function_response"

    actions = getattr(event, "actions", None)
    if actions is not None:
        if getattr(actions, "transfer_to_agent", None):
            return "transfer"
        if getattr(actions, "escalate", False):
            return "escalate"
    return "text"


def _function_calls(event: Any) -> list[Any]:
    """`event.get_function_calls()`, or an empty list for anything that lacks it."""
    getter = getattr(event, "get_function_calls", None)
    if getter is None:
        return []
    try:
        return list(getter() or [])
    except Exception:  # noqa: BLE001 — a malformed event classifies as text, not a crash
        return []


def _function_responses(event: Any) -> list[Any]:
    """`event.get_function_responses()`, or an empty list."""
    getter = getattr(event, "get_function_responses", None)
    if getter is None:
        return []
    try:
        return list(getter() or [])
    except Exception:  # noqa: BLE001
        return []


def _first_text(event: Any) -> str | None:
    """The first textual part of the event's content, ignoring thought parts.

    Thoughts are the model's private reasoning; the demo timeline shows what the agent
    said, and mixing the two would put unedited chain-of-thought on a recorded screen.
    """
    parts = getattr(getattr(event, "content", None), "parts", None) or []
    for part in parts:
        if getattr(part, "thought", False):
            continue
        text = getattr(part, "text", None)
        if text:
            return str(text)
    return None


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


def _jsonable(value: Any, depth: int = 0) -> Any:
    """Coerce anything into something `json.dumps` will accept.

    Tries the cheap path first (the value is already JSON), then walks containers so a
    single bad leaf costs one `repr()` rather than the whole subtree. Pydantic models
    are dumped through `model_dump` before the container walk, because `vars()` on one
    yields private fields nobody wants in a trace.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value

    if depth >= MAX_JSON_DEPTH:
        return repr(value)

    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _jsonable(dump(mode="json"), depth + 1)
        except Exception:  # noqa: BLE001 — fall through to the container walk
            pass

    if isinstance(value, dict):
        return {str(key): _jsonable(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(item, depth + 1) for item in value]

    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


def event_to_dict(event: Any, elapsed_s: float) -> dict[str, Any]:
    """Reduce an ADK `Event` to the flat record the trace format stores.

    Pure, duck-typed and total: every field has a defined value for an event that is
    missing it, so a replayed trace never has to guard for absence. `elapsed_s` is
    passed in rather than read from a clock so this stays a pure function — the tests
    assert on exact records.
    """
    calls = [
        {
            "name": str(getattr(call, "name", "") or ""),
            "args": _jsonable(getattr(call, "args", None) or {}),
        }
        for call in _function_calls(event)
    ]
    responses = [
        {
            "name": str(getattr(response, "name", "") or ""),
            "response": _jsonable(getattr(response, "response", None) or {}),
        }
        for response in _function_responses(event)
    ]

    actions = getattr(event, "actions", None)
    state_delta = _jsonable(getattr(actions, "state_delta", None) or None) if actions else None
    transfer_to = str(getattr(actions, "transfer_to_agent", "") or "") or None if actions else None
    escalate = bool(getattr(actions, "escalate", False)) if actions else False

    try:
        is_final = bool(event.is_final_response())
    except Exception:  # noqa: BLE001 — a display flag is never worth an exception
        is_final = False

    return {
        "elapsed_s": round(float(elapsed_s), 3),
        "author": str(getattr(event, "author", "") or "unknown"),
        "event_type": classify_event(event),
        "content_text": _first_text(event),
        "function_calls": calls,
        "function_responses": responses,
        "state_delta": state_delta or None,
        "transfer_to": transfer_to,
        "escalate": escalate,
        "partial": bool(getattr(event, "partial", False)),
        "is_final": is_final,
    }


# --------------------------------------------------------------------------- #
# The logger
# --------------------------------------------------------------------------- #


class EventLogger:
    """Append ADK events to a JSONL trace, one JSON object per line.

    Not reentrant across traces: `start_trace` on an already-open logger closes the
    previous file first, so a second run cannot interleave into the first one's trace.

    Thread-safe, because `EventStreamConsumer` calls `log_event` from its background
    event-loop thread while Streamlit reads from the main one.
    """

    def __init__(self, output_dir: str = DEFAULT_TRACE_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.run_id: str | None = None
        self.path: Path | None = None
        self.start_time: float | None = None
        self.events_written = 0
        self._handle: Any = None
        self._lock = threading.Lock()

    # -------------------------------------------------------------- lifecycle

    def start_trace(self, run_id: str) -> Path:
        """Open `{output_dir}/{run_id}.jsonl` and start the clock.

        Returns the path so a caller can show it before the run produces anything.
        """
        with self._lock:
            self._close_locked()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.run_id = run_id
            self.path = self.output_dir / f"{run_id}.jsonl"
            # `w`, not `a`: re-running the same run_id replaces its trace rather than
            # producing a file whose second half disagrees with its first.
            self._handle = self.path.open("w", encoding="utf-8")
            self.start_time = time.monotonic()
            self.events_written = 0
            return self.path

    def log_event(self, event: Any) -> None:
        """Serialize one event and write it as a line. Never raises.

        A logger that has not been started is a no-op, so wiring one into
        `EventStreamConsumer` before the first `start_trace` is harmless.
        """
        with self._lock:
            if self._handle is None or self.start_time is None:
                return
            elapsed = time.monotonic() - self.start_time
            try:
                record = event_to_dict(event, elapsed)
                line = json.dumps(record, default=repr)
            except Exception as exc:  # noqa: BLE001 — see module docstring, point 2
                logger.exception("event serialization failed")
                # A trace with a hole in it beats a run killed by its own logging, and
                # beats a silently short trace: the hole says so.
                line = json.dumps(
                    {
                        "elapsed_s": round(elapsed, 3),
                        "author": str(getattr(event, "author", "") or "unknown"),
                        "event_type": "text",
                        "content_text": None,
                        "function_calls": [],
                        "function_responses": [],
                        "state_delta": None,
                        "transfer_to": None,
                        "escalate": False,
                        "partial": False,
                        "is_final": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "repr": repr(event)[:2000],
                    },
                    default=repr,
                )
            try:
                self._handle.write(line + "\n")
                self._handle.flush()  # see module docstring, point 1
                self.events_written += 1
            except Exception:  # noqa: BLE001 — a full disk must not end the run
                logger.exception("event write failed")

    def end_trace(self) -> Path | None:
        """Close the file and return its path, printing where it landed.

        Idempotent: calling it twice, or without a `start_trace`, returns None the
        second time rather than raising.
        """
        with self._lock:
            path = self.path if self._handle is not None else None
            written = self.events_written
            self._close_locked()
        if path is not None:
            print(f"[trace] {written} events → {path}")
        return path

    def _close_locked(self) -> None:
        """Close the handle. Caller holds `self._lock`."""
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            handle.close()
        except Exception:  # noqa: BLE001
            logger.exception("closing the trace file failed")

    # ------------------------------------------------------------ conveniences

    @property
    def is_open(self) -> bool:
        """True between `start_trace` and `end_trace`."""
        return self._handle is not None

    def __enter__(self) -> EventLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.end_trace()
