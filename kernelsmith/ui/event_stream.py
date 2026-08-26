"""Consuming ADK's async event stream from Streamlit's synchronous reruns (spec 10.2).

Streamlit re-executes the whole script top-to-bottom on every interaction, and there is
no event loop it will keep alive between those runs. ADK's `Runner.run_async` is an
async generator that must stay alive for the length of a whole agent run — minutes,
across many reruns. The two models are incompatible, so this module bridges them with
the pattern the spec calls the safe approach:

    Streamlit thread            background thread
    ----------------            -----------------
    start_run()      --submit-->  loop.run_forever()
                                  async for event in runner.run_async(...)
    drain_events()   <--Queue---      queue.put(event)

The `queue.Queue` is the only shared mutable state, and it is thread-safe by
construction. Streamlit never awaits anything; the background loop never touches
Streamlit. A rerun that happens mid-run simply drains whatever has arrived so far.

Run *status* is deliberately kept off the queue. The queue carries ADK `Event` objects
and nothing else, so callers can iterate it without type-sniffing; whether a run is
still going, and why it stopped, are read from `is_running` / `last_error` instead.

On the two-message protocol: an ADK `LoopAgent` cannot transfer back to its parent, so
the Supervisor's turn ends when the RefinementLoop escalates and the upsert + hot-swap
steps run on the *next* turn (see `.claude/rules/implementation-deviations.md`). That
makes it the driver's job — here, `streamlit_app.py` — to call `start_run` a second
time once the first run goes idle. This class deliberately runs one turn at a time and
refuses overlapping runs, which is what makes that follow-up safe to trigger from a
1 Hz refresh loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # Import cost matters: the module is imported on every Streamlit rerun.
    from google.adk.events import Event
    from google.adk.runners import Runner

logger = logging.getLogger(__name__)

#: Tool arguments include whole Triton kernel sources; the panels only need a glimpse.
ARG_PREVIEW_CHARS = 200
#: Model turns are shown in an expander, not read in full.
TEXT_PREVIEW_CHARS = 1200
#: Tool responses the bottom banner is built from, kept whole. Everything else is
#: previewed, so a retrieval hit carrying three kernel sources cannot bloat the UI.
#: `explain_kernel` is here for the same reason: the right column renders the whole
#: explanation, and a 200-char preview of it is not an explanation.
BANNER_TOOLS = frozenset({"verify_kernel", "hotswap_kernel", "explain_kernel"})
#: How long to wait for the background loop to come up before giving up on it.
LOOP_START_TIMEOUT_S = 5.0


# --------------------------------------------------------------------------- #
# Event summaries (pure functions — no Streamlit, no ADK imports at runtime)
# --------------------------------------------------------------------------- #


@dataclass
class EventSummary:
    """The parts of an ADK `Event` the dashboard renders.

    Summarising on drain rather than storing raw events keeps `session_state` bounded:
    a single `verify_kernel` call carries a full kernel source, and a run produces
    dozens of them.
    """

    author: str = "unknown"
    timestamp: float = 0.0
    thought: str = ""
    text: str = ""
    #: (tool_name, preview of args)
    calls: list[tuple[str, str]] = field(default_factory=list)
    #: (tool_name, response) — response is the whole payload only for BANNER_TOOLS.
    responses: list[tuple[str, Any]] = field(default_factory=list)
    escalate: bool = False
    transfer_to: str = ""
    is_final: bool = False

    @property
    def action(self) -> str:
        """One line naming what this event *did*, as opposed to what it said."""
        if self.calls:
            return " · ".join(f"{name}({args})" for name, args in self.calls)
        if self.responses:
            return " · ".join(f"{name} → {_preview(response)}" for name, response in self.responses)
        if self.transfer_to:
            return f"transfer → {self.transfer_to}"
        if self.escalate:
            return "escalate (loop exit)"
        return ""


def summarize_event(event: Any) -> EventSummary:
    """Reduce an ADK `Event` to the fields the panels read. Never raises.

    Duck-typed on purpose: this runs on every event of every run, and a malformed or
    partial event must degrade to an empty panel line rather than break the dashboard.
    """
    summary = EventSummary(
        author=str(getattr(event, "author", "") or "unknown"),
        timestamp=float(getattr(event, "timestamp", 0.0) or 0.0),
    )

    actions = getattr(event, "actions", None)
    if actions is not None:
        summary.escalate = bool(getattr(actions, "escalate", False))
        summary.transfer_to = str(getattr(actions, "transfer_to_agent", "") or "")

    try:
        summary.is_final = bool(event.is_final_response())
    except Exception:  # noqa: BLE001 — a display flag is never worth an exception
        summary.is_final = False

    thoughts: list[str] = []
    texts: list[str] = []
    for part in getattr(getattr(event, "content", None), "parts", None) or []:
        text = getattr(part, "text", None)
        if text:
            (thoughts if getattr(part, "thought", False) else texts).append(str(text))

        call = getattr(part, "function_call", None)
        if call is not None and getattr(call, "name", None):
            summary.calls.append((str(call.name), _preview_args(getattr(call, "args", None))))

        response = getattr(part, "function_response", None)
        if response is not None and getattr(response, "name", None):
            name = str(response.name)
            payload = getattr(response, "response", None)
            summary.responses.append((name, payload if name in BANNER_TOOLS else _preview(payload)))

    summary.thought = _truncate("\n".join(thoughts), TEXT_PREVIEW_CHARS)
    summary.text = _truncate("\n".join(texts), TEXT_PREVIEW_CHARS)
    return summary


def _preview_args(args: Any) -> str:
    """Render tool arguments compactly, with kernel sources cut down to a glimpse."""
    if not isinstance(args, dict) or not args:
        return ""
    rendered = ", ".join(f"{key}={_preview(value, 60)}" for key, value in args.items())
    return _truncate(rendered, ARG_PREVIEW_CHARS)


def _preview(value: Any, limit: int = ARG_PREVIEW_CHARS) -> str:
    """A short one-line rendering of anything at all."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, default=str)
        except (TypeError, ValueError):
            text = str(value)
    return _truncate(" ".join(text.split()), limit)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


# --------------------------------------------------------------------------- #
# The consumer
# --------------------------------------------------------------------------- #


class EventStreamConsumer:
    """A background asyncio loop that drains `Runner.run_async` into a thread-safe queue.

    One instance per Streamlit session, held by `@st.cache_resource` so it survives
    reruns. The thread is a daemon: Streamlit's own shutdown does not wait on it.
    """

    def __init__(self, thread_name: str = "kernelsmith-event-loop") -> None:
        self._queue: queue.Queue[Any] = queue.Queue()
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._lock = threading.Lock()
        self._future: Future | None = None
        self._running = False
        self._error: str | None = None
        self._runs_completed = 0

        self._thread = threading.Thread(target=self._run_forever, name=thread_name, daemon=True)
        self._thread.start()
        if not self._loop_ready.wait(timeout=LOOP_START_TIMEOUT_S):
            raise RuntimeError(
                f"event-loop thread did not start within {LOOP_START_TIMEOUT_S}s; "
                "the dashboard cannot drive the agent tree"
            )

    # ------------------------------------------------------------------ status

    @property
    def is_running(self) -> bool:
        """True while a turn is in flight. The driver waits on this before following up."""
        with self._lock:
            return self._running

    @property
    def last_error(self) -> str | None:
        """Why the last turn stopped early, or None if it finished cleanly."""
        with self._lock:
            return self._error

    @property
    def runs_completed(self) -> int:
        """Turns that have ended, successfully or not. Rises exactly once per turn."""
        with self._lock:
            return self._runs_completed

    @property
    def pending(self) -> int:
        """Events sitting in the queue, not yet drained."""
        return self._queue.qsize()

    # ------------------------------------------------------------------- drive

    def start_run(
        self,
        runner: Runner,
        user_id: str,
        session_id: str,
        message: str,
        *,
        state_delta: dict[str, Any] | None = None,
    ) -> bool:
        """Submit one agent turn to the background loop.

        Returns False without doing anything if a turn is already in flight — the
        dashboard refreshes at 1 Hz and would otherwise launch a second Supervisor over
        the same session on the next tick.

        Args:
            runner: The ADK Runner. Its `session_service` is used to create the session
                on first use, since Runner defaults to `auto_create_session=False`.
            user_id: Session owner.
            session_id: Session to run in. Reused across turns, which is what makes the
                Supervisor's follow-up turn resume from the state the first one left.
            message: The user message for this turn.
            state_delta: Optional seed for `session.state` (the dashboard seeds
                `task_spec` this way on the first turn).
        """
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._error = None

        self._future = asyncio.run_coroutine_threadsafe(
            self._consume(runner, user_id, session_id, message, state_delta), self._loop
        )
        return True

    def drain_events(self, limit: int | None = None) -> list[Event]:
        """Return every event currently queued, without blocking.

        Args:
            limit: Stop after this many events, leaving the rest for the next drain.
                Bounds the work one rerun can do when a burst arrives.
        """
        drained: list[Any] = []
        while limit is None or len(drained) < limit:
            try:
                drained.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return drained

    def cancel(self) -> None:
        """Ask the in-flight turn to stop. The agent tree may take a moment to notice."""
        future = self._future
        if future is not None and not future.done():
            self._loop.call_soon_threadsafe(future.cancel)

    def close(self, timeout: float = LOOP_START_TIMEOUT_S) -> None:
        """Wait out any in-flight turn, then stop the background loop.

        For tests and teardown; the dashboard keeps its consumer for the life of the
        server. Waiting before stopping matters: `loop.stop()` while a turn is still
        queued leaves the coroutine scheduled but never awaited, which surfaces as a
        `RuntimeWarning` and a destroyed pending task.
        """
        future = self._future
        if future is not None:
            try:
                future.result(timeout=timeout)
            except BaseException:  # noqa: BLE001 — a run's own failure is in `last_error`
                self.cancel()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=timeout)

    # --------------------------------------------------------------- internals

    def _run_forever(self) -> None:
        """Thread body: own an event loop and keep it spinning."""
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._loop_ready.set)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    async def _consume(
        self,
        runner: Runner,
        user_id: str,
        session_id: str,
        message: str,
        state_delta: dict[str, Any] | None,
    ) -> None:
        """Drain one `run_async` generator into the queue. Never propagates out of the loop."""
        try:
            await self._ensure_session(runner, user_id, session_id)
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=_to_content(message),
                state_delta=state_delta or None,
            ):
                self._queue.put(event)
        except asyncio.CancelledError:
            with self._lock:
                self._error = "run cancelled"
            raise
        except BaseException as exc:  # noqa: BLE001 — a failed run is a banner, not a crash
            logger.exception("agent run failed")
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._running = False
                self._runs_completed += 1

    @staticmethod
    async def _ensure_session(runner: Runner, user_id: str, session_id: str) -> None:
        """Create the session if it is not there yet; reuse it on every later turn."""
        service = runner.session_service
        try:
            existing = await service.get_session(
                app_name=runner.app_name, user_id=user_id, session_id=session_id
            )
        except Exception:  # noqa: BLE001 — some services raise instead of returning None
            existing = None
        if existing is None:
            await service.create_session(
                app_name=runner.app_name, user_id=user_id, session_id=session_id
            )


def _to_content(message: Any) -> Any:
    """Wrap a plain string into the `types.Content` the Runner expects."""
    if isinstance(message, str):
        from google.genai import types

        return types.Content(role="user", parts=[types.Part(text=message)])
    return message
