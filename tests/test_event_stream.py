"""Unit tests for the dashboard's ADK event bridge (spec 10.2).

No GPU, no Vertex, no Streamlit: a fake Runner stands in for the agent tree so the
threading and summarising can be tested on their own.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from kernelsmith.ui.event_stream import EventStreamConsumer, summarize_event


class FakeSessionService:
    def __init__(self) -> None:
        self.sessions: dict[tuple[str, str, str], object] = {}

    async def get_session(self, *, app_name: str, user_id: str, session_id: str, config=None):
        return self.sessions.get((app_name, user_id, session_id))

    async def create_session(self, *, app_name: str, user_id: str, session_id=None, state=None):
        session = SimpleNamespace(id=session_id, state=state or {})
        self.sessions[(app_name, user_id, session_id)] = session
        return session


class FakeRunner:
    """Yields `count` events, or raises `error` partway through."""

    def __init__(self, count: int = 3, error: Exception | None = None) -> None:
        self.app_name = "test-app"
        self.session_service = FakeSessionService()
        self.count = count
        self.error = error
        self.calls: list[dict] = []

    async def run_async(self, *, user_id, session_id, new_message=None, state_delta=None, **kw):
        self.calls.append(
            {"user_id": user_id, "session_id": session_id, "state_delta": state_delta}
        )
        for i in range(self.count):
            yield SimpleNamespace(author=f"Agent{i}", timestamp=float(i), content=None)
        if self.error is not None:
            raise self.error


@pytest.fixture
def consumer():
    instance = EventStreamConsumer()
    yield instance
    instance.close()


def wait_idle(consumer: EventStreamConsumer, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while consumer.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not consumer.is_running, "run did not finish"


# --------------------------------------------------------------------------- consumer


def test_drains_every_event_onto_the_queue(consumer):
    runner = FakeRunner(count=5)
    assert consumer.start_run(runner, "u", "s", "go")
    wait_idle(consumer)

    drained = consumer.drain_events()
    assert [e.author for e in drained] == [f"Agent{i}" for i in range(5)]
    assert consumer.drain_events() == []  # drain is destructive
    assert consumer.last_error is None


def test_drain_events_is_non_blocking_when_empty(consumer):
    assert consumer.drain_events() == []


def test_drain_events_honours_its_limit(consumer):
    consumer.start_run(FakeRunner(count=10), "u", "s", "go")
    wait_idle(consumer)
    assert len(consumer.drain_events(limit=4)) == 4
    assert len(consumer.drain_events()) == 6


def test_creates_the_session_once_and_reuses_it(consumer):
    runner = FakeRunner(count=1)
    consumer.start_run(runner, "u", "s1", "turn one", state_delta={"task_spec": {"op": "rmsnorm"}})
    wait_idle(consumer)
    consumer.start_run(runner, "u", "s1", "turn two")
    wait_idle(consumer)

    assert len(runner.session_service.sessions) == 1
    assert runner.calls[0]["state_delta"] == {"task_spec": {"op": "rmsnorm"}}
    assert runner.calls[1]["state_delta"] is None
    assert runner.calls[1]["session_id"] == "s1"  # turn 2 resumes turn 1's state


def test_refuses_overlapping_runs(consumer):
    """The dashboard refreshes at 1 Hz; a second Supervisor on the same session is a bug."""
    slow = FakeRunner(count=200)
    assert consumer.start_run(slow, "u", "s", "first")
    if consumer.is_running:
        assert consumer.start_run(slow, "u", "s", "second") is False
    wait_idle(consumer)


def test_a_failing_run_is_reported_not_raised(consumer):
    consumer.start_run(FakeRunner(count=2, error=RuntimeError("vertex is down")), "u", "s", "go")
    wait_idle(consumer)

    assert len(consumer.drain_events()) == 2  # events before the failure survive
    assert "vertex is down" in consumer.last_error
    assert consumer.runs_completed == 1
    assert consumer.start_run(FakeRunner(count=1), "u", "s", "retry")  # and it recovers
    wait_idle(consumer)
    assert consumer.last_error is None


def test_runs_completed_advances_once_per_turn(consumer):
    """`drive_run` fires the follow-up message off this counter."""
    assert consumer.runs_completed == 0
    for expected in (1, 2):
        consumer.start_run(FakeRunner(count=1), "u", "s", "go")
        wait_idle(consumer)
        assert consumer.runs_completed == expected


# --------------------------------------------------------------------------- summaries


def part(**overrides):
    fields = {"text": None, "thought": False, "function_call": None, "function_response": None}
    return SimpleNamespace(**{**fields, **overrides})


def event(parts, author="Coder", **overrides):
    fields = {
        "author": author,
        "timestamp": 1.0,
        "content": SimpleNamespace(parts=parts),
        "actions": SimpleNamespace(escalate=False, transfer_to_agent=None),
        "is_final_response": lambda: False,
    }
    return SimpleNamespace(**{**fields, **overrides})


def test_summary_separates_thoughts_from_text():
    summary = summarize_event(
        event([part(text="thinking about tiles", thought=True), part(text="here is the kernel")])
    )
    assert summary.thought == "thinking about tiles"
    assert summary.text == "here is the kernel"
    assert summary.author == "Coder"


def test_summary_previews_tool_args_so_kernel_source_cannot_flood_the_panel():
    call = SimpleNamespace(name="verify_kernel", args={"kernel_code": "x" * 5000})
    summary = summarize_event(event([part(function_call=call)]))

    name, args = summary.calls[0]
    assert name == "verify_kernel"
    assert len(args) < 250
    assert "verify_kernel(" in summary.action


def test_banner_tool_responses_are_kept_whole():
    """The banner reads violations off the verifier's own payload, not the model's prose."""
    payload = {"reward": -1, "violations": [{"rule_id": 3, "line": 7, "description": "no-op"}]}
    response = SimpleNamespace(name="verify_kernel", response=payload)
    summary = summarize_event(event([part(function_response=response)]))

    assert summary.responses == [("verify_kernel", payload)]


def test_non_banner_tool_responses_are_previewed():
    payload = {"skills": [{"winning_kernel_source": "y" * 5000}]}
    response = SimpleNamespace(name="retrieve_skills_for_agent", response=payload)
    summary = summarize_event(event([part(function_response=response)]))

    name, preview = summary.responses[0]
    assert name == "retrieve_skills_for_agent"
    assert isinstance(preview, str) and len(preview) < 250


def test_summary_records_escalation():
    escalating = event([], author="EscalationChecker")
    escalating.actions = SimpleNamespace(escalate=True, transfer_to_agent=None)
    assert summarize_event(escalating).escalate is True
    assert "escalate" in summarize_event(escalating).action


def test_summary_survives_a_malformed_event():
    """A partial event must degrade to an empty panel line, never break the dashboard."""
    summary = summarize_event(SimpleNamespace())
    assert summary.author == "unknown"
    assert summary.thought == "" and summary.action == ""
