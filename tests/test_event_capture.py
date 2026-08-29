"""`EventLogger` writes a trace a replay can read back (Task 12, Part A).

Everything here uses duck-typed stand-ins rather than real ADK `Event` objects. That is
deliberate: `event_capture` is written to survive whatever ADK hands it, so testing it
against a hand-built `Event` would only prove it works on the happy path that already
works. The stubs here are the shapes that actually break serializers — a Pydantic model
in a `state_delta`, a `__repr__`-only object in a tool argument, a `get_function_calls`
that raises.

The one property that is not obvious from the code and is asserted anyway: every line
is flushed. A run that dies halfway is exactly the run whose trace you need, so the
file has to be readable while the process is still alive.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelsmith.ui.event_capture import (
    EVENT_TYPES,
    EventLogger,
    classify_event,
    event_to_dict,
)

# --------------------------------------------------------------------------- #
# Stand-ins
# --------------------------------------------------------------------------- #


class FakeActions:
    def __init__(self, *, state_delta=None, transfer_to_agent=None, escalate=False):
        self.state_delta = state_delta
        self.transfer_to_agent = transfer_to_agent
        self.escalate = escalate


class FakePart:
    def __init__(self, text=None, thought=False):
        self.text = text
        self.thought = thought


class FakeContent:
    def __init__(self, parts):
        self.parts = parts


class FakeCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class FakeResponse:
    def __init__(self, name, response):
        self.name = name
        self.response = response


class FakeEvent:
    """Enough of an ADK `Event` for `event_capture` to do its job."""

    def __init__(
        self,
        author="Supervisor",
        text=None,
        calls=(),
        responses=(),
        actions=None,
        partial=False,
        final=False,
        thought=None,
    ):
        parts = []
        if thought is not None:
            parts.append(FakePart(text=thought, thought=True))
        if text is not None:
            parts.append(FakePart(text=text))
        self.author = author
        self.content = FakeContent(parts) if parts else None
        self.actions = actions if actions is not None else FakeActions()
        self.partial = partial
        self._calls = list(calls)
        self._responses = list(responses)
        self._final = final

    def get_function_calls(self):
        return self._calls

    def get_function_responses(self):
        return self._responses

    def is_final_response(self):
        return self._final


@pytest.fixture
def logger(tmp_path: Path) -> EventLogger:
    return EventLogger(str(tmp_path / "traces"))


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------------------- #
# Classification — the five types, in the order the spec resolves them
# --------------------------------------------------------------------------- #


def test_a_tool_call_classifies_as_function_call():
    event = FakeEvent(calls=[FakeCall("verify_kernel", {"entrypoint": "f"})])
    assert classify_event(event) == "function_call"


def test_a_tool_response_classifies_as_function_response():
    event = FakeEvent(responses=[FakeResponse("verify_kernel", {"reward": 3})])
    assert classify_event(event) == "function_response"


def test_a_transfer_classifies_as_transfer():
    event = FakeEvent(actions=FakeActions(transfer_to_agent="Profiler"))
    assert classify_event(event) == "transfer"


def test_an_escalation_classifies_as_escalate():
    event = FakeEvent(actions=FakeActions(escalate=True))
    assert classify_event(event) == "escalate"


def test_a_plain_message_classifies_as_text():
    assert classify_event(FakeEvent(text="I'll profile the model first.")) == "text"


def test_a_call_outranks_the_text_that_accompanies_it():
    """An event that both calls a tool and narrates it is a call: the call is the decision."""
    event = FakeEvent(text="Verifying now.", calls=[FakeCall("verify_kernel", {})])
    assert classify_event(event) == "function_call"


def test_every_classification_is_one_of_the_declared_types():
    events = [
        FakeEvent(calls=[FakeCall("t", {})]),
        FakeEvent(responses=[FakeResponse("t", {})]),
        FakeEvent(actions=FakeActions(transfer_to_agent="Coder")),
        FakeEvent(actions=FakeActions(escalate=True)),
        FakeEvent(text="hello"),
        FakeEvent(),
    ]
    assert {classify_event(event) for event in events} <= set(EVENT_TYPES)


def test_an_object_that_is_not_an_event_at_all_classifies_as_text():
    """Never raise on the way into a log file."""
    assert classify_event(object()) == "text"


# --------------------------------------------------------------------------- #
# Record shape
# --------------------------------------------------------------------------- #


def test_the_record_carries_every_declared_field():
    record = event_to_dict(FakeEvent(text="hi"), 1.5)
    assert set(record) == {
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


def test_thought_parts_are_not_captured_as_content():
    """Private reasoning must not land on a recorded screen — only what the agent said."""
    record = event_to_dict(FakeEvent(thought="secretly unsure", text="Profiling now."), 0.0)
    assert record["content_text"] == "Profiling now."


def test_calls_and_responses_are_flattened_to_name_and_payload():
    event = FakeEvent(
        calls=[FakeCall("verify_kernel", {"entrypoint": "rmsnorm_triton"})],
        responses=[FakeResponse("profile_op", {"ai": 0.42})],
    )
    record = event_to_dict(event, 2.0)
    assert record["function_calls"] == [
        {"name": "verify_kernel", "args": {"entrypoint": "rmsnorm_triton"}}
    ]
    assert record["function_responses"] == [{"name": "profile_op", "response": {"ai": 0.42}}]


# --------------------------------------------------------------------------- #
# Serialization fallbacks
# --------------------------------------------------------------------------- #


class Unserializable:
    def __repr__(self):
        return "<Unserializable object>"


def test_a_non_serializable_leaf_falls_back_to_repr():
    event = FakeEvent(calls=[FakeCall("t", {"handle": Unserializable()})])
    record = event_to_dict(event, 0.0)
    assert record["function_calls"][0]["args"]["handle"] == "<Unserializable object>"


def test_the_fallback_is_per_leaf_not_per_container():
    """One awkward value must not reduce a whole state_delta to a string."""
    event = FakeEvent(actions=FakeActions(state_delta={"reward": 3, "handle": Unserializable()}))
    record = event_to_dict(event, 0.0)
    assert record["state_delta"]["reward"] == 3
    assert record["state_delta"]["handle"] == "<Unserializable object>"


def test_a_pydantic_model_in_the_state_delta_is_dumped_not_repred():
    from kernelsmith.memory.schemas import BottleneckFingerprint

    fingerprint = BottleneckFingerprint(
        op_family="norm",
        hardware="L4",
        memory_throughput_gbps=212.4,
        achieved_occupancy=0.62,
        arithmetic_intensity=0.5,
        is_memory_bound=True,
        is_compute_bound=False,
        tile_size_hint=1024,
    )
    record = event_to_dict(FakeEvent(actions=FakeActions(state_delta={"fp": fingerprint})), 0.0)
    assert record["state_delta"]["fp"]["op_family"] == "norm"


def test_an_event_whose_accessors_raise_still_produces_a_record():
    class Hostile(FakeEvent):
        def get_function_calls(self):
            raise RuntimeError("boom")

        def is_final_response(self):
            raise RuntimeError("boom")

    record = event_to_dict(Hostile(text="still here"), 0.0)
    assert record["event_type"] == "text"
    assert record["is_final"] is False


# --------------------------------------------------------------------------- #
# The logger
# --------------------------------------------------------------------------- #


def test_start_trace_creates_the_directory_and_the_file(logger: EventLogger):
    path = logger.start_trace("run-1")
    assert path.exists()
    assert path.name == "run-1.jsonl"
    logger.end_trace()


def test_it_writes_one_valid_json_line_per_event(logger: EventLogger):
    path = logger.start_trace("run-2")
    logger.log_event(FakeEvent(author="Supervisor", text="one"))
    logger.log_event(FakeEvent(author="Profiler", calls=[FakeCall("profile_op", {})]))
    logger.end_trace()

    records = read_lines(path)
    assert [r["author"] for r in records] == ["Supervisor", "Profiler"]
    assert [r["event_type"] for r in records] == ["text", "function_call"]


def test_each_line_is_flushed_before_the_trace_is_closed(logger: EventLogger):
    """A run that dies halfway must still leave a readable trace."""
    path = logger.start_trace("run-3")
    logger.log_event(FakeEvent(text="written"))
    assert len(read_lines(path)) == 1  # read while the handle is still open
    logger.end_trace()


def test_elapsed_seconds_are_monotonic_and_start_near_zero(logger: EventLogger):
    path = logger.start_trace("run-4")
    for _ in range(3):
        logger.log_event(FakeEvent(text="tick"))
    logger.end_trace()

    elapsed = [record["elapsed_s"] for record in read_lines(path)]
    assert elapsed[0] < 1.0
    assert elapsed == sorted(elapsed)


def test_logging_before_start_trace_is_a_no_op(logger: EventLogger):
    logger.log_event(FakeEvent(text="nowhere to go"))
    assert logger.events_written == 0
    assert not logger.is_open


def test_end_trace_closes_the_file_and_is_idempotent(logger: EventLogger):
    logger.start_trace("run-5")
    logger.log_event(FakeEvent(text="one"))
    assert logger.end_trace() is not None
    assert not logger.is_open
    assert logger.end_trace() is None


def test_logging_after_end_trace_is_a_no_op(logger: EventLogger):
    path = logger.start_trace("run-6")
    logger.log_event(FakeEvent(text="one"))
    logger.end_trace()
    logger.log_event(FakeEvent(text="two"))
    assert len(read_lines(path)) == 1


def test_a_second_start_trace_closes_the_first(logger: EventLogger):
    """Two runs must not interleave into one trace."""
    first = logger.start_trace("run-a")
    logger.log_event(FakeEvent(text="a"))
    second = logger.start_trace("run-b")
    logger.log_event(FakeEvent(text="b"))
    logger.end_trace()

    assert [r["content_text"] for r in read_lines(first)] == ["a"]
    assert [r["content_text"] for r in read_lines(second)] == ["b"]


def test_a_serialization_failure_writes_a_marked_line_rather_than_raising(
    logger: EventLogger, monkeypatch
):
    """Point 2 of the module contract: logging never raises into the agent loop."""
    import kernelsmith.ui.event_capture as capture

    def explode(*_args, **_kwargs):
        raise RuntimeError("serializer exploded")

    monkeypatch.setattr(capture, "event_to_dict", explode)
    path = logger.start_trace("run-7")
    logger.log_event(FakeEvent(author="Judge", text="unloggable"))
    logger.end_trace()

    (record,) = read_lines(path)
    assert record["author"] == "Judge"
    assert "serializer exploded" in record["error"]


def test_the_context_manager_ends_the_trace(tmp_path: Path):
    with EventLogger(str(tmp_path)) as logger:
        path = logger.start_trace("run-8")
        logger.log_event(FakeEvent(text="one"))
    assert not logger.is_open
    assert len(read_lines(path)) == 1
