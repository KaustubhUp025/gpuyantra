"""The demo dashboard's pure logic, and that the page renders at all (Task 12, Part C).

`demo_dashboard.py` calls `main()` at import, the way every Streamlit script does, so it
is driven through `AppTest` for the render checks. Importing it directly for the pure
functions works because `main()` is cheap by construction: no agent tree, no Firestore,
no GPU, no network at page load.

What is worth locking down here is the part that cannot be checked by looking at the
screen during a rehearsal:

- **`extract_metrics` reads tool responses and nothing else.** Red line #3. An agent
  claiming a speedup in prose must not be able to move the header metric, and the only
  way to be sure is a test that feeds it prose saying so.
- **The graph highlights exactly one node.** A stale highlight points the audience at
  the wrong agent, which is worse than no highlight.
- **`drive_run` is reached on the live path.** Same trap `streamlit_app.py` documents:
  an early return above it silently skips turn 2, so upsert and hot-swap never run
  while every panel still looks healthy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kernelsmith.ui.demo_dashboard import (
    GRAPH_NODES,
    NODE_ACTIVE_COLOR,
    NODE_IDLE_COLOR,
    Turn,
    build_agent_graph,
    classify_event,
    extract_metrics,
    group_turns,
    highlight_active_agent,
    split_prose_and_code,
    turn_headline,
)

APP = Path(__file__).resolve().parent.parent / "kernelsmith" / "ui" / "demo_dashboard.py"
SAMPLE = Path(__file__).resolve().parent.parent / "data" / "traces" / "sample_run.jsonl"
TIMEOUT_S = 120

NODE_NAMES = [name for name, _ in GRAPH_NODES]


def event(author="Supervisor", kind="text", **extra):
    record = {
        "elapsed_s": extra.pop("elapsed_s", 0.0),
        "author": author,
        "event_type": kind,
        "content_text": extra.pop("content_text", None),
        "function_calls": extra.pop("function_calls", []),
        "function_responses": extra.pop("function_responses", []),
        "state_delta": None,
        "transfer_to": extra.pop("transfer_to", None),
        "escalate": extra.pop("escalate", False),
        "partial": extra.pop("partial", False),
        "is_final": False,
    }
    record.update(extra)
    return record


# --------------------------------------------------------------------------- #
# classify_event is the capture module's, re-exported for the dashboard's callers
# --------------------------------------------------------------------------- #


def test_classify_event_is_available_from_the_dashboard():
    from kernelsmith.ui.event_capture import classify_event as canonical

    assert classify_event is canonical


# --------------------------------------------------------------------------- #
# build_agent_graph / highlight_active_agent
# --------------------------------------------------------------------------- #


def test_the_graph_is_valid_dot_with_every_agent_and_edge():
    dot = build_agent_graph()
    assert dot.startswith("digraph {")
    assert dot.rstrip().endswith("}")
    for name in NODE_NAMES:
        assert f"{name} [" in dot
    assert "Supervisor -> Profiler" in dot
    assert "RefinementLoop -> Judge" in dot


def test_the_graph_parses_as_dot():
    """A DOT string the browser cannot parse is a blank panel mid-recording."""
    graphviz = pytest.importorskip("graphviz")
    source = graphviz.Source(build_agent_graph("Coder"))
    assert "Coder" in source.source


def test_the_refinement_loop_is_drawn_as_a_loop():
    assert "RefinementLoop [shape=box3d" in build_agent_graph()


def test_with_no_active_agent_the_supervisor_keeps_its_root_colour():
    dot = build_agent_graph(None)
    assert 'Supervisor [shape=box, fillcolor="#2563eb"' in dot
    assert NODE_ACTIVE_COLOR not in dot


@pytest.mark.parametrize("agent", NODE_NAMES)
def test_the_active_agent_is_the_only_amber_node(agent: str):
    dot = build_agent_graph(agent)
    amber = [line for line in dot.splitlines() if NODE_ACTIVE_COLOR in line]
    assert len(amber) == 1
    assert amber[0].strip().startswith(agent)


@pytest.mark.parametrize("agent", NODE_NAMES)
def test_highlight_active_agent_recolours_exactly_that_node(agent: str):
    highlighted = highlight_active_agent(build_agent_graph(None), agent)
    for name in NODE_NAMES:
        line = next(row for row in highlighted.splitlines() if row.strip().startswith(f"{name} ["))
        expected = NODE_ACTIVE_COLOR if name == agent else NODE_IDLE_COLOR
        assert expected in line, f"{name} should be {expected}"


def test_highlight_agrees_with_build_for_the_same_agent():
    """One definition of the tree's shape, two ways of colouring it."""
    assert highlight_active_agent(build_agent_graph(None), "Judge") == build_agent_graph("Judge")


def test_highlighting_an_agent_that_is_not_in_the_tree_leaves_it_unchanged():
    """Turn-2 events name things that are not nodes; that must not blank the diagram."""
    dot = build_agent_graph("Coder")
    assert highlight_active_agent(dot, "SomeToolAgent").count(NODE_ACTIVE_COLOR) == 0


def test_highlighting_none_returns_to_the_idle_tree():
    assert highlight_active_agent(build_agent_graph("Coder"), None) == build_agent_graph(None)


# --------------------------------------------------------------------------- #
# group_turns / turn_headline
# --------------------------------------------------------------------------- #


def test_consecutive_events_from_one_agent_are_one_turn():
    turns = group_turns([event("Coder"), event("Coder"), event("Judge")])
    assert [(t.author, len(t.events)) for t in turns] == [("Coder", 2), ("Judge", 1)]


def test_an_agent_that_speaks_twice_gets_two_turns():
    """The Supervisor before and after the loop are two moments, not one."""
    turns = group_turns([event("Supervisor"), event("Coder"), event("Supervisor")])
    assert [t.author for t in turns] == ["Supervisor", "Coder", "Supervisor"]


def test_grouping_an_empty_stream_yields_no_turns():
    assert group_turns([]) == []


def test_a_turns_duration_is_its_first_to_last_span():
    turn = Turn("Judge", [event(elapsed_s=8.5), event(elapsed_s=12.0)])
    assert turn.duration_s == pytest.approx(3.5)


def test_a_tool_call_headlines_the_turn():
    turn = Turn(
        "Judge",
        [event("Judge", "function_call", function_calls=[{"name": "verify_kernel", "args": {}}])],
    )
    assert turn_headline(turn) == "Calling verify_kernel"


def test_a_transfer_headlines_the_turn():
    turn = Turn("Supervisor", [event("Supervisor", "transfer", transfer_to="Profiler")])
    assert turn_headline(turn) == "Delegating to Profiler"


def test_an_escalation_headlines_the_turn():
    turn = Turn("EscalationChecker", [event("EscalationChecker", "escalate", escalate=True)])
    assert "escalating" in turn_headline(turn)


def test_a_decision_outranks_the_prose_that_precedes_it():
    turn = Turn(
        "Judge",
        [
            event("Judge", "text", content_text="Let me check this."),
            event("Judge", "function_call", function_calls=[{"name": "verify_kernel", "args": {}}]),
        ],
    )
    assert turn_headline(turn) == "Calling verify_kernel"


def test_prose_headlines_a_turn_that_decided_nothing():
    turn = Turn("Coder", [event("Coder", content_text="I'll write a fused kernel.\nDetails…")])
    assert turn_headline(turn) == "I'll write a fused kernel."


def test_a_long_first_line_is_truncated_for_the_label():
    turn = Turn("Coder", [event("Coder", content_text="x" * 200)])
    assert len(turn_headline(turn)) <= 90


# --------------------------------------------------------------------------- #
# split_prose_and_code
# --------------------------------------------------------------------------- #


KERNEL = "import triton\nimport triton.language as tl\n\n\n@triton.jit\ndef f(X):\n    tl.load(X)\n"


def test_a_coders_message_splits_into_its_narration_and_its_kernel():
    """The explanation is the one part a non-technical viewer can read; keep it prose."""
    prose, code = split_prose_and_code("RMSNorm is memory-bound. I'll fuse it:\n\n" + KERNEL)
    assert prose == "RMSNorm is memory-bound. I'll fuse it:"
    assert code.startswith("import triton")


def test_plain_prose_stays_entirely_prose():
    prose, code = split_prose_and_code("I'll profile the model first.")
    assert prose == "I'll profile the model first."
    assert code == ""


def test_a_bare_kernel_with_no_narration_is_all_code():
    prose, code = split_prose_and_code(KERNEL)
    assert prose == ""
    assert code.startswith("import triton")


def test_a_sentence_mentioning_def_is_not_split_into_code():
    """One keyword in a sentence is not the start of a module."""
    text = "I will def-initely fuse it, and def is a keyword, but this is not code."
    prose, code = split_prose_and_code(text)
    assert prose == text
    assert code == ""


def test_nothing_is_lost_in_the_split():
    text = "Narration here.\n\n" + KERNEL
    prose, code = split_prose_and_code(text)
    assert set(text.split()) == set((prose + " " + code).split())


# --------------------------------------------------------------------------- #
# extract_metrics
# --------------------------------------------------------------------------- #


def verdict(**overrides):
    payload = {
        "reward": 3,
        "correctness_pass": True,
        "speedup_vs_eager": 7.04,
        "speedup_vs_compile": 1.39,
        "violations": [],
    }
    payload.update(overrides)
    return event(
        "Judge",
        "function_response",
        function_responses=[{"name": "verify_kernel", "response": payload}],
    )


def test_metrics_are_empty_before_anything_is_verified():
    metrics = extract_metrics([event("Supervisor", content_text="starting")])
    assert metrics["speedup"] is None
    assert metrics["reward"] is None
    assert metrics["iteration"] == 0


def test_a_verdict_fills_the_header_metrics():
    metrics = extract_metrics([verdict()])
    assert metrics["reward"] == 3
    assert metrics["speedup"] == pytest.approx(7.04)
    assert metrics["speedup_vs_compile"] == pytest.approx(1.39)
    assert metrics["correct"] is True


def test_agent_prose_claiming_a_speedup_moves_nothing():
    """Red line #3: only the verifier's numbers reach the screen."""
    boast = event("Judge", content_text="This kernel is 40x faster than eager. Reward +3.")
    assert extract_metrics([boast])["speedup"] is None


def test_the_iteration_count_is_the_number_of_verdicts():
    assert extract_metrics([verdict(reward=-1), verdict(reward=1), verdict()])["iteration"] == 3


def test_the_last_verdict_wins():
    metrics = extract_metrics([verdict(reward=-1, speedup_vs_eager=0.0), verdict()])
    assert metrics["reward"] == 3
    assert metrics["speedup"] == pytest.approx(7.04)


def test_violations_are_surfaced_for_the_anti_hack_banner():
    rejected = verdict(
        reward=-1,
        correctness_pass=False,
        violations=[{"rule_id": 1, "line": 12, "description": "torch.nn fallback"}],
    )
    assert extract_metrics([rejected])["violations"][0]["rule_id"] == 1


def test_the_honest_plus_one_is_detected():
    """Correct but not faster — the compute-bound case the demo is proud of reporting."""
    metrics = extract_metrics([verdict(reward=1, speedup_vs_eager=1.01)])
    assert metrics["honest_plus_one"] is True


def test_a_reward_of_three_is_not_the_honest_plus_one():
    assert extract_metrics([verdict()])["honest_plus_one"] is False


def test_a_successful_hotswap_supplies_the_tokens_per_second():
    swap = event(
        "Supervisor",
        "function_response",
        function_responses=[
            {
                "name": "hotswap_kernel",
                "response": {
                    "success": True,
                    "modules_patched": 57,
                    "stats": {"tokens_per_s": 312.0},
                },
            }
        ],
    )
    assert extract_metrics([swap])["tokens_per_s"] == pytest.approx(312.0)


def test_a_refused_hotswap_supplies_nothing():
    swap = event(
        "Supervisor",
        "function_response",
        function_responses=[
            {
                "name": "hotswap_kernel",
                "response": {"success": False, "error": "no module matched"},
            }
        ],
    )
    assert extract_metrics([swap])["tokens_per_s"] is None


def test_a_functiontool_result_wrapper_is_unwrapped():
    """FunctionTool wraps a non-dict return as {"result": ...}; the verdict is inside."""
    wrapped = event(
        "Judge",
        "function_response",
        function_responses=[
            {
                "name": "verify_kernel",
                "response": {
                    "result": {
                        "reward": 3,
                        "correctness_pass": True,
                        "speedup_vs_eager": 7.04,
                        "speedup_vs_compile": 1.39,
                    }
                },
            }
        ],
    )
    assert extract_metrics([wrapped])["reward"] == 3


def test_the_sample_trace_produces_the_metrics_the_demo_claims():
    from kernelsmith.ui.event_replay import load_events

    metrics = extract_metrics(load_events(SAMPLE))
    assert metrics["reward"] == 3
    assert metrics["speedup"] == pytest.approx(7.04)
    assert metrics["tokens_per_s"] == pytest.approx(312.0)
    assert metrics["violations"] is None


# --------------------------------------------------------------------------- #
# The page renders
# --------------------------------------------------------------------------- #


@pytest.fixture
def app():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    return at


def exceptions(at) -> list[str]:
    return [str(element.value) for element in at.exception]


def test_the_demo_dashboard_renders_without_a_gpu_or_credentials(app):
    assert exceptions(app) == []


def test_the_sidebar_offers_live_and_replay(app):
    from kernelsmith.ui.demo_dashboard import MODES

    assert app.sidebar.radio[0].options == list(MODES)


def test_the_header_shows_all_four_metrics(app):
    labels = [metric.label for metric in app.metric]
    assert {"Speedup", "Reward", "Iteration", "Tokens/s"} <= set(labels)


def test_switching_to_replay_lists_the_committed_sample_trace(app):
    app.sidebar.radio[0].set_value("📼 Replay").run()
    assert exceptions(app) == []
    assert "sample_run.jsonl" in app.sidebar.selectbox[0].options


def test_playing_the_sample_trace_builds_the_timeline_and_fills_the_header(app):
    """The fallback demo, end to end, through the real app — at instant speed.

    This is the one test that exercises the replay renderer rather than the functions
    under it: `render_event`, the `st.status` per turn, and the header updating from
    the verdict. At 1x it would take 13.5 seconds; at "instant" it takes none, and the
    rendering path is identical either way.
    """
    app.sidebar.radio[0].set_value("📼 Replay").run()
    app.sidebar.select_slider[0].set_value("instant").run()
    app.sidebar.button[0].click().run()

    assert exceptions(app) == []

    labels = [status.label for status in app.status]
    assert any("Profiler" in label for label in labels)
    assert any("Judge" in label for label in labels)
    assert any("escalating" in label for label in labels)

    values = {metric.label: metric.value for metric in app.metric}
    assert values["Speedup"] == "7.04×"
    assert values["Reward"] == "+3"
    assert values["Tokens/s"] == "312"


# --------------------------------------------------------------------------- #
# The trap streamlit_app.py documents, repeated here
# --------------------------------------------------------------------------- #


def test_the_live_view_drives_turn_two_before_it_can_return():
    """`drive_run` must be reached on every live rerun.

    An early return placed above it — a tempting "nothing new, skip the work" guard —
    silently skips upsert and hot-swap while every panel keeps looking healthy. That is
    unreproducible in a test without Vertex, a GPU and a live server, so this asserts
    the ordering in the source instead, exactly as `test_dashboard_tabs.py` does.
    """
    import inspect

    from kernelsmith.ui.demo_dashboard import render_live

    body = inspect.getsource(render_live)
    drive = body.index("drive_run(consumer)")
    first_return = body.find("\n        return")
    assert first_return == -1 or drive < first_return, (
        "drive_run must run before any early return in render_live"
    )
