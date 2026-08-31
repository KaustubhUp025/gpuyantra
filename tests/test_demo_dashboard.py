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

import re
from pathlib import Path

import pytest

from kernelsmith.ui.demo_dashboard import (
    GRAPH_NODES,
    NARRATIVE,
    NODE_ACTIVE_COLOR,
    NODE_IDLE_COLOR,
    Turn,
    build_agent_graph,
    classify_event,
    extract_metrics,
    format_event_for_display,
    group_turns,
    highlight_active_agent,
    is_noise_event,
    iteration_label,
    merge_live_tokens,
    narrative_after,
    ordered_traces,
    split_prose_and_code,
    tokens_metric,
    trace_summary,
    turn_headline,
    turn_label,
)

APP = Path(__file__).resolve().parent.parent / "kernelsmith" / "ui" / "demo_dashboard.py"
SAMPLE = Path(__file__).resolve().parent.parent / "data" / "traces" / "sample_run.jsonl"
TIMEOUT_S = 120

NODE_NAMES = [name for name, _, _ in GRAPH_NODES]


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
    """The measured 2026-08-30 L4 numbers, end to end through the extractor."""
    from kernelsmith.ui.event_replay import load_events

    metrics = extract_metrics(load_events(SAMPLE))
    assert metrics["reward"] == 3
    assert metrics["speedup"] == pytest.approx(7.24)
    assert metrics["speedup_vs_compile"] == pytest.approx(1.39)
    assert metrics["tokens_per_s"] == pytest.approx(22.9)
    assert metrics["violations"] is None
    assert metrics["escalated"] is True
    assert metrics["hotswap_ok"] is True
    assert metrics["modules_patched"] == 57
    assert metrics["skill_id"] == "rmsnorm_l4_single_pass_register_fused"


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
    # Explicitly, because the picker defaults to the newest REAL captured trace and
    # this test is about the committed fixture's numbers.
    app.sidebar.selectbox[0].set_value("sample_run.jsonl").run()
    app.sidebar.select_slider[0].set_value("instant").run()
    app.sidebar.button[0].click().run()

    assert exceptions(app) == []

    labels = [status.label for status in app.status]
    assert any("Profiler" in label for label in labels)
    assert any("Judge" in label for label in labels)
    assert any("stopping the loop" in label for label in labels)
    # Problem 5.1/5.3: the label carries what happened and how long it took, not
    # "Supervisor — Calling transfer_to_agent".
    assert any("7.24× faster than PyTorch" in label and label.endswith("s)") for label in labels)

    values = {metric.label: metric.value for metric in app.metric}
    assert values["Speedup"] == "7.24×"
    assert values["Reward"] == "+3"
    assert values["Tokens/s"] == "22.9"
    # Problem 3: the loop escalated, so the counter says so rather than sitting at 1/6.
    assert values["Iteration"] == "✅ Done"


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


# --------------------------------------------------------------------------- #
# Task 12b — plain English (problem 1)
# --------------------------------------------------------------------------- #


def response(name, payload, author="Supervisor"):
    return event(
        author, "function_response", function_responses=[{"name": name, "response": payload}]
    )


def call(name, args, author="Supervisor"):
    return event(author, "function_call", function_calls=[{"name": name, "args": args}])


PROFILE_PAYLOAD = {
    "op_family": "norm",
    "hardware": "L4",
    "memory_throughput_gbps": 40.66,
    "arithmetic_intensity": 1.25,
    "is_memory_bound": True,
    "is_compute_bound": False,
    "tile_size_hint": 1024,
    "ridge_point_flops_per_byte": 100.96634455181606,
}


def test_a_transfer_is_described_by_what_the_receiving_agent_is_asked_for():
    display = format_event_for_display(call("transfer_to_agent", {"agent_name": "Profiler"}))
    assert display["headline"] == "🎯 Delegating to Profiler"
    assert "find which part of the model is slow" in display["detail"]


def test_a_profile_call_names_the_op_and_the_shape_it_probes():
    display = format_event_for_display(
        call("profile_op_by_name", {"op_name": "rmsnorm", "batch": 8, "seq_len": 512}, "Profiler")
    )
    assert display["headline"] == "🔍 Measuring rmsnorm"
    assert "8×512" in display["detail"]
    # And it says what rmsnorm actually does, for someone who has never heard of it.
    assert "rescales the numbers" in display["detail"]


def test_a_profile_result_states_the_regime_and_the_distance_to_the_ridge():
    """Same facts as before, in words a viewer can follow without a roofline lecture."""
    display = format_event_for_display(response("profile_op_by_name", PROFILE_PAYLOAD, "Profiler"))
    assert (
        display["headline"]
        == "📊 Normalization is waiting on memory — 1.25 calculations per byte moved"
    )
    assert "1.25 calculations" in display["detail"]
    assert "about 101" in display["detail"]
    assert "81× longer" in display["detail"]  # 100.97 / 1.25
    assert "40.7 GB" in display["detail"]


def test_a_profile_result_invents_no_number_the_payload_did_not_carry():
    """Red line #3 reaches the prose, not only the metrics.

    The task brief's template for this row reads "57 instances, only using 23% of the
    L4's 300 GB/s". Neither figure is in a `profile_op` response — they come from the
    audit table — so the sentence is written without them rather than filled in from
    another measurement of another thing.
    """
    detail = format_event_for_display(response("profile_op_by_name", PROFILE_PAYLOAD, "Profiler"))[
        "detail"
    ]
    assert "57" not in detail
    assert "%" not in detail


def test_a_retrieval_result_names_the_skills_and_the_arm_the_bandit_pulled():
    payload = {
        "count": 2,
        "selected_skill_id": "rmsnorm_l4_single_pass_fused",
        "skills": [
            {"skill_id": "rmsnorm_l4_single_pass_fused"},
            {"skill_id": "rmsnorm_fp16_l4_v1"},
        ],
    }
    display = format_event_for_display(response("retrieve_skills_for_agent", payload))
    assert display["headline"] == "📚 Found 2 solutions to similar problems"
    assert "rmsnorm_fp16_l4_v1" in display["detail"]
    # The bandit, explained by analogy rather than named: "UCB1" means nothing on camera.
    assert "slot machines" in display["detail"]


def test_an_empty_retrieval_is_a_cold_start_not_a_found_zero():
    display = format_event_for_display(
        response("retrieve_skills_for_agent", {"count": 0, "skills": []})
    )
    assert "Nothing like this has been solved before" in display["headline"]
    assert "cold start" in display["detail"]


def test_the_coders_structured_draft_is_shown_as_a_kernel_not_as_json():
    """The Coder speaks JSON. Printed raw it is the least readable thing on the page."""
    import json

    draft = {
        "code": KERNEL,
        "entrypoint": "f",
        "adapter_mapping": [{"kernel_param": "eps", "module_attr": "variance_epsilon"}],
    }
    display = format_event_for_display(event("Coder", "text", content_text=json.dumps(draft)))
    assert display["headline"] == "💻 Writing the new GPU code"
    assert display["code"].startswith("import triton")
    assert "`eps`→`variance_epsilon`" in display["detail"]
    assert "a human writes that by hand" in display["detail"]


def test_a_verdict_response_carries_the_numbers_for_the_metric_cards():
    display = format_event_for_display(
        response(
            "verify_kernel",
            {
                "reward": 3,
                "correctness_pass": True,
                "passed_checks": 15,
                "total_checks": 15,
                "speedup_vs_eager": 7.24,
                "speedup_vs_compile": 1.39,
                "violations": [],
            },
            "Judge",
        )
    )
    assert display["headline"] == "✅ Score +3: 7.24× faster than PyTorch"
    assert "all 15 of them" in display["detail"]
    assert "1.39× faster than PyTorch's own compiler" in display["detail"]
    assert display["verdict"]["reward"] == 3


def test_a_rejected_kernel_is_not_dressed_up():
    display = format_event_for_display(
        response(
            "verify_kernel",
            {"reward": -1, "correctness_pass": False, "violations": [{"rule_id": 1}]},
            "Judge",
        )
    )
    assert display["headline"].startswith("❌")
    assert "REJECTED" in display["detail"]
    assert "found a shortcut" in display["detail"]


def test_a_successful_hotswap_is_the_money_shot():
    display = format_event_for_display(
        response(
            "hotswap_kernel",
            {
                "success": True,
                "modules_patched": 57,
                "parity": {"parity_pass": True, "seeds": 5, "atol": 0.01},
                "stats": {"tokens_per_s": 22.9},
            },
        )
    )
    assert display["headline"] == "🚀 The new code is running live"
    assert "57 layers" in display["detail"]
    assert "22.9 words-worth of text per second" in display["detail"]


def test_a_refused_hotswap_says_the_kernel_is_not_live():
    """The failure mode the committed L4 trace actually contains."""
    display = format_event_for_display(
        response(
            "hotswap_kernel",
            {"success": False, "error": "connection refused", "rolled_back": False},
        )
    )
    assert "NOT running live" in display["headline"]
    assert "connection refused" in display["detail"]


def test_the_string_returning_tools_are_described_not_dumped():
    """`upsert_skill` and `explain_kernel` return a bare string inside `{"result": ...}`."""
    upsert = format_event_for_display(response("upsert_skill", {"result": "upserted"}))
    assert upsert["headline"] == "💾 Saved to the skill library"

    explain = format_event_for_display(
        response("explain_kernel", {"result": "one two three four five"})
    )
    assert explain["headline"] == "📝 Explanation ready"
    assert "5 words" in explain["detail"]
    assert explain["prose"] == "one two three four five"


def test_an_escalation_explains_why_the_loop_stopped():
    display = format_event_for_display(event("EscalationChecker", "escalate", escalate=True))
    assert display["headline"] == "🛑 Good enough — stopping the loop"
    assert "more attempts" in display["detail"]
    assert "not by a language model" in display["detail"]


def test_the_supervisors_closing_markdown_is_routed_to_the_summary_card():
    text = "### Optimization Summary\n\n- **Speedup:** 7.24x vs eager\n"
    display = format_event_for_display(event("Supervisor", "text", content_text=text))
    assert display["headline"] == "🏁 Optimization summary"
    assert display["summary"] == text.strip()


def test_an_unknown_tool_still_gets_a_headline_and_its_raw_payload():
    display = format_event_for_display(call("some_new_tool", {"x": 1}))
    assert display["headline"] == "🔧 Calling some_new_tool"
    assert display["raw"] == {"x": 1}


def test_every_display_has_the_three_documented_keys():
    samples = [
        event("Coder", "text", content_text="prose"),
        event("Supervisor", "transfer", transfer_to="Profiler"),
        event("EscalationChecker", "escalate", escalate=True),
        call("verify_kernel", {"kernel_code": KERNEL}, "Judge"),
        response("verify_kernel", {"reward": 1}, "Judge"),
        response("nothing_known", None),
    ]
    for sample in samples:
        display = format_event_for_display(sample)
        assert {"headline", "detail", "raw"} <= set(display)
        assert isinstance(display["headline"], str) and display["headline"]


def test_every_event_of_every_committed_trace_formats_without_raising():
    """The guard against schema drift in a file that is only read on demo day."""
    from kernelsmith.ui.event_replay import list_traces, load_events

    traces = list_traces(SAMPLE.parent)
    assert traces, "data/traces/ must hold at least the committed fixture"
    for trace in traces:
        for record in load_events(trace):
            display = format_event_for_display(record)
            assert isinstance(display["headline"], str)


# --------------------------------------------------------------------------- #
# Task 12b — noise, labels and narration (problems 4 and 5)
# --------------------------------------------------------------------------- #


def test_the_transfer_echo_is_dropped_but_the_transfer_itself_is_not():
    """ADK emits a `transfer_to_agent` response whose whole payload is `{"result": null}`."""
    echo = response("transfer_to_agent", {"result": None})
    echo["transfer_to"] = "Profiler"
    assert is_noise_event(echo) is True
    assert is_noise_event(call("transfer_to_agent", {"agent_name": "Profiler"})) is False


@pytest.mark.parametrize(
    "sample",
    [
        event("Coder", "text", content_text=""),
        event("Coder", "text", content_text=None),
        event("Coder", "text", content_text="half a sen", partial=True),
    ],
)
def test_empty_and_partial_text_events_are_noise(sample):
    assert is_noise_event(sample) is True


def test_a_status_label_names_the_agent_the_outcome_and_the_duration():
    turn = Turn(
        "Judge",
        [
            call("verify_kernel", {}, "Judge") | {"elapsed_s": 10.0},
            response("verify_kernel", {"reward": 3, "speedup_vs_eager": 7.24}, "Judge")
            | {"elapsed_s": 33.0},
        ],
    )
    assert turn_label(turn, running=False) == (
        "⚖️ Judge — Score +3: 7.24× faster than PyTorch (23s)"
    )


def test_a_running_turn_has_no_duration_in_its_label():
    turn = Turn("Judge", [call("verify_kernel", {}, "Judge")])
    assert turn_label(turn, running=True) == "⚖️ Judge — Testing the new code"


def test_the_label_carries_one_emoji_not_two():
    """The headline's own emoji is stripped: the agent's is already there."""
    turn = Turn("Profiler", [response("profile_op_by_name", PROFILE_PAYLOAD, "Profiler")])
    label = turn_label(turn, running=True)
    assert label.startswith("🔍 Profiler — Normalization is waiting on memory")
    assert "📊" not in label


def test_the_verdict_outranks_the_call_that_produced_it_in_the_label():
    turn = Turn(
        "Judge",
        [
            call("verify_kernel", {}, "Judge"),
            response("verify_kernel", {"reward": 3, "speedup_vs_eager": 7.24}, "Judge"),
        ],
    )
    assert "Score +3" in turn_label(turn, running=False)


@pytest.mark.parametrize(
    ("turn", "key"),
    [
        (
            Turn("Profiler", [response("profile_op_by_name", PROFILE_PAYLOAD, "Profiler")]),
            "profiled",
        ),
        (
            Turn("Supervisor", [response("retrieve_skills_for_agent", {"count": 1, "skills": []})]),
            "retrieved",
        ),
        (Turn("Judge", [response("verify_kernel", {"reward": 3}, "Judge")]), "verified"),
        (
            Turn("EscalationChecker", [event("EscalationChecker", "escalate", escalate=True)]),
            "escalated",
        ),
        (
            Turn(
                "Supervisor", [response("hotswap_kernel", {"success": True, "modules_patched": 57})]
            ),
            "swapped",
        ),
    ],
)
def test_a_finished_turn_is_followed_by_the_narration_for_what_it_did(turn, key):
    assert narrative_after(turn) == NARRATIVE[key]


def test_a_refused_hotswap_is_not_narrated_as_a_deployment():
    turn = Turn("Supervisor", [response("hotswap_kernel", {"success": False, "error": "no"})])
    assert narrative_after(turn) != NARRATIVE["swapped"]


def test_a_turn_that_only_delegated_needs_no_narration():
    turn = Turn("Supervisor", [event("Supervisor", "transfer", transfer_to="Profiler")])
    assert narrative_after(turn) is None


def test_the_narration_is_static_text_with_no_numbers_in_it():
    """It explains why a step matters; a measured value in here could go stale silently."""
    import re

    for text in NARRATIVE.values():
        assert not re.search(r"\d", text), text


# --------------------------------------------------------------------------- #
# Task 12b — the header metrics (problems 2 and 3)
# --------------------------------------------------------------------------- #


def test_the_iteration_counter_counts_verdicts():
    assert iteration_label(extract_metrics([]))[0] == "0/6"
    assert iteration_label(extract_metrics([verdict(reward=-1)]))[0] == "1/6"


def test_an_escalated_loop_reads_as_done():
    events = [verdict(), event("EscalationChecker", "escalate", escalate=True)]
    value, delta = iteration_label(extract_metrics(events))
    assert value == "✅ Done"
    assert delta == "converged in 1"


def test_a_loop_that_spent_its_budget_without_escalating_is_flagged():
    value, delta = iteration_label(extract_metrics([verdict(reward=-1)] * 6))
    assert value == "6/6 ⚠️"
    assert delta == "budget spent"


def test_tokens_are_a_dash_when_nothing_has_measured_them():
    value, delta, _ = tokens_metric(extract_metrics([]))
    assert (value, delta) == ("—", None)


def test_a_reachable_server_reporting_zero_shows_zero_not_a_dash():
    """The bug this replaces: `0.0 or None` blanked a metric that was working."""
    state = {"stats": {"tokens_per_s": 0.0}, "stats_ok": True}
    merged = merge_live_tokens(extract_metrics([]), state)
    value, _, help_text = tokens_metric(merged)
    assert value == "0.0"
    assert "read live from the running server" in help_text


def test_a_replayed_swap_supplies_the_tokens_and_says_where_they_came_from():
    swap = response("hotswap_kernel", {"success": True, "stats": {"tokens_per_s": 22.9}})
    value, _, help_text = tokens_metric(extract_metrics([swap]))
    assert value == "22.9"
    assert "when the new code was loaded" in help_text


def test_the_pre_swap_rate_is_latched_and_becomes_the_delta():
    """The only moment the baseline can be observed is before the swap lands."""
    state = {"stats": {"tokens_per_s": 5.7}, "stats_ok": True, "tokens_before_swap": None}
    merge_live_tokens(extract_metrics([]), state)
    assert state["tokens_before_swap"] == pytest.approx(5.7)

    state["stats"] = {"tokens_per_s": 22.9}
    swap = response("hotswap_kernel", {"success": True, "stats": {"tokens_per_s": 22.9}})
    merged = merge_live_tokens(extract_metrics([swap]), state)
    # Still 5.7: a swap has happened, so the baseline is no longer being updated.
    assert state["tokens_before_swap"] == pytest.approx(5.7)
    assert tokens_metric(merged)[1] == "+17.2 vs before the swap (5.7)"


def test_the_live_server_outranks_the_swap_snapshot():
    """`TokenMeter` clears its window at the swap, so the snapshot is the stalest reading."""
    swap = response("hotswap_kernel", {"success": True, "stats": {"tokens_per_s": 0.0}})
    state = {"stats": {"tokens_per_s": 31.4}, "stats_ok": True}
    merged = merge_live_tokens(extract_metrics([swap]), state)
    assert merged["tokens_per_s"] == pytest.approx(31.4)
    assert merged["tokens_source"] == "server"


def test_blank_metrics_stay_in_step_with_the_extractor():
    from kernelsmith.ui.demo_dashboard import _blank_metrics

    assert set(_blank_metrics()) == set(extract_metrics([]))


# --------------------------------------------------------------------------- #
# Task 12b — the hosted replay (problem 6)
# --------------------------------------------------------------------------- #


def test_real_traces_sort_ahead_of_the_hand_written_fixture(tmp_path: Path):
    """On a fresh clone every mtime is checkout time, so mtime alone is not an order."""
    (tmp_path / "sample_run.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "demo-20260830-101414-58bee2.jsonl").write_text("", encoding="utf-8")
    assert [path.name for path in ordered_traces(tmp_path)][-1] == "sample_run.jsonl"


def test_a_traces_caption_says_whether_its_hot_swap_went_live():
    """Before pressing Play, not as a surprise at the end of the playback."""
    assert "went live on 57 layers" in trace_summary(SAMPLE)
    assert "7.24× faster than PyTorch" in trace_summary(SAMPLE)
    assert "hand-written fixture" in trace_summary(SAMPLE)


def test_an_unreadable_trace_captions_rather_than_raises(tmp_path: Path):
    missing = tmp_path / "gone.jsonl"
    assert "could not be read" in trace_summary(missing)


def test_the_page_defaults_to_replay_when_no_inference_server_answers(monkeypatch):
    """The Cloud Run case: no GPU, no server, a trace baked into the container."""
    import kernelsmith.ui.demo_dashboard as dashboard

    monkeypatch.setattr(dashboard, "script_flags", lambda: set())
    monkeypatch.setattr(dashboard, "inference_server_is_up", lambda: False)
    assert dashboard.default_mode() == dashboard.MODE_REPLAY


def test_a_reachable_server_defaults_to_live(monkeypatch):
    import kernelsmith.ui.demo_dashboard as dashboard

    monkeypatch.setattr(dashboard, "script_flags", lambda: set())
    monkeypatch.setattr(dashboard, "inference_server_is_up", lambda: True)
    assert dashboard.default_mode() == dashboard.MODE_LIVE


def test_the_live_flag_wins_over_the_probe(monkeypatch):
    """`--live` is for the operator who is about to start the server."""
    import kernelsmith.ui.demo_dashboard as dashboard

    monkeypatch.setattr(dashboard, "script_flags", lambda: {"--live"})
    monkeypatch.setattr(dashboard, "inference_server_is_up", lambda: False)
    assert dashboard.default_mode() == dashboard.MODE_LIVE


def test_with_no_traces_at_all_the_page_opens_in_live(monkeypatch):
    """Replay with nothing to replay is a dead end; Live at least has a button."""
    import kernelsmith.ui.demo_dashboard as dashboard

    monkeypatch.setattr(dashboard, "script_flags", lambda: set())
    monkeypatch.setattr(dashboard, "ordered_traces", lambda *args: [])
    monkeypatch.setattr(dashboard, "inference_server_is_up", lambda: False)
    assert dashboard.default_mode() == dashboard.MODE_LIVE


def test_every_committed_trace_plays_through_the_real_app(app):
    """The hosted demo plays whichever trace the picker defaults to — check them all.

    The renderer, not just the formatter: `st.status` per turn, the metric cards, the
    narration, the summary card. At "instant" speed this costs no wall-clock, and it is
    the only automated check that a trace committed for demo day can actually be shown.
    """
    app.sidebar.radio[0].set_value("📼 Replay").run()
    for name in app.sidebar.selectbox[0].options:
        app.sidebar.selectbox[0].set_value(name).run()
        app.sidebar.select_slider[0].set_value("instant").run()
        app.sidebar.button[0].click().run()
        assert exceptions(app) == [], f"{name} raised"
        assert app.status, f"{name} rendered no timeline"


# --------------------------------------------------------------------------- #
# The default view is written for a viewer, not for us
# --------------------------------------------------------------------------- #

#: Terms that carry no meaning for someone watching a four-minute video and cannot be
#: made to carry it in half a sentence. If one of these is the only way to say something,
#: it belongs in the raw fold or in GLOSSARY — not in a headline or a detail line.
JARGON = (
    "atol",
    "rtol",
    "do_bench",
    "UCB1",
    "arithmetic intensity",
    "FLOP/byte",
    "roofline",
    "ridge point",
    "eager",
    "upsert",
    "parity",
    "bandit",
    "escalat",
    "monkey-patch",
    "nn.Module",
    "AST",
    "torch.compile",
)


def test_no_unexplained_jargon_reaches_the_default_view():
    """Every committed trace, every event, headline and detail.

    The raw fold and GLOSSARY are where the vocabulary lives. This is the check that the
    sentence a viewer actually reads does not assume they already know the field — it
    caught "AST scan", "atol=1e-2", "UCB1 bandit" and "vs eager" on the first run.
    """
    from kernelsmith.ui.event_replay import list_traces, load_events

    offences = []
    for trace in list_traces(SAMPLE.parent):
        for record in load_events(trace):
            display = format_event_for_display(record)
            text = f"{display['headline']} {display['detail']}"
            for term in JARGON:
                # Word-boundary, so "faster" is not an "AST" and "compared" is not "par".
                if re.search(rf"(?<![\w.]){re.escape(term)}", text, re.IGNORECASE):
                    offences.append((trace.name, term, display["headline"]))
    assert not offences, offences


def test_the_narration_avoids_the_same_jargon():
    for key, text in NARRATIVE.items():
        for term in JARGON:
            assert not re.search(rf"(?<![\w.]){re.escape(term)}", text, re.IGNORECASE), (
                f"{key}: {term}"
            )


def test_the_glossary_explains_the_terms_the_page_cannot_avoid():
    """Triton, "memory-bound" and "hot-swap" DO appear on screen — so they are defined."""
    from kernelsmith.ui.demo_dashboard import GLOSSARY

    terms = {term.lower() for term, _ in GLOSSARY}
    assert {"triton", "memory-bound", "hot-swap"} <= terms
    for term, meaning in GLOSSARY:
        assert term and len(meaning.split()) >= 10, term


def test_the_agent_roles_are_job_descriptions_not_titles():
    """The panel next to the diagram says what each agent does, in words, always."""
    from kernelsmith.ui.demo_dashboard import AGENT_STYLES

    for name, style in AGENT_STYLES.items():
        # A phrase describing an action, never a one-word title like "orchestrator".
        assert len(style["role"].split()) >= 3, name


# --------------------------------------------------------------------------- #
# Task 13 — one metrics row, a chart, and an honest empty Tokens/s
# --------------------------------------------------------------------------- #


def test_a_run_that_never_reached_a_server_says_so_on_the_tokens_card():
    """Problem 4: "—" alone cannot be told apart from "not measured yet"."""
    swap = response("hotswap_kernel", {"success": False, "error": "connection refused"})
    value, delta, help_text = tokens_metric(extract_metrics([swap]))
    assert value == "—"
    assert delta == "no server"
    assert "never reached a live server" in help_text
    assert "estimated" in help_text  # ... and nothing is estimated in its place


def test_a_before_reading_in_the_trace_becomes_the_delta():
    """Supported if a trace carries one; never invented when it does not."""
    swap = response(
        "hotswap_kernel",
        {
            "success": True,
            "stats": {"tokens_per_s": 22.9, "tokens_per_s_before": 5.7},
        },
    )
    metrics = extract_metrics([swap])
    assert metrics["tokens_before_swap"] == pytest.approx(5.7)
    value, delta, _ = tokens_metric(metrics)
    assert value == "22.9"
    assert delta == "+17.2 vs before the swap (5.7)"


def test_no_before_reading_means_no_delta_rather_than_a_guessed_one():
    swap = response("hotswap_kernel", {"success": True, "stats": {"tokens_per_s": 22.9}})
    assert tokens_metric(extract_metrics([swap]))[1] is None


def test_the_speedup_chart_has_three_bars_none_of_them_hardcoded():
    """Problem 6. PyTorch is 1.0 by definition; the other two come out of the verdict."""
    from kernelsmith.ui.demo_dashboard import speedup_chart_rows

    metrics = extract_metrics(
        [
            verdict(
                speedup_vs_eager=7.24,
                speedup_vs_compile=1.39,
                baseline_ms={"eager_ms": 9.021, "compile_ms": 1.727},
            )
        ]
    )
    rows = speedup_chart_rows(metrics)
    assert [label for label, _ in rows] == ["PyTorch", "PyTorch compiler", "KernelSmith"]
    values = dict(rows)
    assert values["PyTorch"] == 1.0
    assert values["KernelSmith"] == pytest.approx(7.24)
    # 9.021 / 1.727 — the compiler's own speedup, from the verdict's baseline timings.
    assert values["PyTorch compiler"] == pytest.approx(5.223, abs=0.01)


def test_the_compiler_bar_falls_back_to_the_two_ratios():
    from kernelsmith.ui.demo_dashboard import speedup_chart_rows

    metrics = extract_metrics([verdict(speedup_vs_eager=7.24, speedup_vs_compile=1.39)])
    assert dict(speedup_chart_rows(metrics))["PyTorch compiler"] == pytest.approx(
        7.24 / 1.39, abs=0.01
    )


def test_a_verdict_with_no_compiler_number_gets_two_bars_not_a_fabricated_third():
    from kernelsmith.ui.demo_dashboard import speedup_chart_rows

    metrics = extract_metrics([verdict(speedup_vs_eager=7.24, speedup_vs_compile=0.0)])
    assert [label for label, _ in speedup_chart_rows(metrics)] == ["PyTorch", "KernelSmith"]


def test_there_is_no_chart_before_there_is_a_verdict():
    from kernelsmith.ui.demo_dashboard import speedup_chart_rows

    assert speedup_chart_rows(extract_metrics([])) is None


def test_the_graph_labels_are_readable_and_carry_no_bracket():
    """Problem 5: friendly two-line labels, and nothing that breaks the recolour regex."""
    dot = build_agent_graph()
    assert 'label="Quality checker\\ndecides when to stop"' in dot
    assert 'label="Write → test loop\\nup to 6 attempts"' in dot
    assert "delegates to" in dot
    # No "]" inside a label: `highlight_active_agent` recolours a node by rewriting
    # inside its "[...]" attribute list, and a bracket in a label ends that match early.
    from kernelsmith.ui.demo_dashboard import GRAPH_NODES as nodes

    for _name, _shape, label in nodes:
        assert "]" not in label and "[" not in label, label


def test_the_active_node_is_outlined_as_well_as_filled():
    """A fill change alone is easy to miss at video bitrates on a dark theme."""
    dot = build_agent_graph("Coder")
    line = next(row for row in dot.splitlines() if row.strip().startswith("Coder ["))
    assert NODE_ACTIVE_COLOR in line
    assert "penwidth=2" in line


def test_the_dashboard_uses_no_deprecated_width_argument():
    """Problem 3: `use_container_width` printed four lines of warning per interaction."""
    source = APP.read_text(encoding="utf-8")
    assert "use_container_width" not in source


def test_the_page_skeleton_is_the_same_element_sequence_in_both_modes():
    """Problem 2's actual cause: conditional top-level elements shifted every position.

    Asserted in the source rather than on screen, because reproducing it needs two
    overlapping script runs — which is exactly what a long playback plus one click is.
    """
    import inspect

    from kernelsmith.ui.demo_dashboard import render_live, render_replay

    for func in (render_live, render_replay):
        body = inspect.getsource(func)
        assert "build_page_skeleton()" in body, func.__name__
        # No mode may create its own top-level layout any more.
        assert "st.columns(" not in body, func.__name__
        assert "st.divider()" not in body, func.__name__


def test_only_one_metrics_row_is_on_screen_after_a_replay(app):
    """Problem 2, end to end: exactly one Speedup card, not one per script run."""
    app.sidebar.radio[0].set_value("📼 Replay").run()
    app.sidebar.selectbox[0].set_value("sample_run.jsonl").run()
    app.sidebar.select_slider[0].set_value("instant").run()
    app.sidebar.button[0].click().run()

    assert exceptions(app) == []
    header_labels = [m.label for m in app.metric if m.label in {"Speedup", "Iteration", "Tokens/s"}]
    assert header_labels.count("Speedup") == 1
    assert header_labels.count("Iteration") == 1
    assert header_labels.count("Tokens/s") == 1
    # And it is the finished run's row, not a blank one left behind.
    values = {m.label: m.value for m in app.metric}
    assert values["Speedup"] == "7.24×"
    assert values["Iteration"] == "✅ Done"


def test_the_speedup_chart_reaches_the_page_after_a_replay(app):
    """Not the `st.bar_chart` fallback: altair ships with Streamlit, so it should render."""
    app.sidebar.radio[0].set_value("📼 Replay").run()
    app.sidebar.selectbox[0].set_value("sample_run.jsonl").run()
    app.sidebar.select_slider[0].set_value("instant").run()
    app.sidebar.button[0].click().run()

    assert exceptions(app) == []
    assert len(app.get("vega_lite_chart")) == 1
    assert len(app.get("graphviz_chart")) == 1


def test_there_is_no_chart_on_the_page_before_anything_is_measured(app):
    assert app.get("vega_lite_chart") == []
