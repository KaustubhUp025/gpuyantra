"""Agent-tree wiring and the two pieces of real logic in it (spec 4.2 / 13.1).

The LLM turns are not testable offline, but three things are, and all three are places
where a mistake is silent rather than loud:

- the tree's shape and its red-line constraints (max_iterations, Coder tools, Judge
  output_schema),
- the EscalationChecker's stop decision, which is the only thing standing between a
  finished run and six paid iterations,
- the Judge's verdict reconstruction, which must never let a model's claimed reward
  override what the verifier measured.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
from google.adk.agents import LlmAgent, LoopAgent
from google.genai import types

from kernelsmith.agents.coder_agent import build_instruction as coder_instruction
from kernelsmith.agents.escalation import EscalationChecker, build_escalation_checker
from kernelsmith.agents.judge_agent import (
    build_instruction as judge_instruction,
)
from kernelsmith.agents.judge_agent import (
    extract_json_object,
    find_verifier_response,
    reconcile,
    record_verdict,
)
from kernelsmith.agents.profiler_agent import build_instruction as profiler_instruction
from kernelsmith.agents.supervisor import build_instruction as supervisor_instruction
from kernelsmith.agents.supervisor import build_supervisor
from kernelsmith.config import MAX_LOOP_ITERATIONS, PRIMARY_MODEL
from kernelsmith.memory.schemas import Verdict
from kernelsmith.tools.verifier_tool import adapter_mapping_from_draft, verifier_tool

#: State keys the prompt providers interpolate. If one of these survives rendering,
#: the provider fell back to a literal template and the agent is reading a placeholder.
_PLACEHOLDERS = {
    "task_spec",
    "fingerprint",
    "skills",
    "skills_status",
    "feedback",
    "kernel_draft",
    "verdict",
    "best_reward",
    "iteration",
    "sram_kb",
}


# --------------------------------------------------------------------------- fakes


class FakeState(dict):
    """`CallbackContext.state` behaves like a dict for get/set; that is all we use."""


def fake_ctx(state: dict, events: list | None = None, invocation_id: str = "inv-1"):
    """Stand-in for ReadonlyContext / CallbackContext / InvocationContext.

    All three expose `state`; the callback variants also expose `session` and
    `invocation_id`. Building a real InvocationContext needs a Runner and a live
    session service, which would turn a logic test into an integration test.
    """
    session = SimpleNamespace(state=state, events=events or [])
    return SimpleNamespace(state=state, session=session, invocation_id=invocation_id)


def verifier_event(payload: dict, invocation_id: str = "inv-1", name: str = "verify_kernel"):
    from google.adk.events import Event

    return Event(
        author="Judge",
        invocation_id=invocation_id,
        content=types.Content(
            role="user",
            parts=[types.Part.from_function_response(name=name, response=payload)],
        ),
    )


# ----------------------------------------------------------------------- tree shape


@pytest.fixture(scope="module")
def tree() -> LlmAgent:
    return build_supervisor()


def test_root_is_the_supervisor(tree):
    assert isinstance(tree, LlmAgent)
    assert tree.name == "Supervisor"
    assert tree.output_key == "supervisor_summary"


def test_every_agent_uses_the_primary_model(tree):
    def walk(agent):
        if isinstance(agent, LlmAgent):
            yield agent
        for sub in agent.sub_agents:
            yield from walk(sub)

    models = {a.name: a.model for a in walk(tree)}
    assert models == dict.fromkeys(models, PRIMARY_MODEL), models


def test_loop_keeps_its_circuit_breaker(tree):
    """Red line #4: max_iterations is never removed."""
    loop = tree.find_agent("RefinementLoop")
    assert isinstance(loop, LoopAgent)
    assert loop.max_iterations == MAX_LOOP_ITERATIONS
    assert [s.name for s in loop.sub_agents] == ["Coder", "Judge", "EscalationChecker"]


def test_escalation_checker_is_a_base_agent_not_a_tool(tree):
    """ADK #501/#2692/#2808: escalation from a tool or callback does not exit the loop."""
    checker = tree.find_agent("EscalationChecker")
    assert isinstance(checker, EscalationChecker)
    tool_names = {t.name for a in (tree, tree.find_agent("Judge")) for t in a.tools}
    assert not any("escalat" in n for n in tool_names)


def test_coder_has_a_schema_and_no_tools(tree):
    coder = tree.find_agent("Coder")
    assert coder.output_schema is not None
    assert coder.tools == []
    assert coder.disallow_transfer_to_parent
    assert coder.disallow_transfer_to_peers


def test_judge_has_tools_and_no_output_schema(tree):
    """ADK #3969: schema + tools is fragile, so the callback supplies the structure."""
    judge = tree.find_agent("Judge")
    assert judge.output_schema is None
    assert [t.name for t in judge.tools] == ["verify_kernel"]
    assert judge.after_agent_callback is not None


def test_tree_can_be_built_twice():
    """Factories, not singletons: ADK binds one parent per agent instance."""
    assert build_supervisor().find_agent("Coder") is not build_supervisor().find_agent("Coder")


# ------------------------------------------------------------------ instructions


@pytest.mark.parametrize(
    "provider",
    [coder_instruction, judge_instruction, profiler_instruction, supervisor_instruction],
)
def test_instructions_render_on_empty_state(provider):
    """Iteration 1 has no fingerprint, skills or verdict; a `{template}` would KeyError."""
    text = provider(fake_ctx({}))
    assert text
    leftovers = [
        placeholder
        for placeholder in re.findall(r"{([a-z_]+)}", text)
        if placeholder in _PLACEHOLDERS
    ]
    assert not leftovers, leftovers


def test_judge_prompt_keeps_its_literal_json_braces():
    """The reply shape is spelled out in JSON, so `{{`/`}}` must survive .format()."""
    text = judge_instruction(fake_ctx({}))
    assert '{"reward": int, "correctness_pass": bool' in text


def test_coder_prompt_carries_the_judge_feedback():
    state = {
        "task_spec": {"op_name": "rmsnorm", "hidden_size": 1536},
        "verdict": {"next_action": "cast to float32 before rsqrt", "stderr_tail": "nan mismatch"},
    }
    text = coder_instruction(fake_ctx(state))
    assert "cast to float32 before rsqrt" in text
    assert "nan mismatch" in text
    assert "rmsnorm" in text


def test_coder_prompt_renders_a_string_verdict_left_by_output_key():
    """Before the Judge's callback lands, `verdict` is raw model text, not a dict."""
    state = {"verdict": '{"next_action": "add a tail mask", "stop": false}'}
    assert "add a tail mask" in coder_instruction(fake_ctx(state))


# ------------------------------------------------------------------- escalation


async def _decide(state: dict) -> bool:
    checker = build_escalation_checker()
    events = [e async for e in checker._run_async_impl(fake_ctx(state))]
    assert len(events) == 1
    return bool(events[0].actions.escalate)


@pytest.mark.asyncio
async def test_does_not_escalate_on_a_failing_verdict():
    assert await _decide({"verdict": {"reward": -1, "stop": False}, "iteration": 1}) is False


@pytest.mark.asyncio
async def test_does_not_escalate_on_a_correct_but_slow_kernel():
    assert await _decide({"verdict": {"reward": 2, "stop": False}, "iteration": 3}) is False


@pytest.mark.asyncio
async def test_escalates_on_the_winning_reward():
    assert await _decide({"verdict": {"reward": 3, "stop": False}, "iteration": 2}) is True


@pytest.mark.asyncio
async def test_escalates_when_the_judge_says_stop():
    assert await _decide({"verdict": {"reward": 1, "stop": True}, "iteration": 2}) is True


@pytest.mark.asyncio
async def test_escalates_when_the_iteration_budget_is_spent():
    state = {"verdict": {"reward": 1, "stop": False}, "iteration": MAX_LOOP_ITERATIONS}
    assert await _decide(state) is True


@pytest.mark.asyncio
async def test_unparsed_verdict_text_is_not_a_decision():
    """`output_key` writes raw text; a string is not a stop signal, so keep looping."""
    assert await _decide({"verdict": "the kernel looks great, reward 3", "iteration": 1}) is False


@pytest.mark.asyncio
async def test_empty_state_does_not_escalate():
    assert await _decide({}) is False


# ------------------------------------------------------------------ verdict JSON


def test_extracts_bare_json():
    assert extract_json_object('{"reward": 3, "stop": true}')["reward"] == 3


def test_extracts_json_from_a_fenced_block_with_prose():
    text = (
        'Here is my verdict.\n```json\n{"reward": 1, "next_action": "widen the tile"}\n```\nDone.'
    )
    assert extract_json_object(text)["next_action"] == "widen the tile"


def test_ignores_braces_inside_strings():
    text = '{"next_action": "replace {BLOCK} with 1024", "reward": 2}'
    assert extract_json_object(text)["next_action"] == "replace {BLOCK} with 1024"


def test_prefers_the_verdict_object_over_an_earlier_one():
    text = '{"op_name": "rmsnorm"} then {"reward": 2, "next_action": "coalesce loads"}'
    assert extract_json_object(text)["reward"] == 2


def test_unparseable_text_yields_nothing():
    assert extract_json_object("the kernel is fine, ship it") == {}
    assert extract_json_object("") == {}


# ---------------------------------------------------------------- reconciliation


MEASURED = {
    "reward": 1,
    "correctness_pass": True,
    "speedup_vs_eager": 0.97,
    "speedup_vs_compile": 0.80,
    "next_action": "Correct but not fast enough: propose ONE concrete change.",
    "stop": False,
    "stderr_tail": "",
    "latency_ms_by_shape": {"(8, 512)": 0.41},
}


def test_measured_numbers_beat_the_models_claims():
    """Red line #3: the model may not talk a losing kernel into a win."""
    parsed = {"reward": 3, "correctness_pass": True, "speedup_vs_eager": 4.2, "stop": True}
    verdict = reconcile(parsed, MEASURED)
    assert verdict.reward == 1
    assert verdict.speedup_vs_eager == pytest.approx(0.97)
    assert verdict.latency_ms_by_shape == {"(8, 512)": 0.41}


def test_the_model_still_owns_the_fix_instruction():
    parsed = {"reward": 1, "next_action": "increase BLOCK_SIZE to 1024", "stop": False}
    assert reconcile(parsed, MEASURED).next_action == "increase BLOCK_SIZE to 1024"


def test_falls_back_to_the_verifiers_own_next_action():
    assert reconcile({}, MEASURED).next_action == MEASURED["next_action"]


def test_a_winning_reward_forces_stop_even_if_the_model_forgot():
    verdict = reconcile({"stop": False}, MEASURED | {"reward": 3})
    assert verdict.stop is True


def test_no_verifier_call_means_reward_minus_one():
    verdict = reconcile({"reward": 3, "stop": True}, {})
    assert verdict.reward == -1
    assert verdict.correctness_pass is False
    assert verdict.stop is False
    assert "verifier_tool" in verdict.next_action


# ------------------------------------------------------------------- the callback


def test_find_verifier_response_takes_the_latest_call():
    events = [
        verifier_event({"reward": -1, "correctness_pass": False}),
        verifier_event({"reward": 2, "correctness_pass": True}),
    ]
    assert find_verifier_response(fake_ctx({}, events))["reward"] == 2


def test_find_verifier_response_ignores_other_tools_and_invocations():
    events = [
        verifier_event({"reward": 3}, name="profile_op_by_name"),
        verifier_event({"reward": 3}, invocation_id="inv-other"),
    ]
    assert find_verifier_response(fake_ctx({}, events)) == {}


def test_record_verdict_replaces_text_with_a_typed_verdict():
    state = FakeState(
        verdict='```json\n{"reward": 1, "next_action": "widen the tile", "stop": false}\n```',
        kernel_draft={"code": "# kernel v1", "entrypoint": "rmsnorm"},
    )
    record_verdict(fake_ctx(state, [verifier_event(MEASURED)]))

    assert Verdict(**state["verdict"]).next_action == "widen the tile"
    assert state["verdict"]["reward"] == 1
    assert state["iteration"] == 1
    assert state["best_reward"] == 1
    assert state["best_kernel"] == "# kernel v1"


def test_record_verdict_keeps_the_best_kernel_when_a_later_iteration_regresses():
    state = FakeState(
        verdict='{"next_action": "keep going", "stop": false}',
        kernel_draft={"code": "# winner", "entrypoint": "rmsnorm"},
    )
    record_verdict(fake_ctx(state, [verifier_event(MEASURED | {"reward": 2})]))
    assert state["best_reward"] == 2

    state["verdict"] = '{"next_action": "that broke it", "stop": false}'
    state["kernel_draft"] = {"code": "# regression", "entrypoint": "rmsnorm"}
    record_verdict(fake_ctx(state, [verifier_event(MEASURED | {"reward": -1})]))

    assert state["verdict"]["reward"] == -1
    assert state["best_reward"] == 2
    assert state["best_kernel"] == "# winner"
    assert state["iteration"] == 2


def test_record_verdict_survives_a_judge_turn_with_no_tool_call():
    state = FakeState(verdict="I think this kernel is correct.")
    record_verdict(fake_ctx(state, []))
    assert state["verdict"]["reward"] == -1
    assert state["best_reward"] == -1


# --------------------------------------------------------------------------- #
# The deployment contract must survive the trip from Coder to verifier
# --------------------------------------------------------------------------- #
#
# Both hops used to be a free-form `dict[str, str]`, and both silently produced `{}`:
# such a dict compiles to a JSON schema with no named properties, so structured
# generation has nothing to anchor on. Measured against gemini-3.7-flash on one prompt,
# the dict form filled 0/3 times and a list of two-field objects filled 3/3.
#
# An empty contract is not an error anywhere downstream — it just falls back to the
# hard-coded per-op adapter, i.e. the human-written bridge this project claims the agent
# writes for itself. Every green check kept passing while the novel path never ran, so
# these tests exist to make that failure loud.


def test_the_verifier_tool_does_not_ask_the_model_for_the_contract():
    """The Judge must not be able to restate, invent, or drop the mapping."""
    schema = verifier_tool._get_declaration().parameters_json_schema
    properties = set(schema["properties"])

    assert properties == {"kernel_code", "entrypoint", "task_spec"}
    assert "adapter_mapping" not in properties
    assert "tool_context" not in properties


def test_the_verifier_tool_keeps_its_published_name():
    """`find_verifier_response` matches on it and the Judge's prompt names it."""
    assert verifier_tool.name == "verify_kernel"


def test_a_declared_contract_converts_to_the_form_consumers_use():
    draft = {
        "adapter_mapping": [
            {"kernel_param": "weight", "module_attr": "weight"},
            {"kernel_param": "eps", "module_attr": "variance_epsilon"},
        ]
    }
    assert adapter_mapping_from_draft(draft) == {
        "weight": "weight",
        "eps": "variance_epsilon",
    }


def test_a_json_string_draft_is_parsed():
    """`output_key` hands back raw text until the draft is parsed; both shapes arrive."""
    draft = json.dumps(
        {"adapter_mapping": [{"kernel_param": "eps", "module_attr": "variance_epsilon"}]}
    )
    assert adapter_mapping_from_draft(draft) == {"eps": "variance_epsilon"}


def test_the_legacy_mapping_shape_still_deploys():
    """A session or skill row written before the schema changed must not break."""
    assert adapter_mapping_from_draft({"adapter_mapping": {"weight": "weight"}}) == {
        "weight": "weight"
    }


@pytest.mark.parametrize(
    "draft",
    [
        None,
        "not json at all",
        {},
        {"adapter_mapping": None},
        {"adapter_mapping": []},
        {"adapter_mapping": "weight=weight"},
        {"adapter_mapping": [{"kernel_param": "weight"}]},  # half an entry
        {"adapter_mapping": ["weight"]},
        {"adapter_mapping": [{"module_attr": "weight"}]},
    ],
)
def test_an_unusable_contract_degrades_to_empty_rather_than_raising(draft):
    """A malformed contract falls back to the per-op adapter; it never crashes the Judge."""
    assert adapter_mapping_from_draft(draft) == {}


def test_the_coder_is_told_to_emit_the_list_form():
    """The prompt and the schema have to describe the same shape."""
    from kernelsmith.agents.coder_agent import _INSTRUCTION

    assert "kernel_param" in _INSTRUCTION
    assert "module_attr" in _INSTRUCTION


def test_the_judge_is_told_not_to_pass_the_contract():
    """It is read from the draft; a Judge that restates it can corrupt it."""
    from kernelsmith.agents.judge_agent import _INSTRUCTION

    assert "not yours to pass" in _INSTRUCTION.lower()
