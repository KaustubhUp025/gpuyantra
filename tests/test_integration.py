"""End-to-end integration against live gemini-3.7-flash, Firestore and the GPU (spec 13.2).

Everything else in `tests/` is hermetic. This file is the opposite: it spends real
tokens, writes real Firestore rows, and compiles real Triton on a real card, because
the failures it exists to catch only appear when those meet each other — an ADK state
key that no unit test asserts on, a fingerprint the retrieval index rejects, a Coder
whose kernel compiles in isolation but not against the reference binding.

**Budget.** Spec 13.2 caps this at `max_iterations=2`, and the whole module drives the
loop exactly ONCE, in a session-scoped fixture, with every assertion reading that one
result. Rerunning the pipeline per test function would multiply the token spend by the
number of tests for no extra coverage.

**What is asserted, and what deliberately is not.** The pass bar is `reward >= -1`, not
`reward >= 3`. A generated kernel that fails to compile is a legitimate outcome of a
two-iteration budget with a nondeterministic model, and a test that demands a winning
kernel would fail on model variance rather than on a defect. What must hold every time
is that the *pipeline* completes: the profiler returns a well-formed fingerprint,
retrieval answers, the loop terminates via escalation rather than by running out of
iterations uncontrolled, and the verifier returns a scored verdict rather than an
exception. The reward-conditional assertions in `test_a_winning_kernel_reaches_the_
skill_library` then check the things that are only true when a kernel did win.

Skipped automatically without a CUDA device or Google credentials, so `make test-unit`
and CI stay green on a laptop. Run with `make test-int`.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# Guards — this module is skipped, never failed, when its world is absent
# --------------------------------------------------------------------------- #


def _cuda_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _credentials_are_available() -> bool:
    """ADC, explicit service account, or Compute Engine metadata server."""
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return False
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return True
    adc = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    if os.path.exists(adc):
        return True
    # Compute Engine VMs use the metadata server — google.auth.default() finds it
    try:
        from google.auth import default

        creds, _ = default()
        return creds is not None
    except Exception:
        return False


#: Spec 13.2: budget cap. This is a cap, not a removal — red line #4 forbids a
#: LoopAgent with no `max_iterations` at all, not one with a smaller budget.
INTEGRATION_MAX_ITERATIONS = 2

OP_NAME = "rmsnorm"
HIDDEN_SIZE = 1536  # Qwen2.5-1.5B
PROBE_BATCH = 16
PROBE_SEQ = 2048


# --------------------------------------------------------------------------- #
# The one live run
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pipeline() -> dict[str, Any]:
    """Drive profile -> retrieve -> refine once, and return what each stage produced.

    Stages run explicitly rather than through the Supervisor: the Supervisor is an
    LlmAgent that *decides* to call these tools, so driving it here would make the test
    fail whenever the model chose a different order — a prompt regression, not a
    pipeline defect. The Supervisor's own wiring is covered by `test_agents.py`, and
    the two-turn protocol end to end by `run_demo`.
    """
    # Guards live in the fixture body, not as marks: pytest refuses marks on fixtures,
    # and every test in this module depends on this fixture, so skipping here skips all.
    if not _cuda_is_available():
        pytest.skip("no CUDA device")
    if not _credentials_are_available():
        pytest.skip("no Google credentials / GOOGLE_CLOUD_PROJECT")

    # (a) Seed everything, before torch is imported anywhere in this process.
    from kernelsmith.reproducibility import seed_everything

    reproducibility = seed_everything()

    # (b) The task.
    task_spec = {"op_name": OP_NAME, "hidden_size": HIDDEN_SIZE}

    # (c) Profile the op -> BottleneckFingerprint.
    from kernelsmith.tools.profiler_tool import profile_op_by_name

    fingerprint = profile_op_by_name(OP_NAME, PROBE_BATCH, PROBE_SEQ, HIDDEN_SIZE)

    # (d) Retrieve prior skills. An empty list is a valid first-run answer.
    from kernelsmith.tools.retrieval_tool import retrieve_skills_for_agent

    retrieval = retrieve_skills_for_agent(
        op_family=fingerprint.get("op_family", "norm"),
        hardware=fingerprint.get("hardware", "L4"),
        fingerprint_text=fingerprint.get("fingerprint_text", ""),
    )

    # (e) Refinement loop, budget-capped, with real gemini-3.7-flash behind the Coder
    #     and the Judge, and the real subprocess verifier behind the Judge's tool.
    state = _run_refinement_loop(
        task_spec=task_spec,
        fingerprint=fingerprint,
        retrieval=retrieval,
    )

    return {
        "reproducibility": reproducibility,
        "task_spec": task_spec,
        "fingerprint": fingerprint,
        "retrieval": retrieval,
        "state": state,
    }


def _run_refinement_loop(
    task_spec: dict[str, Any],
    fingerprint: dict[str, Any],
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    """Run the real RefinementLoop over a seeded session and return its final state."""
    import asyncio

    from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
    from google.adk.runners import Runner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.genai import types

    from kernelsmith.agents.refinement_loop import build_refinement_loop

    loop = build_refinement_loop()
    # Budget cap for the test only. Asserted below so a future edit that drops
    # `max_iterations` entirely (red line #4) fails here too.
    loop.max_iterations = INTEGRATION_MAX_ITERATIONS
    assert loop.max_iterations == INTEGRATION_MAX_ITERATIONS

    runner = Runner(
        agent=loop,
        app_name="kernelsmith-integration",
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
    )
    user_id = "integration"
    session_id = f"int-{uuid.uuid4().hex[:10]}"

    async def drive() -> dict[str, Any]:
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
            # The loop's sub-agents read these three keys; the Supervisor would
            # normally have written them via its tools' output_keys.
            state={
                "task_spec": task_spec,
                "bottleneck_fingerprint": fingerprint,
                "retrieved_skills": retrieval.get("skills", []),
                "selected_skill_id": retrieval.get("selected_skill_id", ""),
                "iteration": 0,
            },
        )
        message = types.Content(
            role="user",
            parts=[types.Part(text=f"Optimize the {task_spec['op_name']} op for the L4.")],
        )
        async for _event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=message
        ):
            pass

        session = await runner.session_service.get_session(
            app_name=runner.app_name, user_id=user_id, session_id=session_id
        )
        return dict(getattr(session, "state", {}) or {})

    return asyncio.run(drive())


# --------------------------------------------------------------------------- #
# (a) Reproducibility
# --------------------------------------------------------------------------- #


def test_the_run_was_seeded(pipeline):
    """A speedup that needs a lucky seed is not a speedup (spec 11)."""
    from kernelsmith import config

    repro = pipeline["reproducibility"]
    assert repro["seed"] == config.GLOBAL_SEED
    assert repro["cublas_workspace_config"] == config.CUBLAS_WORKSPACE
    assert repro["deterministic_algorithms"] is config.DETERMINISTIC_CUDA
    assert repro["cuda_devices_seeded"] >= 1


def test_deterministic_algorithms_survived_the_timed_baselines(pipeline):
    """`measure_baselines` turns the flag off; it must have turned it back on."""
    import torch

    from kernelsmith import config

    assert torch.are_deterministic_algorithms_enabled() is config.DETERMINISTIC_CUDA


# --------------------------------------------------------------------------- #
# (c) Profiling
# --------------------------------------------------------------------------- #


def test_the_profiler_returns_a_wellformed_fingerprint(pipeline):
    """Every field the retrieval key and the Coder's prompt are built from."""
    from kernelsmith.memory.schemas import BottleneckFingerprint

    fingerprint = pipeline["fingerprint"]
    assert fingerprint.get("error") is None, fingerprint.get("error")

    # It must round-trip through the schema: retrieval and upsert both re-validate it.
    parsed = BottleneckFingerprint.model_validate(
        {k: v for k, v in fingerprint.items() if k in BottleneckFingerprint.model_fields}
    )
    assert parsed.op_family == "norm"
    assert parsed.hardware == "L4"
    assert parsed.arithmetic_intensity > 0
    assert 0.0 < parsed.achieved_occupancy <= 1.0
    assert parsed.memory_throughput_gbps > 0


def test_rmsnorm_is_fingerprinted_as_memory_bound(pipeline):
    """RMSNorm is a row-wise reduction: ~2 flops per byte. If this flips, the Coder is
    being told to optimize for math on a bandwidth-bound op."""
    fingerprint = pipeline["fingerprint"]
    assert fingerprint["is_memory_bound"] is True
    assert fingerprint["is_compute_bound"] is False
    assert fingerprint["arithmetic_intensity"] < fingerprint["ridge_point_flops_per_byte"]


def test_the_fingerprint_text_is_the_retrieval_key(pipeline):
    """The embedded text must carry the fields the composite pre-filter indexes on."""
    text = pipeline["fingerprint"]["fingerprint_text"]
    assert "op=norm" in text
    assert "hw=L4" in text
    assert "mem_bound=True" in text


# --------------------------------------------------------------------------- #
# (d) Retrieval — an empty library is a valid answer
# --------------------------------------------------------------------------- #


def test_retrieval_answers_without_raising(pipeline):
    """A cold or unreachable library degrades to zero skills, never to an exception."""
    retrieval = pipeline["retrieval"]
    assert isinstance(retrieval["skills"], list)
    assert retrieval["count"] == len(retrieval["skills"])
    if retrieval.get("error"):
        pytest.skip(f"skill library unavailable: {retrieval['error']}")


def test_the_bandit_picks_an_arm_when_the_library_is_warm(pipeline):
    """With rows present, UCB1 must select one and put it first for the Coder."""
    retrieval = pipeline["retrieval"]
    if not retrieval["skills"]:
        pytest.skip("cold skill library — nothing to select")

    assert retrieval["selected_skill"] is not None
    assert retrieval["selected_skill_id"]
    assert retrieval["skills"][0]["skill_id"] == retrieval["selected_skill_id"]


def test_retrieved_skills_carry_a_transferable_fix_rule(pipeline):
    """Retrieval is bottleneck-indexed: the `fix_rule` is the part that transfers."""
    skills = pipeline["retrieval"]["skills"]
    if not skills:
        pytest.skip("cold skill library")
    for skill in skills:
        assert skill.get("fix_rule"), f"{skill.get('skill_id')} has no fix_rule"
        assert skill.get("winning_kernel_source")


# --------------------------------------------------------------------------- #
# (e) + (f) The loop ran, terminated, and produced a scored verdict
# --------------------------------------------------------------------------- #


def test_the_loop_stayed_inside_its_budget(pipeline):
    """`max_iterations` is a circuit breaker (red line #4). It must actually bind."""
    iteration = pipeline["state"].get("iteration", 0)
    assert 1 <= iteration <= INTEGRATION_MAX_ITERATIONS, (
        f"loop ran {iteration} iterations against a cap of {INTEGRATION_MAX_ITERATIONS}"
    )


def test_the_coder_produced_a_kernel_draft(pipeline):
    """A live gemini-3.7-flash call reached the Coder and came back with something."""
    draft = pipeline["state"].get("kernel_draft")
    assert draft, "the Coder wrote nothing — check the model id and Vertex auth"


def test_the_judge_returned_a_parsed_verdict(pipeline):
    """The Judge has no `output_schema` (ADK #3969); its callback parses the JSON.

    If this comes back a raw string, that callback did not run and the
    EscalationChecker has been reading unparsed model text as a decision.
    """
    verdict = pipeline["state"].get("verdict")
    assert isinstance(verdict, dict), f"verdict was not parsed into a dict: {type(verdict)}"
    assert "reward" in verdict
    assert "correctness_pass" in verdict


def test_the_final_reward_is_a_valid_score(pipeline):
    """(f) Even a failed attempt is a valid outcome — but it must be SCORED.

    -1 means "the verifier ran and rejected it". A missing or non-integer reward means
    the verifier never returned, which is the failure this asserts against.
    """
    reward = pipeline["state"].get("best_reward")
    assert isinstance(reward, int), f"best_reward is {type(reward)}, not an int: {reward!r}"
    assert -1 <= reward <= 3, f"reward {reward} is outside the milestone range"


def test_the_verifier_measured_rather_than_guessed(pipeline):
    """Speedups must be present and finite whenever correctness passed."""
    verdict = pipeline["state"].get("best_verdict") or pipeline["state"].get("verdict") or {}
    if not verdict.get("correctness_pass"):
        pytest.skip(f"no correct kernel this run (reward {pipeline['state'].get('best_reward')})")

    import math

    for key in ("speedup_vs_eager", "speedup_vs_compile"):
        value = float(verdict.get(key, 0.0))
        assert math.isfinite(value), f"{key} is not finite: {value}"
        assert value > 0.0, f"{key} is {value}: a correct kernel cannot be infinitely slow"


def test_the_winning_kernel_survives_an_independent_full_grid(pipeline):
    """5 seeds x 3 shapes = 15 checks. Red line #3: never weaken the verifier.

    This cannot be read off `state["verdict"]`: the `Verdict` schema deliberately keeps
    only the decision fields, so `total_checks` never reaches session state. Asserting
    against it there would produce a test that skips on every run and reports comfort it
    never earned.

    So the winning kernel is re-verified here, directly. That is a second GPU pass, and
    it buys two things a state read could not: proof that the grid really ran at full
    width, and an independent reproduction of the agent's own result — the kernel scores
    the same way twice, from a fresh subprocess, under the same seed.
    """
    from kernelsmith.config import CORRECTNESS_SEEDS, CORRECTNESS_SHAPES

    state = pipeline["state"]
    reward = state.get("best_reward", -1)
    if reward < 1 or not state.get("best_kernel"):
        pytest.skip(f"no verified kernel to re-check this run (reward {reward})")

    from kernelsmith.tools.verifier_tool import verify_kernel

    result = verify_kernel(
        state["best_kernel"],
        str(state.get("best_entrypoint") or ""),
        pipeline["task_spec"],
        state.get("best_adapter_mapping") or None,
    )

    assert result["total_checks"] == CORRECTNESS_SEEDS * len(CORRECTNESS_SHAPES), (
        f"the grid ran {result['total_checks']} checks, not "
        f"{CORRECTNESS_SEEDS * len(CORRECTNESS_SHAPES)}"
    )
    assert result["passed_checks"] == result["total_checks"]
    assert result["correctness_pass"] is True
    # Reproduction: a kernel whose score moves between identical runs is not a result.
    assert result["reward"] == reward, (
        f"re-verification scored {result['reward']}, the run scored {reward}"
    )


# --------------------------------------------------------------------------- #
# The deployment contract — the novel contribution, live
# --------------------------------------------------------------------------- #


def test_the_coder_declared_a_real_deployment_contract(pipeline):
    """The agent must write the adapter, not fall back to the hard-coded one.

    An empty contract is not an error anywhere downstream — it silently routes the swap
    through the human-written per-op adapter. That failure is invisible unless something
    asserts on it, which is what this is. The contract is checked against the REAL
    `Qwen2RMSNorm`, so a mapping that names attributes the class does not have fails
    here rather than at swap time.
    """
    state = pipeline["state"]
    if int(state.get("best_reward", -1)) < 1:
        pytest.skip(f"no verified kernel this run (reward {state.get('best_reward')})")

    contract = state.get("best_adapter_mapping")
    assert isinstance(contract, dict), f"contract is {type(contract)}: {contract!r}"
    assert contract, (
        "the Coder declared an EMPTY deployment contract — the generic adapter never "
        "ran and the swap would fall back to the hard-coded per-op adapter"
    )

    from kernelsmith.verifier.adapter_mapping import validate_adapter_mapping

    errors = validate_adapter_mapping(OP_NAME, contract)
    assert not errors, f"the declared contract does not validate: {errors}"


def test_the_declared_contract_builds_a_working_generic_adapter(pipeline):
    """End of the novel path: the agent's own contract binds its own kernel."""
    state = pipeline["state"]
    contract = state.get("best_adapter_mapping") or {}
    if int(state.get("best_reward", -1)) < 1 or not contract:
        pytest.skip("no verified kernel with a declared contract this run")

    from kernelsmith.inference_server.patchable_ops import build_forward
    from kernelsmith.inference_server.server import _load_entrypoint

    entry = _load_entrypoint(state["best_kernel"], str(state["best_entrypoint"]))
    forward = build_forward(OP_NAME, entry, contract)

    # The generic adapter, not the per-op fallback. If this ever reads
    # `_rmsnorm_adapter`, a human wrote the bridge and the demo claim is false.
    assert forward.__qualname__.startswith("build_forward_from_mapping"), (
        f"the swap used {forward.__qualname__}, not the declared contract"
    )


# --------------------------------------------------------------------------- #
# (g) A winning kernel reaches the skill library
# --------------------------------------------------------------------------- #


def test_a_winning_kernel_reaches_the_skill_library(pipeline):
    """(g) reward >= 1 means a verified kernel exists, so it must be persistable.

    The RefinementLoop does not upsert — the Supervisor does, on turn 2 (see
    `.claude/rules/implementation-deviations.md`). So this drives the same upsert tool
    the Supervisor would call and asserts the row lands and reads back.
    """
    state = pipeline["state"]
    reward = state.get("best_reward", -1)
    if reward < 1:
        pytest.skip(f"no kernel cleared the +1 milestone this run (reward {reward})")

    assert state.get("best_kernel"), "reward >= 1 but no winning kernel was kept"

    from kernelsmith.memory.firestore_store import delete_skill, get_skill
    from kernelsmith.tools.upsert_tool import upsert_skill

    verdict = state.get("best_verdict") or state.get("verdict") or {}
    fingerprint = pipeline["fingerprint"]
    skill_id = f"integration-{uuid.uuid4().hex[:10]}"

    result = upsert_skill(
        {
            "skill_id": skill_id,
            "op_signature": f"{OP_NAME}_fp16_[B,S,H]",
            "op_family": fingerprint["op_family"],
            "hardware": fingerprint["hardware"],
            "bottleneck_fingerprint": {
                k: v for k, v in fingerprint.items() if k not in ("fingerprint_text",)
            },
            "winning_kernel_source": state["best_kernel"],
            "speedup_vs_eager": float(verdict.get("speedup_vs_eager", 0.0)),
            "speedup_vs_torch_compile": float(verdict.get("speedup_vs_compile", 0.0)),
            "fix_rule": "Written by the integration test from a verified run.",
            "tags": ["integration-test", OP_NAME],
        }
    )
    try:
        assert "error" not in str(result).lower(), result
        stored = get_skill(skill_id)
        assert stored is not None, f"skills/{skill_id} was not written"
        assert stored.winning_kernel_source == state["best_kernel"]
        assert len(stored.embedding) == 768  # red line #8
    finally:
        # The library is a demo asset; a test must not leave rows in it.
        delete_skill(skill_id)


def test_the_bandit_arm_was_credited_exactly_once(pipeline):
    """One run is one pull. The EscalationChecker writes the credit and guards it."""
    state = pipeline["state"]
    if not pipeline["retrieval"].get("selected_skill_id"):
        pytest.skip("cold library — no arm was pulled")
    assert state.get("bandit_credited") is True, (
        "an arm was selected but the EscalationChecker never credited it"
    )
