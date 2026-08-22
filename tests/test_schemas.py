"""Pydantic validation for every structured boundary (spec 3 / 13.1)."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from kernelsmith.memory.schemas import (
    BottleneckFingerprint,
    KernelDraft,
    RunRecord,
    SkillRecord,
    TraceRecord,
    Verdict,
)

VALID_EMBEDDING = [0.1] * 768


def make_fingerprint(**overrides) -> BottleneckFingerprint:
    kwargs = {
        "op_family": "norm",
        "hardware": "L4",
        "memory_throughput_gbps": 212.4,
        "achieved_occupancy": 0.62,
        "arithmetic_intensity": 0.5,
        "is_memory_bound": True,
        "is_compute_bound": False,
        "tile_size_hint": 1024,
    }
    kwargs.update(overrides)
    return BottleneckFingerprint(**kwargs)


def make_skill(**overrides) -> SkillRecord:
    kwargs = {
        "skill_id": "rmsnorm_fp16_l4_v1",
        "op_signature": "rmsnorm_fp16_[B,S,H]",
        "op_family": "norm",
        "hardware": "L4",
        "bottleneck_fingerprint": make_fingerprint(),
        "winning_kernel_source": "import triton  # ...",
        "speedup_vs_eager": 1.71,
        "speedup_vs_torch_compile": 1.06,
        "fix_rule": "Fuse the row reduction and the weight multiply.",
        "embedding": VALID_EMBEDDING,
    }
    kwargs.update(overrides)
    return SkillRecord(**kwargs)


# --------------------------------------------------------------------------- #
# BottleneckFingerprint
# --------------------------------------------------------------------------- #


def test_fingerprint_valid():
    fp = make_fingerprint()
    assert fp.op_family == "norm"
    assert fp.hardware == "L4"  # default applies
    assert fp.is_memory_bound and not fp.is_compute_bound


def test_fingerprint_embedding_text_is_the_retrieval_key():
    text = make_fingerprint().to_embedding_text()
    assert text == "op=norm mem_bound=True ai=0.5 tile=1024 hw=L4"


def test_fingerprint_embedding_text_rounds_arithmetic_intensity():
    assert "ai=12.3" in make_fingerprint(arithmetic_intensity=12.345).to_embedding_text()


@pytest.mark.parametrize("occupancy", [-0.01, 1.01, 5.0])
def test_fingerprint_rejects_occupancy_outside_unit_interval(occupancy):
    with pytest.raises(ValidationError):
        make_fingerprint(achieved_occupancy=occupancy)


def test_fingerprint_rejects_negative_arithmetic_intensity():
    with pytest.raises(ValidationError):
        make_fingerprint(arithmetic_intensity=-1.0)


def test_fingerprint_rejects_zero_tile_size():
    with pytest.raises(ValidationError):
        make_fingerprint(tile_size_hint=0)


def test_fingerprint_rejects_missing_field():
    with pytest.raises(ValidationError):
        BottleneckFingerprint(op_family="norm")


# --------------------------------------------------------------------------- #
# SkillRecord
# --------------------------------------------------------------------------- #


def test_skill_record_valid():
    skill = make_skill()
    assert len(skill.embedding) == 768
    assert skill.bandit_pulls == 0
    assert skill.bandit_total_reward == 0.0
    assert skill.tags == []
    assert skill.created_at.tzinfo is not None


@pytest.mark.parametrize("dim", [0, 767, 769, 3072])
def test_skill_record_rejects_wrong_embedding_dim(dim):
    """Red line #8: a non-768 embedding must never reach Firestore."""
    with pytest.raises(ValidationError) as exc:
        make_skill(embedding=[0.1] * dim)
    assert "768" in str(exc.value)


def test_skill_record_rejects_missing_required_field():
    with pytest.raises(ValidationError) as exc:
        SkillRecord(skill_id="x", op_family="norm", embedding=VALID_EMBEDDING)
    missing = {e["loc"][0] for e in exc.value.errors() if e["type"] == "missing"}
    assert {"op_signature", "bottleneck_fingerprint", "winning_kernel_source"} <= missing


def test_skill_record_rejects_invalid_nested_fingerprint():
    with pytest.raises(ValidationError):
        make_skill(bottleneck_fingerprint={"op_family": "norm"})


def test_skill_record_roundtrips_through_dict():
    original = make_skill()
    restored = SkillRecord(**original.model_dump())
    assert restored.embedding == original.embedding
    assert restored.bottleneck_fingerprint == original.bottleneck_fingerprint


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #


def make_verdict(**overrides) -> Verdict:
    kwargs = {
        "reward": 2,
        "correctness_pass": True,
        "speedup_vs_eager": 1.7,
        "speedup_vs_compile": 1.05,
        "next_action": "STOP",
        "stop": True,
    }
    kwargs.update(overrides)
    return Verdict(**kwargs)


def test_verdict_valid():
    verdict = make_verdict()
    assert verdict.stderr_tail == ""
    assert verdict.latency_ms_by_shape == {}


@pytest.mark.parametrize("reward", [-2, 4, 100])
def test_verdict_rejects_reward_outside_milestone_range(reward):
    """Milestone rewards are -1/+1/+2/+3 — nothing outside [-1, 3] is legal."""
    with pytest.raises(ValidationError):
        make_verdict(reward=reward)


@pytest.mark.parametrize("reward", [-1, 0, 1, 2, 3])
def test_verdict_accepts_full_milestone_range(reward):
    assert make_verdict(reward=reward).reward == reward


def test_verdict_rejects_missing_next_action():
    with pytest.raises(ValidationError):
        Verdict(
            reward=1,
            correctness_pass=True,
            speedup_vs_eager=1.0,
            speedup_vs_compile=1.0,
            stop=False,
        )


# --------------------------------------------------------------------------- #
# KernelDraft
# --------------------------------------------------------------------------- #


def test_kernel_draft_valid():
    draft = KernelDraft(
        code="import triton\n",
        entrypoint="rmsnorm_triton",
        rationale="Memory-bound; fuse the reduction.",
    )
    assert draft.block_sizes == {}


def test_kernel_draft_rejects_missing_rationale():
    """The rationale must reference the fingerprint — it is never optional."""
    with pytest.raises(ValidationError):
        KernelDraft(code="import triton", entrypoint="rmsnorm_triton")


# --------------------------------------------------------------------------- #
# RunRecord / TraceRecord
# --------------------------------------------------------------------------- #


def test_run_record_defaults():
    run = RunRecord(
        run_id="run-1",
        task_ref="kernelbench/L1/rmsnorm",
        started_at=datetime.now(UTC),
    )
    assert run.status == "running"
    assert run.ended_at is None
    assert run.final_reward == -1
    assert run.total_iterations == 0


def test_run_record_rejects_missing_started_at():
    with pytest.raises(ValidationError):
        RunRecord(run_id="run-1", task_ref="kernelbench/L1/rmsnorm")


def test_trace_record_valid():
    trace = TraceRecord(
        iteration=1,
        agent="coder",
        prompt_summary="Optimize RMSNorm",
        response_summary="Emitted fused kernel",
        reward=2,
        latency_ms_by_shape={"(8, 512)": 0.041},
    )
    assert trace.timestamp.tzinfo is not None


def test_trace_record_rejects_non_integer_iteration():
    with pytest.raises(ValidationError):
        TraceRecord(
            iteration="first",
            agent="coder",
            prompt_summary="",
            response_summary="",
            reward=0,
            latency_ms_by_shape={},
        )
