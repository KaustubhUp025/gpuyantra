"""Cross-model skill transfer (Task 10, Part E).

The claim: a kernel learned on Qwen2.5's RMSNorm is retrievable for GPT-2's LayerNorm
and ResNet-50's BatchNorm, because skills are indexed by the roofline FINGERPRINT and
pre-filtered on `(op_family, hardware)` — never on the op's name and never on the model
it came from. A name-keyed cache cannot make that jump; that is the whole point of
bottleneck-indexed retrieval (spec 6.4).

This file proves it at the API level, with Firestore faked, so the assertion is about the
QUERY the system issues and not about whatever happens to be in the live library on the
day. The fake is deliberately strict: it records the pre-filters, so a test cannot pass
by the tool having quietly dropped one and matched everything.

`retrieve_skills` is exercised against a single seeded Qwen2.5 RMSNorm skill, then
queried three times — once per registered architecture — and the same skill must come
back every time.
"""

from __future__ import annotations

from typing import Any

import pytest
from google.cloud.firestore_v1.vector import Vector

from kernelsmith.config import EMBEDDING_DIM, FIRESTORE_COLLECTION_SKILLS, MODEL_REGISTRY
from kernelsmith.memory.schemas import BottleneckFingerprint
from kernelsmith.tools import retrieval_tool as retrieval
from kernelsmith.tools.profiler_tool import family_from_name
from kernelsmith.tools.retrieval_tool import retrieve_skills, retrieve_skills_for_agent

#: One skill, learned on Qwen2.5-1.5B's RMSNorm. Note `op_signature` names rmsnorm and
#: `tags` names the model: neither is used as a filter, and that is the test.
QWEN_RMSNORM_SKILL: dict[str, Any] = {
    "skill_id": "rmsnorm_fp16_l4_v1",
    "op_signature": "rmsnorm_fp16_[B,S,H]",
    "op_family": "norm",
    "hardware": "L4",
    "winning_kernel_source": "@triton.jit\ndef _rmsnorm_fwd(...): ...",
    "fix_rule": "Fuse the square-sum, rsqrt and weight multiply into one pass.",
    "speedup_vs_eager": 6.92,
    "speedup_vs_torch_compile": 1.36,
    "tags": ["qwen2.5-1.5b", "Qwen2RMSNorm"],
    "embedding": Vector([0.01] * EMBEDDING_DIM),
    "vector_distance": 0.04,
    "bandit_pulls": 3,
    "bandit_total_reward": 9.0,
}

#: The bottleneck each architecture's normalization presents. Different models, different
#: class names, same family and same side of the ridge point.
TARGETS = {
    "qwen2.5-1.5b": ("Qwen2RMSNorm", 1536),
    "gpt2": ("LayerNorm", 768),
    "resnet50": ("BatchNorm2d", 2048),
}


class FakeDoc:
    def __init__(self, doc_id: str, data: dict[str, Any]):
        self.id = doc_id
        self._data = data

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


class FakeQuery:
    """Records every pre-filter, so a dropped filter cannot masquerade as a match."""

    def __init__(self, docs: list[FakeDoc]):
        self.docs = docs
        self.filters: list[tuple[str, str, Any]] = []
        self.nearest_kwargs: dict[str, Any] | None = None

    def where(self, field: str, op: str, value: Any) -> FakeQuery:
        self.filters.append((field, op, value))
        return self

    def find_nearest(self, **kwargs: Any) -> FakeQuery:
        self.nearest_kwargs = kwargs
        return self

    def stream(self):
        """Honour the recorded equality pre-filters, the way Firestore would."""
        matching = [
            doc
            for doc in self.docs
            if all(doc.to_dict().get(field) == value for field, _, value in self.filters)
        ]
        limit = (self.nearest_kwargs or {}).get("limit", len(matching))
        return iter(matching[:limit])


class FakeDb:
    def __init__(self, docs: list[FakeDoc]):
        self.query = FakeQuery(docs)
        self.collections: list[str] = []

    def collection(self, name: str) -> FakeQuery:
        self.collections.append(name)
        return self.query


@pytest.fixture
def library() -> FakeDb:
    """A library holding exactly one skill: the Qwen2.5 RMSNorm kernel."""
    return FakeDb([FakeDoc(QWEN_RMSNORM_SKILL["skill_id"], QWEN_RMSNORM_SKILL)])


@pytest.fixture
def mock_embed(monkeypatch) -> list[str]:
    """Stand in for gemini-embedding-001; record what text was embedded."""
    embedded: list[str] = []

    def fake_embed_768(text: str) -> list[float]:
        embedded.append(text)
        return [0.0] * (EMBEDDING_DIM - 1) + [1.0]

    monkeypatch.setattr(retrieval, "embed_768", fake_embed_768)
    return embedded


def fingerprint_for(module_type: str, hidden_size: int) -> BottleneckFingerprint:
    """The fingerprint a memory-bound norm of this width produces on the L4."""
    from kernelsmith.tools.profiler_tool import compute_tile_hint

    return BottleneckFingerprint(
        op_family=family_from_name(module_type),
        hardware="L4",
        memory_throughput_gbps=0.0,
        achieved_occupancy=0.5,
        arithmetic_intensity=0.9,
        is_memory_bound=True,
        is_compute_bound=False,
        tile_size_hint=compute_tile_hint(hidden_size),
    )


# --------------------------------------------------------------------------- #
# The three architectures all classify into the same family
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("module_type", "hidden"), list(TARGETS.values()))
def test_every_architectures_norm_classifies_into_op_family_norm(module_type, hidden):
    """If this failed, the pre-filter alone would make transfer impossible."""
    assert fingerprint_for(module_type, hidden).op_family == "norm"


def test_the_registry_and_the_targets_table_agree():
    assert set(TARGETS) == set(MODEL_REGISTRY)


# --------------------------------------------------------------------------- #
# The transfer itself
# --------------------------------------------------------------------------- #


def test_a_qwen_rmsnorm_skill_is_retrieved_for_gpt2s_layernorm(library, mock_embed):
    """The headline: a skill learned on one architecture, retrieved for another."""
    fingerprint = fingerprint_for("LayerNorm", 768)

    skills = retrieve_skills(
        fingerprint.op_family,
        fingerprint.hardware,
        fingerprint.to_embedding_text(),
        db=library,
    )

    assert [skill["skill_id"] for skill in skills] == ["rmsnorm_fp16_l4_v1"]
    assert skills[0]["op_signature"] == "rmsnorm_fp16_[B,S,H]"  # learned on rmsnorm
    assert "qwen2.5-1.5b" in skills[0]["tags"]  # and on another model


@pytest.mark.parametrize(("model", "target"), sorted(TARGETS.items()))
def test_the_same_skill_is_retrieved_for_every_registered_architecture(
    model, target, library, mock_embed
):
    module_type, hidden = target
    fingerprint = fingerprint_for(module_type, hidden)

    skills = retrieve_skills(
        fingerprint.op_family,
        fingerprint.hardware,
        fingerprint.to_embedding_text(),
        db=library,
    )

    assert [skill["skill_id"] for skill in skills] == ["rmsnorm_fp16_l4_v1"], (
        f"{model}'s {module_type} could not reach the Qwen2.5 norm skill"
    )


def test_the_query_pre_filters_on_family_and_hardware_and_nothing_else(library, mock_embed):
    """Filtering on the op name or the model would break transfer by construction."""
    fingerprint = fingerprint_for("LayerNorm", 768)
    retrieve_skills(
        fingerprint.op_family, fingerprint.hardware, fingerprint.to_embedding_text(), db=library
    )

    assert library.collections == [FIRESTORE_COLLECTION_SKILLS]
    assert library.query.filters == [("op_family", "==", "norm"), ("hardware", "==", "L4")]
    filtered_fields = {field for field, _, _ in library.query.filters}
    assert "op_signature" not in filtered_fields
    assert "tags" not in filtered_fields
    assert "skill_id" not in filtered_fields


def test_what_gets_embedded_is_the_bottleneck_and_carries_no_model_or_op_name(library, mock_embed):
    """The retrieval key is WHY the op is slow. If it named the op, GPT-2 would miss."""
    fingerprint = fingerprint_for("LayerNorm", 768)
    retrieve_skills(
        fingerprint.op_family, fingerprint.hardware, fingerprint.to_embedding_text(), db=library
    )

    assert len(mock_embed) == 1
    text = mock_embed[0]
    assert text == fingerprint.to_embedding_text()
    assert "mem_bound=True" in text
    for leaked in ("layernorm", "gpt2", "qwen", "rmsnorm", "batchnorm"):
        assert leaked not in text.lower(), f"{leaked!r} leaked into the retrieval key"


def test_gpt2_and_qwen_produce_the_same_retrieval_key_when_the_bottleneck_matches():
    """Same family, same boundedness, same tile hint — literally the same query vector."""
    qwen = fingerprint_for("Qwen2RMSNorm", 1536)
    gpt2 = fingerprint_for("LayerNorm", 1536)
    assert qwen.to_embedding_text() == gpt2.to_embedding_text()


def test_a_different_bottleneck_does_not_retrieve_the_norm_skill(library, mock_embed):
    """Transfer is not indiscriminate: an mlp fingerprint must not reach a norm skill."""
    skills = retrieve_skills(
        "mlp", "L4", "op=mlp mem_bound=False ai=682.0 tile=1024 hw=L4", db=library
    )
    assert skills == []


def test_a_different_hardware_does_not_retrieve_the_l4_skill(library, mock_embed):
    """A kernel tuned for the L4's ridge point is not evidence about another GPU."""
    assert (
        retrieve_skills(
            "norm", "A100", "op=norm mem_bound=True ai=0.9 tile=1024 hw=A100", db=library
        )
        == []
    )


# --------------------------------------------------------------------------- #
# Through the agent-facing tool, including the bandit
# --------------------------------------------------------------------------- #


def test_the_agent_tool_returns_the_transferred_skill_as_the_bandits_pick(
    library, mock_embed, monkeypatch
):
    monkeypatch.setattr(
        retrieval, "retrieve_skills", lambda *a, **k: retrieve_skills(*a[:3], db=library)
    )
    fingerprint = fingerprint_for("LayerNorm", 768)

    result = retrieve_skills_for_agent(
        fingerprint.op_family, fingerprint.hardware, fingerprint.to_embedding_text()
    )

    assert result["count"] == 1
    assert result["selected_skill_id"] == "rmsnorm_fp16_l4_v1"
    assert result["skills"][0]["skill_id"] == "rmsnorm_fp16_l4_v1"


def test_a_cold_library_is_a_cold_start_not_an_error(mock_embed, monkeypatch):
    """Every architecture starts cold once. That must read as "write it from scratch"."""
    empty = FakeDb([])
    monkeypatch.setattr(
        retrieval, "retrieve_skills", lambda *a, **k: retrieve_skills(*a[:3], db=empty)
    )

    result = retrieve_skills_for_agent(
        "norm", "L4", "op=norm mem_bound=True ai=0.9 tile=1024 hw=L4"
    )

    assert result["skills"] == []
    assert result["count"] == 0
    assert result["selected_skill"] is None
    assert "error" not in result


def test_the_embedding_stays_out_of_the_retrieved_payload(library, mock_embed):
    """768 floats are the key, not context for the Coder."""
    skills = retrieve_skills(
        "norm", "L4", "op=norm mem_bound=True ai=0.9 tile=1024 hw=L4", db=library
    )
    assert "embedding" not in skills[0]
    assert skills[0]["winning_kernel_source"]  # but the kernel itself does come through
