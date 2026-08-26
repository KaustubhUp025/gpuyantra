"""Bottleneck-indexed retrieval with Firestore faked (spec 6.4 / 13.1).

The pre-filter is the part that can silently rot: Firestore vector search accepts only
EQUALITY pre-filters, and `op_family`/`hardware` must be exactly the fields that
precede `embedding` in the composite index (spec 6.2). If a filter is dropped or an
inequality creeps in, the query still runs — it just returns the wrong neighbours, or
nothing, with no error. So these tests assert the query shape, not just the result.
"""

from typing import Any

import pytest
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

from kernelsmith.config import EMBEDDING_DIM, FIRESTORE_COLLECTION_SKILLS, RETRIEVAL_TOP_K
from kernelsmith.tools import retrieval_tool as retrieval
from kernelsmith.tools.retrieval_tool import retrieve_skills, retrieve_skills_for_agent

FINGERPRINT_TEXT = "op=norm mem_bound=True ai=1.2 tile=1024 hw=L4"

SEED_SKILL = {
    "skill_id": "rmsnorm_fp16_l4_v1",
    "op_signature": "rmsnorm_fp16_[B,S,H]",
    "op_family": "norm",
    "hardware": "L4",
    "winning_kernel_source": "@triton.jit\ndef _rmsnorm_fwd(...): ...",
    "fix_rule": "Fuse the square-sum, rsqrt, and weight multiply into one pass.",
    "speedup_vs_eager": 1.71,
    "speedup_vs_torch_compile": 1.06,
    "embedding": Vector([0.01] * EMBEDDING_DIM),
    "vector_distance": 0.04,
}


class FakeDoc:
    def __init__(self, doc_id: str, data: dict[str, Any]):
        self.id = doc_id
        self._data = data

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


class FakeQuery:
    """Records every filter and find_nearest argument the tool applies."""

    def __init__(self, docs: list[FakeDoc]):
        self.docs = docs
        self.filters: list[tuple[str, str, Any]] = []
        self.nearest_kwargs: dict[str, Any] | None = None

    def where(self, field: str, op: str, value: Any) -> "FakeQuery":
        self.filters.append((field, op, value))
        return self

    def find_nearest(self, **kwargs: Any) -> "FakeQuery":
        self.nearest_kwargs = kwargs
        return self

    def stream(self):
        limit = (self.nearest_kwargs or {}).get("limit", len(self.docs))
        return iter(self.docs[:limit])


class FakeDb:
    def __init__(self, docs: list[FakeDoc]):
        self.query = FakeQuery(docs)
        self.collections: list[str] = []

    def collection(self, name: str) -> FakeQuery:
        self.collections.append(name)
        return self.query


@pytest.fixture
def fake_db() -> FakeDb:
    return FakeDb([FakeDoc(SEED_SKILL["skill_id"], SEED_SKILL)])


@pytest.fixture
def mock_embed(monkeypatch) -> list[str]:
    """Stand in for gemini-embedding-001; record what text got embedded."""
    embedded: list[str] = []

    def fake_embed_768(text: str) -> list[float]:
        embedded.append(text)
        return [0.0] * (EMBEDDING_DIM - 1) + [1.0]  # unit norm, 768 dims

    monkeypatch.setattr(retrieval, "embed_768", fake_embed_768)
    return embedded


# --------------------------------------------------------------------------- #
# Pre-filter shape
# --------------------------------------------------------------------------- #


def test_equality_prefilters_are_applied_on_op_family_and_hardware(fake_db, mock_embed):
    retrieve_skills("norm", "L4", FINGERPRINT_TEXT, db=fake_db)

    assert fake_db.collections == [FIRESTORE_COLLECTION_SKILLS]
    assert fake_db.query.filters == [("op_family", "==", "norm"), ("hardware", "==", "L4")]


def test_prefilters_are_equality_only(fake_db, mock_embed):
    """Firestore vector search rejects inequality pre-filters outright."""
    retrieve_skills("mlp", "L4", FINGERPRINT_TEXT, db=fake_db)
    assert all(op == "==" for _, op, _ in fake_db.query.filters)


def test_find_nearest_uses_cosine_over_the_embedding_field(fake_db, mock_embed):
    retrieve_skills("norm", "L4", FINGERPRINT_TEXT, db=fake_db)

    kwargs = fake_db.query.nearest_kwargs
    assert kwargs["vector_field"] == "embedding"
    assert kwargs["distance_measure"] is DistanceMeasure.COSINE
    assert kwargs["limit"] == RETRIEVAL_TOP_K
    assert kwargs["distance_result_field"] == "vector_distance"


def test_k_overrides_the_default_limit(fake_db, mock_embed):
    retrieve_skills("norm", "L4", FINGERPRINT_TEXT, k=1, db=fake_db)
    assert fake_db.query.nearest_kwargs["limit"] == 1


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #


def test_the_fingerprint_text_is_what_gets_embedded(fake_db, mock_embed):
    """Retrieval is by bottleneck, not by op name — the query vector must prove it."""
    retrieve_skills("norm", "L4", FINGERPRINT_TEXT, db=fake_db)
    assert mock_embed == [FINGERPRINT_TEXT]


def test_query_vector_is_a_768_dim_firestore_vector(fake_db, mock_embed):
    retrieve_skills("norm", "L4", FINGERPRINT_TEXT, db=fake_db)

    query_vector = fake_db.query.nearest_kwargs["query_vector"]
    assert isinstance(query_vector, Vector)
    assert len(list(query_vector)) == EMBEDDING_DIM


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


def test_results_are_plain_dicts_carrying_the_kernel_source(fake_db, mock_embed):
    results = retrieve_skills("norm", "L4", FINGERPRINT_TEXT, db=fake_db)

    assert isinstance(results, list)
    assert len(results) == 1
    skill = results[0]
    assert isinstance(skill, dict)
    assert skill["skill_id"] == "rmsnorm_fp16_l4_v1"
    assert skill["fix_rule"].startswith("Fuse")
    assert skill["speedup_vs_eager"] == pytest.approx(1.71)
    assert skill["vector_distance"] == pytest.approx(0.04)


def test_the_embedding_is_stripped_from_results(fake_db, mock_embed):
    """768 floats are the retrieval key, not context worth spending on the Coder."""
    assert "embedding" not in retrieve_skills("norm", "L4", FINGERPRINT_TEXT, db=fake_db)[0]


def test_missing_skill_id_falls_back_to_the_document_id(mock_embed):
    data = {k: v for k, v in SEED_SKILL.items() if k != "skill_id"}
    db = FakeDb([FakeDoc("doc-id-from-firestore", data)])

    assert retrieve_skills("norm", "L4", FINGERPRINT_TEXT, db=db)[0]["skill_id"] == (
        "doc-id-from-firestore"
    )


def test_a_cold_library_returns_an_empty_list(mock_embed):
    """Nothing learned yet is a normal state on run one, not an error."""
    assert retrieve_skills("rope", "L4", FINGERPRINT_TEXT, db=FakeDb([])) == []


# --------------------------------------------------------------------------- #
# Agent-facing wrapper
# --------------------------------------------------------------------------- #


def test_agent_wrapper_returns_skills_and_count(monkeypatch, fake_db, mock_embed):
    monkeypatch.setattr(
        retrieval, "skills_collection", lambda db=None: fake_db.collection("skills")
    )

    payload = retrieve_skills_for_agent("norm", "L4", FINGERPRINT_TEXT)
    assert payload["count"] == 1
    assert payload["skills"][0]["skill_id"] == "rmsnorm_fp16_l4_v1"
    assert "error" not in payload


def test_agent_wrapper_reports_firestore_failures_as_data(monkeypatch, mock_embed):
    """An unreachable library must not abort the run — the Coder can still write a kernel."""

    def unavailable(db=None):
        raise RuntimeError("503 Firestore unavailable")

    monkeypatch.setattr(retrieval, "skills_collection", unavailable)

    payload = retrieve_skills_for_agent("norm", "L4", FINGERPRINT_TEXT)
    assert payload == {
        "skills": [],
        "count": 0,
        # No arm was pulled, so the EscalationChecker credits nothing (spec 9).
        "selected_skill_id": "",
        "selected_skill": None,
        "error": "RuntimeError: 503 Firestore unavailable",
    }
