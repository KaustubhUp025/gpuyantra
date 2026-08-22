"""embed_768: dimension assertion + L2-normalization (spec 6.3 / red line #8).

The Vertex call is mocked throughout — these are unit tests and must run without
credentials, network, or spend. The live path is covered by the integration suite.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from kernelsmith import config
from kernelsmith.memory import embeddings


class FakeEmbedContent:
    """Records the call and replays a canned vector, like models.embed_content."""

    def __init__(self, values: list[float]):
        self.values = values
        self.calls: list[dict] = []

    def __call__(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(embeddings=[SimpleNamespace(values=list(self.values))])


@pytest.fixture
def fake_api(monkeypatch):
    """Install a fake client; returns a function to set the canned response."""

    def install(values: list[float]) -> FakeEmbedContent:
        embed_content = FakeEmbedContent(values)
        client = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
        monkeypatch.setattr(embeddings, "_get_client", lambda: client)
        monkeypatch.setattr(embeddings, "_client", None)
        return embed_content

    return install


def rng_vector(n: int, seed: int = 0, scale: float = 7.0) -> list[float]:
    """Deliberately un-normalized: gemini-embedding-001 does NOT normalize sub-3072 output."""
    return (np.random.default_rng(seed).standard_normal(n) * scale).tolist()


def test_returns_768_dims(fake_api):
    fake_api(rng_vector(768))
    assert len(embeddings.embed_768("op=norm mem_bound=True")) == 768


def test_output_is_unit_norm(fake_api):
    """Trap 2: sub-3072 vectors arrive un-normalized; embed_768 must fix that."""
    raw = rng_vector(768, seed=1)
    fake_api(raw)
    assert not math.isclose(np.linalg.norm(raw), 1.0, abs_tol=1e-6)  # precondition

    vec = embeddings.embed_768("anything")

    assert math.isclose(np.linalg.norm(vec), 1.0, abs_tol=1e-6)


def test_preserves_direction(fake_api):
    """Normalization must only rescale — cosine similarity with the raw vector is 1.0."""
    raw = rng_vector(768, seed=2)
    fake_api(raw)

    vec = embeddings.embed_768("anything")

    cosine = np.dot(vec, raw) / (np.linalg.norm(vec) * np.linalg.norm(raw))
    assert math.isclose(cosine, 1.0, abs_tol=1e-9)


def test_truncates_oversized_vector_to_768(fake_api):
    """Trap 1: output_dimensionality is silently ignored on some client paths.

    MRL guarantees the leading 768 dims are usable, so truncate then re-normalize.
    """
    raw = rng_vector(3072, seed=3)
    fake_api(raw)

    vec = embeddings.embed_768("anything")

    assert len(vec) == 768
    assert math.isclose(np.linalg.norm(vec), 1.0, abs_tol=1e-6)
    expected = np.array(raw[:768]) / np.linalg.norm(raw[:768])
    assert np.allclose(vec, expected, atol=1e-12)


def test_rejects_undersized_vector(fake_api):
    """Truncation cannot rescue a short vector — fail loudly rather than pad."""
    fake_api(rng_vector(512, seed=4))

    with pytest.raises(AssertionError, match="Expected 768"):
        embeddings.embed_768("anything")


def test_rejects_zero_norm_vector(fake_api):
    """A zero vector would divide by zero and poison COSINE retrieval."""
    fake_api([0.0] * 768)

    with pytest.raises(AssertionError, match="Zero-norm"):
        embeddings.embed_768("anything")


def test_requests_correct_model_and_dimensionality(fake_api):
    embed_content = fake_api(rng_vector(768, seed=5))

    embeddings.embed_768("op=norm mem_bound=True ai=0.5 tile=1024 hw=L4")

    assert len(embed_content.calls) == 1
    call = embed_content.calls[0]
    assert call["model"] == config.EMBEDDING_MODEL == "gemini-embedding-001"
    assert call["config"]["output_dimensionality"] == config.EMBEDDING_DIM == 768
    assert call["contents"] == "op=norm mem_bound=True ai=0.5 tile=1024 hw=L4"


def test_output_is_plain_python_floats(fake_api):
    """Firestore's Vector() and Pydantic both reject numpy scalars."""
    fake_api(rng_vector(768, seed=6))

    vec = embeddings.embed_768("anything")

    assert isinstance(vec, list)
    assert all(type(x) is float for x in vec)
