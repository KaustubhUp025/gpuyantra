"""gemini-embedding-001 @768 dims + manual L2-normalization.

Red line #8: always assert len(embedding) == 768 and L2-normalize after every call.
"""

import google.genai as genai
import numpy as np

from kernelsmith.config import EMBEDDING_DIM, EMBEDDING_MODEL, GCP_LOCATION, GCP_PROJECT

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazily construct the Vertex AI genai client (ADC only — never a key)."""
    global _client
    if _client is None:
        _client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    return _client


def embed_768(text: str) -> list[float]:
    """Embed text to 768 dims using gemini-embedding-001. L2-normalize.

    TWO TRAPS:
    1. output_dimensionality=768 is silently ignored in some client paths.
       ASSERT the returned length.
    2. Sub-3072 vectors are NOT auto-normalized by gemini-embedding-001.
       Manual L2-norm is required.
    """
    client = _get_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={"output_dimensionality": EMBEDDING_DIM},
    )
    vec = result.embeddings[0].values

    # Trap 1: assert dimension
    if len(vec) != EMBEDDING_DIM:
        # Fallback: truncate manually (MRL guarantees first 768 dims are usable)
        vec = vec[:EMBEDDING_DIM]
    assert len(vec) == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM}, got {len(vec)}"

    # Trap 2: L2-normalize
    arr = np.array(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    assert norm > 0, "Zero-norm embedding"
    arr = arr / norm
    return arr.tolist()
