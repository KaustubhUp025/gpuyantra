"""Bottleneck-indexed skill retrieval (spec 6.4). The core novelty.

Skills are retrieved by WHY an op is slow — the roofline fingerprint text — not by the
op's name. A skill learned on RMSNorm surfaces for RoPE when both are memory-bound
row-wise reductions with the same tile hint, which is exactly the transfer a
name-keyed cache can never make.

Firestore vector search constraints that shape this file:
  - Pre-filters must be EQUALITY only, and their fields must precede the vector field
    in the composite index (spec 6.2). `op_family` and `hardware` are those fields.
  - COSINE distance requires unit vectors on both sides; `embed_768` L2-normalizes.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import FunctionTool
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

from kernelsmith.config import RETRIEVAL_TOP_K
from kernelsmith.memory.embeddings import embed_768
from kernelsmith.memory.firestore_store import skills_collection

#: The 768 floats are the retrieval key, not context for the Coder. Stripping them
#: keeps a top-3 response ~2 KB instead of ~40 KB of digits.
_STRIPPED_FIELDS = ("embedding",)


def retrieve_skills(
    op_family: str,
    hardware: str,
    fingerprint_text: str,
    k: int = RETRIEVAL_TOP_K,
    db: Any = None,
) -> list[dict[str, Any]]:
    """Find prior winning kernels whose bottleneck matches this one.

    Args:
        op_family: Equality pre-filter — "norm" | "rope" | "mlp" | "elementwise" |
            "reduction".
        hardware: Equality pre-filter — "L4".
        fingerprint_text: `BottleneckFingerprint.to_embedding_text()`. This is what
            gets embedded and compared, so it must describe the BOTTLENECK.
        k: Number of neighbours to return.
        db: Firestore client override, for tests.

    Returns:
        Up to `k` skill dicts, nearest first, each carrying `winning_kernel_source`,
        `fix_rule`, `speedup_vs_eager`, and `vector_distance` (0 = identical
        bottleneck). The `embedding` field is stripped. Returns [] when nothing has
        been learned yet for this family — a cold library is not an error.
    """
    query_vec = Vector(embed_768(fingerprint_text))
    query = (
        skills_collection(db)
        .where("op_family", "==", op_family)
        .where("hardware", "==", hardware)
        .find_nearest(
            vector_field="embedding",
            query_vector=query_vec,
            distance_measure=DistanceMeasure.COSINE,
            limit=k,
            distance_result_field="vector_distance",
        )
    )
    return [_to_skill_dict(doc) for doc in query.stream()]


def _to_skill_dict(doc: Any) -> dict[str, Any]:
    """Firestore snapshot -> plain dict, embedding stripped, doc id preserved."""
    data = dict(doc.to_dict() or {})
    for field in _STRIPPED_FIELDS:
        data.pop(field, None)
    doc_id = getattr(doc, "id", None)
    if doc_id and not data.get("skill_id"):
        data["skill_id"] = doc_id
    return data


def retrieve_skills_for_agent(
    op_family: str,
    hardware: str,
    fingerprint_text: str,
    k: int = RETRIEVAL_TOP_K,
) -> dict[str, Any]:
    """Retrieve prior winning kernels for a bottleneck like this one.

    Searches the learned skill library by bottleneck fingerprint, not by op name, so a
    kernel that worked on a similar bottleneck (same boundedness, same tile shape) is
    returned even if it was learned on a different operation. Use the returned
    `fix_rule` and `winning_kernel_source` as a starting point, never verbatim.

    Args:
        op_family: "norm", "rope", "mlp", "elementwise", or "reduction".
        hardware: GPU the skill was learned on. Always "L4" here.
        fingerprint_text: The bottleneck fingerprint text from the profiler.
        k: How many prior skills to return (default 3).

    Returns:
        {"skills": [...], "count": int}, nearest bottleneck first. An empty list means
        nothing has been learned for this family yet — write the kernel from scratch.
    """
    try:
        skills = retrieve_skills(op_family, hardware, fingerprint_text, k)
    except Exception as exc:  # noqa: BLE001 — a cold or unreachable library is not fatal
        return {"skills": [], "count": 0, "error": f"{type(exc).__name__}: {exc}"}
    return {"skills": skills, "count": len(skills)}


#: Registered on the Supervisor (spec 4.2), which writes the result to
#: `session.state["retrieved_skills"]` for the Coder.
retrieval_tool = FunctionTool(retrieve_skills_for_agent)
