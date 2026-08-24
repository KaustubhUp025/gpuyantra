"""Skill write-back with dedupe (spec 6.5).

One row per `skill_id`, and the row that survives is the one with the higher
`speedup_vs_eager`. Without that rule the library fills with worse re-derivations of
kernels it already knows, and retrieval starts handing the Coder a regression as its
starting point.

The embedding is computed HERE, from the bottleneck fingerprint, because the caller is
an LLM: no agent can emit 768 normalized floats, and a zero or random vector would
silently destroy retrieval quality for every later run.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import FunctionTool

from kernelsmith.memory.embeddings import embed_768
from kernelsmith.memory.firestore_store import upsert_skill as _store_upsert_skill
from kernelsmith.memory.schemas import BottleneckFingerprint, SkillRecord


def upsert_skill(skill_data: dict) -> str:
    """Save a verified winning kernel to the shared skill library.

    Call this only for a kernel the verifier has already scored — a wrong or slower
    kernel must never enter the library. If a skill with the same `skill_id` already
    exists with an equal or better `speedup_vs_eager`, the existing one is kept.

    Args:
        skill_data: A SkillRecord as a dict: skill_id, op_signature, op_family,
            hardware, bottleneck_fingerprint, winning_kernel_source, speedup_vs_eager,
            speedup_vs_torch_compile, fix_rule, and optional tags. The 768-dim
            embedding is computed here from the fingerprint — never pass one.

    Returns:
        "upserted" if the library was updated, "kept_existing" if the stored kernel was
        already at least as fast, or "error: <reason>" if the record was invalid.
    """
    try:
        record = _to_record(skill_data)
    except Exception as exc:  # noqa: BLE001 — a malformed record is data, not a crash
        return f"error: {type(exc).__name__}: {exc}"

    try:
        return _store_upsert_skill(record)
    except Exception as exc:  # noqa: BLE001 — a Firestore outage must not kill the run
        return f"error: {type(exc).__name__}: {exc}"


def _to_record(skill_data: dict[str, Any]) -> SkillRecord:
    """Validate into a SkillRecord, embedding the fingerprint if needed."""
    data = dict(skill_data)

    fingerprint = data.get("bottleneck_fingerprint")
    if not isinstance(fingerprint, BottleneckFingerprint):
        if not isinstance(fingerprint, dict):
            raise ValueError("skill_data['bottleneck_fingerprint'] must be a dict")
        fingerprint = BottleneckFingerprint(**fingerprint)
    data["bottleneck_fingerprint"] = fingerprint

    embedding = data.get("embedding")
    if not embedding:
        # The retrieval key is the bottleneck text, so the write and the query must
        # embed the SAME string (spec 6.4).
        data["embedding"] = embed_768(fingerprint.to_embedding_text())

    return SkillRecord(**data)


#: Registered on the Supervisor (spec 4.2), called once the RefinementLoop returns.
upsert_tool = FunctionTool(upsert_skill)
