"""Firestore client, collection refs, CRUD, and composite-vector-index bootstrap.

Layout (spec 6.1):
  skills/{skill_id}                  -> SkillRecord, `embedding` stored as Vector(768)
  runs/{run_id}                      -> RunRecord
  runs/{run_id}/traces/{auto_id}     -> TraceRecord

Auth is ADC only. Never a service-account key file (red line #5).
"""

from __future__ import annotations

import shlex
import subprocess
from typing import Any

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector

from kernelsmith.config import (
    EMBEDDING_DIM,
    FIRESTORE_COLLECTION_RUNS,
    FIRESTORE_COLLECTION_SKILLS,
    FIRESTORE_DATABASE,
    FIRESTORE_SUBCOLLECTION_TRACES,
    GCP_PROJECT,
)
from kernelsmith.memory.schemas import RunRecord, SkillRecord, TraceRecord

_db: firestore.Client | None = None


def get_db() -> firestore.Client:
    """Process-wide Firestore client (ADC)."""
    global _db
    if _db is None:
        _db = firestore.Client(project=GCP_PROJECT, database=FIRESTORE_DATABASE)
    return _db


def skills_collection(db: firestore.Client | None = None):
    return (db or get_db()).collection(FIRESTORE_COLLECTION_SKILLS)


def runs_collection(db: firestore.Client | None = None):
    return (db or get_db()).collection(FIRESTORE_COLLECTION_RUNS)


def traces_collection(run_id: str, db: firestore.Client | None = None):
    return runs_collection(db).document(run_id).collection(FIRESTORE_SUBCOLLECTION_TRACES)


# --------------------------------------------------------------------------- #
# Serialization helpers
# --------------------------------------------------------------------------- #


def skill_to_document(rec: SkillRecord) -> dict[str, Any]:
    """SkillRecord -> Firestore doc. The embedding MUST become a Vector for find_nearest."""
    assert len(rec.embedding) == EMBEDDING_DIM, (
        f"Embedding must be {EMBEDDING_DIM}-dim, got {len(rec.embedding)}"
    )
    doc = rec.model_dump()
    doc["embedding"] = Vector(rec.embedding)
    return doc


def document_to_skill(doc: dict[str, Any]) -> SkillRecord:
    """Firestore doc -> SkillRecord. Vector -> plain list of floats."""
    data = dict(doc)
    embedding = data.get("embedding")
    if isinstance(embedding, Vector) or embedding is not None:
        data["embedding"] = list(embedding)
    data.pop("vector_distance", None)  # injected by find_nearest, not part of the schema
    return SkillRecord(**data)


# --------------------------------------------------------------------------- #
# Skills CRUD
# --------------------------------------------------------------------------- #


def put_skill(rec: SkillRecord, db: firestore.Client | None = None) -> str:
    """Unconditional write of a skill (doc id = skill_id). Returns the skill_id."""
    skills_collection(db).document(rec.skill_id).set(skill_to_document(rec))
    return rec.skill_id


def get_skill(skill_id: str, db: firestore.Client | None = None) -> SkillRecord | None:
    snap = skills_collection(db).document(skill_id).get()
    if not snap.exists:
        return None
    return document_to_skill(snap.to_dict())


def upsert_skill(rec: SkillRecord, db: firestore.Client | None = None) -> str:
    """Dedupe by skill_id: keep whichever version has the higher speedup_vs_eager.

    Returns "upserted" or "kept_existing".
    """
    ref = skills_collection(db).document(rec.skill_id)
    snap = ref.get()
    if snap.exists:
        existing = snap.to_dict() or {}
        if float(existing.get("speedup_vs_eager", 0.0)) >= rec.speedup_vs_eager:
            return "kept_existing"
    ref.set(skill_to_document(rec))
    return "upserted"


def delete_skill(skill_id: str, db: firestore.Client | None = None) -> None:
    skills_collection(db).document(skill_id).delete()


def list_skills(
    op_family: str | None = None,
    hardware: str | None = None,
    limit: int = 50,
    db: firestore.Client | None = None,
) -> list[SkillRecord]:
    """Equality pre-filters only — Firestore vector search forbids inequality filters."""
    query = skills_collection(db)
    if op_family is not None:
        query = query.where("op_family", "==", op_family)
    if hardware is not None:
        query = query.where("hardware", "==", hardware)
    return [document_to_skill(doc.to_dict()) for doc in query.limit(limit).stream()]


def update_bandit_stats(skill_id: str, reward: float, db: firestore.Client | None = None) -> None:
    """Record one bandit pull for a skill (used by the skill-selection bandit, spec 9)."""
    skills_collection(db).document(skill_id).update(
        {
            "bandit_pulls": firestore.Increment(1),
            "bandit_total_reward": firestore.Increment(float(reward)),
        }
    )


# --------------------------------------------------------------------------- #
# Runs + traces CRUD
# --------------------------------------------------------------------------- #


def put_run(rec: RunRecord, db: firestore.Client | None = None) -> str:
    runs_collection(db).document(rec.run_id).set(rec.model_dump())
    return rec.run_id


def get_run(run_id: str, db: firestore.Client | None = None) -> RunRecord | None:
    snap = runs_collection(db).document(run_id).get()
    if not snap.exists:
        return None
    return RunRecord(**snap.to_dict())


def update_run(run_id: str, fields: dict[str, Any], db: firestore.Client | None = None) -> None:
    runs_collection(db).document(run_id).update(fields)


def list_runs(limit: int = 20, db: firestore.Client | None = None) -> list[RunRecord]:
    """Most recent runs first — the dashboard's run-history table (spec 10.1)."""
    query = (
        runs_collection(db)
        .order_by("started_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )
    return [RunRecord(**doc.to_dict()) for doc in query.stream()]


def append_trace(run_id: str, rec: TraceRecord, db: firestore.Client | None = None) -> str:
    _, ref = traces_collection(run_id, db).add(rec.model_dump())
    return ref.id


def list_traces(run_id: str, db: firestore.Client | None = None) -> list[TraceRecord]:
    docs = traces_collection(run_id, db).order_by("iteration").stream()
    return [TraceRecord(**doc.to_dict()) for doc in docs]


# --------------------------------------------------------------------------- #
# Index bootstrap (spec 6.2) — run ONCE, never on demo day
# --------------------------------------------------------------------------- #

INDEX_COMMAND: list[str] = [
    "gcloud",
    "firestore",
    "indexes",
    "composite",
    "create",
    f"--project={GCP_PROJECT}",
    f"--collection-group={FIRESTORE_COLLECTION_SKILLS}",
    "--query-scope=COLLECTION",
    "--field-config=field-path=op_family,order=ASCENDING",
    "--field-config=field-path=hardware,order=ASCENDING",
    f'--field-config=vector-config={{"dimension":"{EMBEDDING_DIM}","flat":"{{}}"}},field-path=embedding',
    # No shell quoting here: subprocess runs argv directly, so "(default)" needs no escaping.
    f"--database={FIRESTORE_DATABASE}",
]


def index_command_str() -> str:
    """The exact gcloud command that creates the composite vector index."""
    return " ".join(shlex.quote(part) for part in INDEX_COMMAND)


def bootstrap_index(dry_run: bool = True) -> str:
    """Create the composite vector index for `skills`.

    The equality pre-filter fields (op_family, hardware) MUST precede the vector field.
    Index builds take minutes — call this once at setup, never during a demo.

    Returns the command output (or the command itself when dry_run=True).
    """
    if dry_run:
        return index_command_str()
    proc = subprocess.run(INDEX_COMMAND, capture_output=True, text=True)
    if proc.returncode != 0 and "already exists" not in (proc.stderr or ""):
        raise RuntimeError(f"Index creation failed:\n{proc.stderr}")
    return proc.stdout or proc.stderr
