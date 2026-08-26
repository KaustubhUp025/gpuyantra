"""Bottleneck-indexed skill retrieval (spec 6.4). The core novelty.

Skills are retrieved by WHY an op is slow — the roofline fingerprint text — not by the
op's name. A skill learned on RMSNorm surfaces for RoPE when both are memory-bound
row-wise reductions with the same tile hint, which is exactly the transfer a
name-keyed cache can never make.

Firestore vector search constraints that shape this file:
  - Pre-filters must be EQUALITY only, and their fields must precede the vector field
    in the composite index (spec 6.2). `op_family` and `hardware` are those fields.
  - COSINE distance requires unit vectors on both sides; `embed_768` L2-normalizes.

Retrieval answers "which skills are relevant"; the UCB1 bandit below answers "which of
them do we actually start from" (spec 9). Nearest-by-distance alone would lock the
system onto whichever kernel happened to be seeded first: a skill that has never been
tried has no evidence against it, only no evidence for it. UCB1 pulls unpulled arms
first and thereafter trades the arm's mean reward off against how little we know about
it, and `update_bandit_stats` feeds the verifier's reward back so the trade-off is made
on measured outcomes rather than on the Coder's opinion of them.
"""

from __future__ import annotations

import math
from typing import Any

from google.adk.tools import FunctionTool
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

from kernelsmith.config import RETRIEVAL_TOP_K
from kernelsmith.memory.embeddings import embed_768
from kernelsmith.memory.firestore_store import skills_collection
from kernelsmith.memory.firestore_store import update_bandit_stats as _store_update_bandit_stats

#: sqrt(2), the standard UCB1 exploration constant. Higher explores more.
UCB1_C = 1.41

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


# --------------------------------------------------------------------------- #
# UCB1 bandit over skills (spec 9)
# --------------------------------------------------------------------------- #


def total_bandit_pulls(skills: list[dict[str, Any]]) -> int:
    """Total pulls across the candidate arms — the `t` in UCB1's ln(t) term."""
    return sum(max(0, _as_int(skill.get("bandit_pulls"))) for skill in skills)


def ucb1_select(
    skills: list[dict[str, Any]],
    total_pulls: int,
    c: float = UCB1_C,
) -> dict[str, Any] | None:
    """Pick one skill by UCB1: `mean_reward + c * sqrt(ln(total_pulls) / n_pulls)`.

    Optimism under uncertainty. An arm with zero pulls has an unbounded exploration
    term, so it is returned immediately rather than scored — that is what stops the
    library from ossifying around whichever kernel was seeded first.

    Args:
        skills: Candidate skill dicts, each optionally carrying `bandit_pulls` and
            `bandit_total_reward` (both default to 0 — an unpulled arm).
        total_pulls: Pulls across ALL arms, i.e. `total_bandit_pulls(skills)`. Values
            below 1 are clamped up, since ln(0) is undefined and ln(1) = 0 correctly
            says "no evidence yet, exploration adds nothing".
        c: Exploration constant. sqrt(2) ~= 1.41 is the textbook value.

    Returns:
        The selected skill dict, or None when `skills` is empty — a cold library is a
        normal state, not an error.
    """
    if not skills:
        return None

    t = max(1, int(total_pulls))
    log_t = math.log(t)

    best_ucb = -float("inf")
    best_skill: dict[str, Any] | None = None
    for skill in skills:
        n = _as_int(skill.get("bandit_pulls"))
        if n <= 0:
            return skill  # Explore unpulled arms first.
        mean = _as_float(skill.get("bandit_total_reward")) / n
        ucb = mean + c * math.sqrt(log_t / n)
        if ucb > best_ucb:
            best_ucb = ucb
            best_skill = skill
    return best_skill


def update_bandit_stats(skill_id: str, reward: int, db: Any = None) -> dict[str, Any]:
    """Credit one pull of `skill_id` with the reward the VERIFIER measured.

    Two Firestore `Increment`s in one update, so concurrent runs cannot lose a pull to
    a read-modify-write race. The reward must be the verifier's, never a model's
    self-report (red line #3): the bandit is a memory of what was measured.

    Returns:
        {"updated": bool, "skill_id": str, "reward": int} — plus "error" when the write
        failed. Never raises: losing a bandit update must not kill a run whose kernel
        is already verified.
    """
    if not skill_id:
        return {
            "updated": False,
            "skill_id": skill_id,
            "reward": int(reward),
            "error": "no skill_id",
        }
    try:
        _store_update_bandit_stats(skill_id, float(reward), db=db)
    except Exception as exc:  # noqa: BLE001 — a Firestore outage is not a run failure
        return {
            "updated": False,
            "skill_id": skill_id,
            "reward": int(reward),
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"updated": True, "skill_id": skill_id, "reward": int(reward)}


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if result != result else result  # NaN -> 0.0


# --------------------------------------------------------------------------- #
# ADK tool surface
# --------------------------------------------------------------------------- #


def retrieve_skills_for_agent(
    op_family: str,
    hardware: str,
    fingerprint_text: str,
    k: int = RETRIEVAL_TOP_K,
) -> dict[str, Any]:
    """Retrieve prior winning kernels for a bottleneck like this one.

    Searches the learned skill library by bottleneck fingerprint, not by op name, so a
    kernel that worked on a similar bottleneck (same boundedness, same tile shape) is
    returned even if it was learned on a different operation. A UCB1 bandit then picks
    ONE of them as the arm to pull — start from `selected_skill`, and use the rest only
    as context. Use the returned `fix_rule` and `winning_kernel_source` as a starting
    point, never verbatim.

    Args:
        op_family: "norm", "rope", "mlp", "elementwise", or "reduction".
        hardware: GPU the skill was learned on. Always "L4" here.
        fingerprint_text: The bottleneck fingerprint text from the profiler.
        k: How many prior skills to return (default 3).

    Returns:
        {"skills": [...], "count": int, "selected_skill_id": str,
        "selected_skill": {...} | None}, with the bandit's pick first in `skills`. An
        empty list means nothing has been learned for this family yet — write the
        kernel from scratch.
    """
    try:
        skills = retrieve_skills(op_family, hardware, fingerprint_text, k)
    except Exception as exc:  # noqa: BLE001 — a cold or unreachable library is not fatal
        return {
            "skills": [],
            "count": 0,
            "selected_skill_id": "",
            "selected_skill": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    selected = ucb1_select(skills, total_bandit_pulls(skills))
    if selected is not None:
        # The Coder reads the list in order and the bandit's pick is the arm being
        # pulled, so it leads. The others stay as context, nearest-bottleneck first.
        skills = [selected] + [s for s in skills if s is not selected]

    return {
        "skills": skills,
        "count": len(skills),
        "selected_skill_id": str((selected or {}).get("skill_id", "")),
        "selected_skill": selected,
    }


#: Registered on the Supervisor (spec 4.2), which writes the result to
#: `session.state["retrieved_skills"]` for the Coder.
retrieval_tool = FunctionTool(retrieve_skills_for_agent)
