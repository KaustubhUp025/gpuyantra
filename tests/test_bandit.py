"""UCB1 skill selection and its Firestore write-back (spec 9 / 13.1).

Two things are being protected here. First, that an arm nobody has tried is always
tried: a library that only ever pulls its best-known kernel stops learning, and the
seed skill's warm start (3 pulls at mean 3.0) is strong enough to shut out every later
skill if exploration is broken. Second, that the credit written back is an INCREMENT
rather than a read-modify-write — two runs finishing at once must not lose a pull.

Firestore is faked. What is asserted is the shape of the update, because a plain
`{"bandit_pulls": 4}` would pass any test that only checked the resulting number in a
single-threaded fake.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from google.cloud import firestore

from kernelsmith.tools.retrieval_tool import (
    UCB1_C,
    total_bandit_pulls,
    ucb1_select,
    update_bandit_stats,
)


def arm(skill_id: str, pulls: int, total_reward: float) -> dict[str, Any]:
    return {
        "skill_id": skill_id,
        "bandit_pulls": pulls,
        "bandit_total_reward": total_reward,
        "winning_kernel_source": "# ...",
    }


# --------------------------------------------------------------------------- ucb1


def test_all_arms_unpulled_returns_the_first():
    """Nothing is known about any of them, so the scan returns on the first arm."""
    arms = [arm("a", 0, 0.0), arm("b", 0, 0.0), arm("c", 0, 0.0)]
    assert ucb1_select(arms, total_pulls=0) is arms[0]


def test_unpulled_arm_beats_a_high_mean_arm():
    """Optimism under uncertainty: an untried kernel outranks a proven one."""
    arms = [arm("proven", 20, 60.0), arm("untried", 0, 0.0)]
    assert ucb1_select(arms, total_bandit_pulls(arms))["skill_id"] == "untried"


def test_equal_pulls_selects_the_highest_mean():
    """With the exploration term identical across arms, only the mean can decide."""
    arms = [arm("low", 5, 5.0), arm("high", 5, 14.0), arm("mid", 5, 10.0)]
    assert ucb1_select(arms, total_bandit_pulls(arms))["skill_id"] == "high"


def test_exploration_bonus_shrinks_as_an_arm_is_pulled():
    """c * sqrt(ln t / n) must fall with n — otherwise UCB1 never converges."""

    def bonus(n: int, total: int) -> float:
        return UCB1_C * math.sqrt(math.log(total) / n)

    assert bonus(2, 100) > bonus(10, 100) > bonus(50, 100)


def test_a_rarely_pulled_arm_can_outrank_a_better_mean():
    """The whole point of the bonus: 2 pulls at mean 2.0 beats 200 pulls at mean 2.5."""
    arms = [arm("popular", 200, 500.0), arm("rare", 2, 4.0)]
    assert ucb1_select(arms, total_bandit_pulls(arms))["skill_id"] == "rare"

    # ... but not once the rare arm has been sampled enough to know it is worse.
    arms = [arm("popular", 200, 500.0), arm("rare", 150, 300.0)]
    assert ucb1_select(arms, total_bandit_pulls(arms))["skill_id"] == "popular"


def test_empty_library_selects_nothing():
    """A cold library is a normal state, not an error."""
    assert ucb1_select([], total_pulls=0) is None


def test_missing_bandit_fields_are_treated_as_unpulled():
    """A skill written before the bandit existed has no stats and must be explored."""
    legacy = {"skill_id": "legacy"}
    assert ucb1_select([arm("known", 5, 15.0), legacy], total_pulls=5) is legacy


def test_total_pulls_below_one_is_clamped():
    """ln(0) is undefined; a first-ever selection must still return an arm."""
    assert ucb1_select([arm("only", 1, 3.0)], total_pulls=0)["skill_id"] == "only"


def test_garbage_bandit_stats_do_not_raise():
    """State can arrive from Firestore or a model; neither is trusted to be typed."""
    arms = [{"skill_id": "bad", "bandit_pulls": "three", "bandit_total_reward": None}]
    assert ucb1_select(arms, total_pulls=3) is arms[0]  # parsed as unpulled -> explored


def test_seed_warm_start_does_not_shut_out_a_new_skill():
    """The seeded arm (3 pulls, mean 3.0) must not starve a freshly learned kernel."""
    arms = [arm("rmsnorm_fp16_l4_v1", 3, 9.0), arm("newly_learned", 0, 0.0)]
    assert ucb1_select(arms, total_bandit_pulls(arms))["skill_id"] == "newly_learned"


# ------------------------------------------------------------------ write-back


class FakeDocument:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.raises: Exception | None = None

    def update(self, fields: dict[str, Any]) -> None:
        if self.raises is not None:
            raise self.raises
        self.updates.append(fields)


class FakeCollection:
    def __init__(self) -> None:
        self.documents: dict[str, FakeDocument] = {}
        self.requested: list[str] = []

    def document(self, doc_id: str) -> FakeDocument:
        self.requested.append(doc_id)
        return self.documents.setdefault(doc_id, FakeDocument())


class FakeDb:
    def __init__(self, collection: FakeCollection) -> None:
        self._collection = collection
        self.names: list[str] = []

    def collection(self, name: str) -> FakeCollection:
        self.names.append(name)
        return self._collection


@pytest.fixture
def collection() -> FakeCollection:
    return FakeCollection()


def test_update_bandit_stats_increments_both_counters(collection: FakeCollection):
    result = update_bandit_stats("rmsnorm_fp16_l4_v1", 3, db=FakeDb(collection))

    assert result == {"updated": True, "skill_id": "rmsnorm_fp16_l4_v1", "reward": 3}
    assert collection.requested == ["rmsnorm_fp16_l4_v1"]

    (fields,) = collection.documents["rmsnorm_fp16_l4_v1"].updates
    assert set(fields) == {"bandit_pulls", "bandit_total_reward"}
    # Increments, not assignments: concurrent runs must not lose a pull.
    assert isinstance(fields["bandit_pulls"], firestore.Increment)
    assert isinstance(fields["bandit_total_reward"], firestore.Increment)
    assert fields["bandit_pulls"].value == 1
    assert fields["bandit_total_reward"].value == 3.0


def test_a_negative_reward_is_credited_as_a_pull(collection: FakeCollection):
    """A failed run is evidence too — otherwise every arm's mean only ever rises."""
    update_bandit_stats("bad_skill", -1, db=FakeDb(collection))

    (fields,) = collection.documents["bad_skill"].updates
    assert fields["bandit_pulls"].value == 1
    assert fields["bandit_total_reward"].value == -1.0


def test_a_firestore_outage_is_reported_not_raised(collection: FakeCollection):
    """The kernel is already verified; losing its bandit credit must not kill the run."""
    collection.documents["skill"] = FakeDocument()
    collection.documents["skill"].raises = RuntimeError("deadline exceeded")

    result = update_bandit_stats("skill", 2, db=FakeDb(collection))

    assert result["updated"] is False
    assert "RuntimeError: deadline exceeded" in result["error"]


def test_an_empty_skill_id_writes_nothing(collection: FakeCollection):
    """A cold library leaves `selected_skill_id` empty; that is not a document path."""
    result = update_bandit_stats("", 3, db=FakeDb(collection))

    assert result["updated"] is False
    assert collection.requested == []


# ------------------------------------------------- selection inside the tool


def test_the_tool_puts_the_bandit_pick_first(monkeypatch):
    """The Coder reads the skill list in order, so the arm being pulled must lead it."""
    from kernelsmith.tools import retrieval_tool as retrieval

    nearest, unpulled = arm("nearest", 10, 20.0), arm("unpulled", 0, 0.0)
    monkeypatch.setattr(retrieval, "retrieve_skills", lambda *a, **k: [nearest, unpulled])

    payload = retrieval.retrieve_skills_for_agent("norm", "L4", "op=norm mem_bound=True")

    assert payload["selected_skill_id"] == "unpulled"
    assert [s["skill_id"] for s in payload["skills"]] == ["unpulled", "nearest"]
    # Nothing is dropped: the runners-up stay as context, nearest-bottleneck first.
    assert payload["count"] == 2


# ----------------------------------------------------- credit at end of a run


async def _escalate(state: dict[str, Any]) -> Any:
    """Run the EscalationChecker over `state` and return its single event."""
    from types import SimpleNamespace

    from kernelsmith.agents.escalation import build_escalation_checker

    ctx = SimpleNamespace(
        state=state,
        session=SimpleNamespace(state=state, events=[]),
        invocation_id="inv-1",
    )
    events = [event async for event in build_escalation_checker()._run_async_impl(ctx)]
    assert len(events) == 1
    return events[0]


@pytest.fixture
def recorded_credits(monkeypatch) -> list[tuple[str, int]]:
    """Capture bandit credits instead of writing them to Firestore."""
    credits: list[tuple[str, int]] = []

    def fake_update(skill_id: str, reward: int) -> dict[str, Any]:
        credits.append((skill_id, reward))
        return {"updated": True, "skill_id": skill_id, "reward": reward}

    from kernelsmith.agents import escalation

    monkeypatch.setattr(escalation, "update_bandit_stats", fake_update)
    return credits


@pytest.mark.asyncio
async def test_the_pulled_arm_is_credited_once_the_loop_finishes(recorded_credits):
    """One run is one pull, credited with the run's BEST reward, not the last one."""
    event = await _escalate(
        {
            "verdict": {"reward": 1, "stop": True},
            "best_reward": 3,
            "selected_skill_id": "rmsnorm_fp16_l4_v1",
            "iteration": 4,
        }
    )

    assert recorded_credits == [("rmsnorm_fp16_l4_v1", 3)]
    assert event.actions.escalate is True
    assert event.actions.state_delta == {"bandit_credited": True}


@pytest.mark.asyncio
async def test_no_credit_while_the_loop_is_still_running(recorded_credits):
    """Crediting per iteration would count six pulls for one experiment."""
    event = await _escalate(
        {
            "verdict": {"reward": -1, "stop": False},
            "best_reward": -1,
            "selected_skill_id": "rmsnorm_fp16_l4_v1",
            "iteration": 1,
        }
    )

    assert recorded_credits == []
    assert event.actions.escalate is False


@pytest.mark.asyncio
async def test_an_already_credited_run_is_not_credited_twice(recorded_credits):
    """The checker can run again on a resumed invocation; the pull must not double."""
    await _escalate(
        {
            "verdict": {"reward": 3, "stop": True},
            "best_reward": 3,
            "selected_skill_id": "rmsnorm_fp16_l4_v1",
            "bandit_credited": True,
        }
    )

    assert recorded_credits == []


@pytest.mark.asyncio
async def test_a_cold_library_credits_nothing(recorded_credits):
    """Nothing was retrieved, so no arm was pulled — there is nothing to credit."""
    event = await _escalate({"verdict": {"reward": 3, "stop": True}, "best_reward": 3})

    assert recorded_credits == []
    assert event.actions.escalate is True


@pytest.mark.asyncio
async def test_a_failed_credit_still_marks_the_run_as_credited(recorded_credits, monkeypatch):
    """Retrying could double-count a write that actually landed; one attempt, then stop."""
    from kernelsmith.agents import escalation

    monkeypatch.setattr(
        escalation,
        "update_bandit_stats",
        lambda skill_id, reward: {"updated": False, "error": "503"},
    )

    event = await _escalate(
        {"verdict": {"reward": 3, "stop": True}, "best_reward": 3, "selected_skill_id": "s"}
    )

    assert event.actions.escalate is True
    assert event.actions.state_delta == {"bandit_credited": True}
