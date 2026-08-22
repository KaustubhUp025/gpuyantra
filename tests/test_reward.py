"""Milestone reward boundary cases (spec 5.5 / 13.1).

The threshold is a strict `>`, so 1.05 exactly (with SPEEDUP_THRESHOLD=0.05) is NOT a
win. Noise must never be rewarded, and the boundary must not drift.
"""

import math

import pytest

from kernelsmith.config import SPEEDUP_THRESHOLD
from kernelsmith.verifier.reward import compute_reward


def test_threshold_is_five_percent():
    """Red line #3: the gate does not move without an explicit spec change."""
    assert SPEEDUP_THRESHOLD == 0.05


@pytest.mark.parametrize(
    ("eager", "compile_"),
    [(0.5, 0.5), (1.0, 1.0), (1.06, 1.06), (374.0, 374.0), (math.nan, math.nan)],
)
def test_incorrect_is_always_minus_one(eager, compile_):
    """Correctness is a gate, not a term: no speedup can buy a wrong kernel out of -1."""
    assert compute_reward(False, eager, compile_) == -1


def test_correct_but_slower_is_plus_one():
    assert compute_reward(True, 0.95, 0.90) == 1


def test_correct_and_exactly_break_even_is_plus_one():
    assert compute_reward(True, 1.0, 1.0) == 1


def test_beats_eager_only_is_plus_two():
    assert compute_reward(True, 1.06, 0.99) == 2


def test_beats_both_is_plus_three():
    assert compute_reward(True, 1.06, 1.06) == 3


def test_below_threshold_is_plus_one():
    """1.04 is a 4% gain — inside measurement noise, so not a milestone."""
    assert compute_reward(True, 1.04, 1.04) == 1


def test_exactly_at_threshold_is_plus_one():
    """The boundary case: 1.05 is not `> 1.05`, so it scores +1, not +2."""
    assert compute_reward(True, 1.05, 1.05) == 1


def test_just_over_threshold_is_a_milestone():
    """The first value strictly past the gate must clear it."""
    just_over = 1.0 + SPEEDUP_THRESHOLD + 1e-9
    assert compute_reward(True, just_over, 1.0) == 2
    assert compute_reward(True, just_over, just_over) == 3


def test_beating_compile_without_beating_eager_is_not_plus_three():
    """+3 requires clearing BOTH bars; compile alone cannot promote a marginal kernel."""
    assert compute_reward(True, 1.02, 2.0) == 1


def test_zero_speedup_is_plus_one():
    """A candidate time that could not be measured scores 0.0 speedup (timing.py)."""
    assert compute_reward(True, 0.0, 0.0) == 1


@pytest.mark.parametrize("reward_args", [(True, 1.06, 1.06), (True, 1.06, 0.9), (True, 0.9, 0.9)])
def test_reward_is_an_int_in_the_milestone_set(reward_args):
    reward = compute_reward(*reward_args)
    assert isinstance(reward, int)
    assert reward in {-1, 1, 2, 3}
