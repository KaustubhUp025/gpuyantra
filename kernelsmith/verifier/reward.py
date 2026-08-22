"""CUDA Agent milestone reward, gated on all-seed correctness (spec 5.5).

    -1  incorrect (or rejected by the static checker / sandbox)
    +1  correct, not meaningfully faster
    +2  beats eager by more than SPEEDUP_THRESHOLD
    +3  beats BOTH eager and torch.compile by more than SPEEDUP_THRESHOLD

Correctness is a gate, not a term: no amount of speedup can lift a wrong kernel above
-1. That is the whole point of the milestone shape.
"""

from kernelsmith.config import SPEEDUP_THRESHOLD

REWARD_INCORRECT = -1
REWARD_CORRECT = 1
REWARD_BEATS_EAGER = 2
REWARD_BEATS_COMPILE = 3


def compute_reward(
    correctness_pass: bool,
    speedup_vs_eager: float,
    speedup_vs_compile: float,
) -> int:
    """Milestone reward in {-1, +1, +2, +3}.

    The threshold comparison is strictly greater than `1.0 + SPEEDUP_THRESHOLD`, so a
    speedup sitting exactly at the 5% line scores +1, not +2. Measurement noise must
    never be rewarded.
    """
    if not correctness_pass:
        return REWARD_INCORRECT
    if speedup_vs_eager <= 1.0:
        return REWARD_CORRECT
    if speedup_vs_eager > 1.0 + SPEEDUP_THRESHOLD:
        if speedup_vs_compile > 1.0 + SPEEDUP_THRESHOLD:
            return REWARD_BEATS_COMPILE
        return REWARD_BEATS_EAGER
    return REWARD_CORRECT
