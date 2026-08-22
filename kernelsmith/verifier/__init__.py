"""The verifier: the trust anchor (spec 5).

Import order note — `correctness` and `timing` pull in torch/triton, so they are NOT
re-exported here. Import them directly from the sandbox script that needs them; the
static checker and reward logic stay importable with no GPU stack present.
"""

from kernelsmith.verifier.reward import compute_reward
from kernelsmith.verifier.static_checker import check_static

__all__ = ["check_static", "compute_reward"]
