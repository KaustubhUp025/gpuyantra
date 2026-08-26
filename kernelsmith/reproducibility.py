"""The reproducibility contract, in one callable (spec 11).

The field's credibility problem is the reason this file exists: Sakana's 3.13x became
1.49x and KernelBench-Verified's 1.43x became 0.88x once anyone re-ran them. Both
collapses are seeding and baseline artifacts, not fraud. A speedup that only appears
under one lucky seed is not a speedup, so every entry point — `run_demo`, the sandbox
runner, and the tests — starts from the same fixed state.

Order matters here and is not cosmetic:

1. `CUBLAS_WORKSPACE_CONFIG` is read by cuBLAS when its handle is first created. Set
   after that point it is ignored, and `use_deterministic_algorithms(True)` then throws
   on the first GEMM. It is therefore set FIRST, before anything touches CUDA.
2. `use_deterministic_algorithms(True)` comes last, once the seeds are in place.

One deliberate exception, from `.claude/rules/implementation-deviations.md`: the
deterministic flag penalizes eager/torch.compile baselines by ~23% while leaving Triton
untouched, so `measure_baselines()` turns it off around the timed comparison only. The
flag stays on everywhere else — correctness checks, the agent loop, the demo.
"""

from __future__ import annotations

import os
import random
from typing import Any

from kernelsmith import config


def seed_everything(seed: int = config.GLOBAL_SEED) -> dict[str, Any]:
    """Pin every source of randomness this system has. Call before anything else.

    Sets the Python, NumPy and torch (CPU + all CUDA devices) RNGs, exports
    `CUBLAS_WORKSPACE_CONFIG`, and enables deterministic algorithm selection.

    Args:
        seed: The seed to use everywhere. Defaults to `config.GLOBAL_SEED`.

    Returns:
        What was actually applied — seed, the workspace config, whether deterministic
        algorithms are on, and how many CUDA devices were seeded. Useful to log at the
        top of a run so a recorded demo says what state it ran under.
    """
    # 1. cuBLAS workspace FIRST: read at handle creation, ignored afterwards.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.CUBLAS_WORKSPACE

    random.seed(seed)

    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op without CUDA; never raises

    # TF32 on for honest baselines (KernelBench-Verified): an un-TF32'd eager baseline
    # manufactures a speedup out of nothing.
    torch.set_float32_matmul_precision("high")

    if config.DETERMINISTIC_CUDA:
        torch.use_deterministic_algorithms(True)

    return {
        "seed": seed,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": bool(config.DETERMINISTIC_CUDA),
        "cuda_devices_seeded": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
