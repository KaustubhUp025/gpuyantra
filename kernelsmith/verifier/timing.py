"""Honest timing: do_bench with adequate warmup + the two baselines (spec 5.2).

Red line #9: warmup >= 150. Triton's default warmup=25 underestimates by ~30%
(Triton issue #2306), which would inflate every speedup we report.

Red line #10: never torch.compile before monkey-patching. `measure_baselines` compiles
the REFERENCE op, so callers must measure baselines before any hot-swap is applied.
"""

from collections.abc import Callable
from typing import Any

import torch
import triton.testing

from kernelsmith.config import DO_BENCH_REP, DO_BENCH_WARMUP


def bench_kernel(
    fn: Callable[[], Any],
    warmup: int = DO_BENCH_WARMUP,
    rep: int = DO_BENCH_REP,
) -> float:
    """Median wall time of `fn` in milliseconds.

    `fn` takes no arguments — bind the input with a lambda or functools.partial.
    """
    if warmup < DO_BENCH_WARMUP:
        raise ValueError(
            f"warmup={warmup} is below the required minimum {DO_BENCH_WARMUP}: "
            "short warmup underestimates latency by ~30% (Triton issue #2306)"
        )
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median"))


def measure_baselines(
    reference_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
) -> dict[str, float]:
    """Time the two honest baselines for `reference_fn` at input `x`.

    1. Eager + TF32 (`matmul_precision='high'`) — the KernelBench-Verified baseline.
       Timing against TF32-off eager would hand us a free ~2x on any matmul.
    2. torch.compile(mode="reduce-overhead") — the bar for the +3 milestone.

    Returns {"eager_ms": float, "compile_ms": float}.
    """
    torch.set_float32_matmul_precision("high")
    eager_ms = bench_kernel(lambda: reference_fn(x))

    compiled = torch.compile(reference_fn, mode="reduce-overhead")
    # Warm the compile cache outside the timed region: the first call pays
    # dynamo + inductor compilation, and CUDA graphs need a capture run.
    for _ in range(3):
        compiled(x)
    torch.cuda.synchronize()
    compile_ms = bench_kernel(lambda: compiled(x))

    return {"eager_ms": eager_ms, "compile_ms": compile_ms}


def compute_speedups(
    eager_ms: float,
    compile_ms: float,
    candidate_ms: float,
) -> dict[str, float]:
    """Speedup ratios, >1.0 meaning the candidate is faster than the baseline.

    A non-positive candidate time is not a win, it is a broken measurement (a kernel
    that never ran, or a timer fooled by an extra stream), so it scores 0.0.
    """
    if candidate_ms <= 0:
        return {"speedup_vs_eager": 0.0, "speedup_vs_compile": 0.0}
    return {
        "speedup_vs_eager": eager_ms / candidate_ms,
        "speedup_vs_compile": compile_ms / candidate_ms,
    }
