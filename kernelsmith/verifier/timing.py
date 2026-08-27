"""Honest timing: do_bench with adequate warmup + the two baselines (spec 5.2).

Red line #9: warmup >= 150. Triton's default warmup=25 underestimates by ~30%
(Triton issue #2306), which would inflate every speedup we report.

Red line #10: never torch.compile before monkey-patching. `measure_baselines` compiles
the REFERENCE op, so callers must measure baselines before any hot-swap is applied.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
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


@contextmanager
def _nondeterministic_for_timing() -> Iterator[None]:
    """Turn deterministic algorithm selection off for the duration of a timed baseline.

    `torch.use_deterministic_algorithms(True)` forces slower cuBLAS/cuDNN codepaths and
    costs the eager and torch.compile baselines ~23% — while leaving a Triton candidate
    untouched, because Triton generates its own PTX and never consults the flag. Timing
    the baselines under it therefore manufactures a speedup out of a measurement
    artifact: 8.52x vs eager with the flag on, 6.9x with it off.

    The flag stays ON everywhere else — correctness checks, the agent loop, the demo.
    Only the timed comparison runs without it, and the previous setting (including
    `warn_only`) is restored even if benchmarking raises.
    """
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)


def measure_baselines(
    reference_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
) -> dict[str, float]:
    """Time the two honest baselines for `reference_fn` at input `x`.

    1. Eager + TF32 (`matmul_precision='high'`) — the KernelBench-Verified baseline.
       Timing against TF32-off eager would hand us a free ~2x on any matmul.
    2. torch.compile(mode="reduce-overhead") — the bar for the +3 milestone.

    Both are timed with deterministic algorithms OFF, so the baselines are not
    handicapped by a flag the Triton candidate never pays. See
    `_nondeterministic_for_timing`.

    Returns {"eager_ms": float, "compile_ms": float}.
    """
    torch.set_float32_matmul_precision("high")

    with _nondeterministic_for_timing():
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
