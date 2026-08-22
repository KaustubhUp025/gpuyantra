"""Correctness gate: 5 seeds x 3 shapes, torch.allclose(atol=rtol=1e-2) (spec 5.1).

Every one of the 15 checks must pass. A single failure means reward = -1, no timing,
no hot-swap. This is the trust anchor of the whole system.

Red line #3: never weaken this file — not fewer seeds, not looser tolerance, not a
dropped NaN/Inf or shape/dtype guard.

Red line #2: this module is imported by the sandbox script, which runs in a separate
subprocess. Never call it with generated code in the main process.
"""

from collections.abc import Callable
from typing import Any

import torch

from kernelsmith.config import ATOL, CORRECTNESS_SEEDS, CORRECTNESS_SHAPES, RTOL

TensorFn = Callable[[torch.Tensor], torch.Tensor]


def check_correctness(
    reference_fn: TensorFn,
    candidate_fn: TensorFn,
    hidden_size: int,
    device: str = "cuda",
) -> dict[str, Any]:
    """Compare a candidate forward against the reference over every seed x shape.

    Args:
        reference_fn: The torch reference implementation, `x -> y`.
        candidate_fn: The generated kernel wrapper, same signature.
        hidden_size: Model hidden dim; the third axis of the generated input.
        device: Device to allocate inputs on. "cuda" in production.

    Returns:
        {
          "correctness_pass": bool,   # True only if all 15 checks passed
          "total_checks": int,        # CORRECTNESS_SEEDS * len(CORRECTNESS_SHAPES)
          "passed_checks": int,
          "failed_cases": [ {"seed": int, "shape": [b, s], "reason": str}, ... ],
        }
    """
    failed_cases: list[dict[str, Any]] = []
    total_checks = CORRECTNESS_SEEDS * len(CORRECTNESS_SHAPES)

    for seed in range(CORRECTNESS_SEEDS):
        torch.manual_seed(seed)
        for batch, seq_len in CORRECTNESS_SHAPES:
            x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=torch.float16)
            reason = _check_one(reference_fn, candidate_fn, x)
            if reason is not None:
                failed_cases.append({"seed": seed, "shape": [batch, seq_len], "reason": reason})

    return {
        "correctness_pass": not failed_cases,
        "total_checks": total_checks,
        "passed_checks": total_checks - len(failed_cases),
        "failed_cases": failed_cases,
    }


def _check_one(reference_fn: TensorFn, candidate_fn: TensorFn, x: torch.Tensor) -> str | None:
    """Run one comparison. Returns None if it passed, else why it failed.

    A candidate that raises is a failure, not a crash: the caller needs all 15
    verdicts, and an exception here would hide the other 14.
    """
    try:
        ref_out = reference_fn(x)
    except Exception as exc:  # noqa: BLE001 — a broken reference is a hard failure too
        return f"reference raised {type(exc).__name__}: {exc}"

    try:
        cand_out = candidate_fn(x)
    except Exception as exc:  # noqa: BLE001 — any candidate exception is just a failed check
        return f"candidate raised {type(exc).__name__}: {exc}"

    if not isinstance(cand_out, torch.Tensor):
        return f"candidate returned {type(cand_out).__name__}, expected torch.Tensor"

    # Guard: NaN/Inf. Sakana's exploits routinely produce non-finite output that
    # still slips through a naive allclose on a subset of elements.
    if not torch.isfinite(ref_out).all():
        return "reference produced NaN/Inf"
    if not torch.isfinite(cand_out).all():
        return "candidate produced NaN/Inf"

    # Guard: shape/dtype. An upcast to fp32 would beat the tolerance dishonestly.
    if cand_out.shape != ref_out.shape:
        return (
            f"shape mismatch: candidate {tuple(cand_out.shape)} vs reference {tuple(ref_out.shape)}"
        )
    if cand_out.dtype != ref_out.dtype:
        return f"dtype mismatch: candidate {cand_out.dtype} vs reference {ref_out.dtype}"

    # Tolerance check (CUDA Agent uses exactly atol=rtol=1e-2).
    if not torch.allclose(cand_out, ref_out, atol=ATOL, rtol=RTOL):
        max_abs = (cand_out.float() - ref_out.float()).abs().max().item()
        return f"allclose failed at atol={ATOL} rtol={RTOL} (max abs diff {max_abs:.6g})"

    return None
