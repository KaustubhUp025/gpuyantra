"""The live inference server and its hot-swap endpoint (spec 8.1).

This is the process the demo points at: a real Qwen2.5-1.5B answering real requests,
whose RMSNorm gets replaced under it, mid-flight, by a kernel the agent tree wrote.

Three things make that safe rather than reckless:

1. **One lock.** `/generate` and `/swap` share `_SWAP_LOCK`, so a patch can never land
   between two decode steps of an in-flight request.
2. **A parity gate.** After patching, and before the swap is reported as successful,
   the new forward is compared against the saved original over 5 seeds at
   atol=rtol=1e-2. A single mismatch auto-rolls-back and the swap is refused. This is
   spec 13.4: no op is served to a user until a torch reference agrees with it.
3. **The static checker.** The same AST rules the verifier uses run again here, so a
   kernel that never went through the verifier cannot reach `exec` by calling `/swap`
   directly.

On red line #2: executing kernel source in this process is the point of a hot-swap —
the ban is on the VERIFIER running unvetted code, which still happens only in the
sandbox subprocess. What lands here has already been sandboxed, scored and re-checked.

The model is never `torch.compile`d: patching a compiled graph silently no-ops.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from kernelsmith.config import (
    ATOL,
    CORRECTNESS_SEEDS,
    GPU_PROBE_TIMEOUT_S,
    RTOL,
    SERVED_MODEL,
    SWAP_PARITY_SHAPE,
)
from kernelsmith.inference_server.models import TokenMeter, load_model
from kernelsmith.inference_server.patchable_ops import (
    build_forward,
    find_modules,
    resolve_class_name,
    rollback_op,
    swap_op,
)
from kernelsmith.verifier.static_checker import check_static

#: Guards every mutation of the live model AND every generation against it. One lock,
#: not two: the whole point is that they exclude each other.
_SWAP_LOCK = asyncio.Lock()


@dataclass
class ServerState:
    """Everything the endpoints share. Populated by the lifespan handler."""

    model: Any = None
    tokenizer: Any = None
    meter: TokenMeter | None = None
    #: op_name -> {module_name: TRUE original forward}. Keyed on the first swap of an
    #: op and never overwritten, so rollback always reaches the stock implementation
    #: even after a second kernel is swapped on top of the first.
    originals: dict[str, dict[str, Callable]] = field(default_factory=dict)


STATE = ServerState()


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    tokens: int
    time_ms: float


class StatsResponse(BaseModel):
    tokens_per_s: float
    tokens_total: int
    active_kernel: str
    last_swap_ts: float | None


class SwapRequest(BaseModel):
    op_name: str
    kernel_source: str
    entrypoint: str


class RollbackRequest(BaseModel):
    op_name: str


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, at startup, and warm it up before anyone is timed."""
    del app
    model, tokenizer = load_model()
    STATE.model = model
    STATE.tokenizer = tokenizer
    STATE.meter = TokenMeter(model, tokenizer)
    STATE.meter.warmup()
    yield
    STATE.model = None
    STATE.tokenizer = None
    STATE.meter = None
    STATE.originals.clear()


app = FastAPI(title="KernelSmith inference server", lifespan=lifespan)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> dict[str, Any]:
    """Generate a completion. Holds the swap lock, so no patch lands mid-decode."""
    meter = _require_meter()
    async with _SWAP_LOCK:
        # generate() is blocking and GPU-bound; off the event loop it goes, or /stats
        # stops answering for the duration of every request.
        return await asyncio.to_thread(
            meter.generate, request.prompt, request.max_tokens, request.temperature
        )


@app.get("/stats", response_model=StatsResponse)
async def stats() -> dict[str, Any]:
    """Rolling throughput and which kernel is live. The dashboard polls this ~1/s."""
    meter = _require_meter()
    return meter.stats()


@app.post("/swap")
async def swap(request: SwapRequest) -> dict[str, Any]:
    """Hot-swap a kernel into the live model, or refuse and roll back.

    A refusal is a 200 with `success: false` and a reason: a kernel that fails parity
    is an expected answer from this endpoint, not a server fault. Only a server that
    cannot serve at all (no model loaded) is an HTTP error.
    """
    model = _require_model()
    meter = _require_meter()

    async with _SWAP_LOCK:
        result = await asyncio.to_thread(
            apply_swap,
            model,
            request.op_name,
            request.kernel_source,
            request.entrypoint,
            STATE.originals.get(_op_key(request.op_name)),
        )
        if result["success"]:
            # setdefault, never assign: on a second swap of the same op the handles we
            # just saved point at the PREVIOUS generated forward, not the stock one.
            STATE.originals.setdefault(_op_key(request.op_name), result.pop("originals"))
            meter.record_swap(_op_key(request.op_name))
        result["stats"] = meter.stats()
    return result


@app.post("/rollback")
async def rollback(request: RollbackRequest) -> dict[str, Any]:
    """Restore the stock forwards for one op."""
    model = _require_model()
    meter = _require_meter()

    async with _SWAP_LOCK:
        originals = STATE.originals.pop(_op_key(request.op_name), None)
        if not originals:
            return {
                "success": False,
                "op_name": request.op_name,
                "error": f"no active swap for op {request.op_name!r}",
                "stats": meter.stats(),
            }
        restored = rollback_op(model, originals)
        _refresh_active_kernel(meter)
        return {
            "success": True,
            "op_name": request.op_name,
            "modules_restored": restored,
            "stats": meter.stats(),
        }


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness plus a real GPU probe — a wedged L4 must not read as healthy."""
    gpu = await asyncio.to_thread(gpu_status)
    return {
        "status": "ok" if gpu.get("available") else "degraded",
        "model": SERVED_MODEL,
        "model_loaded": STATE.model is not None,
        "active_kernel": STATE.meter.active_kernel if STATE.meter else "none",
        "gpu": gpu,
    }


# --------------------------------------------------------------------------- #
# Swap mechanics (pure functions — no FastAPI, no globals, so they are testable)
# --------------------------------------------------------------------------- #


def apply_swap(
    model: Any,
    op_name: str,
    kernel_source: str,
    entrypoint: str,
    baseline: dict[str, Callable] | None = None,
) -> dict[str, Any]:
    """Patch `op_name` with `entrypoint` from `kernel_source`, keeping it only if parity holds.

    Steps b–f of the `/swap` protocol: save the current forwards, load the kernel,
    rebind, parity-check, and roll back on any mismatch. The caller owns the lock.

    Args:
        model: The live model.
        op_name: A key of PATCHABLE_OPS.
        kernel_source: Complete Python source of the verified kernel.
        entrypoint: Name of the wrapper function inside that source.
        baseline: The TRUE original forwards, when an earlier swap for this op is
            already live. Parity is always measured against these, never against a
            previously swapped-in kernel.

    Returns:
        On success: {"success": True, "originals": {...}, "modules_patched": int,
        "parity": {...}}. On failure: {"success": False, "error": str,
        "rolled_back": bool, ...} — the model is left exactly as it was found.
    """
    try:
        class_name = resolve_class_name(op_name)
    except KeyError as exc:
        return _refused(op_name, str(exc))

    violations = check_static(kernel_source)
    if violations:
        detail = "; ".join(f"rule {rule} (line {line}): {desc}" for rule, line, desc in violations)
        return _refused(op_name, f"static checker rejected the kernel: {detail}")

    try:
        new_forward = build_forward(op_name, _load_entrypoint(kernel_source, entrypoint))
    except Exception as exc:  # noqa: BLE001 — bad kernel source is data, not an outage
        return _refused(op_name, f"could not load entrypoint {entrypoint!r}: {exc}")

    targets = find_modules(model, class_name)
    if not targets:
        # Silence here would be the worst outcome: a "successful" swap that patched
        # nothing and changed no throughput.
        return _refused(op_name, f"no module whose class name contains {class_name!r}")

    originals = swap_op(model, class_name, new_forward)
    probe_name, probe_module = targets[0]
    reference = (baseline or {}).get(probe_name) or originals[probe_name]

    try:
        parity = check_parity(probe_module, reference)
    except Exception as exc:  # noqa: BLE001 — a kernel that explodes is a failed parity
        rollback_op(model, originals)
        return _refused(op_name, f"parity check raised {type(exc).__name__}: {exc}", True)

    if not parity["parity_pass"]:
        rollback_op(model, originals)
        return {
            "success": False,
            "op_name": op_name,
            "error": "numeric parity failed: " + "; ".join(parity["failures"][:3]),
            "rolled_back": True,
            "parity": parity,
        }

    return {
        "success": True,
        "op_name": op_name,
        "class_name": class_name,
        "modules_patched": len(originals),
        "originals": originals,
        "parity": parity,
        "rolled_back": False,
    }


def check_parity(
    module: Any,
    original_forward: Callable,
    seeds: int = CORRECTNESS_SEEDS,
    shape: tuple[int, int] = SWAP_PARITY_SHAPE,
) -> dict[str, Any]:
    """Compare the module's current forward against `original_forward` over `seeds` seeds.

    Deliberately the same contract as the verifier's correctness gate (atol=rtol=1e-2,
    NaN/Inf, shape and dtype guards), just on the live weights instead of synthetic
    ones — a kernel can pass the sandbox and still be wrong against a real weight
    distribution.
    """
    param = next(module.parameters())
    batch, seq_len = shape
    probe_shape = (batch, seq_len, int(param.shape[-1]))

    failures: list[str] = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        x = torch.randn(*probe_shape, device=param.device, dtype=param.dtype)
        with torch.inference_mode():
            reference = original_forward(x)
            candidate = module.forward(x)
        reason = _compare(reference, candidate)
        if reason is not None:
            failures.append(f"seed {seed}: {reason}")

    return {
        "parity_pass": not failures,
        "seeds": seeds,
        "shape": list(probe_shape),
        "atol": ATOL,
        "rtol": RTOL,
        "failures": failures,
    }


def gpu_status() -> dict[str, Any]:
    """Ask nvidia-smi what the GPU is doing. Never raises; a dead GPU is a report."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=GPU_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    if completed.returncode != 0:
        return {"available": False, "error": completed.stderr.strip()[-200:]}
    return {
        "available": True,
        "device": completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else "",
        "torch_cuda": torch.cuda.is_available(),
    }


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _load_entrypoint(kernel_source: str, entrypoint: str) -> Callable:
    """Execute verified kernel source and pull out its wrapper function."""
    namespace: dict[str, Any] = {"__name__": "kernelsmith_hotswap_kernel"}
    exec(compile(kernel_source, "<hotswap-kernel>", "exec"), namespace)  # noqa: S102
    candidate = namespace.get(entrypoint)
    if not callable(candidate):
        defined = ", ".join(n for n in namespace if not n.startswith("_"))
        raise AttributeError(f"{entrypoint!r} is not a callable in the kernel; defined: {defined}")
    return candidate


def _compare(reference: Any, candidate: Any) -> str | None:
    """None if the two outputs agree, else why they do not."""
    if not isinstance(candidate, torch.Tensor):
        return f"kernel returned {type(candidate).__name__}, expected torch.Tensor"
    if not torch.isfinite(candidate).all():
        return "kernel produced NaN/Inf"
    if candidate.shape != reference.shape:
        return f"shape {tuple(candidate.shape)} vs reference {tuple(reference.shape)}"
    if candidate.dtype != reference.dtype:
        return f"dtype {candidate.dtype} vs reference {reference.dtype}"
    if not torch.allclose(candidate, reference, atol=ATOL, rtol=RTOL):
        max_abs = (candidate.float() - reference.float()).abs().max().item()
        return f"allclose failed at atol={ATOL} rtol={RTOL} (max abs diff {max_abs:.6g})"
    return None


def _refused(op_name: str, error: str, rolled_back: bool = False) -> dict[str, Any]:
    return {"success": False, "op_name": op_name, "error": error, "rolled_back": rolled_back}


def _op_key(op_name: str) -> str:
    return op_name.strip().lower()


def _refresh_active_kernel(meter: TokenMeter) -> None:
    """After a rollback, report whatever swap (if any) is still live."""
    remaining = sorted(STATE.originals)
    if remaining:
        meter.record_swap(remaining[-1])
    else:
        meter.record_rollback()


def _require_model() -> Any:
    if STATE.model is None:
        raise HTTPException(status_code=503, detail="model is not loaded yet")
    return STATE.model


def _require_meter() -> TokenMeter:
    if STATE.meter is None:
        raise HTTPException(status_code=503, detail="model is not loaded yet")
    return STATE.meter
