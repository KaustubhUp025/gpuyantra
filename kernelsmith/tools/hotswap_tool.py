"""The Supervisor's hand on the live server: POST /swap (spec 8.3).

This is the last step of a run and the only one whose blast radius is a user-facing
process. It is deliberately thin: the server owns the parity gate and the rollback, so
this tool's whole job is to carry a verified kernel across the process boundary and
report — without softening — what came back.

Two rules the prompt cannot be trusted to enforce alone are enforced here as data:
a swap failure is returned as `success: False` with the server's reason attached, and
an unreachable server is a failure, never a silent success.
"""

from __future__ import annotations

from typing import Any

import httpx
from google.adk.tools import FunctionTool

from kernelsmith.config import HOTSWAP_TIMEOUT_S, INFERENCE_HOST, INFERENCE_PORT


def swap_url() -> str:
    return f"http://{INFERENCE_HOST}:{INFERENCE_PORT}/swap"


def hotswap_kernel(kernel_source: str, entrypoint: str, op_name: str) -> dict:
    """Hot-swap a verified kernel into the running inference server.

    Call this ONLY for a kernel the verifier scored >= +2 (correct and faster than
    eager). The server re-runs the static checker, patches every matching module,
    checks numeric parity against the original forward over 5 seeds, and rolls back
    automatically if parity fails — so a refusal here means the kernel is not safe to
    serve, not that the call was malformed.

    Args:
        kernel_source: Complete Python source of the winning kernel, verbatim.
        entrypoint: Name of the wrapper function inside that source.
        op_name: Which op to patch: "rmsnorm", "swiglu" or "rope".

    Returns:
        {"success": bool, "op_name": str, ...}. On success: "modules_patched" and
        "stats" (tokens_per_s, tokens_total, active_kernel, last_swap_ts) from the live
        server. On failure: "error" with the server's reason and "rolled_back" saying
        whether the model was restored. Report the result as-is; a failed swap must
        never be described as a success.
    """
    payload = {"op_name": op_name, "kernel_source": kernel_source, "entrypoint": entrypoint}
    try:
        response = httpx.post(swap_url(), json=payload, timeout=HOTSWAP_TIMEOUT_S)
    except httpx.HTTPError as exc:
        return _failure(op_name, f"inference server unreachable at {swap_url()}: {exc}")

    if response.status_code != 200:
        return _failure(
            op_name,
            f"inference server returned HTTP {response.status_code}: {response.text[:300]}",
        )

    try:
        body = response.json()
    except ValueError as exc:
        return _failure(op_name, f"inference server returned non-JSON: {exc}")

    if not isinstance(body, dict):
        return _failure(
            op_name, f"inference server returned {type(body).__name__}, expected object"
        )

    body.setdefault("op_name", op_name)
    body["success"] = bool(body.get("success", False))
    return body


def _failure(op_name: str, error: str) -> dict[str, Any]:
    return {"success": False, "op_name": op_name, "error": error, "rolled_back": False}


#: Registered on the Supervisor (spec 4.2), called once a winning kernel is saved.
hotswap_tool = FunctionTool(hotswap_kernel)
