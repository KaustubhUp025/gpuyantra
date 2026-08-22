"""Subprocess sandbox for every generated candidate (spec 5.4).

Red line #2: generated Triton code NEVER runs in the ADK, Streamlit, or inference
server process. It runs here, in a throwaway subprocess, with a scrubbed environment
and a hard timeout, and the GPU is health-probed afterwards.

The subprocess contract: the verification script prints one JSON object to stdout as
its last line. Anything else — a crash, a hang, garbage on stdout — is reward -1.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from kernelsmith.config import (
    CUBLAS_WORKSPACE,
    GPU_HEALTH_PROBE_TIMEOUT_S,
    SANDBOX_TIMEOUT_S,
)

#: Throwaway cwd. Generated code may litter it; nothing here is ever trusted.
SANDBOX_DIR = Path("/tmp/kernelsmith_sandbox")

#: `bash scripts/gpu_reset.sh` — last resort when the GPU is wedged.
GPU_RESET_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gpu_reset.sh"

#: Keep stderr in the reward payload bounded; the Coder only needs the tail.
STDERR_TAIL_CHARS = 4000

#: Trivial known-answer probe: 1 + 2 == 3 on the GPU.
_HEALTH_PROBE_SRC = (
    "import torch;"
    "a=torch.ones(1,device='cuda');"
    "b=torch.full((1,),2.0,device='cuda');"
    "torch.cuda.synchronize();"
    "assert (a+b).item()==3.0"
)


def run_in_sandbox(script_path: str, timeout: int = SANDBOX_TIMEOUT_S) -> dict[str, Any]:
    """Run a verification script in an isolated subprocess and return its reward dict.

    - Scrubbed env: only PATH, CUDA_VISIBLE_DEVICES, HOME, CUBLAS_WORKSPACE_CONFIG.
    - Hard timeout: `subprocess.run` SIGKILLs the child on TimeoutExpired (Triton can
      ignore SIGTERM), and `start_new_session=True` puts it in its own process group
      so the kill cannot reach back into us.
    - GPU health probe after EVERY candidate, pass or fail.

    Never raises for candidate misbehaviour — a bad candidate is a -1, not an outage.
    """
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_safe_env(),
            cwd=str(SANDBOX_DIR),
            start_new_session=True,
            check=False,
        )
        result = parse_result(completed.stdout, completed.stderr, completed.returncode)
    except subprocess.TimeoutExpired as exc:
        result = {
            "reward": -1,
            "correctness_pass": False,
            "error": "timeout_sigkill",
            "stderr_tail": _tail(_as_text(exc.stderr)),
        }
    except OSError as exc:
        result = {
            "reward": -1,
            "correctness_pass": False,
            "error": "spawn_failed",
            "stderr_tail": str(exc),
        }

    # Health probe after every candidate — a wedged GPU poisons every later run.
    if not gpu_health_probe():
        result["gpu_wedged"] = True
        result["gpu_reset_attempted"] = run_gpu_reset()

    return result


def parse_result(stdout: str, stderr: str, returncode: int = 0) -> dict[str, Any]:
    """Turn subprocess output into a reward dict.

    The script's JSON is trusted for the numbers only when it exited cleanly; a
    non-zero exit is -1 regardless of what it managed to print.
    """
    stderr_tail = _tail(stderr)

    if returncode != 0:
        return {
            "reward": -1,
            "correctness_pass": False,
            "error": f"nonzero_exit:{returncode}",
            "stderr_tail": stderr_tail,
        }

    payload = _last_json_object(stdout)
    if payload is None:
        return {
            "reward": -1,
            "correctness_pass": False,
            "error": "unparseable_output",
            "stderr_tail": stderr_tail,
        }

    payload.setdefault("reward", -1)
    payload.setdefault("correctness_pass", False)
    payload.setdefault("stderr_tail", stderr_tail)
    return payload


def gpu_health_probe(timeout: int = GPU_HEALTH_PROBE_TIMEOUT_S) -> bool:
    """Run a trivial known-answer kernel in a subprocess. False means the GPU is wedged.

    Out-of-process on purpose: an in-process probe against a hung GPU would hang the
    caller, and the timeout is the whole point of the probe.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _HEALTH_PROBE_SRC],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_safe_env(),
            start_new_session=True,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return completed.returncode == 0


def run_gpu_reset() -> bool:
    """Invoke scripts/gpu_reset.sh. True if the script ran and exited cleanly."""
    if not GPU_RESET_SCRIPT.exists():
        return False
    bash = shutil.which("bash") or "/bin/bash"
    try:
        completed = subprocess.run(
            [bash, str(GPU_RESET_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=SANDBOX_TIMEOUT_S,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return completed.returncode == 0


def _safe_env() -> dict[str, str]:
    """The only four variables a candidate is allowed to see."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "HOME": os.environ.get("HOME", str(SANDBOX_DIR)),
        "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE,
    }


def _last_json_object(stdout: str) -> dict[str, Any] | None:
    """Parse the last JSON object printed on stdout, ignoring any chatter before it."""
    if not stdout:
        return None
    candidates = [stdout, *reversed(stdout.strip().splitlines())]
    for chunk in candidates:
        chunk = chunk.strip()
        if not chunk.startswith("{"):
            continue
        try:
            parsed = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _tail(text: str | None) -> str:
    return _as_text(text)[-STDERR_TAIL_CHARS:]
