"""The ADK-facing verifier: static check -> sandbox -> reward (spec 5).

This is the only path by which a generated kernel is allowed to run. It never
executes candidate code in this process (red line #2): the candidate is written to a
throwaway file, a generated runner script is written beside it, and
`sandbox.run_in_sandbox` spawns both in a scrubbed subprocess with a hard timeout.

Four defences, in order of cost:
  1. `static_checker.check_static` — purely syntactic, costs nothing, cannot be fooled
     by runtime behaviour. Any violation is -1 with no execution at all.
  2. `adapter_mapping.validate_adapter_mapping` — deterministic `hasattr` check of the
     deployment contract the Coder declared, against the real module class. Also -1
     with no execution: a mapping naming an attribute that does not exist would fail
     inside a hot forward on a live server, and that is far too late to find out.
  3. `correctness.check_correctness` — 5 seeds x 3 shapes at atol=rtol=1e-2, inside
     the sandbox. Timing is only measured after all 15 checks pass.
  4. `reward.compute_reward` — recomputed HERE from the parsed numbers. Whatever the
     subprocess printed as its own `reward` is never trusted: the candidate controls
     that stdout.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from google.adk.tools import FunctionTool

from kernelsmith.config import GCP_PROJECT, SANDBOX_TIMEOUT_S
from kernelsmith.verifier.adapter_mapping import validate_adapter_mapping
from kernelsmith.verifier.reward import compute_reward
from kernelsmith.verifier.sandbox import SANDBOX_DIR, run_in_sandbox
from kernelsmith.verifier.static_checker import check_static

#: Repo root, injected into the sandbox's sys.path so `import kernelsmith` works
#: under the scrubbed environment (PYTHONPATH is deliberately not passed through).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])

#: Replaced with a JSON blob when the runner is materialized.
_CONFIG_TOKEN = "__KERNELSMITH_CONFIG__"

_RUNNER_TEMPLATE = '''\
"""GENERATED verification runner. Runs ONLY inside the sandbox subprocess.

Contract with verifier/sandbox.py: print exactly one JSON object as the last line of
stdout. Anything else — a crash, a hang, garbage — is reward -1.
"""

import json
import os
import random
import sys
import traceback

CONFIG = json.loads(r"""__KERNELSMITH_CONFIG__""")

# kernelsmith.config reads GOOGLE_CLOUD_PROJECT strictly, and the sandbox env is
# scrubbed to four variables. Re-inject the (non-secret) project id before importing.
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", CONFIG["gcp_project"])
sys.path.insert(0, CONFIG["repo_root"])


def run() -> dict:
    import importlib.util

    import numpy as np
    import torch

    from kernelsmith.config import CORRECTNESS_SHAPES, DETERMINISTIC_CUDA, GLOBAL_SEED
    from kernelsmith.tools.profiler_tool import build_op
    from kernelsmith.verifier.correctness import check_correctness
    from kernelsmith.verifier.reward import compute_reward
    from kernelsmith.verifier.timing import bench_kernel, compute_speedups, measure_baselines

    # Reproducibility contract (spec 11): a speedup that needs a lucky seed is a lie.
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if DETERMINISTIC_CUDA:
        torch.use_deterministic_algorithms(True)
    # TF32 on for the honest eager baseline (KernelBench-Verified).
    torch.set_float32_matmul_precision("high")

    device = CONFIG["device"]
    hidden = CONFIG["hidden_size"]
    dtype = getattr(torch, CONFIG["dtype"])
    binding = build_op(CONFIG["op_name"], hidden, device, dtype)

    spec = importlib.util.spec_from_file_location("candidate_kernel", CONFIG["kernel_path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, CONFIG["entrypoint"]):
        raise AttributeError(
            "entrypoint " + CONFIG["entrypoint"] + " not found in the kernel; defined names: "
            + ", ".join(n for n in vars(module) if not n.startswith("_"))
        )
    candidate = binding.bind(getattr(module, CONFIG["entrypoint"]))

    checks = check_correctness(binding.reference, candidate, hidden, device=device)
    result = {
        "correctness_pass": checks["correctness_pass"],
        "total_checks": checks["total_checks"],
        "passed_checks": checks["passed_checks"],
        "failed_cases": checks["failed_cases"][:5],
        "speedup_vs_eager": 0.0,
        "speedup_vs_torch_compile": 0.0,
        "latency_ms_by_shape": {},
        "baseline_ms": {},
    }
    if not checks["correctness_pass"]:
        result["reward"] = -1
        return result

    # Timing only runs on a kernel that is already known to be correct.
    for batch, seq_len in CORRECTNESS_SHAPES:
        x = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)
        result["latency_ms_by_shape"][str(batch) + "x" + str(seq_len)] = bench_kernel(
            lambda x=x: candidate(x)
        )

    # Headline shape = the largest, where memory traffic actually dominates.
    batch, seq_len = CORRECTNESS_SHAPES[-1]
    x = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)
    baselines = measure_baselines(binding.reference, x)
    speedups = compute_speedups(
        baselines["eager_ms"],
        baselines["compile_ms"],
        result["latency_ms_by_shape"][str(batch) + "x" + str(seq_len)],
    )

    result["baseline_ms"] = baselines
    result["headline_shape"] = str(batch) + "x" + str(seq_len)
    result["speedup_vs_eager"] = speedups["speedup_vs_eager"]
    result["speedup_vs_torch_compile"] = speedups["speedup_vs_compile"]
    result["reward"] = compute_reward(
        True, speedups["speedup_vs_eager"], speedups["speedup_vs_compile"]
    )
    return result


try:
    payload = run()
except BaseException as exc:  # noqa: BLE001 — any failure is a -1, reported as data
    traceback.print_exc()
    payload = {
        "reward": -1,
        "correctness_pass": False,
        "error": type(exc).__name__ + ": " + str(exc),
        "traceback_tail": traceback.format_exc()[-2000:],
    }

print(json.dumps(payload))
'''


def verify_kernel(
    kernel_code: str,
    entrypoint: str,
    task_spec: dict,
    adapter_mapping: dict | None = None,
) -> dict:
    """Verify a generated Triton kernel: contract first, then correctness, then speed.

    Runs the static reward-hack checker and validates the declared deployment contract,
    then executes the kernel in an isolated subprocess against the reference op named
    by the task spec — 5 seeds x 3 shapes at atol=rtol=1e-2 — and only times it if all
    15 checks pass.

    Args:
        kernel_code: Complete Python source of the candidate (the @triton.jit kernel
            plus its Python wrapper).
        entrypoint: Name of the wrapper function inside `kernel_code` to call.
        task_spec: {"op_name": one of "rmsnorm" | "layernorm" | "softmax" | "silu" |
            "rope" | "mlp", "hidden_size": int, optional "device", optional "dtype"}.
        adapter_mapping: The candidate's `adapter_mapping` — kernel parameter name ->
            module attribute name, e.g. {"weight": "weight", "eps": "variance_epsilon"}.
            Checked against the real module class before anything runs. Pass it exactly
            as the Coder wrote it; omit it only when the draft declared none.

    Returns:
        A dict with the Verdict fields — reward (-1..3), correctness_pass,
        speedup_vs_eager, speedup_vs_compile, next_action, stop, stderr_tail,
        latency_ms_by_shape — plus diagnostics: violations, failed_cases, baseline_ms.
        A reward of -1 means the kernel is wrong or was rejected; read stderr_tail and
        failed_cases for the reason.
    """
    violations = check_static(kernel_code)
    if violations:
        return _rejected_verdict(violations)

    op_name_for_mapping = str(task_spec.get("op_name", "")).strip()
    mapping_errors = validate_adapter_mapping(op_name_for_mapping, adapter_mapping)
    if mapping_errors:
        # Layer 2 of the three-layer contract model: no sandbox, no execution. The
        # kernel may be perfect; the contract that deploys it is not.
        return _mapping_rejected_verdict(mapping_errors)

    op_name = str(task_spec.get("op_name", "")).strip()
    hidden_size = int(task_spec.get("hidden_size", 0) or 0)
    if not op_name or hidden_size <= 0:
        return _error_verdict(
            "task_spec needs a non-empty 'op_name' and a positive 'hidden_size'; "
            f"got op_name={op_name!r} hidden_size={hidden_size}"
        )

    run_id = uuid.uuid4().hex[:8]
    kernel_path = _write(f"kernel_{run_id}.py", kernel_code)
    runner_path = _write(
        f"runner_{run_id}.py",
        _RUNNER_TEMPLATE.replace(
            _CONFIG_TOKEN,
            json.dumps(
                {
                    "kernel_path": str(kernel_path),
                    "entrypoint": entrypoint,
                    "op_name": op_name,
                    "hidden_size": hidden_size,
                    "device": str(task_spec.get("device", "cuda")),
                    "dtype": str(task_spec.get("dtype", "float16")),
                    "repo_root": _REPO_ROOT,
                    "gcp_project": GCP_PROJECT,
                }
            ),
        ),
    )

    raw = run_in_sandbox(str(runner_path), timeout=int(task_spec.get("timeout", SANDBOX_TIMEOUT_S)))
    return _verdict_from_sandbox(raw)


def _write(name: str, content: str) -> Path:
    """Write one file into the throwaway sandbox dir. Nothing here is ever trusted."""
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    path = SANDBOX_DIR / name
    path.write_text(content)
    return path


def _verdict_from_sandbox(raw: dict[str, Any]) -> dict[str, Any]:
    """Turn the sandbox payload into Verdict fields, recomputing the reward here.

    The subprocess's own `reward` is discarded: a candidate that prints
    `{"reward": 3}` and nothing else must still score -1.
    """
    correctness_pass = bool(raw.get("correctness_pass", False))
    speedup_vs_eager = _as_float(raw.get("speedup_vs_eager"))
    speedup_vs_compile = _as_float(raw.get("speedup_vs_torch_compile"))
    reward = compute_reward(correctness_pass, speedup_vs_eager, speedup_vs_compile)

    verdict: dict[str, Any] = {
        "reward": reward,
        "correctness_pass": correctness_pass,
        "speedup_vs_eager": speedup_vs_eager,
        "speedup_vs_compile": speedup_vs_compile,
        "next_action": _default_next_action(reward, raw),
        "stop": reward >= 3,
        "stderr_tail": str(raw.get("stderr_tail", ""))[-500:],
        "latency_ms_by_shape": raw.get("latency_ms_by_shape") or {},
        "violations": [],
    }
    for key in (
        "error",
        "traceback_tail",
        "failed_cases",
        "passed_checks",
        "total_checks",
        "baseline_ms",
        "headline_shape",
        "gpu_wedged",
    ):
        if key in raw:
            verdict[key] = raw[key]
    return verdict


def _rejected_verdict(violations: list[tuple[int, int, str]]) -> dict[str, Any]:
    """Static-checker rejection: reward -1 with no execution at all."""
    detail = "; ".join(f"rule {rule} (line {line}): {desc}" for rule, line, desc in violations)
    return {
        "reward": -1,
        "correctness_pass": False,
        "speedup_vs_eager": 0.0,
        "speedup_vs_compile": 0.0,
        "next_action": f"Rewrite the kernel to remove these rejected patterns: {detail}",
        "stop": False,
        "stderr_tail": f"static checker rejected the candidate: {detail}"[-500:],
        "latency_ms_by_shape": {},
        "violations": [
            {"rule_id": rule, "line": line, "description": desc} for rule, line, desc in violations
        ],
    }


def _mapping_rejected_verdict(errors: list[str]) -> dict[str, Any]:
    """Adapter-contract rejection: reward -1, sandbox skipped entirely."""
    detail = "; ".join(errors)
    return {
        "reward": -1,
        "correctness_pass": False,
        "speedup_vs_eager": 0.0,
        "speedup_vs_compile": 0.0,
        "next_action": (
            "Fix adapter_mapping so every value names a real attribute of the module "
            f"being patched: {detail}"
        ),
        "stop": False,
        "stderr_tail": f"adapter_mapping rejected: {detail}"[-500:],
        "latency_ms_by_shape": {},
        "violations": [],
        "adapter_mapping_errors": errors,
    }


def _error_verdict(message: str) -> dict[str, Any]:
    return {
        "reward": -1,
        "correctness_pass": False,
        "speedup_vs_eager": 0.0,
        "speedup_vs_compile": 0.0,
        "next_action": f"Fix the request before retrying: {message}",
        "stop": False,
        "stderr_tail": message[-500:],
        "latency_ms_by_shape": {},
        "violations": [],
        "error": message,
    }


def _default_next_action(reward: int, raw: dict[str, Any]) -> str:
    """A starting point for the Judge, which owns the final `next_action` (spec 4.2)."""
    if reward >= 3:
        return "STOP"
    if reward == -1:
        if raw.get("failed_cases"):
            return f"Fix correctness: {raw['failed_cases'][0].get('reason', 'check failed')}"
        return f"Fix the failure: {raw.get('error') or str(raw.get('stderr_tail', ''))[-200:]}"
    return "Correct but not fast enough: propose ONE concrete performance change."


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    # NaN loses every comparison in compute_reward, but 0.0 says what happened.
    return 0.0 if result != result else result


#: Registered on the Judge agent (spec 4.2). The Judge has NO output_schema — ADK
#: cannot combine tools with a schema (bug #3969) — so it parses the Verdict itself.
verifier_tool = FunctionTool(verify_kernel)
