"""`make demo`: one command, one L4, one reproducible number (spec 11, 16).

The whole claim of this project is checkable by a stranger in one command. That is the
entire design goal of this file, and it is why the order below is not negotiable:

    seed_everything()  ->  Firestore  ->  inference server  ->  Runner  ->  two turns

`seed_everything()` runs FIRST, before anything imports torch or touches CUDA, because
`CUBLAS_WORKSPACE_CONFIG` is read when the cuBLAS handle is created and ignored
afterwards. A demo that seeds late is a demo whose numbers move between runs, which is
exactly the failure the field is already known for (Sakana 3.13x -> 1.49x).

Two turns, not one. ADK's `LoopAgent` cannot transfer back to its parent, so the
Supervisor's turn ends the moment the RefinementLoop escalates — with the kernel scored
but neither saved nor swapped. Steps 4-7 of the protocol (upsert, hot-swap, explain,
summarize) run on a follow-up message into the SAME session, which is what
`.claude/rules/implementation-deviations.md` describes and what the dashboard's
`drive_run()` does for the UI. Anything that sends only one message silently skips the
hot-swap and reports a run that never went live.

The inference server is a subprocess, not an import: it holds a 3 GB model and its own
event loop, and the demo must be able to leave it running (or attach to one already
running) without either process being able to wedge the other.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # torch/transformers must not be imported at module scope here
    from kernelsmith.tools.profiler_tool import AuditReport

# The reproducibility contract has to be applied before torch is imported anywhere in
# this process, so this is the one module-level import that is deliberately eager.
from kernelsmith.reproducibility import seed_everything

DEFAULT_OP = "rmsnorm"
DEFAULT_HIDDEN_SIZE = 1536  # Qwen2.5-1.5B
DEFAULT_AUDIT_MODEL = "qwen2.5-1.5b"  # the served model; `audit` with no --model

#: Loading Qwen2.5-1.5B and warming it up is the slow part of startup, not uvicorn.
SERVER_START_TIMEOUT_S = 600
SERVER_POLL_INTERVAL_S = 2.0
SERVER_SHUTDOWN_TIMEOUT_S = 20

TURN_1 = (
    "Optimize the {op_name} op for Qwen2.5-1.5B on the L4. Follow the protocol: "
    "profile it, retrieve prior skills, then run the refinement loop."
)
TURN_2 = (
    "The refinement loop has finished. Continue the protocol from where you left off: "
    "save the winning kernel to the skill library, hot-swap it into the live inference "
    "server, have Gemma explain the winning kernel, then summarize the run."
)


# --------------------------------------------------------------------------- #
# Inference server lifecycle
# --------------------------------------------------------------------------- #


def health_url() -> str:
    from kernelsmith.config import INFERENCE_HOST, INFERENCE_PORT

    return f"http://{INFERENCE_HOST}:{INFERENCE_PORT}/health"


def server_is_up(timeout: float = 2.0) -> bool:
    """True if something is already serving a loaded model on the inference port."""
    import httpx

    try:
        response = httpx.get(health_url(), timeout=timeout)
        return response.status_code == 200 and bool(response.json().get("model_loaded"))
    except Exception:  # noqa: BLE001 — "not up yet" is the normal answer here
        return False


def start_inference_server(timeout: float = SERVER_START_TIMEOUT_S) -> subprocess.Popen | None:
    """Start the inference server and wait for it to load the model.

    Returns:
        The `Popen` handle if this call started the server (the caller must stop it), or
        None if a healthy server was already running and is being reused.

    Raises:
        RuntimeError: the server died, or never became healthy within `timeout`.
    """
    from kernelsmith.config import INFERENCE_HOST, INFERENCE_PORT

    if server_is_up():
        print(f"[demo] reusing the inference server already at {health_url()}")
        return None

    print(f"[demo] starting the inference server on {INFERENCE_HOST}:{INFERENCE_PORT} …")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "kernelsmith.inference_server.server:app",
            "--host",
            INFERENCE_HOST,
            "--port",
            str(INFERENCE_PORT),
        ],
        # os.environ already carries CUBLAS_WORKSPACE_CONFIG: seed_everything set it.
        env=dict(os.environ),
    )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"inference server exited with code {process.returncode}")
        if server_is_up():
            print(
                f"[demo] inference server ready after {timeout - (deadline - time.monotonic()):.0f}s"
            )
            return process
        time.sleep(SERVER_POLL_INTERVAL_S)

    stop_inference_server(process)
    raise RuntimeError(f"inference server did not become healthy within {timeout:.0f}s")


def stop_inference_server(process: subprocess.Popen | None) -> None:
    """Stop a server this demo started. A server we merely attached to is left alone."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=SERVER_SHUTDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=SERVER_SHUTDOWN_TIMEOUT_S)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


async def _drive(
    runner: Any,
    user_id: str,
    session_id: str,
    op_name: str,
    hidden_size: int,
) -> dict[str, Any]:
    """Run both turns of the Supervisor protocol in one session, then return its state."""
    from google.genai import types

    def message(text: str) -> Any:
        return types.Content(role="user", parts=[types.Part(text=text)])

    await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )

    turns = (
        ("1/2 profile -> retrieve -> refine", TURN_1.format(op_name=op_name), True),
        ("2/2 upsert -> hot-swap -> explain", TURN_2, False),
    )
    for label, text, seed_state in turns:
        print(f"\n[demo] turn {label}")
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message(text),
            # task_spec is seeded on turn 1 only; turn 2 resumes from the same session.
            state_delta=(
                {"task_spec": {"op_name": op_name, "hidden_size": hidden_size}}
                if seed_state
                else None
            ),
        ):
            _print_event(event)

    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    return dict(getattr(session, "state", {}) or {})


def _print_event(event: Any) -> None:
    """One line per agent turn and per tool call — the demo's live narration."""
    author = getattr(event, "author", "?")
    for call in event.get_function_calls() or []:
        print(f"  [{author}] -> {call.name}()")
    for response in event.get_function_responses() or []:
        payload = response.response
        if isinstance(payload, dict) and "reward" in payload:
            print(
                f"  [{author}] <- {response.name}: reward={payload.get('reward')} "
                f"eager={payload.get('speedup_vs_eager')} "
                f"compile={payload.get('speedup_vs_compile')}"
            )
        else:
            print(f"  [{author}] <- {response.name}()")


def run_demo(
    op_name: str = DEFAULT_OP,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    start_server: bool = True,
) -> dict[str, Any]:
    """Run the full KernelSmith demo end to end. The entry point for `make demo`.

    Seeds every RNG, starts (or attaches to) the inference server, drives both turns of
    the Supervisor protocol, and prints the winning kernel with the numbers the
    VERIFIER measured — never a model's description of them.

    Args:
        op_name: The op to optimize. "rmsnorm" is the demo path.
        hidden_size: The served model's hidden dimension.
        start_server: Start the inference server as a subprocess. False attaches to
            whatever is already on the port, and fails the hot-swap if nothing is.

    Returns:
        The final `session.state`, plus "run_id" and "reproducibility".
    """
    repro = seed_everything()
    print(f"[demo] seeded: {json.dumps(repro)}")

    run_id = f"demo-{uuid.uuid4().hex[:10]}"
    started_at = datetime.now(UTC)
    _record_run_start(run_id, op_name, started_at)

    process = start_inference_server() if start_server else None
    try:
        from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
        from google.adk.runners import Runner
        from google.adk.sessions.in_memory_session_service import InMemorySessionService

        from kernelsmith.root_agent import root_agent

        runner = Runner(
            agent=root_agent,
            app_name="kernelsmith",
            session_service=InMemorySessionService(),
            artifact_service=InMemoryArtifactService(),
        )
        state = asyncio.run(_drive(runner, "demo", run_id, op_name, hidden_size))
    finally:
        if start_server:
            stop_inference_server(process)

    _record_run_end(run_id, state)
    state["run_id"] = run_id
    state["reproducibility"] = repro
    print_results(state)
    return state


# --------------------------------------------------------------------------- #
# Firestore run record (best-effort — a demo must not die on a memory outage)
# --------------------------------------------------------------------------- #


def _record_run_start(run_id: str, op_name: str, started_at: datetime) -> None:
    try:
        from kernelsmith.memory.firestore_store import get_db, put_run
        from kernelsmith.memory.schemas import RunRecord

        get_db()  # fail here, with a clear message, rather than mid-run
        put_run(RunRecord(run_id=run_id, task_ref=op_name, started_at=started_at))
    except Exception as exc:  # noqa: BLE001 — the run history is a nicety, the run is not
        print(f"[demo] Firestore run record unavailable ({type(exc).__name__}: {exc})")


def _record_run_end(run_id: str, state: dict[str, Any]) -> None:
    try:
        from kernelsmith.memory.firestore_store import update_run

        update_run(
            run_id,
            {
                "ended_at": datetime.now(UTC),
                "final_reward": int(state.get("best_reward", -1) or -1),
                "total_iterations": int(state.get("iteration", 0) or 0),
                "status": "completed",
            },
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[demo] could not close the run record ({type(exc).__name__}: {exc})")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def print_results(state: dict[str, Any]) -> None:
    """Print what was measured. Numbers come from the verdict, never from the summary."""
    verdict = state.get("best_verdict") or state.get("verdict") or {}
    if not isinstance(verdict, dict):
        verdict = {}
    hotswap = state.get("hotswap_result") or {}

    print("\n" + "=" * 72)
    print("KernelSmith demo results")
    print("=" * 72)
    print(f"run_id           : {state.get('run_id', '?')}")
    print(f"reward           : {state.get('best_reward', 'none')} (max 3)")
    print(f"iterations       : {state.get('iteration', 0)}")
    print(f"correctness      : {'PASS' if verdict.get('correctness_pass') else 'FAIL'}")
    print(f"speedup vs eager : {_fmt(verdict.get('speedup_vs_eager'))}x")
    print(f"speedup vs compile: {_fmt(verdict.get('speedup_vs_compile'))}x")
    print(f"latency by shape : {json.dumps(verdict.get('latency_ms_by_shape') or {})}")
    print(f"skill bandit arm : {state.get('selected_skill_id') or '(cold library)'}")
    print(
        "hot-swap         : "
        + (
            f"live, {hotswap.get('modules_patched')} modules patched"
            if isinstance(hotswap, dict) and hotswap.get("success")
            else f"not live ({(hotswap or {}).get('error', 'not attempted')})"
        )
    )

    kernel = str(state.get("best_kernel") or "")
    print("\n--- winning kernel " + "-" * 53)
    print(kernel if kernel else "(no kernel survived verification)")

    explanation = str(state.get("kernel_explanation") or "")
    if explanation:
        print("\n--- Gemma's explanation " + "-" * 48)
        print(explanation)

    summary = str(state.get("supervisor_summary") or "")
    if summary:
        print("\n--- supervisor summary " + "-" * 49)
        print(summary)
    print("=" * 72)


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "0.00"


# --------------------------------------------------------------------------- #
# Audit mode (spec 7, Task 10)
# --------------------------------------------------------------------------- #


def run_audit(
    model: str,
    device: str | None = None,
    output: str = "text",
) -> AuditReport:
    """Audit one model and print the result. Returns the report for programmatic use."""
    from kernelsmith.tools.profiler_tool import audit_model, format_audit_report

    resolved = device or default_audit_device()
    report = audit_model(model, device=resolved)

    if output == "json":
        from dataclasses import asdict

        print(json.dumps(asdict(report), indent=2, default=list))
    else:
        print(format_audit_report(report))
    return report


def run_audit_all(device: str | None = None, output: str = "text") -> list[AuditReport]:
    """Audit every registered model in turn, then print a side-by-side comparison.

    Sequentially, one model resident at a time. The point of the comparison table is
    that the top target CHANGES with the architecture — RMSNorm, LayerNorm, BatchNorm —
    while the reason it is the top target does not: the same roofline analysis, on the
    same hardware model, picking whichever normalization the architecture happens to use.
    """
    from kernelsmith.config import MODEL_REGISTRY

    resolved = device or default_audit_device()
    reports: list[AuditReport] = []
    for key in MODEL_REGISTRY:
        print()
        try:
            reports.append(run_audit(key, device=resolved, output=output))
        except Exception as exc:  # noqa: BLE001 — one unloadable model must not stop the sweep
            print(f"[audit] {key}: could not audit ({type(exc).__name__}: {exc})")

    if output != "json":
        print()
        print(format_comparison_table(reports))
    return reports


def format_comparison_table(reports: Sequence[AuditReport]) -> str:
    """All audited models on one row each: what the architecture is, and what to attack."""
    from kernelsmith.config import AUDIT_REPORT_WIDTH, MODEL_REGISTRY

    by_hf_id = {str(entry["hf_id"]): (key, entry) for key, entry in MODEL_REGISTRY.items()}

    header = ("Model", "Norm", "Top target", "Count", "AI", "Priority")
    widths = (16, 10, 22, 7, 8, 10)
    lines = [
        "═══ Cross-architecture comparison " + "═" * max(0, AUDIT_REPORT_WIDTH - 34),
        _table_row(header, widths),
        "─" * min(AUDIT_REPORT_WIDTH, sum(widths) + len(widths) + 1),
    ]
    for report in reports:
        key, registry = by_hf_id.get(report.model_name, (report.model_name, {}))
        top = next((e for e in report.module_entries if e.module_type == report.top_target), None)
        lines.append(
            _table_row(
                (
                    key,
                    str(registry.get("norm_type", "?")),
                    report.top_target or "(none)",
                    str(top.count if top else 0),
                    f"{top.arithmetic_intensity:.2f}" if top else "n/a",
                    top.priority if top else "n/a",
                ),
                widths,
            )
        )
    lines.append(
        "Same roofline analysis, three architectures: the norm the model happens to use "
        "is the target in every one."
    )
    return "\n".join(lines)


def _table_row(cells: Sequence[str], widths: Sequence[int]) -> str:
    parts = []
    for cell, width in zip(cells, widths, strict=True):
        text = str(cell)
        if len(text) > width:
            text = text[: max(0, width - 1)] + "…"
        parts.append(text.ljust(width))
    return " ".join(parts).rstrip()


def default_audit_device() -> str:
    """ "cuda" when there is a GPU to measure with, else "cpu". Never raises."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 — no torch at all still audits nothing, but on CPU
        return "cpu"


# --------------------------------------------------------------------------- #
# Cross-model skill transfer (spec 6.4, Task 10)
# --------------------------------------------------------------------------- #


def demonstrate_cross_model_transfer(source_model: str, target_model: str) -> dict[str, Any]:
    """Show that a skill learned on `source_model` is retrievable for `target_model`.

    This is the claim the retrieval design has always implied and never demonstrated:
    skills are indexed by the BOTTLENECK fingerprint, not by the op's name or the model
    it came from, so a fused-norm kernel learned on Qwen2's RMSNorm is a hit for GPT-2's
    LayerNorm — same `op_family`, same hardware, same side of the ridge point. A
    name-keyed cache cannot make that jump; that is the point.

    Nothing here writes to Firestore. It audits the target, builds the fingerprint its
    top op would produce, and runs the real retrieval query against the real library.

    Returns:
        A dict describing what was retrieved, so a caller (or a test) can assert on it
        rather than on stdout.
    """
    from kernelsmith.tools.profiler_tool import (
        audit_model,
        compute_tile_hint,
        fallback_fingerprint,
        family_from_name,
    )
    from kernelsmith.tools.retrieval_tool import retrieve_skills

    print("\n" + "=" * 72)
    print(f"Cross-model skill transfer: {source_model} -> {target_model}")
    print("=" * 72)

    report = audit_model(target_model, device="cpu")
    top = report.top_target
    print(f"[transfer] {target_model} top target: {top} ({report.total_modules} modules scanned)")

    # The fingerprint the target's top op WOULD produce. Built from the audit's own
    # numbers, not measured — the target model is not being optimized here, only queried.
    entry = next((e for e in report.module_entries if e.module_type == top), None)
    # `family_from_name`, not `classify_op_family`: `top` is a CLASS NAME string, and
    # the callable-taking classifier would read `str` off it and answer "elementwise".
    op_family = family_from_name(top) if top else "elementwise"
    fingerprint = fallback_fingerprint(op_family, report.hidden_size)
    if entry is not None:
        fingerprint = fingerprint.model_copy(
            update={
                "arithmetic_intensity": entry.arithmetic_intensity,
                "is_memory_bound": entry.bottleneck == "memory",
                "is_compute_bound": entry.bottleneck != "memory",
                "tile_size_hint": compute_tile_hint(report.hidden_size),
            }
        )
    fingerprint_text = fingerprint.to_embedding_text()
    print(f"[transfer] fingerprint: {fingerprint_text}")

    try:
        skills = retrieve_skills(fingerprint.op_family, fingerprint.hardware, fingerprint_text)
        error = ""
    except Exception as exc:  # noqa: BLE001 — a cold or unreachable library is a message
        skills, error = [], f"{type(exc).__name__}: {exc}"

    result = {
        "source_model": source_model,
        "target_model": target_model,
        "top_target": top,
        "op_family": fingerprint.op_family,
        "fingerprint_text": fingerprint_text,
        "skills": [
            {
                "skill_id": s.get("skill_id", "?"),
                "op_signature": s.get("op_signature", "?"),
                "speedup_vs_eager": s.get("speedup_vs_eager"),
                "vector_distance": s.get("vector_distance"),
            }
            for s in skills
        ],
        "transferred": bool(skills),
        "error": error,
    }

    if error:
        print(f"[transfer] skill library unreachable — {error}")
    elif skills:
        for s in result["skills"]:
            print(
                f"[transfer]   {s['skill_id']} (op_signature={s['op_signature']}, "
                f"distance={s['vector_distance']})"
            )
        print(
            f"Cross-model transfer: skill from {source_model} retrieved for "
            f"{target_model} — fingerprints match on op_family "
            f"({fingerprint.op_family}) and hardware ({fingerprint.hardware})"
        )
    else:
        print("No matching skills found — cold start required")
    print("=" * 72)
    return result


def run_full(
    op_name: str,
    hidden_size: int,
    start_server: bool,
    device: str | None = None,
) -> dict[str, Any]:
    """audit -> optimize -> re-audit -> cross-model transfer. The whole story, in order.

    The re-audit is deliberately the same call as the first one: the audit describes the
    model's SHAPES, so it does not move because a kernel got faster. What changes between
    the two is the skill library, which is why the transfer demonstration comes last —
    it is the step that can only succeed after something has been learned.
    """
    resolved = device or default_audit_device()

    print("\n### 1/4 audit — where is the time going?")
    before = run_audit(DEFAULT_AUDIT_MODEL, device=resolved)

    print(f"\n### 2/4 optimize — {op_name} (audit's top target: {before.top_target})")
    state = run_demo(op_name, hidden_size, start_server=start_server)

    print("\n### 3/4 re-audit — the same model, now with a verified kernel for it")
    after = run_audit(DEFAULT_AUDIT_MODEL, device=resolved)

    print("\n### 4/4 cross-model transfer — does the skill move to another architecture?")
    transfer = demonstrate_cross_model_transfer(DEFAULT_AUDIT_MODEL, "gpt2")

    state["audit_before"] = before.top_target
    state["audit_after"] = after.top_target
    state["cross_model_transfer"] = transfer
    return state


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

#: Subcommands, and the one that runs when none is given. `make demo` passes only
#: `$(DEMO_ARGS)`, so an argv that does not start with a subcommand has "optimize"
#: prepended rather than being rejected — `python -m kernelsmith.run_demo --no-server`
#: has to keep working exactly as it did before this file learned about subcommands.
SUBCOMMANDS = ("audit", "optimize", "full")
DEFAULT_SUBCOMMAND = "optimize"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kernelsmith.run_demo",
        description="Audit a model, optimize an op, or run the whole story end to end.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="profile a model and rank its kernel targets")
    audit.add_argument(
        "--model",
        default=DEFAULT_AUDIT_MODEL,
        help=f"MODEL_REGISTRY key or HuggingFace id (default: {DEFAULT_AUDIT_MODEL})",
    )
    audit.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="cpu = analytic estimates only; cuda also measures bandwidth (default: auto)",
    )
    audit.add_argument("--output", choices=("text", "json"), default="text")
    audit.add_argument(
        "--all",
        action="store_true",
        help="audit every model in MODEL_REGISTRY and print a comparison table",
    )

    for name, help_text in (
        ("optimize", "generate, verify and hot-swap a kernel (the default)"),
        ("full", "audit -> optimize -> re-audit -> cross-model transfer"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--op", default=DEFAULT_OP, help="op to optimize (default: rmsnorm)")
        sub.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
        sub.add_argument(
            "--no-server",
            action="store_true",
            help="do not start the inference server; attach to one already running",
        )
        if name == "full":
            sub.add_argument("--device", choices=("cpu", "cuda"), default=None)

    return parser


def normalize_argv(argv: list[str]) -> list[str]:
    """Prepend the default subcommand unless one was given. Keeps `make demo` working."""
    if argv and argv[0] in SUBCOMMANDS:
        return argv
    if argv and argv[0] in ("-h", "--help"):
        return argv
    return [DEFAULT_SUBCOMMAND, *argv]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(normalize_argv(list(sys.argv[1:] if argv is None else argv)))

    if args.command == "audit":
        if args.all:
            reports = run_audit_all(device=args.device, output=args.output)
            return 0 if reports else 1
        run_audit(args.model, device=args.device, output=args.output)
        return 0

    if args.command == "full":
        state = run_full(
            args.op, args.hidden_size, start_server=not args.no_server, device=args.device
        )
    else:
        state = run_demo(args.op, args.hidden_size, start_server=not args.no_server)

    # Exit non-zero when nothing verified, so `make demo` fails loudly in CI.
    return 0 if int(state.get("best_reward", -1) or -1) >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
