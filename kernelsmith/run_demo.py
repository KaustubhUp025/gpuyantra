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
from datetime import UTC, datetime
from typing import Any

# The reproducibility contract has to be applied before torch is imported anywhere in
# this process, so this is the one module-level import that is deliberately eager.
from kernelsmith.reproducibility import seed_everything

DEFAULT_OP = "rmsnorm"
DEFAULT_HIDDEN_SIZE = 1536  # Qwen2.5-1.5B

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the KernelSmith demo end to end.")
    parser.add_argument("--op", default=DEFAULT_OP, help="op to optimize (default: rmsnorm)")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="do not start the inference server; attach to one already running",
    )
    args = parser.parse_args(argv)

    state = run_demo(args.op, args.hidden_size, start_server=not args.no_server)
    # Exit non-zero when nothing verified, so `make demo` fails loudly in CI.
    return 0 if int(state.get("best_reward", -1) or -1) >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
