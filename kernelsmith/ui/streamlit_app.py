"""The KernelSmith dashboard (spec 10.1).

Three columns and a banner:

    ┌───────────────┬───────────────────────┬──────────────────┐
    │ Agent Activity│  live tokens/sec      │  Skill Library   │
    │  Supervisor   │  rolling latency      │  Run History     │
    │  Profiler     │  [Start Optimization] │                  │
    │  Coder        │                       │                  │
    │  Judge        │                       │                  │
    └───────────────┴───────────────────────┴──────────────────┘
    │  🚫 REJECTED — reward hack   /   ✅ Kernel hot-swapped     │

Three facts about this file are load-bearing, and all three come from
`.claude/rules/implementation-deviations.md`:

1. **The run takes two messages, not one.** ADK's `LoopAgent` cannot transfer back to
   its parent, so the Supervisor's turn ends the moment the RefinementLoop escalates —
   with the winning kernel scored but neither saved nor swapped. The driver has to send
   a follow-up message to get steps 4-6 (upsert → hot-swap → summary). That driver is
   `_drive_run()` below: it watches the consumer go idle and fires turn 2 by itself, so
   the operator presses one button and the two-turn protocol is invisible.

2. **`TokenMeter` clears its rolling window on a swap.** `/stats` after a swap reports
   the *new* kernel's throughput only, which is the honest number but also means
   tokens/s legitimately drops to 0.0 until the next generation. The latency chart
   inserts a real gap at the swap boundary instead of drawing a line across it — the
   before/after discontinuity is the thing the demo is claiming.

3. **Nothing heavy is imported at page load.** Building the agent tree pulls in torch
   and Triton; a dashboard that cannot open without a GPU is useless for rehearsing the
   layout. The Runner is built on the first click and cached from then on.

Everything async lives behind `EventStreamConsumer`. This module never awaits.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from typing import Any

import streamlit as st

from kernelsmith.ui.event_stream import EventStreamConsumer, EventSummary, summarize_event

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REFRESH_MS = 1000
#: Spec 10.1: the latency chart is a rolling window, not a full history.
CHART_POINTS = 30
#: Bounded so an overnight session cannot exhaust memory.
MAX_EVENTS = 2000
#: Drained per rerun. A burst spreads over a few refreshes rather than stalling one.
DRAIN_LIMIT = 200
STATS_TIMEOUT_S = 1.0
SKILL_TABLE_TTL_S = 15

AGENT_PANELS = ("Supervisor", "Profiler", "Coder", "Judge")

DEFAULT_OP = "rmsnorm"
DEFAULT_HIDDEN_SIZE = 1536  # Qwen2.5-1.5B hidden size

TURN_1 = (
    "Optimize the {op_name} op for Qwen2.5-1.5B on the L4. Follow the protocol: "
    "profile it, retrieve prior skills, then run the refinement loop."
)
TURN_2 = (
    "The refinement loop has finished. Continue the protocol from where you left off: "
    "save the winning kernel to the skill library and hot-swap it into the live "
    "inference server, then summarize the run."
)


# --------------------------------------------------------------------------- #
# Singletons (survive Streamlit reruns)
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_consumer() -> EventStreamConsumer:
    """The background event loop. Cheap to build — no agent imports, no GPU."""
    return EventStreamConsumer()


@st.cache_resource(show_spinner="Building the KernelSmith agent tree…")
def get_runner() -> Any:
    """The ADK Runner over the shared agent tree.

    Built lazily, on the first Start click, because importing `root_agent` drags in
    torch, Triton and the Vertex clients. `cache_resource` keeps it for the life of the
    server, which matters: `build_supervisor()` binds every sub-agent to a parent, so a
    second tree per rerun would be both wasteful and wrong.
    """
    from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
    from google.adk.runners import Runner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService

    from kernelsmith.root_agent import root_agent

    return Runner(
        agent=root_agent,
        app_name="kernelsmith",
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
    )


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #


def init_state() -> None:
    """Seed every key this script reads, so no branch has to guard for absence."""
    state = st.session_state
    state.setdefault("events", [])
    state.setdefault("panels", {})  # agent name -> EventSummary
    state.setdefault("samples", deque(maxlen=CHART_POINTS))  # (label, ms_per_token|None)
    state.setdefault("stats", {})
    state.setdefault("stats_error", "")
    state.setdefault("last_swap_ts", None)
    state.setdefault("tps_before_swap", None)
    state.setdefault("tps_after_swap", None)
    state.setdefault("user_id", "operator")
    state.setdefault("session_id", "")
    state.setdefault("turn", 0)
    state.setdefault("awaiting_followup", False)
    state.setdefault("runs_seen", 0)
    state.setdefault("run_error", "")
    state.setdefault("rejection", None)
    state.setdefault("swap", None)
    state.setdefault("explanation", "")
    state.setdefault("autorefresh_error", "")


# --------------------------------------------------------------------------- #
# Polling
# --------------------------------------------------------------------------- #


def poll_stats() -> None:
    """Fetch `/stats` and fold it into the chart. A dead server is a caption, not a crash."""
    import httpx

    from kernelsmith.config import INFERENCE_HOST, INFERENCE_PORT

    url = f"http://{INFERENCE_HOST}:{INFERENCE_PORT}/stats"
    try:
        response = httpx.get(url, timeout=STATS_TIMEOUT_S)
        response.raise_for_status()
        stats = response.json()
    except Exception as exc:  # noqa: BLE001 — the inference server is optional for the UI
        st.session_state["stats_error"] = f"{type(exc).__name__}: {exc}"
        return

    if not isinstance(stats, dict):
        st.session_state["stats_error"] = f"/stats returned {type(stats).__name__}"
        return

    st.session_state["stats_error"] = ""
    st.session_state["stats"] = stats
    record_sample(stats)


def record_sample(stats: dict[str, Any]) -> None:
    """Append one latency point, breaking the line where a swap happened.

    `TokenMeter.record_swap` clears the rolling window, so the samples either side of a
    swap describe two different kernels. Joining them with a line would draw a smooth
    ramp across the exact boundary the demo is claiming; a `None` leaves a visible gap.
    """
    state = st.session_state
    swap_ts = stats.get("last_swap_ts")
    label = time.strftime("%H:%M:%S")

    if swap_ts != state["last_swap_ts"]:
        if state["last_swap_ts"] is not None or state["samples"]:
            state["tps_before_swap"] = _last_tps(state["samples"])
        state["last_swap_ts"] = swap_ts
        state["samples"].append((label, None))  # the discontinuity

    tps = _as_float(stats.get("tokens_per_s"))
    if tps > 0:
        state["samples"].append((label, 1000.0 / tps))
        if state["last_swap_ts"] is not None:
            state["tps_after_swap"] = tps


def _last_tps(samples: deque) -> float | None:
    """Recover tokens/s from the most recent charted latency point."""
    for _, ms in reversed(samples):
        if ms:
            return 1000.0 / ms
    return None


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# Event ingestion
# --------------------------------------------------------------------------- #


def ingest_events(consumer: EventStreamConsumer) -> None:
    """Drain the queue into `session_state`, updating panels and banners as we go."""
    state = st.session_state
    for event in consumer.drain_events(limit=DRAIN_LIMIT):
        summary = summarize_event(event)
        state["events"].append(summary)
        if summary.author in AGENT_PANELS:
            state["panels"][summary.author] = summary
        update_banners(summary)

    if len(state["events"]) > MAX_EVENTS:
        del state["events"][: len(state["events"]) - MAX_EVENTS]


def update_banners(summary: EventSummary) -> None:
    """Latch the two events the bottom banner exists for.

    Both are read from the *tool response* rather than from any agent's prose. The
    verifier's numbers are the only ones allowed on screen (red line #3), and a model
    that describes a failed swap as a success must not be able to turn the banner green.
    """
    state = st.session_state
    for name, payload in summary.responses:
        if not isinstance(payload, dict):
            continue
        if name == "verify_kernel":
            violations = payload.get("violations")
            if violations:
                state["rejection"] = {
                    "violations": violations,
                    "detail": _violation_text(violations),
                    "at": summary.timestamp,
                }
        elif name == "hotswap_kernel":
            if payload.get("success"):
                swap_stats = payload.get("stats")
                state["swap"] = {
                    "op_name": payload.get("op_name", "?"),
                    "modules_patched": payload.get("modules_patched"),
                    "tps": _as_float((swap_stats or {}).get("tokens_per_s")),
                    "at": summary.timestamp,
                }
            else:
                state["run_error"] = (
                    f"hot-swap refused: {payload.get('error', 'unknown reason')}"
                    + (" (rolled back)" if payload.get("rolled_back") else "")
                )
        elif name == "explain_kernel":
            # FunctionTool wraps the str return as {"result": ...}. The tool reports its
            # own failures as text starting with "error:", so show them as a caption
            # rather than swallowing them.
            state["explanation"] = str(payload.get("result", "")).strip()


def _violation_text(violations: Any) -> str:
    """Render the static checker's findings for the banner."""
    if not isinstance(violations, list):
        return str(violations)
    parts = []
    for violation in violations:
        if isinstance(violation, dict):
            parts.append(
                f"rule {violation.get('rule_id', '?')} "
                f"(line {violation.get('line', '?')}): {violation.get('description', '')}"
            )
        else:
            parts.append(str(violation))
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# Run driver — the two-message protocol
# --------------------------------------------------------------------------- #


def start_optimization(op_name: str, hidden_size: int) -> None:
    """Kick off turn 1: profile → retrieve → refinement loop."""
    state = st.session_state
    try:
        runner = get_runner()
    except Exception as exc:  # noqa: BLE001 — a missing GPU or ADC must not blank the page
        state["run_error"] = f"could not build the agent tree — {type(exc).__name__}: {exc}"
        return

    state["session_id"] = f"ui-{uuid.uuid4().hex[:12]}"
    state["turn"] = 1
    state["awaiting_followup"] = True
    state["run_error"] = ""
    state["rejection"] = None
    state["swap"] = None
    state["explanation"] = ""
    state["tps_before_swap"] = _last_tps(state["samples"])
    state["tps_after_swap"] = None

    started = get_consumer().start_run(
        runner,
        state["user_id"],
        state["session_id"],
        TURN_1.format(op_name=op_name),
        state_delta={"task_spec": {"op_name": op_name, "hidden_size": hidden_size}},
    )
    if not started:
        state["run_error"] = "a run is already in flight"
        state["awaiting_followup"] = False
    state["runs_seen"] = get_consumer().runs_completed


def drive_run(consumer: EventStreamConsumer) -> None:
    """Send the follow-up message once turn 1 ends (implementation-deviations.md).

    The RefinementLoop escalates, the Supervisor's turn ends, and steps 4-6 of the
    protocol have not run yet. Every step is idempotent and reads `session.state`, so
    resuming is just another message into the same session.
    """
    state = st.session_state
    completed = consumer.runs_completed
    if completed <= state["runs_seen"] or consumer.is_running:
        return
    state["runs_seen"] = completed

    error = consumer.last_error
    if error:
        state["run_error"] = error
        state["awaiting_followup"] = False
        return

    if not state["awaiting_followup"]:
        return

    state["awaiting_followup"] = False
    state["turn"] = 2
    try:
        runner = get_runner()
    except Exception as exc:  # noqa: BLE001
        state["run_error"] = f"{type(exc).__name__}: {exc}"
        return
    consumer.start_run(runner, state["user_id"], state["session_id"], TURN_2)
    state["runs_seen"] = consumer.runs_completed


# --------------------------------------------------------------------------- #
# Firestore-backed tables
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=SKILL_TABLE_TTL_S, show_spinner=False)
def load_skills() -> tuple[list[dict[str, Any]], str]:
    """Skill-library rows, or an empty table and the reason it is empty."""
    try:
        from kernelsmith.memory.firestore_store import list_skills

        return [
            {
                "name": skill.skill_id,
                "op_family": skill.op_family,
                "speedup": round(skill.speedup_vs_eager, 2),
                "reward": round(
                    skill.bandit_total_reward / skill.bandit_pulls if skill.bandit_pulls else 0.0,
                    2,
                ),
            }
            for skill in list_skills()
        ], ""
    except Exception as exc:  # noqa: BLE001 — Firestore is optional for a layout rehearsal
        return [], f"{type(exc).__name__}: {exc}"


@st.cache_data(ttl=SKILL_TABLE_TTL_S, show_spinner=False)
def load_runs() -> tuple[list[dict[str, Any]], str]:
    """Run-history rows, newest first."""
    try:
        from kernelsmith.memory.firestore_store import list_runs

        return [
            {
                "run_id": run.run_id,
                "task": run.task_ref,
                "reward": run.final_reward,
                "iters": run.total_iterations,
                "status": run.status,
            }
            for run in list_runs()
        ], ""
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_agent_panels() -> None:
    """Left column: what each agent last thought and last did."""
    st.subheader("Agent Activity")
    panels: dict[str, EventSummary] = st.session_state["panels"]
    for name in AGENT_PANELS:
        summary = panels.get(name)
        marker = "🟢" if summary else "⚪"
        with st.expander(f"{marker} {name}", expanded=summary is not None):
            if summary is None:
                st.caption("no events yet")
                continue
            st.caption(time.strftime("%H:%M:%S", time.localtime(summary.timestamp)))
            st.markdown("**Last thought**")
            st.write(summary.thought or summary.text or "_(no text this turn)_")
            st.markdown("**Last action**")
            st.code(summary.action or "—", language="text")


def render_center(consumer: EventStreamConsumer) -> None:
    """Middle column: the live throughput metric, the latency chart, and the button."""
    state = st.session_state
    stats = state["stats"]

    tps = _as_float(stats.get("tokens_per_s"))
    delta = None
    before = state["tps_before_swap"]
    if before and tps > 0:
        delta = f"{tps - before:+.1f} vs pre-swap"
    st.metric("Tokens / sec (live)", f"{tps:.1f}" if stats else "—", delta=delta)

    columns = st.columns(2)
    columns[0].metric("Active kernel", str(stats.get("active_kernel", "—")))
    columns[1].metric("Tokens generated", f"{int(_as_float(stats.get('tokens_total'))):,}")

    if state["stats_error"]:
        st.caption(f"inference server unreachable — {state['stats_error']}")

    st.markdown("**Rolling latency (ms/token)**")
    samples = list(state["samples"])
    if any(ms for _, ms in samples):
        st.line_chart(
            {"ms/token": [ms for _, ms in samples]},
            x_label="last 30 polls",
            y_label="ms/token",
            height=200,
        )
    else:
        st.caption("no generations measured yet — the chart fills as /stats reports tokens")

    st.divider()
    disabled = consumer.is_running
    op_name = st.selectbox("Op", ("rmsnorm", "swiglu"), disabled=disabled)
    hidden_size = st.number_input(
        "hidden_size", min_value=64, max_value=8192, value=DEFAULT_HIDDEN_SIZE, disabled=disabled
    )
    if st.button(
        "🚀 Start Optimization",
        type="primary",
        use_container_width=True,
        disabled=disabled,
    ):
        start_optimization(op_name, int(hidden_size))
        st.rerun()

    if consumer.is_running:
        st.info(f"Turn {state['turn']} of 2 running — {len(state['events'])} events so far")
        if st.button("Cancel run", use_container_width=True):
            consumer.cancel()
    elif state["turn"]:
        st.caption(f"idle after turn {state['turn']} — {len(state['events'])} events")


def render_library() -> None:
    """Right column: what the system has learned, and what it has attempted."""
    st.subheader("Skill Library")
    skills, skills_error = load_skills()
    if skills:
        st.dataframe(skills, use_container_width=True, hide_index=True)
    else:
        st.caption(skills_error or "no skills saved yet")

    render_explanation()

    st.subheader("Run History")
    runs, runs_error = load_runs()
    if runs:
        st.dataframe(runs, use_container_width=True, hide_index=True)
    else:
        st.caption(runs_error or "no runs recorded yet")


def render_explanation() -> None:
    """Right column: Gemma's plain-English read of the winning kernel (spec 15).

    A different model family from the agents, so it is a second opinion on the kernel
    rather than the Coder marking its own homework — and it is the answer to "I can't
    write GPU kernels": the system wrote one and then explained it back.
    """
    explanation = st.session_state["explanation"]
    if not explanation:
        return

    st.subheader("Kernel Explained (Gemma 4)")
    if explanation.startswith("error:"):
        st.caption(f"explanation unavailable — {explanation}")
        return
    with st.expander("What the winning kernel does", expanded=True):
        st.markdown(explanation)


def render_banner() -> None:
    """Bottom banner: the reward-hack rejection and the successful hot-swap."""
    state = st.session_state

    if state["run_error"]:
        st.warning(f"⚠️ {state['run_error']}")

    rejection = state["rejection"]
    if rejection:
        st.error(f"### 🚫 REJECTED — reward hack: {rejection['detail']}")

    swap = state["swap"]
    if swap:
        before = state["tps_before_swap"]
        after = state["tps_after_swap"] or swap["tps"]
        if before and after:
            throughput = f"Tokens/sec: {before:.1f} → {after:.1f}"
        elif after:
            throughput = f"Tokens/sec: {after:.1f} (no pre-swap baseline measured)"
        else:
            throughput = "Tokens/sec: pending — generate to measure the new kernel"
        st.success(
            f"### ✅ Kernel hot-swapped! {throughput}\n"
            f"op `{swap['op_name']}` · {swap['modules_patched']} modules patched"
        )


def render_autorefresh() -> None:
    """Refresh at 1 Hz, or fall back to a button if the component is unavailable."""
    state = st.session_state
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=REFRESH_MS, key="kernelsmith-refresh")
        return
    except Exception as exc:  # noqa: BLE001 — a missing component is not a broken dashboard
        state["autorefresh_error"] = f"{type(exc).__name__}: {exc}"

    st.warning(
        "Auto-refresh is unavailable "
        f"({state['autorefresh_error']}) — refresh manually to see new events."
    )
    st.button("🔄 Refresh", use_container_width=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(page_title="KernelSmith", page_icon="⚒️", layout="wide")
    init_state()

    st.title("⚒️ KernelSmith")
    st.caption(
        "An ADK agent tree writes a Triton kernel, the verifier scores it, "
        "and a passing kernel is hot-swapped into a live Qwen2.5-1.5B."
    )

    try:
        consumer = get_consumer()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not start the event-stream thread: {type(exc).__name__}: {exc}")
        st.stop()
        return

    render_autorefresh()

    try:
        ingest_events(consumer)
        drive_run(consumer)
    except Exception as exc:  # noqa: BLE001 — a bad event must not blank the dashboard
        st.session_state["run_error"] = f"event processing failed: {type(exc).__name__}: {exc}"

    poll_stats()

    left, center, right = st.columns([1, 2, 1])
    with left:
        render_agent_panels()
    with center:
        render_center(consumer)
    with right:
        render_library()

    st.divider()
    render_banner()


main()
