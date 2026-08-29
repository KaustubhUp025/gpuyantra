"""The recording dashboard (Task 12) — a separate app from `streamlit_app.py`.

    streamlit run kernelsmith/ui/demo_dashboard.py --server.port 8502

`streamlit_app.py` is the operator's dashboard: three columns, dense, every number on
screen at once. This one is built for a 1080p screen recording watched by someone who
has never seen the system, so it trades density for legibility. One visual carries it —
a vertical agent timeline that builds up turn by turn, like a Cloud Trace waterfall with
the jargon removed.

CLAUDE.md rule 14: this is a NEW file. `streamlit_app.py` is not touched, and the two
run side by side on 8501 and 8502.

Three design decisions worth stating, because each of them is the difference between a
demo that records cleanly and one that does not:

1. **Both modes render through the same `render_event`.** Live events arrive from
   `EventStreamConsumer.drain_events()`; replayed ones arrive from
   `event_replay.pace_events()`. Both are the flat dicts `event_capture` defines, so
   nothing downstream knows or cares which it is looking at. A replay that rendered
   through a second code path would be a rehearsal of the wrong thing.

2. **Live mode re-renders the whole timeline every rerun; replay mode renders once,
   progressively.** Streamlit re-executes the script top-to-bottom on every refresh, so
   live mode has no choice: it groups `session_state["timeline_events"]` into turns and
   draws them all. Replay blocks inside one script run and appends to its containers as
   the generator yields, which is what makes the timeline visibly *build* on camera.
   Auto-refresh is therefore switched off during a replay — a 1 Hz rerun would restart
   the playback from the top, forever.

3. **Every number on screen comes from a tool response, never from an agent's prose.**
   Red line #3. The Judge can describe a kernel however it likes; the Speedup metric
   moves only when `verify_kernel` returns one.

On `st.set_page_config`: it has no `theme` parameter. Dark is set by
`--theme.base dark` in the Makefile target and reinforced by the CSS below, so opening
the app by hand still records correctly.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st

from kernelsmith.ui.event_capture import DEFAULT_TRACE_DIR, EventLogger, classify_event
from kernelsmith.ui.event_replay import list_traces, load_events, pace_events, total_duration_s

__all__ = [
    "AGENT_STYLES",
    "Turn",
    "build_agent_graph",
    "classify_event",
    "extract_metrics",
    "group_turns",
    "highlight_active_agent",
    "split_prose_and_code",
    "turn_headline",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REFRESH_MS = 1000
DRAIN_LIMIT = 200
STATS_TIMEOUT_S = 1.0
#: Bounded so a long rehearsal cannot exhaust the browser session.
MAX_TIMELINE_EVENTS = 1500
#: Kernel sources reach four figures of characters; the timeline shows the shape of one.
CODE_PREVIEW_CHARS = 700
TEXT_PREVIEW_CHARS = 600

MODE_LIVE = "🔴 Live"
MODE_REPLAY = "📼 Replay"
MODES = (MODE_LIVE, MODE_REPLAY)

SPEED_CHOICES = {"0.5×": 0.5, "1×": 1.0, "2×": 2.0, "instant": 0.0}

#: Colour, emoji and a plain-English role per agent. The colours are the ones the task
#: brief specifies; the roles exist because the audience is not expected to know what a
#: "Profiler" is from the name alone.
AGENT_STYLES: dict[str, dict[str, str]] = {
    "Supervisor": {"color": "#2563eb", "emoji": "🎯", "role": "orchestrates the run"},
    "Profiler": {"color": "#10b981", "emoji": "🔍", "role": "measures the bottleneck"},
    "Coder": {"color": "#f59e0b", "emoji": "💻", "role": "writes the Triton kernel"},
    "Judge": {"color": "#ef4444", "emoji": "⚖️", "role": "verifies it independently"},
    "EscalationChecker": {"color": "#8b5cf6", "emoji": "🛑", "role": "decides when to stop"},
    "RefinementLoop": {"color": "#6b7280", "emoji": "🔁", "role": "iterates coder ↔ judge"},
    "user": {"color": "#6b7280", "emoji": "👤", "role": "the operator"},
}
DEFAULT_STYLE = {"color": "#6b7280", "emoji": "⚙️", "role": ""}

#: Nodes of the agent tree, in the order graphviz should rank them.
GRAPH_NODES = (
    ("Supervisor", "box"),
    ("Profiler", "box"),
    ("RefinementLoop", "box3d"),
    ("Coder", "box"),
    ("Judge", "box"),
    ("EscalationChecker", "box"),
)
GRAPH_EDGES = (
    ("Supervisor", "Profiler", "delegate"),
    ("Supervisor", "RefinementLoop", "delegate"),
    ("RefinementLoop", "Coder", ""),
    ("RefinementLoop", "Judge", ""),
    ("RefinementLoop", "EscalationChecker", ""),
)
NODE_IDLE_COLOR = "#374151"
NODE_ACTIVE_COLOR = "#f59e0b"
NODE_ROOT_COLOR = "#2563eb"

TURN_1 = (
    "Optimize the {op_name} op for Qwen2.5-1.5B on the L4. Follow the protocol: "
    "profile it, retrieve prior skills, then run the refinement loop."
)
TURN_2 = (
    "The refinement loop has finished. Continue the protocol from where you left off: "
    "save the winning kernel to the skill library and hot-swap it into the live "
    "inference server, then summarize the run."
)
DEFAULT_OP = "rmsnorm"
DEFAULT_HIDDEN_SIZE = 1536


CSS = """
<style>
  /* Tighter than the default so the timeline starts above the fold at 1080p, but not
     so tight that Streamlit's fixed header clips the title — 3.5rem clears it. */
  .block-container { padding-top: 3.5rem; padding-bottom: 2rem; max-width: 100%; }

  /* Video cleanliness: nothing on screen that is not the product. The Deploy button
     and the toolbar are the two that show up in a 1080p capture of the top-right. */
  #MainMenu,
  footer,
  header [data-testid="stStatusWidget"],
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stAppDeployButton"] { display: none !important; }

  /* st.status labels are the timeline's headlines — they carry the demo. */
  details[data-testid="stExpander"] summary p,
  div[data-testid="stExpander"] summary p { font-size: 1.12rem; font-weight: 600; }

  /* Metrics readable from across a room. */
  div[data-testid="stMetricValue"] {
      font-size: 2.1rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  div[data-testid="stMetricLabel"] { font-size: 0.95rem; letter-spacing: .04em; }

  .ks-title { font-size: 2.4rem; font-weight: 700; line-height: 1.1; margin: 0; }
  .ks-sub   { font-size: 1.0rem; opacity: .70; margin: .25rem 0 0 0; }
  .ks-agent { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
              font-weight: 700; }
  .ks-elapsed { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                opacity: .6; font-size: .9rem; }
  code, pre, .stCode { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
</style>
"""


# --------------------------------------------------------------------------- #
# Pure logic — no Streamlit below this line until the render section
# --------------------------------------------------------------------------- #


def style_for(author: str) -> dict[str, str]:
    """Colour/emoji/role for an agent, with a neutral default for anything unknown."""
    return AGENT_STYLES.get(author, DEFAULT_STYLE)


def build_agent_graph(active: str | None = None) -> str:
    """The agent tree as a graphviz DOT string, with `active` highlighted.

    A string, not a `graphviz.Digraph`: `st.graphviz_chart` renders DOT in the browser,
    so the dashboard draws its tree with no system graphviz installed and no import to
    fail mid-recording. The `graphviz` dependency exists for anyone who wants to render
    the same DOT to a file offline.

    The Supervisor keeps its blue when nothing is active — it is the root, and a tree
    with every node grey reads as a tree that is switched off.
    """
    lines = [
        "digraph {",
        "  rankdir=TB",
        '  bgcolor="transparent"',
        '  node [shape=box, style="rounded,filled", fontname="monospace", fontsize=12]',
        '  edge [fontname="monospace", fontsize=9, color="#9ca3af", fontcolor="#9ca3af"]',
    ]
    for name, shape in GRAPH_NODES:
        if name == active:
            fill = NODE_ACTIVE_COLOR
        elif name == "Supervisor" and active is None:
            fill = NODE_ROOT_COLOR
        else:
            fill = NODE_IDLE_COLOR
        lines.append(f'  {name} [shape={shape}, fillcolor="{fill}", fontcolor="white"]')
    for source, target, label in GRAPH_EDGES:
        suffix = f' [label="{label}"]' if label else ""
        lines.append(f"  {source} -> {target}{suffix}")
    lines.append("}")
    return "\n".join(lines)


def highlight_active_agent(dot: str, agent: str | None) -> str:
    """Recolour one node of an existing DOT string, leaving every other line untouched.

    Separate from `build_agent_graph` because the two answer different questions:
    building is "what is the tree", highlighting is "who is talking". Rewriting a DOT
    that has already been built is what the live path does on every event, and doing it
    textually means the tree's shape has exactly one definition.

    An `agent` that is not in the graph returns the DOT unchanged — the Supervisor's
    turn-2 events name agents that are not nodes, and that must not blank the diagram.
    """
    known = {name for name, _ in GRAPH_NODES}
    result = dot
    for name in known:
        fill = NODE_ACTIVE_COLOR if name == agent else NODE_IDLE_COLOR
        if name == "Supervisor" and agent is None:
            fill = NODE_ROOT_COLOR
        result = re.sub(
            rf'(^\s*{re.escape(name)}\s*\[[^\]]*?fillcolor=")[^"]*(")',
            rf"\g<1>{fill}\g<2>",
            result,
            count=1,
            flags=re.MULTILINE,
        )
    return result


@dataclass
class Turn:
    """One contiguous run of events from a single agent — one `st.status` on screen."""

    author: str
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def start_s(self) -> float:
        return float(self.events[0].get("elapsed_s") or 0.0) if self.events else 0.0

    @property
    def end_s(self) -> float:
        return float(self.events[-1].get("elapsed_s") or 0.0) if self.events else 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def group_turns(events: list[dict[str, Any]]) -> list[Turn]:
    """Collapse a flat event list into one `Turn` per contiguous author.

    Contiguous, not per-agent: the Supervisor speaks at the start and again after the
    loop escalates, and those are two turns on the timeline because they are two
    different moments in the story.
    """
    turns: list[Turn] = []
    for event in events:
        author = str(event.get("author") or "unknown")
        if not turns or turns[-1].author != author:
            turns.append(Turn(author=author))
        turns[-1].events.append(event)
    return turns


def turn_headline(turn: Turn) -> str:
    """One line naming what this agent turn *did*, for the `st.status` label.

    Ordered by how much the event settles: a tool call is a decision, an escalation is a
    decision, prose is commentary. The first decisive event in the turn wins, because
    the label is written before the turn has finished.
    """
    for event in turn.events:
        kind = event.get("event_type")
        if kind == "function_call":
            names = [call.get("name", "?") for call in event.get("function_calls") or []]
            return f"Calling {', '.join(names) or 'a tool'}"
        if kind == "transfer":
            return f"Delegating to {event.get('transfer_to') or '?'}"
        if kind == "escalate":
            return "Target met — escalating out of the loop"
        if kind == "function_response":
            names = [r.get("name", "?") for r in event.get("function_responses") or []]
            return f"Result from {', '.join(names) or 'a tool'}"
    for event in turn.events:
        text = (event.get("content_text") or "").strip()
        if text:
            first = text.splitlines()[0]
            return first if len(first) <= 90 else first[:87] + "…"
    return style_for(turn.author).get("role", "working") or "working"


def extract_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Pull the header numbers out of the event stream. Tool responses only.

    Red line #3: an agent describing its kernel as "roughly seven times faster" must not
    be able to move the Speedup metric. Only `verify_kernel` and `hotswap_kernel`
    responses are read, and the last one of each wins — the final verdict of a run is
    the one that counts, not its best intermediate.

    Returns a dict with `speedup`, `reward`, `iteration`, `tokens_per_s`, `correct`,
    `violations` and `honest_plus_one`, each None/False when nothing has set it.
    """
    metrics: dict[str, Any] = {
        "speedup": None,
        "speedup_vs_compile": None,
        "reward": None,
        "iteration": 0,
        "tokens_per_s": None,
        "correct": None,
        "violations": None,
        "honest_plus_one": False,
    }
    for event in events:
        for response in event.get("function_responses") or []:
            name = response.get("name")
            payload = response.get("response")
            if not isinstance(payload, dict):
                continue
            # FunctionTool wraps a non-dict return as {"result": ...}; a dict return
            # arrives as-is. Unwrap only when the wrapper is all there is.
            if set(payload) == {"result"} and isinstance(payload["result"], dict):
                payload = payload["result"]

            if name == "verify_kernel":
                metrics["iteration"] += 1
                if payload.get("reward") is not None:
                    metrics["reward"] = payload.get("reward")
                metrics["correct"] = payload.get("correctness_pass")
                metrics["speedup"] = _as_float(payload.get("speedup_vs_eager")) or None
                metrics["speedup_vs_compile"] = _as_float(payload.get("speedup_vs_compile")) or None
                violations = payload.get("violations")
                metrics["violations"] = violations or None
                # The honest case the demo is proud of: correct, verified, not faster.
                metrics["honest_plus_one"] = bool(
                    payload.get("reward") == 1 and payload.get("correctness_pass")
                )
            elif name == "hotswap_kernel" and payload.get("success"):
                stats = payload.get("stats") or {}
                metrics["tokens_per_s"] = _as_float(stats.get("tokens_per_s")) or None
    return metrics


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n…"


def _looks_like_code(text: str) -> bool:
    """Whether a blob should be rendered as a code block rather than as prose."""
    markers = ("@triton.jit", "import triton", "def ", "tl.load", "torch.")
    return sum(marker in text for marker in markers) >= 2


#: A line that starts a Python module. The Coder's turns are prose *then* a kernel, and
#: this is where one ends and the other begins.
_CODE_START = re.compile(r"^\s*(import |from |@|def |class )")


def split_prose_and_code(text: str) -> tuple[str, str]:
    """Split an agent's message into its narration and the kernel it is narrating.

    The Coder writes "RMSNorm is memory-bound … I'll fuse it:" and then pastes a module.
    Rendering the whole blob as one `st.code` puts the explanation in a monospace box
    with syntax colouring applied to English — which is the one part of the timeline a
    non-technical viewer was supposed to be able to read.

    Returns `(prose, code)`; either half may be empty.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _CODE_START.match(line):
            prose = "\n".join(lines[:index]).strip()
            code = "\n".join(lines[index:]).strip()
            # A lone `def ...` inside a sentence is not the start of a module.
            if code and _looks_like_code(code):
                return prose, code
            break
    return ("", text) if _looks_like_code(text) else (text, "")


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #


def init_state() -> None:
    """Seed every key the script reads, so no branch has to guard for absence."""
    state = st.session_state
    state.setdefault("mode", MODE_LIVE)
    state.setdefault("timeline_events", [])
    state.setdefault("active_agent", None)
    state.setdefault("user_id", "operator")
    state.setdefault("session_id", "")
    state.setdefault("awaiting_followup", False)
    state.setdefault("runs_seen", 0)
    state.setdefault("run_error", "")
    state.setdefault("trace_path", "")
    state.setdefault("stats", {})
    state.setdefault("celebrated", False)
    state.setdefault("replay_done", False)
    state.setdefault("replaying", False)


# --------------------------------------------------------------------------- #
# Singletons (survive Streamlit reruns)
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_logger() -> EventLogger:
    """The trace writer. Wired into the consumer once, at construction."""
    return EventLogger(DEFAULT_TRACE_DIR)


@st.cache_resource(show_spinner=False)
def get_consumer() -> Any:
    """The background event loop, with capture wired in (Task 12, Part D).

    Imported lazily: `event_stream` spins up a thread at construction, and the replay
    path has no use for one.
    """
    from kernelsmith.ui.event_stream import EventStreamConsumer

    return EventStreamConsumer(event_logger=get_logger())


@st.cache_resource(show_spinner="Building the KernelSmith agent tree…")
def get_runner() -> Any:
    """The ADK Runner. Built on the first Start click — importing it drags in torch."""
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
# Rendering — the half both modes share
# --------------------------------------------------------------------------- #


def render_event(target: Any, event: dict[str, Any]) -> None:
    """Draw one event inside an already-open `st.status`.

    This is the function that makes live and replay identical: both call it, with the
    same dicts, into the same kind of container. `target` is a DeltaGenerator — an
    `st.status` handle — so calls append to that status rather than to the page.
    """
    kind = event.get("event_type")

    if kind == "function_call":
        for call in event.get("function_calls") or []:
            target.markdown(f"**Calling** `{call.get('name', '?')}`")
            args = call.get("args") or {}
            if args:
                with target.expander("Arguments"):
                    _render_args(st, args)
        return

    if kind == "function_response":
        for response in event.get("function_responses") or []:
            name = response.get("name", "?")
            payload = response.get("response")
            _render_response_highlights(target, name, payload)
            with target.expander(f"Result from `{name}`"):
                st.json(payload)
        return

    if kind == "transfer":
        destination = event.get("transfer_to") or "?"
        style = style_for(destination)
        target.markdown(f"↳ handing off to **{style['emoji']} {destination}** — {style['role']}")
        return

    if kind == "escalate":
        target.markdown("**Escalating** — the loop's exit condition is met.")
        return

    # kind == "text"
    text = (event.get("content_text") or "").strip()
    if not text:
        return
    if event.get("partial"):
        # Streaming chunks are fragments of a sentence. The final, non-partial event
        # carries the whole thing, so rendering both would print the turn twice.
        return
    prose, code = split_prose_and_code(text)
    if prose:
        target.markdown(_truncate(prose, TEXT_PREVIEW_CHARS))
    if code:
        target.code(_truncate(code, CODE_PREVIEW_CHARS), language="python")


def _render_args(container: Any, args: dict[str, Any]) -> None:
    """Show tool arguments, with kernel sources as syntax-highlighted code."""
    scalars = {}
    for key, value in args.items():
        if isinstance(value, str) and _looks_like_code(value):
            container.markdown(f"`{key}`")
            container.code(_truncate(value, CODE_PREVIEW_CHARS), language="python")
        else:
            scalars[key] = value
    if scalars:
        container.json(scalars)


def _render_response_highlights(target: Any, name: str, payload: Any) -> None:
    """Surface the two responses the audience is meant to read, as metrics.

    Everything else stays folded inside its expander. A verdict rendered as raw JSON is
    unreadable on a recording; the same three numbers as `st.metric` are the shot.
    """
    if not isinstance(payload, dict):
        return
    if set(payload) == {"result"} and isinstance(payload["result"], dict):
        payload = payload["result"]

    if name == "verify_kernel":
        from kernelsmith.config import CORRECTNESS_SEEDS, CORRECTNESS_SHAPES

        checks = CORRECTNESS_SEEDS * len(CORRECTNESS_SHAPES)
        passed = payload.get("correctness_pass")
        columns = target.columns(3)
        columns[0].metric(
            "Correctness",
            f"{checks}/{checks} ✅" if passed else f"failed ({checks} checks)",
        )
        speedup = _as_float(payload.get("speedup_vs_eager"))
        columns[1].metric("Speedup", f"{speedup:.2f}×" if speedup else "—")
        reward = payload.get("reward")
        columns[2].metric("Reward", f"{reward:+d}" if isinstance(reward, int) else "—")
    elif name == "hotswap_kernel" and payload.get("success"):
        stats = payload.get("stats") or {}
        target.markdown(
            f"Kernel deployed to the inference server — "
            f"**{payload.get('modules_patched', '?')} modules patched**"
            + (
                f", now at **{_as_float(stats.get('tokens_per_s')):.0f} tokens/s**"
                if stats.get("tokens_per_s")
                else ""
            )
        )


def open_turn(container: Any, author: str, headline: str, *, running: bool) -> Any:
    """Create the `st.status` for one agent turn and return its handle."""
    style = style_for(author)
    label = f"{style['emoji']} {author} — {headline}"
    return container.status(
        label,
        state="running" if running else "complete",
        expanded=True,
    )


def render_turn(container: Any, turn: Turn, *, running: bool) -> Any:
    """Draw a whole finished turn: one status, then every event inside it."""
    status = open_turn(container, turn.author, turn_headline(turn), running=running)
    for event in turn.events:
        render_event(status, event)
    if turn.duration_s > 0:
        status.caption(f"{turn.duration_s:.1f}s")
    return status


def render_timeline(container: Any, events: list[dict[str, Any]], *, running: bool) -> None:
    """Draw the whole timeline from scratch — the live path, once per rerun."""
    turns = group_turns(events)
    for index, turn in enumerate(turns):
        last = index == len(turns) - 1
        render_turn(container, turn, running=running and last)


# --------------------------------------------------------------------------- #
# Header, graph, banners
# --------------------------------------------------------------------------- #


def render_header(metrics: dict[str, Any]) -> None:
    """Title on the left, four metric cards on the right."""
    title, cards = st.columns([1.35, 2])
    with title:
        st.markdown(
            '<p class="ks-title">KernelSmith</p>'
            '<p class="ks-sub">GPU Kernel Optimization Agent</p>',
            unsafe_allow_html=True,
        )
    with cards:
        speed, reward, iteration, tokens = st.columns(4)
        speed.metric(
            "Speedup",
            f"{metrics['speedup']:.2f}×" if metrics["speedup"] else "—",
            help="vs eager PyTorch, measured by the verifier",
        )
        reward.metric(
            "Reward",
            f"{metrics['reward']:+d}" if isinstance(metrics["reward"], int) else "—",
        )
        from kernelsmith.config import MAX_LOOP_ITERATIONS

        iteration.metric("Iteration", f"{metrics['iteration']}/{MAX_LOOP_ITERATIONS}")
        tokens.metric(
            "Tokens/s",
            f"{metrics['tokens_per_s']:.0f}" if metrics["tokens_per_s"] else "—",
        )


def render_graph(active: str | None) -> None:
    """The agent tree, with whoever is currently speaking highlighted in amber."""
    st.markdown("##### Agent tree")
    st.graphviz_chart(build_agent_graph(active), use_container_width=True)
    if active:
        style = style_for(active)
        st.markdown(
            f"<span class='ks-agent' style='color:{style['color']}'>{style['emoji']} {active}</span>"
            f" — {style['role']}",
            unsafe_allow_html=True,
        )
    else:
        st.caption("Idle — press Start Run.")


def render_banners(metrics: dict[str, Any]) -> None:
    """The bottom bar: the anti-reward-hack alert and the honest +1 note."""
    violations = metrics.get("violations")
    if violations:
        st.error(f"🚫 ANTI-REWARD-HACK: static AST checker blocked {_violation_text(violations)}")
    if metrics.get("honest_plus_one"):
        st.info(
            "ℹ️ Reward +1: the kernel is correct but not faster. The MLP is "
            "compute-bound — Tensor Cores already dominate, so there is nothing for a "
            "hand-written kernel to reclaim. KernelSmith reports that rather than "
            "claiming a speedup."
        )


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


def maybe_celebrate(metrics: dict[str, Any]) -> None:
    """Balloons on the first successful hot-swap of the session, and only the first."""
    if st.session_state["celebrated"]:
        return
    if metrics.get("tokens_per_s"):
        st.session_state["celebrated"] = True
        st.balloons()


# --------------------------------------------------------------------------- #
# Live mode
# --------------------------------------------------------------------------- #


def poll_stats() -> None:
    """Fetch `/stats`. A dead inference server leaves the metric at '—'."""
    import httpx

    from kernelsmith.config import INFERENCE_HOST, INFERENCE_PORT

    try:
        response = httpx.get(
            f"http://{INFERENCE_HOST}:{INFERENCE_PORT}/stats", timeout=STATS_TIMEOUT_S
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:  # noqa: BLE001 — the server is optional for a rehearsal
        return
    if isinstance(payload, dict):
        st.session_state["stats"] = payload


def start_run(op_name: str, hidden_size: int) -> None:
    """Kick off turn 1, opening a fresh trace for it."""
    state = st.session_state
    try:
        runner = get_runner()
    except Exception as exc:  # noqa: BLE001 — a missing GPU must not blank the page
        state["run_error"] = f"could not build the agent tree — {type(exc).__name__}: {exc}"
        return

    run_id = f"demo-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    state["session_id"] = run_id
    state["timeline_events"] = []
    state["active_agent"] = None
    state["run_error"] = ""
    state["celebrated"] = False
    state["awaiting_followup"] = True

    try:
        state["trace_path"] = str(get_logger().start_trace(run_id))
    except Exception as exc:  # noqa: BLE001 — an unwritable trace dir must not stop a run
        state["trace_path"] = ""
        state["run_error"] = f"trace capture disabled — {type(exc).__name__}: {exc}"

    consumer = get_consumer()
    if not consumer.start_run(
        runner,
        state["user_id"],
        run_id,
        TURN_1.format(op_name=op_name),
        state_delta={"task_spec": {"op_name": op_name, "hidden_size": hidden_size}},
    ):
        state["run_error"] = "a run is already in flight"
        state["awaiting_followup"] = False
    state["runs_seen"] = consumer.runs_completed


def drive_run(consumer: Any) -> None:
    """Send the follow-up message once turn 1 ends.

    The same two-message protocol `streamlit_app.py` drives, for the same reason: an
    ADK `LoopAgent` cannot transfer back to its parent, so upsert and hot-swap run on
    the next turn. A demo dashboard that skipped this would record a run that never
    went live — which is precisely the shot the recording exists for.
    """
    state = st.session_state
    completed = consumer.runs_completed
    if completed <= state["runs_seen"] or consumer.is_running:
        return
    state["runs_seen"] = completed

    if consumer.last_error:
        state["run_error"] = consumer.last_error
        state["awaiting_followup"] = False
        get_logger().end_trace()
        return

    if not state["awaiting_followup"]:
        get_logger().end_trace()
        return

    state["awaiting_followup"] = False
    try:
        runner = get_runner()
    except Exception as exc:  # noqa: BLE001
        state["run_error"] = f"{type(exc).__name__}: {exc}"
        return
    consumer.start_run(runner, state["user_id"], state["session_id"], TURN_2)
    state["runs_seen"] = consumer.runs_completed


def ingest_events(consumer: Any) -> None:
    """Drain the queue into `session_state` as flat dicts.

    Converted here rather than stored raw: the timeline renders dicts, and a replayed
    trace has no ADK objects to offer it. One shape on screen, from both sources.
    """
    from kernelsmith.ui.event_capture import event_to_dict

    state = st.session_state
    for event in consumer.drain_events(limit=DRAIN_LIMIT):
        try:
            record = event_to_dict(event, _live_elapsed())
        except Exception as exc:  # noqa: BLE001 — a bad event is a line, not a blank page
            state["run_error"] = f"event processing failed: {type(exc).__name__}: {exc}"
            continue
        state["timeline_events"].append(record)
        if record["author"] in AGENT_STYLES:
            state["active_agent"] = record["author"]

    overflow = len(state["timeline_events"]) - MAX_TIMELINE_EVENTS
    if overflow > 0:
        del state["timeline_events"][:overflow]


def _live_elapsed() -> float:
    """Seconds since the trace started, or 0 when nothing is being recorded."""
    logger_ = get_logger()
    if logger_.start_time is None:
        return 0.0
    return time.monotonic() - logger_.start_time


def render_autorefresh() -> None:
    """Refresh at 1 Hz, or fall back to a button if the component is unavailable."""
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=REFRESH_MS, key="kernelsmith-demo-refresh")
    except Exception:  # noqa: BLE001 — a missing component is not a broken dashboard
        st.button("🔄 Refresh", key="manual-refresh")


def render_live() -> None:
    """The live view: drain, drive, render, refresh.

    Owns its own layout rather than being handed columns, because the header has to be
    laid out ABOVE them — Streamlit places elements in call order, so columns created
    in `main()` would put the metric cards underneath the timeline.
    """
    state = st.session_state
    consumer = get_consumer()
    ingest_events(consumer)
    # Before any early return, always: the two-message protocol depends on it.
    drive_run(consumer)
    poll_stats()

    metrics = extract_metrics(state["timeline_events"])
    if metrics["tokens_per_s"] is None and state["stats"].get("tokens_per_s"):
        metrics["tokens_per_s"] = _as_float(state["stats"]["tokens_per_s"]) or None

    render_header(metrics)
    st.divider()

    left, right = st.columns([1, 2.5])
    with left:
        render_graph(state["active_agent"] if consumer.is_running else None)
    with right:
        st.markdown("##### Agent timeline")
        if not state["timeline_events"]:
            st.info("Press **Start Run** in the sidebar. The timeline builds as the agents work.")
        render_timeline(st.container(), state["timeline_events"], running=consumer.is_running)

    st.divider()
    if state["run_error"]:
        st.warning(state["run_error"])
    if state["trace_path"]:
        st.caption(f"Recording to `{state['trace_path']}`")
    render_banners(metrics)
    maybe_celebrate(metrics)
    render_autorefresh()


# --------------------------------------------------------------------------- #
# Replay mode
# --------------------------------------------------------------------------- #


def render_replay(trace: Path | None, speed: float, play: bool) -> None:
    """The replay view. Renders progressively while playing, statically afterwards.

    No auto-refresh here: a 1 Hz rerun landing mid-playback would restart the trace from
    the top and the timeline would never finish building.
    """
    state = st.session_state
    if trace is None:
        st.info(
            f"No traces yet. Run one in Live mode, or drop a `.jsonl` in `{DEFAULT_TRACE_DIR}/`."
        )
        return

    events = load_events(trace)
    metrics = extract_metrics(events)

    if not play:
        # Static preview: the finished state, so the layout can be framed before the
        # camera rolls without sitting through a playback.
        render_header(metrics if state["replay_done"] else _blank_metrics())
        st.divider()
        left, right = st.columns([1, 2.5])
        with left:
            render_graph(None)
        with right:
            st.markdown("##### Agent timeline")
            if state["replay_done"]:
                render_timeline(st.container(), events, running=False)
            else:
                st.info(
                    f"**{trace.name}** — {len(events)} events, "
                    f"{total_duration_s(events):.1f}s. Press **▶ Play**."
                )
        st.divider()
        if state["replay_done"]:
            render_banners(metrics)
        return

    # --- playing -------------------------------------------------------------
    state["replay_done"] = False
    header_slot = st.empty()
    with header_slot.container():
        render_header(_blank_metrics())
    st.divider()

    left, right = st.columns([1, 2.5])
    with left:
        graph_slot = st.empty()
        caption_slot = st.empty()
        graph_slot.graphviz_chart(build_agent_graph(None), use_container_width=True)
    with right:
        st.markdown("##### Agent timeline")
        timeline = st.container()

    seen: list[dict[str, Any]] = []
    status = None
    current_author: str | None = None
    turn_start = 0.0

    for event in pace_events(events, speed):
        seen.append(event)
        author = str(event.get("author") or "unknown")

        if status is None or author != current_author:
            if status is not None:
                status.update(state="complete")
            current_author = author
            turn_start = float(event.get("elapsed_s") or 0.0)
            # The headline needs the turn's decisive event, which has not arrived yet;
            # a one-event Turn gives the best label available at this instant, and the
            # update below rewrites it once the turn's real decision is known.
            status = open_turn(timeline, author, turn_headline(Turn(author, [event])), running=True)
            if author in {name for name, _ in GRAPH_NODES}:
                graph_slot.graphviz_chart(build_agent_graph(author), use_container_width=True)
                style = style_for(author)
                caption_slot.markdown(
                    f"<span class='ks-agent' style='color:{style['color']}'>"
                    f"{style['emoji']} {author}</span> — {style['role']}",
                    unsafe_allow_html=True,
                )
        else:
            # Relabel as the turn reveals what it was actually doing.
            turn = Turn(author, [e for e in seen if e.get("author") == author])
            style = style_for(author)
            status.update(label=f"{style['emoji']} {author} — {turn_headline(turn)}")

        render_event(status, event)
        with header_slot.container():
            render_header(extract_metrics(seen))

    if status is not None:
        elapsed = float(seen[-1].get("elapsed_s") or 0.0) - turn_start
        if elapsed > 0:
            status.caption(f"{elapsed:.1f}s")
        status.update(state="complete")

    # Back to the idle tree, and clear the "who is speaking" caption with it — a
    # highlight left behind after the run points the audience at an agent that
    # finished twenty seconds ago.
    graph_slot.graphviz_chart(build_agent_graph(None), use_container_width=True)
    caption_slot.caption("Replay complete.")
    with header_slot.container():
        render_header(metrics)

    st.divider()
    render_banners(metrics)
    maybe_celebrate(metrics)
    state["replay_done"] = True
    state["replaying"] = False


def _blank_metrics() -> dict[str, Any]:
    """Header metrics before anything has been measured — all dashes, no zeros."""
    return {
        "speedup": None,
        "speedup_vs_compile": None,
        "reward": None,
        "iteration": 0,
        "tokens_per_s": None,
        "correct": None,
        "violations": None,
        "honest_plus_one": False,
    }


# --------------------------------------------------------------------------- #
# Sidebar + main
# --------------------------------------------------------------------------- #


def render_sidebar() -> dict[str, Any]:
    """Mode switch and its controls. Returns what the chosen mode needs."""
    st.sidebar.markdown("### KernelSmith")
    mode = st.sidebar.radio("Mode", MODES, key="mode")
    st.sidebar.divider()

    controls: dict[str, Any] = {"mode": mode}
    if mode == MODE_LIVE:
        op_name = st.sidebar.text_input("Op", value=DEFAULT_OP)
        hidden_size = st.sidebar.number_input(
            "Hidden size", value=DEFAULT_HIDDEN_SIZE, min_value=1, step=64
        )
        controls["op_name"] = op_name
        controls["hidden_size"] = int(hidden_size)
        if st.sidebar.button("▶ Start Run", type="primary", use_container_width=True):
            start_run(op_name, int(hidden_size))
        st.sidebar.caption(f"Trace → `{DEFAULT_TRACE_DIR}/`")
    else:
        traces = list_traces(DEFAULT_TRACE_DIR)
        if traces:
            names = [path.name for path in traces]
            chosen = st.sidebar.selectbox("Trace", names, index=0)
            controls["trace"] = next(path for path in traces if path.name == chosen)
        else:
            controls["trace"] = None
        label = st.sidebar.select_slider("Speed", list(SPEED_CHOICES), value="1×")
        controls["speed"] = SPEED_CHOICES[label]
        controls["play"] = st.sidebar.button(
            "▶ Play", type="primary", use_container_width=True, disabled=controls["trace"] is None
        )
    return controls


def main() -> None:
    st.set_page_config(
        page_title="KernelSmith Demo",
        page_icon="⚒️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    controls = render_sidebar()

    if controls["mode"] == MODE_LIVE:
        render_live()
    else:
        render_replay(controls.get("trace"), controls["speed"], controls["play"])


main()
