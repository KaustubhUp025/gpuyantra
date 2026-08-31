"""The recording dashboard (Task 12) — a separate app from `streamlit_app.py`.

NAMING (Task 13b). On screen this is **gpuyantra**, the project; **KernelSmith** is the
agent tree inside it that does the work. So the page title, the header and the sidebar
mark say gpuyantra, while anything naming who profiles, writes, verifies or deploys says
KernelSmith — including the agent-map nodes and the chart's kernel bar, which are
KernelSmith internals and correctly labelled as such. The Python package, its modules and
the Firestore collections stay `kernelsmith`: code is not a judge-facing surface.

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

Task 12b turned the timeline from developer output into something a first-time viewer can
follow: `format_event_for_display` gives every event a headline and two or three
sentences of plain English, the raw JSON moves behind a per-event "Show raw" fold,
`narrative_after` explains between turns *why* the next step follows, and the page opens
in Replay mode wherever no inference server answers — which is the hosted Cloud Run case,
where the trace is baked into the container and a judge only has to press Play.

The one rule that survived that rewrite unchanged: a sentence may only interpolate a
number the payload actually carried. Where a friendlier template wanted a figure the tool
did not return, the sentence was rewritten instead of filled in from the audit tables.

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
    "NARRATIVE",
    "Turn",
    "build_agent_graph",
    "classify_event",
    "default_mode",
    "extract_metrics",
    "format_event_for_display",
    "group_turns",
    "highlight_active_agent",
    "is_noise_event",
    "iteration_label",
    "narrative_after",
    "ordered_traces",
    "split_prose_and_code",
    "trace_summary",
    "turn_headline",
    "turn_label",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REFRESH_MS = 1000
DRAIN_LIMIT = 200
STATS_TIMEOUT_S = 1.0
#: The health probe that decides whether Live mode is even possible (Task 12b, problem 6).
#: Short: it runs on every rerun of a page that has to stay at 1 Hz.
HEALTH_TIMEOUT_S = 0.6
HEALTH_CACHE_S = 10
#: Bounded so a long rehearsal cannot exhaust the browser session.
MAX_TIMELINE_EVENTS = 1500
#: Kernel sources reach four figures of characters; the timeline shows the shape of one.
CODE_PREVIEW_CHARS = 700
TEXT_PREVIEW_CHARS = 600

MODE_LIVE = "🔴 Live"
MODE_REPLAY = "📼 Replay"
MODES = (MODE_LIVE, MODE_REPLAY)

SPEED_CHOICES = {"0.5×": 0.5, "1×": 1.0, "2×": 2.0, "instant": 0.0}

#: ADK's built-in delegation tool. Its *response* carries nothing but `{"result": null}`,
#: so the timeline renders the call and drops the echo (see `is_noise_event`).
TRANSFER_TOOL = "transfer_to_agent"

#: The hand-written fixture. Real captured traces sort ahead of it in the picker.
SAMPLE_TRACE_NAME = "sample_run.jsonl"

#: Static narration, keyed by what a turn just finished doing (Task 12b, problem 4).
#: These do not depend on any measured value — they say why the next step matters, which
#: is the one thing the event stream itself cannot tell a first-time viewer.
NARRATIVE: dict[str, str] = {
    "profiled": (
        "That is the slow part found — and found by measuring, not by guessing. Before "
        "asking anyone to write new code, the Supervisor checks whether this system has "
        "already solved a problem shaped like this one."
    ),
    "retrieved": (
        "The code it found goes to the Coder as a starting point. This is the memory "
        "working: the system gets better at an operation each time it meets one like it."
    ),
    "kernel_written": (
        "The Coder has written the GPU code. It does not get to mark its own homework — "
        "a separate agent now checks whether the code is correct, whether it is really "
        "faster, and whether it cheated to look faster."
    ),
    "verified": (
        "The new code has been tested and scored. Next, a plain piece of code — no AI "
        "involved — decides whether that score is good enough to stop, or whether the "
        "Coder should try again."
    ),
    "escalated": (
        "The loop is done. What follows is the part that makes this more than a "
        "benchmark: the winning code is saved for future runs and loaded into the "
        "language-model server that is already running and answering requests."
    ),
    "swapped": (
        "The new code is now serving real requests, in a server that was never "
        "restarted. If its answers had drifted from the original, the server would have "
        "put the old code back by itself."
    ),
}

#: Colour, emoji and a one-line job description per agent. The colours are the ones the
#: task brief specifies; the job descriptions are written for someone who has never heard
#: of a "Profiler" and should not have to guess from the name.
AGENT_STYLES: dict[str, dict[str, str]] = {
    "Supervisor": {"color": "#2563eb", "emoji": "🎯", "role": "runs the whole process"},
    "Profiler": {"color": "#10b981", "emoji": "🔍", "role": "finds the slowest part"},
    "Coder": {"color": "#f59e0b", "emoji": "💻", "role": "writes the GPU code"},
    "Judge": {"color": "#ef4444", "emoji": "⚖️", "role": "tests it, independently"},
    "EscalationChecker": {"color": "#8b5cf6", "emoji": "🛑", "role": "decides when to stop"},
    "RefinementLoop": {"color": "#6b7280", "emoji": "🔁", "role": "write → test → try again"},
    "user": {"color": "#6b7280", "emoji": "👤", "role": "the person driving this"},
}
DEFAULT_STYLE = {"color": "#6b7280", "emoji": "⚙️", "role": ""}

#: Nodes of the agent tree: (id, shape, what it says on screen). The id is the ADK agent
#: name — events arrive under it and `highlight_active_agent` matches on it — while the
#: label is what a viewer reads. Two lines per node: who, then what they do.
#:
#: No "]" in a label, ever: `highlight_active_agent` recolours a node by rewriting inside
#: its `[...]` attribute list, and a bracket in a label would end that match early.
GRAPH_NODES = (
    ("Supervisor", "box", "Supervisor\\nruns the process"),
    ("Profiler", "box", "Profiler\\nfinds the slow part"),
    ("RefinementLoop", "box3d", "Write → test loop\\nup to 6 attempts"),
    ("Coder", "box", "Coder\\nwrites GPU code"),
    ("Judge", "box", "Judge\\ntests it"),
    ("EscalationChecker", "box", "Quality checker\\ndecides when to stop"),
)
GRAPH_EDGES = (
    ("Supervisor", "Profiler", "delegates to"),
    ("Supervisor", "RefinementLoop", "delegates to"),
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
  .block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 100%; }

  /* Video cleanliness: nothing on screen that is not the product. The Deploy button
     and the toolbar are the two that show up in a 1080p capture of the top-right. */
  #MainMenu,
  footer,
  header [data-testid="stStatusWidget"],
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stAppDeployButton"] { display: none !important; }

  /* --- hierarchy (Task 13, problem 7) ------------------------------------- */
  /* The header metrics are the loudest thing on the page, and they are cards rather
     than four numbers floating in the background. */
  [data-testid="stMetric"] {
      background: rgba(255, 255, 255, .035);
      border: 1px solid rgba(255, 255, 255, .09);
      border-radius: 10px;
      padding: .7rem .85rem .55rem .85rem;
  }
  div[data-testid="stMetricValue"] {
      font-size: 1.9rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      line-height: 1.15;
  }
  div[data-testid="stMetricLabel"] p {
      font-size: .82rem; letter-spacing: .06em; text-transform: uppercase; opacity: .72;
  }
  div[data-testid="stMetricDelta"] { font-size: .8rem; }

  /* Section headings: quiet, so they organize without competing with the numbers. */
  h5 { font-size: .84rem !important; letter-spacing: .1em; text-transform: uppercase;
       opacity: .55; margin: .2rem 0 .55rem 0 !important; font-weight: 600; }

  /* --- the timeline ------------------------------------------------------- */
  /* st.status labels are the timeline's headlines — they carry the demo. */
  details[data-testid="stExpander"] summary p,
  div[data-testid="stExpander"] summary p { font-size: 1.06rem; font-weight: 600; }

  /* One step = one card, with room to breathe. Streamlit renders st.status as an
     expander, so this is what stops the timeline reading as one long grey wall. */
  div[data-testid="stExpander"] {
      border-radius: 10px;
      border-color: rgba(255, 255, 255, .10) !important;
      margin-bottom: .5rem;
      background: rgba(255, 255, 255, .02);
  }
  /* ... except the folds NESTED inside a step ("Show the exact data behind this"),
     which should read as a footnote to the step, not as a step of their own. */
  div[data-testid="stExpander"] div[data-testid="stExpander"] {
      background: transparent;
      border-color: rgba(255, 255, 255, .07) !important;
  }
  div[data-testid="stExpander"] div[data-testid="stExpander"] summary p {
      font-size: .84rem; font-weight: 500; opacity: .65; text-transform: none;
  }

  .ks-title { font-size: 2.3rem; font-weight: 700; line-height: 1.1; margin: 0;
              letter-spacing: -.01em; }
  .ks-sub   { font-size: .97rem; opacity: .68; margin: .3rem 0 0 0; line-height: 1.5;
              max-width: 62ch; }
  .ks-agent { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
              font-weight: 700; }
  .ks-elapsed { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                opacity: .6; font-size: .9rem; }

  /* The narration between two agent turns. Indented and rule-marked so it reads as
     the dashboard talking, not as something an agent said. */
  .ks-narrative { border-left: 3px solid #3b82f6; padding: .15rem 0 .15rem .8rem;
                  margin: .1rem 0 1.1rem .2rem; font-size: .93rem; opacity: .8;
                  line-height: 1.5; }
  /* Plain-English detail under an event headline. */
  .ks-detail { font-size: .93rem; opacity: .8; line-height: 1.55;
               margin: .15rem 0 .55rem 0; }

  code, pre, .stCode { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  /* Inline code inside a plain-English sentence should not shout. */
  .ks-detail code, .ks-narrative code {
      background: rgba(255, 255, 255, .07); padding: .05rem .3rem; border-radius: 4px;
      font-size: .88em; color: #e5e7eb;
  }
  [data-testid="stCaptionContainer"] p { line-height: 1.55; }
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
        "  nodesep=0.45",
        "  ranksep=0.55",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica,Arial,sans-serif",'
        ' fontsize=12, margin="0.22,0.14", penwidth=0]',
        '  edge [fontname="Helvetica,Arial,sans-serif", fontsize=10, color="#6b7280",'
        ' fontcolor="#9ca3af", arrowsize=0.8, penwidth=1.1]',
    ]
    for name, shape, label in GRAPH_NODES:
        lines.append(_node_line(name, shape, label, active))
    for source, target, label in GRAPH_EDGES:
        suffix = f' [label="{label}"]' if label else ""
        lines.append(f"  {source} -> {target}{suffix}")
    lines.append("}")
    return "\n".join(lines)


def _node_line(name: str, shape: str, label: str, active: str | None) -> str:
    """One node's DOT line. The single definition of how a node looks in either state.

    Both `build_agent_graph` and `highlight_active_agent` go through here, so the two can
    never drift into drawing the same agent differently — which on screen would read as
    two different agents being active.

    The active node gets a light outline as well as the amber fill: on a dark theme at
    video bitrates a fill change alone is easy to miss, and which agent is working is the
    one thing this diagram exists to say.
    """
    if name == active:
        fill, text, outline, width = NODE_ACTIVE_COLOR, "#111827", "#fef3c7", 2
    else:
        fill = NODE_ROOT_COLOR if (name == "Supervisor" and active is None) else NODE_IDLE_COLOR
        text, outline, width = "white", "#00000000", 0
    return (
        f'  {name} [shape={shape}, fillcolor="{fill}", fontcolor="{text}", '
        f'color="{outline}", penwidth={width}, label="{label}"]'
    )


def highlight_active_agent(dot: str, agent: str | None) -> str:
    """Recolour one node of an existing DOT string, leaving every other line untouched.

    Separate from `build_agent_graph` because the two answer different questions:
    building is "what is the tree", highlighting is "who is talking". Rewriting a DOT
    that has already been built is what the live path does on every event, and doing it
    textually means the tree's shape has exactly one definition.

    An `agent` that is not in the graph returns the DOT unchanged — the Supervisor's
    turn-2 events name agents that are not nodes, and that must not blank the diagram.
    """
    result = dot
    for name, shape, label in GRAPH_NODES:
        result = re.sub(
            rf"^\s*{re.escape(name)}\s*\[[^\]]*\]",
            lambda _match, n=name, sh=shape, la=label: _node_line(n, sh, la, agent),
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
        "tokens_source": None,
        "tokens_before_swap": None,
        "correct": None,
        "violations": None,
        "honest_plus_one": False,
        "escalated": False,
        "hotswap_ok": None,
        "hotswap_error": None,
        "modules_patched": None,
        "skill_id": None,
        "latency_ms_by_shape": None,
        "baseline_ms": None,
        "headline_shape": None,
        "checks": None,
    }
    for event in events:
        if event.get("event_type") == "escalate" or event.get("escalate"):
            metrics["escalated"] = True
        for call in event.get("function_calls") or []:
            if not isinstance(call, dict) or call.get("name") != "upsert_skill":
                continue
            skill = (call.get("args") or {}).get("skill_data")
            if isinstance(skill, dict) and skill.get("skill_id"):
                metrics["skill_id"] = skill["skill_id"]
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
                if isinstance(payload.get("passed_checks"), int) and isinstance(
                    payload.get("total_checks"), int
                ):
                    metrics["checks"] = (payload["passed_checks"], payload["total_checks"])
                metrics["latency_ms_by_shape"] = payload.get("latency_ms_by_shape") or None
                metrics["baseline_ms"] = payload.get("baseline_ms") or None
                metrics["headline_shape"] = payload.get("headline_shape") or None
            elif name == "hotswap_kernel":
                if payload.get("success"):
                    metrics["hotswap_ok"] = True
                    metrics["hotswap_error"] = None
                    metrics["modules_patched"] = payload.get("modules_patched")
                    stats = payload.get("stats") or {}
                    tokens = _as_float(stats.get("tokens_per_s")) or None
                    if tokens:
                        metrics["tokens_per_s"] = tokens
                        metrics["tokens_source"] = "swap"
                    # A before-reading, if whoever wrote the trace captured one. The
                    # server does not return this today, so it is read defensively and
                    # never synthesised: no key, no baseline, no delta on the card.
                    before = _as_float(
                        stats.get("tokens_per_s_before") or payload.get("tokens_per_s_before")
                    )
                    if before:
                        metrics["tokens_before_swap"] = before
                else:
                    metrics["hotswap_ok"] = False
                    metrics["hotswap_error"] = payload.get("error")
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
# Plain English (Task 12b, problem 1)
# --------------------------------------------------------------------------- #
#
# A judge watching a four-minute video cannot read a JSON dump. Every event therefore
# gets a headline and two or three sentences of explanation, and the raw payload moves
# behind a "Show raw" toggle — kept, because the credibility of this project rests on
# the numbers being checkable, but never the default view.
#
# The rule the whole section obeys: **every interpolated number comes out of the
# payload.** Red line #3 does not stop at the header metrics. Where a template would
# read better with a figure the tool did not return (how many modules a profile covers,
# what fraction of the L4's bandwidth it reached), the sentence is written without it
# rather than filled in from the audit table or from memory.


def unwrap_payload(payload: Any) -> Any:
    """Unwrap ADK's `{"result": ...}` FunctionTool envelope when that is all there is."""
    if isinstance(payload, dict) and set(payload) == {"result"}:
        return payload["result"]
    return payload


def is_noise_event(event: dict[str, Any]) -> bool:
    """Whether an event carries nothing a viewer should be shown.

    Three kinds, all of them real and all of them in the committed L4 trace:

    - the `transfer_to_agent` *response*, whose whole payload is `{"result": null}` —
      the call one line above it already said where control went;
    - a `text` event with no text (ADK emits one to close a streamed turn);
    - a `partial` streaming chunk, which the final event repeats in full.
    """
    if event.get("partial"):
        return True
    kind = event.get("event_type")
    if kind == "function_response":
        responses = event.get("function_responses") or []
        return bool(responses) and all(
            str(response.get("name") or "") == TRANSFER_TOOL for response in responses
        )
    if kind == "text":
        return not (event.get("content_text") or "").strip()
    return False


def format_event_for_display(event: dict[str, Any]) -> dict[str, Any]:
    """One event as `{headline, detail, raw}` — plus what the renderer needs to draw it.

    Always returns those three keys. `raw` is the payload the headline was derived from
    (the tool's args or response, the message text), so "Show raw" shows the source of
    the sentence above it rather than the whole ADK envelope.

    Four optional keys tell the renderer what kind of body to draw, because a kernel and
    a summary do not belong in the same widget: `code` (render as Python), `prose`
    (render as markdown), `verdict` (render as metric cards), `summary` (render as the
    end-of-run card). Absent means "no body, the detail says it all".
    """
    kind = event.get("event_type")
    author = str(event.get("author") or "unknown")

    if kind == "function_call":
        call = _first(event.get("function_calls"))
        name = str(call.get("name") or "")
        args = call.get("args") or {}
        if name == TRANSFER_TOOL:
            return _delegation(str(args.get("agent_name") or event.get("transfer_to") or "?"), args)
        return _describe_call(name, args)

    if kind == "function_response":
        response = _first(event.get("function_responses"))
        name = str(response.get("name") or "")
        payload = unwrap_payload(response.get("response"))
        if name == TRANSFER_TOOL:
            return _delegation(str(event.get("transfer_to") or "?"), payload)
        return _describe_response(name, payload)

    if kind == "transfer":
        return _delegation(str(event.get("transfer_to") or "?"), {})

    if kind == "escalate":
        return {
            "headline": "🛑 Good enough — stopping the loop",
            "detail": (
                "The score clears the bar, so the loop stops here instead of spending more "
                "attempts (and more model calls) trying to improve on it. This decision is "
                "made by ordinary code, not by a language model, so it cannot be talked "
                "out of it. The result is also recorded against the starting point that "
                "was used, so future runs know which starting points pay off."
            ),
            "raw": event.get("state_delta") or {},
        }

    return _describe_text(author, (event.get("content_text") or "").strip())


def _first(items: Any) -> dict[str, Any]:
    """The first dict of a list, or an empty one. Traces come off disk; nothing is trusted."""
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                return item
    return {}


def _delegation(target: str, raw: Any) -> dict[str, Any]:
    """A hand-off. The detail names what the receiving agent is being asked for."""
    reasons = {
        "Profiler": (
            "The Supervisor asks the Profiler to find which part of the model is slow. It "
            "measures the model running on the GPU rather than guessing from the code."
        ),
        "RefinementLoop": (
            "Now the write-and-test loop starts: the Coder writes GPU code, the Judge "
            "tests it, and a checker decides whether another attempt is worth it. The "
            "number of attempts is capped, so it cannot loop forever."
        ),
        "Coder": "The Coder is asked to write new GPU code for the slow operation.",
        "Judge": (
            "The Judge is asked to test the new code. It is a different agent from the one "
            "that wrote it, on purpose — nobody marks their own homework here."
        ),
        "Supervisor": (
            "Back to the Supervisor, which does the rest: look up past work, run the loop, "
            "save the result, and put it into the live server."
        ),
    }
    style = style_for(target)
    return {
        "headline": f"🎯 Delegating to {target}",
        "detail": reasons.get(
            target, f"Handing control to {target} — {style.get('role', 'a sub-agent')}."
        ),
        "raw": raw if isinstance(raw, dict) else {"transfer_to": target},
    }


def _checks_text() -> str:
    """How many correctness checks there are, spelled out — read from config, not typed.

    "15 checks: 5 sets of random numbers at 3 different input sizes" rather than
    "5 seeds × 3 shapes". Same fact; one of them can be read by someone who has never
    written a test with a random seed in it.
    """
    try:
        from kernelsmith.config import CORRECTNESS_SEEDS, CORRECTNESS_SHAPES

        shapes = len(CORRECTNESS_SHAPES)
        return (
            f"{CORRECTNESS_SEEDS * shapes} checks: {CORRECTNESS_SEEDS} sets of random "
            f"numbers at {shapes} different input sizes"
        )
    except Exception:  # noqa: BLE001 — a missing project env must not blank the timeline
        return "15 checks of random inputs at 3 different input sizes"


#: What the ops this system can optimize actually DO, for a viewer who has never seen
#: the inside of a transformer. Anything not listed here is named without a gloss rather
#: than described wrongly.
OP_IN_WORDS = {
    "rmsnorm": "the step that rescales the numbers flowing between layers",
    "rms_norm": "the step that rescales the numbers flowing between layers",
    "layernorm": "the step that rescales the numbers flowing between layers",
    "norm": "the step that rescales the numbers flowing between layers",
    "swiglu": "the feed-forward block, the widest part of each layer",
    "mlp": "the feed-forward block, the widest part of each layer",
    "softmax": "the step that turns attention scores into weights",
}


def _in_words(op: str) -> str:
    """ " (the step that …)" for a known op, or nothing at all for one we cannot name."""
    meaning = OP_IN_WORDS.get(str(op).strip().lower())
    return f" — {meaning}" if meaning else ""


def _describe_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """A tool call, in the words of what it is about to do."""
    raw = dict(args)

    if name in {"profile_op", "profile_op_by_name"}:
        op = args.get("op_name") or args.get("op_family") or "the op"
        shape = _shape_text(args)
        return {
            "headline": f"🔍 Measuring {op}",
            "detail": (
                f"`{op}`{_in_words(op)}. Timing it on the GPU{shape} to find out what is "
                "holding it back: moving data around, or doing the actual maths. The ones "
                "held back by data movement are the ones new code can speed up."
            ),
            "raw": raw,
        }

    if name in {"retrieve_skills", "retrieve_skills_for_agent"}:
        fingerprint = args.get("fingerprint_text") or "the bottleneck fingerprint"
        return {
            "headline": "📚 Searching skill library",
            "detail": (
                "Looking through everything this system has written before for a similar "
                "problem. It searches by *why* an operation is slow — the description "
                f"`{fingerprint}` — and not by its name, which is what lets a solution "
                "found for one model be reused on a completely different one."
            ),
            "raw": raw,
        }

    if name == "verify_kernel":
        code = str(args.get("kernel_code") or "")
        return {
            "headline": "⚖️ Testing the new code",
            "detail": (
                f"Three questions, in order. Does it give the same answers as PyTorch "
                f"({_checks_text()})? Is it genuinely faster, timed "
                "against both normal PyTorch and PyTorch's own compiler? And did it cheat "
                "— for example by quietly calling PyTorch and taking the credit? The code "
                "under test runs in a separate, locked-down process, never in this one."
            ),
            "raw": raw,
            "code": code,
        }

    if name == "upsert_skill":
        skill = (args.get("skill_data") or {}) if isinstance(args.get("skill_data"), dict) else {}
        skill_id = skill.get("skill_id") or args.get("skill_id") or "a new skill"
        return {
            "headline": "💾 Saving to skill library",
            "detail": (
                f"Filing the tested code away as `{skill_id}`, labelled by the kind of "
                "slowness it fixes, so a later run that hits something similar can start "
                "from it instead of from a blank page."
            ),
            "raw": raw,
        }

    if name == "hotswap_kernel":
        op = args.get("op_name") or "the op"
        mapping = args.get("adapter_mapping") or {}
        contract = (
            " The wiring the Coder wrote for itself: "
            + ", ".join(f"`{key}`→`{value}`" for key, value in mapping.items())
            + "."
            if isinstance(mapping, dict) and mapping
            else ""
        )
        return {
            "headline": "🔄 Loading the new code into the live server",
            "detail": (
                f"Swapping the new code into every `{op}` step of the language model that "
                "is already loaded and answering requests — no restart and no redeploy. "
                "The server checks the answers still match the old code first, and puts "
                f"the old code back if they do not.{contract}"
            ),
            "raw": raw,
        }

    if name == "explain_kernel":
        return {
            "headline": "📝 Asking for a plain-English explanation",
            "detail": (
                "A second, larger model (Gemma) is asked to explain in plain language what "
                "the new code does and why it is faster. This is a bonus step — if it "
                "fails, the run still counts."
            ),
            "raw": raw,
        }

    return {
        "headline": f"🔧 Calling {name or 'a tool'}",
        "detail": "",
        "raw": raw,
    }


def _shape_text(args: dict[str, Any]) -> str:
    """ " at 8×512×1536" from whatever shape arguments the call happened to carry."""
    batch, seq, hidden = args.get("batch"), args.get("seq_len"), args.get("hidden_size")
    parts = [str(value) for value in (batch, seq, hidden) if isinstance(value, int)]
    return f" at {'×'.join(parts)}" if len(parts) >= 2 else ""


def _describe_response(name: str, payload: Any) -> dict[str, Any]:
    """A tool result, in the words of what it found.

    Two of these tools return a bare string inside ADK's `{"result": ...}` envelope —
    `upsert_skill` returns `"upserted"`, `explain_kernel` returns the whole explanation —
    so they are handled before the dict guard below. Sending them through it printed
    "Result from explain_kernel" with two thousand words of Gemma as the subtitle.
    """
    if name == "upsert_skill":
        return {
            "headline": "💾 Saved to the skill library",
            "detail": (
                "Saved. The next time this system meets an operation that is slow for the "
                "same reason — in this model or in a completely different one — it will "
                "find this code and start from it."
            ),
            "raw": payload if isinstance(payload, dict) else {"result": payload},
        }
    if name == "explain_kernel":
        text = (
            str(payload.get("result") or payload.get("explanation") or "")
            if isinstance(payload, dict)
            else str(payload or "")
        )
        return {
            "headline": "📝 Explanation ready",
            "detail": (
                f"{len(text.split())} words back from Gemma on why the new code is faster. "
                "This is the one step of the run whose output is meant for a human to read "
                "rather than for a machine to check."
            ),
            "raw": {"result": text},
            "prose": text,
        }
    if not isinstance(payload, dict):
        return {
            "headline": f"↩︎ Result from {name or 'a tool'}",
            "detail": "" if payload in (None, "") else str(payload)[:300],
            "raw": {"response": payload},
        }
    raw = dict(payload)

    if name in {"profile_op", "profile_op_by_name"}:
        return _describe_profile(payload, raw)
    if name in {"retrieve_skills", "retrieve_skills_for_agent"}:
        return _describe_retrieval(payload, raw)
    if name == "verify_kernel":
        return _describe_verdict(payload, raw)
    if name == "hotswap_kernel":
        return _describe_hotswap(payload, raw)

    return {"headline": f"↩︎ Result from {name or 'a tool'}", "detail": "", "raw": raw}


#: The families the profiler reports, in words. Unknown families keep their own name —
#: better an unfamiliar word than a confidently wrong description.
FAMILY_IN_WORDS = {
    "norm": "Normalization",
    "mlp": "The feed-forward block",
    "swiglu": "The feed-forward block",
    "attention": "Attention",
    "elementwise": "This element-by-element step",
    "reduction": "This summing step",
}


def _describe_profile(payload: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    """What the measurement found, said the way you would say it out loud."""
    raw_family = payload.get("op_family") or payload.get("module_type") or "the op"
    family = FAMILY_IN_WORDS.get(str(raw_family).strip().lower(), str(raw_family))
    intensity = _as_float(payload.get("arithmetic_intensity"))
    bandwidth = _as_float(payload.get("memory_throughput_gbps"))
    ridge = _as_float(payload.get("ridge_point_flops_per_byte") or payload.get("ridge_point"))
    memory_bound = bool(payload.get("is_memory_bound"))
    tile = payload.get("tile_size_hint")

    regime = "waiting on memory" if memory_bound else "busy doing maths"
    headline = f"📊 {family} is {regime}"
    if intensity:
        headline += f" — {intensity:.2f} calculations per byte moved"

    sentences = []
    if intensity and ridge:
        sentences.append(
            f"For every byte it reads, this operation does only {intensity:.2f} "
            f"calculations. This GPU needs about {ridge:.0f} before its maths units become "
            f"the limit, so the chip is idle roughly {ridge / intensity:.0f}× longer than "
            "it needs to be, waiting for data to arrive."
        )
    if bandwidth:
        sentences.append(
            f"It is moving {bandwidth:.1f} GB of data per second, out of the roughly 300 "
            "GB/s this card can manage."
        )
    if memory_bound:
        sentences.append(
            "Doing all the steps in one pass over the data — instead of one pass per step "
            "— removes most of that waiting. That is what the new code has to do."
        )
    else:
        sentences.append(
            "This one is already limited by its own arithmetic, which the GPU's dedicated "
            "matrix hardware is already doing about as fast as it can. Hand-written code "
            "has little left to win here, and the system says so instead of claiming a win."
        )
    if isinstance(tile, int):
        sentences.append(f"Suggested chunk size for the Coder to work in: {tile} values.")

    return {"headline": headline, "detail": " ".join(sentences), "raw": raw}


def _describe_retrieval(payload: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    """What came back from the skill library, and which arm the bandit pulled."""
    skills = payload.get("skills") or []
    count = payload.get("count", len(skills) if isinstance(skills, list) else 0)
    ids = [
        str(skill.get("skill_id"))
        for skill in skills
        if isinstance(skill, dict) and skill.get("skill_id")
    ]
    selected = payload.get("selected_skill_id")

    if not ids:
        return {
            "headline": "📚 Nothing like this has been solved before",
            "detail": (
                "A cold start, so the Coder writes this one from the measurements alone. "
                "Whatever passes the tests becomes the starting point for next time."
            ),
            "raw": raw,
        }

    detail = "Past solutions found: " + ", ".join(f"`{skill_id}`" for skill_id in ids[:4]) + "."
    if selected:
        detail += (
            f" `{selected}` goes first, chosen the way you would choose between slot "
            "machines: mostly back what has worked, but keep trying what has barely been "
            "tried. The Coder sees it as a starting point, not as an answer."
        )
    return {
        "headline": f"📚 Found {count} solution{'s' if count != 1 else ''} to similar problems",
        "detail": detail,
        "raw": raw,
    }


def _describe_verdict(payload: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    """The Judge's verdict: the payoff event, and the one that gets metric cards."""
    reward = payload.get("reward")
    eager = _as_float(payload.get("speedup_vs_eager"))
    compile_ = _as_float(payload.get("speedup_vs_compile"))
    passed = payload.get("passed_checks")
    total = payload.get("total_checks")
    correct = payload.get("correctness_pass")
    violations = payload.get("violations") or []

    mark = "✅" if isinstance(reward, int) and reward > 0 else "❌"
    headline = f"{mark} Score {reward:+d}" if isinstance(reward, int) else f"{mark} Verdict"
    if eager:
        headline += f": {eager:.2f}× faster than PyTorch"

    checks = f"all {total} of them" if isinstance(passed, int) and passed == total else None
    if checks is None:
        checks = (
            f"{passed} of {total}"
            if isinstance(passed, int) and isinstance(total, int)
            else ("passed" if correct else "failed")
        )
    parts = [f"Same answers as PyTorch on {checks}."]
    if eager:
        parts.append(f"{eager:.2f}× faster than normal PyTorch.")
    if compile_:
        parts.append(f"{compile_:.2f}× faster than PyTorch's own compiler.")
    parts.append(
        "No cheating found in the code."
        if not violations
        else f"REJECTED — the code scan found a shortcut: {violations}."
    )
    if isinstance(reward, int):
        parts.append(
            "Top score: correct, and faster than both things it was compared against."
            if reward >= 3
            else "The score decides whether the Coder gets another attempt."
        )
    if payload.get("stderr_tail"):
        parts.append(
            "Anything the code printed when it failed is sent back to the Coder as a hint."
        )

    return {
        "headline": headline,
        "detail": " ".join(parts),
        "raw": raw,
        "verdict": payload,
    }


def _describe_hotswap(payload: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    """The money shot, or an honest refusal. Never a refusal dressed as a success."""
    if not payload.get("success"):
        error = str(payload.get("error") or "the server refused the change").strip().rstrip(".")
        rolled_back = payload.get("rolled_back")
        return {
            "headline": "⚠️ Not loaded — the new code is NOT running live",
            "detail": (
                f"The server reported: {error}. "
                + (
                    "The model was put back exactly as it was, bit for bit."
                    if rolled_back
                    else "Nothing was changed, so there is nothing to undo."
                )
                + " The code passed its tests; it just did not reach a live server. The "
                "system reports that plainly rather than calling it a success."
            ),
            "raw": raw,
        }

    patched = payload.get("modules_patched")
    stats = payload.get("stats") or {}
    parity = payload.get("parity") or {}
    tokens = _as_float(stats.get("tokens_per_s"))

    parts = []
    if isinstance(patched, int):
        parts.append(
            f"{patched} layers of the running language model now use the new code, swapped "
            "in while the model stayed loaded in GPU memory."
        )
    if parity.get("parity_pass"):
        parts.append(
            "Before keeping the change, the server checked that the new code still gives "
            f"the same answers as the old one on the model's real weights, over "
            f"{parity.get('seeds', 'several')} random inputs."
        )
    if tokens:
        parts.append(f"The server is now producing {tokens:.1f} words-worth of text per second.")
    parts.append("No restart, and no request was dropped while it happened.")

    return {
        "headline": "🚀 The new code is running live",
        "detail": " ".join(parts),
        "raw": raw,
    }


def _describe_text(author: str, text: str) -> dict[str, Any]:
    """An agent's own message: a draft, a verdict, the final summary, or plain prose.

    The Coder and the Judge speak in structured JSON (their `output_schema`), which is
    the least readable thing on the page if it is printed as-is. Parsed here, so the
    Coder's turn shows its kernel and the Judge's shows its numbers.
    """
    parsed = _maybe_json(text)

    if isinstance(parsed, dict) and parsed.get("code"):
        code = str(parsed["code"])
        bindings = parsed.get("adapter_mapping") or []
        contract = ""
        if isinstance(bindings, list) and bindings:
            pairs = [
                f"`{binding.get('kernel_param')}`→`{binding.get('module_attr')}`"
                for binding in bindings
                if isinstance(binding, dict)
            ]
            contract = (
                " It also writes its own wiring instructions — which value inside the live "
                "model feeds which input of its code (" + ", ".join(pairs) + "). In every "
                "other published system, a human writes that by hand."
            )
        return {
            "headline": "💻 Writing the new GPU code",
            "detail": (
                "A complete, runnable GPU program written in Triton — Python that compiles "
                f"down to code the graphics card executes directly (it starts "
                f"`{_signature_line(code)}`).{contract}"
            ),
            "raw": parsed,
            "code": code,
        }

    if isinstance(parsed, dict) and "reward" in parsed:
        reward = parsed.get("reward")
        action = str(parsed.get("next_action") or "").upper()
        return {
            "headline": (
                f"⚖️ Result written down: score {reward:+d}"
                if isinstance(reward, int)
                else "⚖️ Result written down"
            ),
            "detail": (
                f"The Judge records its decision — next step: {action or 'n/a'}. Every "
                "number here came from actually running and timing the code, not from the "
                "Judge's opinion of how the code looks."
            ),
            "raw": parsed,
            "verdict": parsed,
        }

    if _is_summary(author, text):
        return {
            "headline": "🏁 Optimization summary",
            "detail": "",
            "raw": {"text": text},
            "summary": text,
        }

    prose, code = split_prose_and_code(text)
    first = (prose or text).strip().splitlines()[0] if (prose or text).strip() else ""
    return {
        "headline": _truncate_line(first, 90) or f"💬 {author}",
        "detail": "",
        "raw": {"text": text},
        "prose": prose,
        "code": code,
    }


def _maybe_json(text: str) -> Any:
    """`json.loads`, or None for anything that is not a JSON object."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    import json

    try:
        return json.loads(stripped)
    except ValueError:
        return None


def _is_summary(author: str, text: str) -> bool:
    """Whether this is the Supervisor's end-of-run summary rather than ordinary prose."""
    if author != "Supervisor":
        return False
    lowered = text.lower()
    return "optimization summary" in lowered or (
        text.lstrip().startswith("#") and "speedup" in lowered
    )


def _signature_line(code: str) -> str:
    """The first line worth quoting from a kernel — its decorator or its first `def`."""
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("@") or stripped.startswith("def "):
            return _truncate_line(stripped, 70)
    for line in code.splitlines():
        if line.strip():
            return _truncate_line(line.strip(), 70)
    return "kernel"


def _truncate_line(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------- #
# Turn-level presentation (Task 12b, problems 3, 4 and 5)
# --------------------------------------------------------------------------- #

#: Which event in a turn the status label should describe, most settled first. A verdict
#: outranks the call that produced it: "Verified: 7.24× vs eager" is the shot, "Calling
#: verify_kernel" is the setup.
_DECISIVE_ORDER = (
    ("function_response", "verify_kernel"),
    ("function_response", "hotswap_kernel"),
    ("escalate", None),
    ("function_response", None),
    ("function_call", None),
    ("transfer", None),
    ("text", None),
)


def decisive_event(turn: Turn) -> dict[str, Any] | None:
    """The one event in `turn` whose headline should label it."""
    events = [event for event in turn.events if not is_noise_event(event)] or turn.events
    for kind, tool in _DECISIVE_ORDER:
        for event in events:
            if event.get("event_type") != kind:
                continue
            if tool is None:
                return event
            names = [
                str(item.get("name") or "")
                for item in (event.get("function_responses") or event.get("function_calls") or [])
                if isinstance(item, dict)
            ]
            if tool in names:
                return event
    return events[0] if events else None


def turn_label(turn: Turn, *, running: bool) -> str:
    """The `st.status` label: who is acting, what they did, and how long it took.

    "⚖️ Judge — Reward +3: 7.24× vs eager (23s)". The duration lives here rather than in
    a caption inside the body, where it read as one more number in a pile of them.
    """
    style = style_for(turn.author)
    event = decisive_event(turn)
    headline = format_event_for_display(event)["headline"] if event else ""
    text = _strip_leading_emoji(headline) or style.get("role", "working") or "working"
    suffix = "" if running or turn.duration_s < 0.5 else f" ({turn.duration_s:.0f}s)"
    return f"{style['emoji']} {turn.author} — {text}{suffix}"


_EMOJI_PREFIX = re.compile(r"^[^\w`(\[]+\s*")


def _strip_leading_emoji(headline: str) -> str:
    """Drop a headline's own emoji — the label already carries the agent's."""
    return _EMOJI_PREFIX.sub("", headline).strip()


def narrative_after(turn: Turn) -> str | None:
    """The static caption that follows a finished turn, or None if it needs none.

    Keyed on what the turn accomplished, not on who ran it: the Supervisor's hot-swap
    and the Profiler's measurement are both "a step whose consequence the audience needs
    spelled out", and the Coder's kernel is only narrated once it exists.
    """
    keys: list[str] = []
    for event in turn.events:
        kind = event.get("event_type")
        if kind == "escalate":
            keys.append("escalated")
            continue
        if kind == "function_response":
            for response in event.get("function_responses") or []:
                if not isinstance(response, dict):
                    continue
                name = str(response.get("name") or "")
                payload = unwrap_payload(response.get("response"))
                if name in {"profile_op", "profile_op_by_name"}:
                    keys.append("profiled")
                elif name in {"retrieve_skills", "retrieve_skills_for_agent"}:
                    keys.append("retrieved")
                elif name == "verify_kernel":
                    keys.append("verified")
                elif (
                    name == "hotswap_kernel"
                    and isinstance(payload, dict)
                    and payload.get("success")
                ):
                    keys.append("swapped")
        elif kind == "text" and turn.author == "Coder":
            display = format_event_for_display(event)
            if display.get("code"):
                keys.append("kernel_written")
    # The last thing the turn did is the one the caption should follow on from.
    return NARRATIVE.get(keys[-1]) if keys else None


def iteration_label(metrics: dict[str, Any]) -> tuple[str, str | None]:
    """The Iteration header metric as `(value, delta)` (Task 12b, problem 3).

    Three states worth distinguishing, because "1/6" forever was indistinguishable from
    a stuck loop: counting up, finished early (what a good run looks like), and out of
    budget without escalating (what a bad one looks like, and it must not read as done).
    """
    try:
        from kernelsmith.config import MAX_LOOP_ITERATIONS as cap
    except Exception:  # noqa: BLE001
        cap = 6
    done = int(metrics.get("iteration") or 0)
    if metrics.get("escalated"):
        return "✅ Done", f"converged in {done}" if done else "target met"
    if done >= cap:
        return f"{done}/{cap} ⚠️", "budget spent"
    return f"{done}/{cap}", None


# --------------------------------------------------------------------------- #
# Traces and modes (Task 12b, problem 6 — the hosted replay)
# --------------------------------------------------------------------------- #


def script_flags() -> set[str]:
    """Flags passed after `--` on the `streamlit run` command line.

    Only one is honoured: `--live`, which forces Live mode even where nothing is
    listening on the inference port. Everything else is ignored rather than rejected —
    Streamlit puts its own arguments through here too.
    """
    import sys

    return {argument for argument in sys.argv[1:] if argument.startswith("--")}


def ordered_traces(trace_dir: str | Path = DEFAULT_TRACE_DIR) -> list[Path]:
    """Traces for the picker: real captured runs newest-first, the fixture last.

    `list_traces` sorts by mtime, which on a fresh clone is checkout time — every file
    within a second of every other, so the "most recent trace" would be whichever one
    git happened to write last. Pinning the hand-written fixture to the end makes the
    default selection deterministic in a container: a real run if there is one, the
    fixture only if there is not.
    """
    traces = list_traces(trace_dir)
    real = [path for path in traces if path.name != SAMPLE_TRACE_NAME]
    fixture = [path for path in traces if path.name == SAMPLE_TRACE_NAME]
    return real + fixture


def trace_summary(path: Path) -> str:
    """One line describing what a trace contains, for the picker's caption.

    Whether the hot-swap in it went live is the fact a viewer most needs stated up
    front, because a trace recorded with `--no-server` ends in an honest refusal, and
    that must be visible *before* someone presses Play rather than as a surprise at the
    end of it.
    """
    try:
        events = load_events(path)
    except Exception as exc:  # noqa: BLE001 — an unreadable trace is a caption, not a crash
        return f"could not be read — {type(exc).__name__}"
    metrics = extract_metrics(events)
    parts = [f"{len(events)} steps", f"{total_duration_s(events):.0f}s"]
    if metrics.get("speedup"):
        parts.append(f"{metrics['speedup']:.2f}× faster than PyTorch")
    if metrics.get("hotswap_ok") is True:
        patched = metrics.get("modules_patched")
        parts.append(f"went live on {patched} layers" if patched else "the new code went live")
    elif metrics.get("hotswap_ok") is False:
        parts.append("never reached a live server")
    if path.name == SAMPLE_TRACE_NAME:
        parts.append("hand-written fixture")
    return " · ".join(parts)


def default_mode() -> str:
    """Which mode a cold page opens in.

    Replay unless there is something live to show: no `--live` flag, no inference server
    answering, and at least one trace on disk. That is exactly the Cloud Run case — the
    container has the traces baked in and no GPU — so a judge who opens the hosted URL
    lands on a page with a Play button rather than on a Live tab that can never start.
    """
    if "--live" in script_flags():
        return MODE_LIVE
    if not ordered_traces():
        return MODE_LIVE
    return MODE_LIVE if inference_server_is_up() else MODE_REPLAY


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #


def init_state() -> None:
    """Seed every key the script reads, so no branch has to guard for absence."""
    state = st.session_state
    state.setdefault("mode", default_mode())
    state.setdefault("timeline_events", [])
    state.setdefault("stats_ok", False)
    state.setdefault("tokens_before_swap", None)
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


def render_event(target: Any, event: dict[str, Any], metrics: dict[str, Any] | None = None) -> None:
    """Draw one event inside an already-open `st.status`.

    This is the function that makes live and replay identical: both call it, with the
    same dicts, into the same kind of container. `target` is a DeltaGenerator — an
    `st.status` handle — so calls append to that status rather than to the page.

    Task 12b: the body is a headline, two or three sentences of plain English, and only
    then — folded, and only where it is the payoff — the data. The raw payload is one
    click away on every event, because "trust us" is not a demo of a verifier.
    """
    if is_noise_event(event):
        # A duplicate hand-off echo or an empty streamed chunk. Dropping it is not
        # hiding anything: the event it duplicates is rendered one line above.
        return

    display = format_event_for_display(event)
    target.markdown(f"**{display['headline']}**")
    if display.get("detail"):
        target.markdown(f"<div class='ks-detail'>{display['detail']}</div>", unsafe_allow_html=True)

    verdict = display.get("verdict")
    if isinstance(verdict, dict):
        _render_verdict_metrics(target, verdict)

    summary = display.get("summary")
    if summary:
        # The card's numbers come from the run's verdict, which this event does not
        # carry — the caller passes the metrics it has already extracted.
        render_summary_card(target, str(summary), metrics or extract_metrics([event]))

    if display.get("prose"):
        target.markdown(_truncate(str(display["prose"]), TEXT_PREVIEW_CHARS))
    if display.get("code"):
        target.code(_truncate(str(display["code"]), CODE_PREVIEW_CHARS), language="python")

    _render_raw(target, event, display)


def _render_raw(target: Any, event: dict[str, Any], display: dict[str, Any]) -> None:
    """The "Show raw" fold. Collapsed everywhere; there is no auto-expanded JSON left.

    Problem 5 asked for Results to open automatically on the Judge's verdict. They do —
    as the three metric cards above, which is the same information in the form a camera
    can read. The JSON underneath stays folded for whoever wants to check it.
    """
    raw = display.get("raw")
    if not raw:
        return
    label = "Show the exact data behind this" + (
        f" · {_raw_source(event)}" if _raw_source(event) else ""
    )
    with target.expander(label, expanded=False):
        if isinstance(raw, dict):
            _render_args(st, raw)
        else:
            st.json(raw)


def _raw_source(event: dict[str, Any]) -> str:
    """Which tool the raw payload belongs to, for the toggle's label."""
    for key in ("function_calls", "function_responses"):
        for item in event.get(key) or []:
            if isinstance(item, dict) and item.get("name"):
                return str(item["name"])
    return ""


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


def _render_verdict_metrics(target: Any, payload: dict[str, Any]) -> None:
    """The verifier's verdict as three metric cards — the payoff, not a JSON dump.

    A verdict rendered as raw JSON is unreadable on a recording; the same three numbers
    as `st.metric` are the shot. Correctness first, because a fast wrong kernel is worth
    nothing, and the speedup card names both baselines rather than only the flattering one.
    """
    from kernelsmith.config import CORRECTNESS_SEEDS, CORRECTNESS_SHAPES

    total = payload.get("total_checks")
    if not isinstance(total, int):
        total = CORRECTNESS_SEEDS * len(CORRECTNESS_SHAPES)
    passed = payload.get("passed_checks")
    if not isinstance(passed, int):
        passed = total if payload.get("correctness_pass") else 0

    columns = target.columns(3)
    columns[0].metric(
        "Same answers",
        f"{passed}/{total} ✅" if passed == total and total else f"{passed}/{total} ❌",
        help=(
            f"{CORRECTNESS_SEEDS} sets of random numbers × {len(CORRECTNESS_SHAPES)} input "
            "sizes, compared against PyTorch to within one part in a hundred."
        ),
    )
    eager = _as_float(payload.get("speedup_vs_eager"))
    compile_ = _as_float(payload.get("speedup_vs_compile"))
    columns[1].metric(
        "Faster than PyTorch",
        f"{eager:.2f}×" if eager else "—",
        delta=f"{compile_:.2f}× vs PyTorch compiler" if compile_ else None,
        delta_color="normal" if compile_ and compile_ >= 1 else "inverse",
    )
    reward = payload.get("reward")
    columns[2].metric(
        "Score",
        f"{reward:+d}" if isinstance(reward, int) else "—",
        help="+3 correct and faster than both baselines · +1 correct · −1 rejected",
    )

    latency = payload.get("latency_ms_by_shape")
    baseline = payload.get("baseline_ms") or {}
    if isinstance(latency, dict) and latency:
        shape = payload.get("headline_shape") or max(latency, key=lambda key: str(key))
        kernel_ms = _as_float(latency.get(shape))
        eager_ms = _as_float(baseline.get("eager_ms"))
        line = f"On the largest input tested (`{shape}`): {kernel_ms:.3f} ms per call"
        if eager_ms:
            line += f", where PyTorch takes {eager_ms:.3f} ms"
        target.caption(
            line + ". Median of 200 timed runs after 150 warm-up runs, so it is not a "
            "one-off lucky measurement."
        )


def render_summary_card(target: Any, text: str, metrics: dict[str, Any]) -> None:
    """The end-of-run summary as a card, with the agent's prose folded underneath.

    Problem 5.4: the Supervisor's summary is a wall of markdown bullets carrying six
    numbers at the same weight. The card promotes the four that matter and keeps the
    prose available — but the card's numbers are read from the *verdict*, not parsed out
    of the sentence, so a summary that misquotes its own run cannot move them.
    """
    card = target.container(border=True)
    card.markdown("#### 🏁 Run complete")

    columns = card.columns(4)
    speedup = metrics.get("speedup")
    columns[0].metric("Faster than PyTorch", f"{speedup:.2f}×" if speedup else "—")
    compile_ = metrics.get("speedup_vs_compile")
    columns[1].metric("Faster than its compiler", f"{compile_:.2f}×" if compile_ else "—")
    reward = metrics.get("reward")
    columns[2].metric("Score", f"{reward:+d}" if isinstance(reward, int) else "—")
    patched = metrics.get("modules_patched")
    columns[3].metric(
        "Layers now live",
        f"{patched}"
        if isinstance(patched, int)
        else ("0" if metrics.get("hotswap_ok") is False else "—"),
    )

    if metrics.get("skill_id"):
        card.caption(f"Saved for future runs as `{metrics['skill_id']}`.")
    if metrics.get("hotswap_ok") is False and metrics.get("hotswap_error"):
        card.warning(
            "The new code passed every test but was not loaded into a live server: "
            f"{metrics['hotswap_error']}"
        )

    with card.expander("Read the agent's own write-up of the run", expanded=False):
        st.markdown(text)


def open_turn(container: Any, author: str, headline: str, *, running: bool) -> Any:
    """Create the `st.status` for one agent turn and return its handle.

    `headline` is taken as the whole label when it already names the agent — `turn_label`
    builds those — so the two callers cannot disagree about the format.
    """
    style = style_for(author)
    label = (
        headline
        if headline.startswith(style["emoji"])
        else f"{style['emoji']} {author} — {headline}"
    )
    return container.status(
        label,
        state="running" if running else "complete",
        expanded=True,
    )


def render_turn(
    container: Any,
    turn: Turn,
    *,
    running: bool,
    metrics: dict[str, Any] | None = None,
    narrate: bool = True,
) -> Any:
    """Draw a whole finished turn: one status, then every event inside it.

    The elapsed time is in the label (problem 5.3), not in a caption at the bottom of
    the body — inside the status it was one more number competing with the verdict.
    """
    status = open_turn(container, turn.author, turn_label(turn, running=running), running=running)
    for event in turn.events:
        render_event(status, event, metrics)
    if narrate and not running:
        caption = narrative_after(turn)
        if caption:
            container.markdown(f"<div class='ks-narrative'>{caption}</div>", unsafe_allow_html=True)
    return status


def render_timeline(container: Any, events: list[dict[str, Any]], *, running: bool) -> None:
    """Draw the whole timeline from scratch — the live path, once per rerun."""
    metrics = extract_metrics(events)
    turns = group_turns(events)
    for index, turn in enumerate(turns):
        last = index == len(turns) - 1
        render_turn(container, turn, running=running and last, metrics=metrics)


# --------------------------------------------------------------------------- #
# Header, graph, banners
# --------------------------------------------------------------------------- #


@dataclass
class PageSlots:
    """The page's fixed furniture: one slot per region, created in the same order always.

    Task 13, problem 2. The dashboard used to show two rows of metrics at once — a final
    row and a blank one. The cause was not a second `render_header` call but *conditional
    top-level elements*: the replay hero appeared only in some runs, so every element
    after it changed position between runs, and Streamlit — mid-playback, where two script
    runs genuinely overlap — could not match the new run's elements to the old ones and
    left the stale ones on screen.

    So the top-level sequence is now identical in every run and in both modes: header,
    orientation line, divider, notice, two columns, divider, banners. Regions that
    sometimes have nothing to say are `st.empty()` placeholders that get filled or
    cleared, never elements that come and go.
    """

    header: Any
    notice: Any
    graph: Any
    who: Any
    chart: Any
    timeline: Any
    banners: Any


def build_page_skeleton() -> PageSlots:
    """Lay out the page once. Callers fill the slots; nobody adds top-level elements."""
    header = st.empty()
    st.caption(ORIENTATION)
    st.divider()
    notice = st.empty()

    left, right = st.columns([1.25, 2.3], gap="large")
    with left:
        st.markdown("##### Who is working")
        graph = st.empty()
        who = st.empty()
        chart = st.empty()
    with right:
        st.markdown("##### What is happening, step by step")
        timeline = st.container()

    st.divider()
    banners = st.container()
    return PageSlots(
        header=header,
        notice=notice,
        graph=graph,
        who=who,
        chart=chart,
        timeline=timeline,
        banners=banners,
    )


#: The one line of orientation that sits under the header in every mode.
ORIENTATION = (
    "Read the right-hand column top to bottom: each box is one agent taking one action, "
    "in plain English, and the blue notes between them say why the next step follows."
)


def render_header(metrics: dict[str, Any], target: Any = None) -> None:
    """Title on the left, four metric cards on the right. Into `target` if given."""
    container = target if target is not None else st
    title, cards = container.columns([1.35, 2])
    with title:
        st.markdown(
            '<p class="ks-title">gpuyantra</p>'
            '<p class="ks-sub">Making a running AI model faster, and proving it — '
            "powered by the <strong>KernelSmith</strong> agent tree</p>",
            unsafe_allow_html=True,
        )
    with cards:
        speed, reward, iteration, tokens = cards.columns(4)
        speedup = metrics.get("speedup")
        compile_ = metrics.get("speedup_vs_compile")
        speed.metric(
            "Speedup",
            f"{speedup:.2f}×" if speedup else "—",
            delta=f"{compile_:.2f}× vs PyTorch compiler" if compile_ else None,
            delta_color="normal" if compile_ and compile_ >= 1 else "inverse",
            help=(
                "How many times faster the new code is than standard PyTorch, timed by the "
                "tester. It only ever moves when a measurement says so — never because an "
                "agent claimed a number."
            ),
        )
        reward.metric(
            "Reward",
            f"{metrics['reward']:+d}" if isinstance(metrics.get("reward"), int) else "—",
            help=(
                "The tester's score. +3 = correct and faster than both baselines · "
                "+1 = correct but no faster · −1 = wrong, or caught cheating."
            ),
        )
        value, delta = iteration_label(metrics)
        iteration.metric(
            "Iteration",
            value,
            delta=delta,
            help="Attempts used out of the maximum the loop is allowed.",
        )
        tokens_value, tokens_delta, tokens_help = tokens_metric(metrics)
        tokens.metric("Tokens/s", tokens_value, delta=tokens_delta, help=tokens_help)


def tokens_metric(metrics: dict[str, Any]) -> tuple[str, str | None, str]:
    """The Tokens/s card as `(value, delta, help)` (Task 12b problem 2, Task 13 problem 4).

    Three sources, in the order they become trustworthy, and the help text always says
    which one is on screen:

    - the live server's `/stats`, polled once per rerun. A reachable server with a *zero*
      rolling window still shows `0.0`, not "—": zero throughput on a server that is up
      is a fact, and blanking it is what made this metric look broken;
    - the `hotswap_kernel` response's `stats`, which is where a replayed trace gets it —
      plus a `tokens_per_s_before` reading beside it if one was captured, which is where
      the before/after pair comes from;
    - nothing, before either exists.

    When there is nothing, the card says *why* rather than leaving a bare dash: a run
    recorded with no inference server (the `--no-server` traces) can never have a
    throughput number, and that is a different situation from a run that has not reached
    the swap yet. Task 13, problem 4.
    """
    value = metrics.get("tokens_per_s")
    source = metrics.get("tokens_source")
    if value is None:
        if metrics.get("hotswap_ok") is False:
            return (
                "—",
                "no server",
                "This run never reached a live server, so no throughput was measured. The "
                "kernel passed every correctness and speed test; it was simply not "
                "deployed. Nothing is estimated in its place.",
            )
        return (
            "—",
            None,
            "Nothing to measure yet: no live server is running, and the new code has not "
            "been loaded into one in this run.",
        )

    baseline = metrics.get("tokens_before_swap")
    delta = None
    if isinstance(baseline, (int, float)) and baseline > 0 and value != baseline:
        delta = f"{value - baseline:+.1f} vs before the swap ({baseline:.1f})"

    origin = {
        "server": "read live from the running server",
        "swap": "reported by the server when the new code was loaded",
    }.get(str(source), "from this run")
    return (
        f"{value:.1f}",
        delta,
        f"How much text the model is producing per second — {origin}.",
    )


#: Five words the rest of the page cannot avoid, explained once. Collapsed by default:
#: it is there for a judge who clicks it, not in the way of the recording.
GLOSSARY = (
    (
        "GPU kernel",
        "A small program that runs on the graphics card. A model like this one is "
        "thousands of kernel launches per word it writes.",
    ),
    (
        "Memory-bound",
        "The kernel is fast at maths but spends its time waiting for data to arrive from "
        "the card's memory. Those are the ones worth rewriting.",
    ),
    (
        "Triton",
        "A way of writing GPU kernels in Python instead of in C++/CUDA. It is what the "
        "Coder agent writes.",
    ),
    (
        "PyTorch / its compiler",
        "The two things the new code is timed against: ordinary PyTorch, and PyTorch's "
        "own optimizing compiler (`torch.compile`). Beating only the first one would be "
        "an easy win, so both are reported.",
    ),
    (
        "Hot-swap",
        "Replacing code inside a program that is already running, without restarting it. "
        "Here: swapping kernels inside a language-model server mid-flight.",
    ),
)


def render_legend() -> None:
    """The glossary, in the sidebar.

    Task 13, problem 7: it used to sit between the header and the timeline, where it was
    the second thing on the page and pushed the demo below the fold. In the sidebar it is
    still one click away and never competes with the run.
    """
    st.sidebar.divider()
    with st.sidebar.expander("New to this? Five words explained", expanded=False):
        for term, meaning in GLOSSARY:
            st.markdown(f"**{term}** — {meaning}")


def render_graph(slots: PageSlots, active: str | None, *, idle_hint: str = "") -> None:
    """The agent map, with whoever is working highlighted, into the page's slots."""
    slots.graph.graphviz_chart(build_agent_graph(active), width="stretch")
    if active:
        style = style_for(active)
        slots.who.markdown(
            f"<span class='ks-agent' style='color:{style['color']}'>{style['emoji']} {active}</span>"
            f" — {style['role']}",
            unsafe_allow_html=True,
        )
    else:
        slots.who.caption(idle_hint or "Nobody working right now.")


# --------------------------------------------------------------------------- #
# The speedup chart (Task 13, problem 6)
# --------------------------------------------------------------------------- #

#: Bar colours: two greys-and-blue for the things being beaten, amber for the result.
CHART_COLORS = ("#6b7280", "#3b82f6", "#f59e0b")
BASELINE_LABEL = "PyTorch"
COMPILER_LABEL = "PyTorch compiler"
KERNEL_LABEL = "KernelSmith"


def speedup_chart_rows(metrics: dict[str, Any]) -> list[tuple[str, float]] | None:
    """The three bars, or None before there is a verdict to draw.

    Nothing here is hardcoded and nothing is invented:

    - PyTorch is 1.0 by definition — it is what the other two are divided by;
    - the compiler bar is `eager_ms / compile_ms` from the verdict's own baseline
      timings, falling back to `speedup_vs_eager / speedup_vs_compile`, which is the same
      quantity computed from the two ratios the verifier reported;
    - the kernel bar is `speedup_vs_eager`, exactly as measured.

    If the verdict carries neither the baseline timings nor both ratios, the compiler bar
    is left out rather than guessed at — two honest bars beat three with a fiction in the
    middle.
    """
    speedup = _as_float(metrics.get("speedup"))
    if not speedup:
        return None

    rows = [(BASELINE_LABEL, 1.0)]
    baseline = metrics.get("baseline_ms") or {}
    eager_ms = _as_float(baseline.get("eager_ms")) if isinstance(baseline, dict) else 0.0
    compile_ms = _as_float(baseline.get("compile_ms")) if isinstance(baseline, dict) else 0.0
    vs_compile = _as_float(metrics.get("speedup_vs_compile"))
    if eager_ms and compile_ms:
        rows.append((COMPILER_LABEL, eager_ms / compile_ms))
    elif vs_compile:
        rows.append((COMPILER_LABEL, speedup / vs_compile))
    rows.append((KERNEL_LABEL, speedup))
    return rows


def render_speedup_chart(target: Any, metrics: dict[str, Any]) -> None:
    """Three bars: PyTorch, its compiler, and the agent's kernel. Cleared before a verdict.

    Rendered into a slot that exists from the first frame, so it fills in when the Judge
    reports rather than pushing the rest of the page down when it appears.
    """
    rows = speedup_chart_rows(metrics)
    if not rows:
        target.empty()
        return

    body = target.container()
    body.markdown("##### How much faster")
    try:
        import altair as alt
        import pandas as pd

        frame = pd.DataFrame(rows, columns=["what", "times"])
        order = [label for label, _ in rows]
        base = alt.Chart(frame).encode(
            x=alt.X("times:Q", title="× faster than PyTorch", axis=alt.Axis(grid=False)),
            y=alt.Y("what:N", sort=order, title=None),
        )
        bars = base.mark_bar(cornerRadiusEnd=4, height=26).encode(
            color=alt.Color(
                "what:N",
                scale=alt.Scale(domain=order, range=list(CHART_COLORS[: len(order)])),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("what:N", title=""),
                alt.Tooltip("times:Q", title="× faster", format=".2f"),
            ],
        )
        labels = base.mark_text(align="left", dx=6, color="#e5e7eb", fontSize=13).encode(
            text=alt.Text("times:Q", format=".2f")
        )
        body.altair_chart((bars + labels).properties(height=len(rows) * 46), width="stretch")
    except Exception:  # noqa: BLE001 — a charting library must never break the demo
        body.bar_chart({label: [value] for label, value in rows}, height=180, width="stretch")
    body.caption(
        "Both comparisons come from the same timed run. The middle bar is PyTorch's own "
        "compiler, measured against the same baseline — beating only plain PyTorch would "
        "be the easy half of the claim."
    )


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
    """Fetch `/stats` once per rerun — i.e. at ~1 Hz while Live mode is refreshing.

    `stats_ok` is what distinguishes "the server says zero" from "there is no server",
    and the Tokens/s card renders those two differently on purpose.
    """
    import httpx

    from kernelsmith.config import INFERENCE_HOST, INFERENCE_PORT

    state = st.session_state
    try:
        response = httpx.get(
            f"http://{INFERENCE_HOST}:{INFERENCE_PORT}/stats", timeout=STATS_TIMEOUT_S
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:  # noqa: BLE001 — the server is optional for a rehearsal
        state["stats_ok"] = False
        return
    if isinstance(payload, dict):
        state["stats"] = payload
        state["stats_ok"] = True


@st.cache_data(ttl=HEALTH_CACHE_S, show_spinner=False)
def inference_server_is_up() -> bool:
    """Whether `/health` answers. Cached, because it gates a decision made every rerun.

    Used for one thing only: choosing the default mode on a cold page (problem 6). A
    Cloud Run container has no GPU and no server, so it opens in Replay; the VM has
    both, so it opens in Live.
    """
    import httpx

    from kernelsmith.config import INFERENCE_HOST, INFERENCE_PORT

    try:
        response = httpx.get(
            f"http://{INFERENCE_HOST}:{INFERENCE_PORT}/health", timeout=HEALTH_TIMEOUT_S
        )
        return response.status_code == 200
    except Exception:  # noqa: BLE001 — an unreachable server is the answer, not an error
        return False


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


def merge_live_tokens(metrics: dict[str, Any], state: Any) -> dict[str, Any]:
    """Fold the polled `/stats` into the run's metrics, and remember the pre-swap rate.

    A live server outranks the swap response's snapshot: the snapshot was taken the
    instant the kernel landed, when `TokenMeter` had just cleared its rolling window, so
    it is the *least* informative reading of the run. The pre-swap rate is latched the
    last time the server answered while no swap had happened yet, which is the only
    moment it can be observed — after the swap it is gone.
    """
    merged = dict(metrics)
    server_tokens = _as_float(state["stats"].get("tokens_per_s")) if state.get("stats_ok") else None
    if state.get("stats_ok"):
        if merged.get("hotswap_ok") is not True and server_tokens:
            state["tokens_before_swap"] = server_tokens
        merged["tokens_per_s"] = server_tokens if server_tokens is not None else 0.0
        merged["tokens_source"] = "server"
    merged["tokens_before_swap"] = state.get("tokens_before_swap")
    return merged


def render_live() -> None:
    """The live view: drain, drive, render, refresh.

    Everything goes into the shared skeleton (`build_page_skeleton`), so Live and Replay
    place their elements identically and switching modes does not reshuffle the page.
    """
    state = st.session_state
    consumer = get_consumer()
    ingest_events(consumer)
    # Before any early return, always: the two-message protocol depends on it.
    drive_run(consumer)
    poll_stats()

    metrics = merge_live_tokens(extract_metrics(state["timeline_events"]), state)
    slots = build_page_skeleton()

    render_header(metrics, slots.header.container())
    render_graph(
        slots,
        state["active_agent"] if consumer.is_running else None,
        idle_hint="Nobody yet — press Start Run.",
    )
    render_speedup_chart(slots.chart, metrics)

    if not state["timeline_events"]:
        slots.timeline.info(
            "Press **Start Run** in the sidebar. Each step appears here as the agents "
            "work, with a plain-English explanation of what it means."
        )
    render_timeline(slots.timeline, state["timeline_events"], running=consumer.is_running)

    # One slot, two kinds of news: an error is a warning, a trace path is a footnote.
    # Both go in the same place so neither shifts the layout when it appears.
    if state["run_error"]:
        slots.notice.warning(state["run_error"])
    elif state["trace_path"]:
        slots.notice.caption(f"Recording this run to `{state['trace_path']}`")

    with slots.banners:
        render_banners(metrics)
        maybe_celebrate(metrics)
    render_autorefresh()


# --------------------------------------------------------------------------- #
# Replay mode
# --------------------------------------------------------------------------- #

HERO_PITCH = (
    "A language model is running on a GPU. In the next minute, the KernelSmith agent tree "
    "finds the slowest piece of it, writes faster code for that piece, tests the new code "
    "against the original, and loads it into the running server — without restarting it. "
    "Every number you see was measured on a real NVIDIA L4 GPU; this is a recording of "
    "that run."
)


def render_replay(trace: Path | None, speed: float, play: bool) -> None:
    """The replay view: progressive while playing, static before and after.

    No auto-refresh here: a 1 Hz rerun landing mid-playback would restart the trace from
    the top and the timeline would never finish building.
    """
    state = st.session_state
    if trace is None:
        slots = build_page_skeleton()
        render_header(_blank_metrics(), slots.header.container())
        render_graph(slots, None, idle_hint="Nothing to replay.")
        slots.notice.info(
            f"No recordings yet. Run one in Live mode, or drop a `.jsonl` in "
            f"`{DEFAULT_TRACE_DIR}/`."
        )
        return

    events = load_events(trace)
    metrics = extract_metrics(events)
    slots = build_page_skeleton()

    if not play and not state["replay_done"]:
        # The hosted case: a judge opens the URL and needs one obvious thing to press.
        # It lives in the notice slot, which exists in every run — so the button
        # disappearing when playback starts moves nothing else on the page.
        hero = slots.notice.container()
        hero.markdown(f'<p class="ks-sub">{HERO_PITCH}</p>', unsafe_allow_html=True)
        columns = hero.columns([1, 3])
        play = columns[0].button("▶ Play Demo", type="primary", width="stretch", key="play-main")
        columns[1].caption(f"`{trace.name}` — {trace_summary(trace)}")

    if not play:
        # Static: the finished state after a playback, the invitation before one. Either
        # way exactly one metrics row — the run's numbers if it has finished, dashes if
        # it has not, never both.
        render_header(
            metrics if state["replay_done"] else _blank_metrics(), slots.header.container()
        )
        render_graph(
            slots,
            None,
            idle_hint="Replay finished."
            if state["replay_done"]
            else "Waiting for you to press Play.",
        )
        if state["replay_done"]:
            render_speedup_chart(slots.chart, metrics)
            render_timeline(slots.timeline, events, running=False)
            with slots.banners:
                render_banners(metrics)
        else:
            slots.timeline.info(
                f"**{trace.name}** — {len(events)} steps, {total_duration_s(events):.0f} "
                "seconds of real agent work. Press **▶ Play Demo**."
            )
        return

    # --- playing -------------------------------------------------------------
    state["replay_done"] = False
    slots.notice.empty()
    render_header(_blank_metrics(), slots.header.container())
    render_graph(slots, None, idle_hint="Starting…")

    seen: list[dict[str, Any]] = []
    status = None
    current_author: str | None = None
    turn_events: list[dict[str, Any]] = []
    node_names = {name for name, _, _ in GRAPH_NODES}

    def close_turn() -> None:
        """Finish the open status: final label, final state, then its narration."""
        if status is None or current_author is None:
            return
        turn = Turn(current_author, list(turn_events))
        status.update(label=turn_label(turn, running=False), state="complete")
        caption = narrative_after(turn)
        if caption:
            slots.timeline.markdown(
                f"<div class='ks-narrative'>{caption}</div>", unsafe_allow_html=True
            )

    for event in pace_events(events, speed):
        seen.append(event)
        author = str(event.get("author") or "unknown")
        running_metrics = extract_metrics(seen)

        if status is None or author != current_author:
            close_turn()
            current_author = author
            turn_events = [event]
            # The label needs the turn's decisive event, which has not arrived yet; a
            # one-event Turn gives the best label available at this instant, and the
            # update below rewrites it as the turn reveals what it was doing.
            status = open_turn(
                slots.timeline,
                author,
                turn_label(Turn(author, turn_events), running=True),
                running=True,
            )
            if author in node_names:
                render_graph(slots, author)
        else:
            turn_events.append(event)
            status.update(label=turn_label(Turn(author, turn_events), running=True))

        render_event(status, event, running_metrics)
        render_header(running_metrics, slots.header.container())
        render_speedup_chart(slots.chart, running_metrics)

    close_turn()

    # Back to the idle map, and clear the "who is working" line with it — a highlight
    # left behind after the run points the audience at an agent that finished a minute ago.
    render_graph(slots, None, idle_hint="Replay complete.")
    render_header(metrics, slots.header.container())
    render_speedup_chart(slots.chart, metrics)
    with slots.banners:
        render_banners(metrics)
        maybe_celebrate(metrics)
    state["replay_done"] = True
    state["replaying"] = False


def _blank_metrics() -> dict[str, Any]:
    """Header metrics before anything has been measured — all dashes, no zeros.

    Derived from `extract_metrics([])` rather than written out again, so a metric added
    to the extractor cannot go missing from the pre-run header.
    """
    return extract_metrics([])


# --------------------------------------------------------------------------- #
# Sidebar + main
# --------------------------------------------------------------------------- #


def render_sidebar() -> dict[str, Any]:
    """Mode switch and its controls. Returns what the chosen mode needs."""
    st.sidebar.markdown("### gpuyantra")
    st.sidebar.caption("The KernelSmith agent tree, live or replayed.")
    mode = st.sidebar.radio(
        "Mode",
        MODES,
        key="mode",
        help=(
            "Live runs the agents for real (needs a GPU and cloud access). Replay plays "
            "back a recording of a real run and needs nothing."
        ),
    )
    st.sidebar.divider()

    controls: dict[str, Any] = {"mode": mode}
    if mode == MODE_LIVE:
        op_name = st.sidebar.text_input(
            "Operation to optimize",
            value=DEFAULT_OP,
            help="Which piece of the model to point the agents at.",
        )
        hidden_size = st.sidebar.number_input(
            "Hidden size", value=DEFAULT_HIDDEN_SIZE, min_value=1, step=64
        )
        controls["op_name"] = op_name
        controls["hidden_size"] = int(hidden_size)
        if st.sidebar.button("▶ Start Run", type="primary", width="stretch"):
            start_run(op_name, int(hidden_size))
        st.sidebar.caption(f"Trace → `{DEFAULT_TRACE_DIR}/`")
    else:
        traces = ordered_traces()
        if traces:
            names = [path.name for path in traces]
            chosen = st.sidebar.selectbox("Recorded run", names, index=0)
            controls["trace"] = next(path for path in traces if path.name == chosen)
            st.sidebar.caption(trace_summary(controls["trace"]))
        else:
            controls["trace"] = None
        label = st.sidebar.select_slider(
            "Playback speed",
            list(SPEED_CHOICES),
            value="1×",
            help="1× is the real timing of the recorded run.",
        )
        controls["speed"] = SPEED_CHOICES[label]
        controls["play"] = st.sidebar.button(
            "▶ Play", type="primary", width="stretch", disabled=controls["trace"] is None
        )
        if not inference_server_is_up():
            st.sidebar.caption(
                "No model server running on this machine, so Live mode has nothing to "
                "talk to. Replay needs only the recording — which is why the hosted "
                "version of this page runs in this mode."
            )
    return controls


def main() -> None:
    st.set_page_config(
        page_title="gpuyantra",
        page_icon="⚒️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    controls = render_sidebar()
    # Sidebar-only, and after the controls: the glossary is reference material, not the
    # first thing either mode should put in front of a viewer (Task 13, problem 7).
    render_legend()

    if controls["mode"] == MODE_LIVE:
        render_live()
    else:
        render_replay(controls.get("trace"), controls["speed"], controls["play"])


main()
