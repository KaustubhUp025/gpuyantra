"""The Profiler: a thin LlmAgent wrapper around `profiler_tool` (spec 4.2).

All the intelligence is in the tool — one honest `do_bench` measurement plus analytic
FLOP/byte counts, placed against the L4 roofline. The agent exists so that the
fingerprint write shows up as a visible ADK Event in the dashboard, and so the
Supervisor can delegate rather than call the tool inline.

The one piece of real work here is keeping `bottleneck_fingerprint` a *dict*, as spec
4.1 types it. `output_key` alone would store the model's prose summary of the tool
result; downstream, `retrieval_tool` needs the exact `fingerprint_text` that the skill
library was written with (spec 6.4), and prose does not round-trip. So the raw tool
response is stashed under a `temp:` key when it returns and promoted over the
`output_key` string in the after-agent callback, which fires last and therefore wins.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext

from kernelsmith import config
from kernelsmith.agents.state_view import render
from kernelsmith.sampling import deterministic_config
from kernelsmith.tools.profiler_tool import profiler_tool

#: `temp:` state is visible for the rest of the invocation but never persisted.
RAW_FINGERPRINT_KEY = "temp:profiler_fingerprint"

_INSTRUCTION = """\
You measure why an operation is slow on an NVIDIA L4. You never write kernels.

TASK:
{task_spec}

Call profiler_tool exactly once with the op_name, batch, seq_len and hidden_size from
the task. If the task does not name a probe shape, use batch=8 and seq_len=512, which
is the middle of the verifier's three shapes.

Then reply with one short paragraph stating whether the op is memory-bound or
compute-bound, its arithmetic intensity against the L4 ridge point, and the tile size
hint. Do not restate the whole JSON — the exact fingerprint is recorded automatically.

Finally hand control back to the Supervisor so it can retrieve prior skills.
"""


def build_instruction(ctx: ReadonlyContext) -> str:
    """Render the Profiler prompt from session state."""
    return _INSTRUCTION.format(task_spec=render(ctx.state.get("task_spec"), empty="(not set)"))


def capture_fingerprint(
    *,
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict[str, Any],
) -> None:
    """after_tool_callback: stash the raw fingerprint dict. Returns None to keep the response."""
    if (
        tool.name == profiler_tool.name
        and isinstance(tool_response, dict)
        and "op_family" in tool_response
    ):
        tool_context.state[RAW_FINGERPRINT_KEY] = dict(tool_response)
    return None


def promote_fingerprint(callback_context: CallbackContext) -> None:
    """after_agent_callback: put the measured dict where `output_key` left the prose."""
    raw = callback_context.state.get(RAW_FINGERPRINT_KEY)
    if isinstance(raw, dict) and raw:
        callback_context.state["bottleneck_fingerprint"] = raw
    return None


def build_profiler_agent() -> LlmAgent:
    """Fresh Profiler. Transfer to the parent stays enabled so control returns upward."""
    return LlmAgent(
        name="Profiler",
        model=config.PRIMARY_MODEL,
        description=(
            "Measures an op on the L4 and records its roofline bottleneck fingerprint, "
            "which is also the skill-library retrieval key."
        ),
        instruction=build_instruction,
        tools=[profiler_tool],
        generate_content_config=deterministic_config(),
        output_key="bottleneck_fingerprint",
        after_tool_callback=capture_fingerprint,
        after_agent_callback=promote_fingerprint,
        disallow_transfer_to_peers=True,
    )
