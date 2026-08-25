"""The Supervisor: root agent and run protocol (spec 4.2).

Assembly is a factory rather than a module-level singleton because ADK binds
`parent_agent` on every sub-agent at construction time; a second `build_supervisor()`
over shared instances would raise. Each call builds a whole fresh tree.

A note on ADK control flow, because it shapes the prompt. When an `LlmAgent` delegates
to a sub-agent, ADK runs that sub-agent and the parent's turn ends — control comes back
only if the sub-agent transfers back, and a `LoopAgent` cannot. The Profiler therefore
hands control back explicitly, but the RefinementLoop cannot, so the invocation ends
when the loop escalates. Steps 4 and 5 of the protocol (save the skill, hot-swap it)
run on the *next* turn: the caller sends one follow-up message and the Supervisor,
whose prompt is rebuilt from state on every turn, picks up where it left off. That is
why the instruction below is a state machine over `session.state` rather than a linear
script — every step is idempotent and skippable.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext

from kernelsmith import config
from kernelsmith.agents.profiler_agent import build_profiler_agent
from kernelsmith.agents.refinement_loop import build_refinement_loop
from kernelsmith.agents.state_view import render
from kernelsmith.tools.retrieval_tool import retrieval_tool
from kernelsmith.tools.upsert_tool import upsert_tool

_INSTRUCTION = """\
You are KernelSmith's supervisor. You orchestrate; you NEVER write Triton code yourself.

TASK:
{task_spec}

WHAT HAS HAPPENED SO FAR (read this before acting — steps already done must not repeat):
- bottleneck_fingerprint: {fingerprint}
- retrieved_skills: {skills_status}
- refinement result: best_reward={best_reward}, iterations={iteration}
- last verdict: {verdict}

PROTOCOL — do the first step below that is not yet done, then stop:
1. If bottleneck_fingerprint is not set, delegate to the Profiler.
2. If retrieved_skills has not been fetched, call retrieval_tool with the fingerprint's
   op_family, hardware and fingerprint_text. An empty result is normal on a cold
   library and is not an error.
3. If the refinement has not run, hand off to the RefinementLoop.
4. If the refinement has finished (best_reward is set) and best_reward >= 1, call
   upsert_tool once to save the winning kernel, passing:
     skill_id: "<op_name>_<hardware>_<short description of the fix>"
     op_signature: e.g. "rmsnorm_fp16_[B,S,H]"
     op_family, hardware: from the fingerprint
     bottleneck_fingerprint: the fingerprint object, unchanged
     winning_kernel_source: best_kernel from state, verbatim and complete
     speedup_vs_eager, speedup_vs_torch_compile: from the winning verdict
     fix_rule: one sentence naming the optimization that won
   Never pass an embedding — it is computed from the fingerprint on write.
   Never upsert a kernel with best_reward < 1: an unverified or slower kernel must not
   enter the library.
5. Then write a short summary: the op, the fingerprint verdict, the reward, the
   measured speedups, and whether the skill was saved. Be exact about numbers and
   never round a speedup upward.

Report the verifier's numbers as they are. A kernel that did not beat torch.compile did
not beat torch.compile.
"""


def build_instruction(ctx: ReadonlyContext) -> str:
    """Render the Supervisor prompt from session state, so each turn resumes the protocol."""
    state = ctx.state
    skills = state.get("retrieved_skills")
    if isinstance(skills, list):
        skills_status = f"fetched, {len(skills)} prior skill(s)"
    else:
        skills_status = "not fetched yet"

    return _INSTRUCTION.format(
        task_spec=render(state.get("task_spec"), empty="(not set — ask the user for the op)"),
        fingerprint=render(state.get("bottleneck_fingerprint"), empty="not set"),
        skills_status=skills_status,
        best_reward=render(state.get("best_reward"), empty="not set"),
        iteration=render(state.get("iteration"), empty="0"),
        verdict=render(state.get("verdict"), empty="none yet"),
    )


def capture_retrieved_skills(
    *,
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict[str, Any],
) -> None:
    """after_tool_callback: publish retrieval results to state for the Coder (spec 4.1).

    The Coder has no tools, so retrieved skills only reach it through `session.state`.
    Returns None so the model still sees the tool's own response.
    """
    if tool.name == retrieval_tool.name and isinstance(tool_response, dict):
        skills = tool_response.get("skills")
        tool_context.state["retrieved_skills"] = skills if isinstance(skills, list) else []
    return None


def build_supervisor() -> LlmAgent:
    """Build the whole KernelSmith agent tree and return its root."""
    return LlmAgent(
        name="Supervisor",
        model=config.PRIMARY_MODEL,
        description=(
            "Root agent. Profiles an op, retrieves prior skills, runs the refinement "
            "loop, and saves the winning kernel back to the skill library."
        ),
        instruction=build_instruction,
        # hotswap_tool joins this list in Task 5, together with protocol step 5
        # ("if best_reward >= 2, patch the live inference server").
        tools=[retrieval_tool, upsert_tool],
        sub_agents=[build_profiler_agent(), build_refinement_loop()],
        output_key="supervisor_summary",
        after_tool_callback=capture_retrieved_skills,
    )
