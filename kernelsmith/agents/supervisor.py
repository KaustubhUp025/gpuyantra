"""The Supervisor: root agent and run protocol (spec 4.2).

Assembly is a factory rather than a module-level singleton because ADK binds
`parent_agent` on every sub-agent at construction time; a second `build_supervisor()`
over shared instances would raise. Each call builds a whole fresh tree.

A note on ADK control flow, because it shapes the prompt. When an `LlmAgent` delegates
to a sub-agent, ADK runs that sub-agent and the parent's turn ends — control comes back
only if the sub-agent transfers back, and a `LoopAgent` cannot. The Profiler therefore
hands control back explicitly, but the RefinementLoop cannot, so the invocation ends
when the loop escalates. Steps 4 through 6 of the protocol (save the skill, hot-swap
it, summarize) run on the *next* turn: the caller sends one follow-up message and the Supervisor,
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
from kernelsmith.tools.hotswap_tool import hotswap_tool
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
- winning entrypoint: {best_entrypoint}
- last verdict: {verdict}
- hot-swap: {hotswap_status}

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
5. If best_reward >= 2 and the skill has been saved, call hotswap_tool ONCE to patch
   the live inference server:
     kernel_source: best_kernel from state, verbatim and complete
     entrypoint: the winning entrypoint above
     op_name: the op being optimized ("rmsnorm", "swiglu" or "rope")
   A reward of +1 is correct but not faster than eager — never hot-swap it. The server
   parity-checks the kernel against the original forward and rolls back on its own if
   it disagrees; if the result says success=false or rolled_back=true, report that
   plainly and do NOT retry the swap.
6. Then write a short summary: the op, the fingerprint verdict, the reward, the
   measured speedups, whether the skill was saved, and whether the kernel is now live
   on the inference server. Be exact about numbers and never round a speedup upward.

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
        best_entrypoint=render(state.get("best_entrypoint"), empty="not set"),
        iteration=render(state.get("iteration"), empty="0"),
        verdict=render(state.get("verdict"), empty="none yet"),
        hotswap_status=render(state.get("hotswap_result"), empty="not attempted"),
    )


def capture_tool_results(
    *,
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict[str, Any],
) -> None:
    """after_tool_callback: publish tool results into state (spec 4.1).

    Two things have to outlive the turn they happened in. Retrieved skills, because
    the Coder has no tools and can only see them through `session.state`. And the
    hot-swap result, because the Supervisor is resumed from state on the follow-up
    turn — without it, a second turn would re-swap a kernel that is already live.

    Returns None so the model still sees each tool's own response.
    """
    if not isinstance(tool_response, dict):
        return None
    if tool.name == retrieval_tool.name:
        skills = tool_response.get("skills")
        tool_context.state["retrieved_skills"] = skills if isinstance(skills, list) else []
    elif tool.name == hotswap_tool.name:
        tool_context.state["hotswap_result"] = tool_response
    return None


def build_supervisor() -> LlmAgent:
    """Build the whole KernelSmith agent tree and return its root."""
    return LlmAgent(
        name="Supervisor",
        model=config.PRIMARY_MODEL,
        description=(
            "Root agent. Profiles an op, retrieves prior skills, runs the refinement "
            "loop, saves the winning kernel to the skill library, and hot-swaps it "
            "into the live inference server."
        ),
        instruction=build_instruction,
        tools=[retrieval_tool, upsert_tool, hotswap_tool],
        sub_agents=[build_profiler_agent(), build_refinement_loop()],
        output_key="supervisor_summary",
        after_tool_callback=capture_tool_results,
    )
