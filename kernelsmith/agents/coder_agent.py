"""The Coder: writes exactly one Triton kernel per loop iteration (spec 4.2).

The Coder has NO tools. That is a hard constraint, not an oversight: an agent whose job
is to emit a `KernelDraft` should have exactly one way to end its turn, and giving it
tools reopens the fragile schema+tools path (ADK #3969) that the Judge already has to
work around. Transfer is disabled in both directions for the same reason — inside a
`LoopAgent` a transfer would skip the Judge, and an unjudged kernel must never reach
state.

Everything the Coder knows arrives through `session.state`: the task, the profiler's
bottleneck fingerprint, prior winning kernels from the skill library, and the Judge's
feedback from the previous iteration.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext

from kernelsmith import config
from kernelsmith.agents.state_view import FEEDBACK_CHARS, as_dict, render, render_skills
from kernelsmith.memory.schemas import KernelDraft

_INSTRUCTION = """\
You write ONE Triton kernel to optimize the operation described in the task below.

TASK:
{task_spec}

BOTTLENECK ANALYSIS (from the profiler — this is WHY the op is slow):
{fingerprint}

PRIOR WINNING KERNELS FOR SIMILAR BOTTLENECKS (adapt, never copy verbatim):
{skills}

PREVIOUS JUDGE FEEDBACK (address this specific point before anything else):
{feedback}

RULES:
- Return ONLY valid KernelDraft JSON.
- The kernel MUST use @triton.jit and tl.load/tl.store — never call torch.nn or F.*.
- Do NOT include try/except blocks.
- Do NOT use torch.empty for outputs — always write to the output via tl.store.
- Do NOT spawn extra CUDA streams or threads.
- Reuse self.weight from the original module (it is already on cuda with correct dtype).
- For RMSNorm: upcast to float32 for variance, compute rsqrt, multiply by weight, downcast.
- Target BLOCK_SIZE that keeps the working set in L4 SRAM (~{sram_kb}KB per SM).

`entrypoint` must name a plain Python wrapper function defined in `code` that takes the
same arguments as the reference op and returns the same tensor. The verifier imports
`code` and calls that function directly.
"""


def build_instruction(ctx: ReadonlyContext) -> str:
    """Render the Coder prompt from session state (see `state_view` for why not a template)."""
    state = ctx.state
    verdict = as_dict(state, "verdict")
    feedback = str(verdict.get("next_action", "")).strip()
    stderr_tail = str(verdict.get("stderr_tail", "")).strip()
    if stderr_tail:
        feedback = f"{feedback}\n\nVerifier stderr tail:\n{stderr_tail}".strip()

    return _INSTRUCTION.format(
        task_spec=render(state.get("task_spec"), empty="(not set)"),
        fingerprint=render(
            state.get("bottleneck_fingerprint"),
            empty="(not profiled — assume memory-bound and optimize for bandwidth)",
        ),
        skills=render_skills(
            state.get("retrieved_skills"),
            empty="(none — nothing has been learned for this bottleneck yet)",
        ),
        feedback=render(
            feedback,
            empty="(this is the first attempt — no feedback yet)",
            limit=FEEDBACK_CHARS,
        ),
        sram_kb=config.L4_SRAM_KB_PER_SM,
    )


def build_coder_agent() -> LlmAgent:
    """Fresh Coder. Factory, not a module singleton: ADK binds one parent per instance."""
    return LlmAgent(
        name="Coder",
        model=config.PRIMARY_MODEL,
        description="Writes one Triton kernel draft per iteration from the bottleneck fingerprint.",
        instruction=build_instruction,
        output_key="kernel_draft",
        output_schema=KernelDraft,
        # An unjudged kernel must never escape the loop, in either direction.
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        # Red line: an agent with output_schema gets no tools here (spec 4.2, ADK #3969).
        tools=[],
    )
