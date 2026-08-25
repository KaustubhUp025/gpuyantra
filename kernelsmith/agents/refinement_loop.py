"""The RefinementLoop: Coder -> Judge -> EscalationChecker, up to six times (spec 4.2).

`max_iterations` is a circuit breaker, not a tuning knob (red line #4). Without it a
loop that never escalates runs until the credits are gone; six iterations is the budget
the cost model in spec 18 is built on.

Exactly one `LoopAgent` level exists in this tree. Nested loops escalate through every
enclosing loop at once in ADK (#2692), so the Supervisor is deliberately an LlmAgent
rather than an outer loop.

Fallback if `LoopAgent` misbehaves at runtime — it does not exit on escalate, or throws
OTel context errors — is to replace it with a custom `BaseAgent` running an explicit
`while` loop that yields each sub-agent's events (spec 4.2).
"""

from __future__ import annotations

from google.adk.agents import LoopAgent

from kernelsmith.agents.coder_agent import build_coder_agent
from kernelsmith.agents.escalation import build_escalation_checker
from kernelsmith.agents.judge_agent import build_judge_agent
from kernelsmith.config import MAX_LOOP_ITERATIONS


def build_refinement_loop() -> LoopAgent:
    """Assemble a fresh refinement loop with its three sub-agents in order."""
    return LoopAgent(
        name="RefinementLoop",
        description=(
            "Writes, verifies and refines a Triton kernel until it beats eager and "
            "torch.compile, or until the iteration budget runs out."
        ),
        sub_agents=[
            build_coder_agent(),
            build_judge_agent(),
            build_escalation_checker(),
        ],
        max_iterations=MAX_LOOP_ITERATIONS,  # NEVER remove: red line #4.
    )
