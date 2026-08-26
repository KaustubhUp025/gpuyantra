"""Loop exit control (spec 4.2).

This is a `BaseAgent`, deliberately — never a tool and never a callback. Setting
`tool_context.actions.escalate = True` from inside a tool or callback is broken in ADK
(issues #501, #2692, #2808, #2988): it either fails to terminate the enclosing
`LoopAgent`, throws OpenTelemetry context errors, or escalates every enclosing loop at
once. The documented, robust pattern is a dedicated sub-agent that yields one Event
carrying `actions.escalate`, which is exactly what `LoopAgent` watches for.

The checker never calls a model, so it costs nothing and cannot fail in a way that
traps the loop. It has one side effect, and it is here because this is the only point
in the tree that knows a run is OVER: crediting the bandit arm that was pulled (spec 9)
with the reward the verifier measured. `best_reward` is only final once the loop
escalates, and one run is one pull — crediting per iteration would count six pulls for
one experiment. The write is idempotent (guarded by `bandit_credited` in state), runs
off the event loop, and swallows its own failures: a Firestore outage must not trap a
loop whose kernel is already verified.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from kernelsmith.config import MAX_LOOP_ITERATIONS
from kernelsmith.tools.retrieval_tool import update_bandit_stats

logger = logging.getLogger(__name__)

#: Reward that ends the run: correct AND faster than both eager and torch.compile.
WINNING_REWARD = 3


def _as_int(value: Any, default: int) -> int:
    """Coerce a state value to int. Model-authored state is never trusted to be typed."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class EscalationChecker(BaseAgent):
    """Reads `verdict` from state and escalates to exit the RefinementLoop when done.

    Escalates on any of three conditions, in order of authority:

    1. the Judge asked to stop (`verdict.stop`),
    2. the verified reward already hit the +3 milestone, or
    3. the iteration counter reached `MAX_LOOP_ITERATIONS`.

    Condition 3 is belt-and-suspenders: `LoopAgent.max_iterations` is the real circuit
    breaker (red line #4). This one also catches the case where a nested runner loses
    the loop's own counter.
    """

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        verdict = state.get("verdict") or {}
        if not isinstance(verdict, dict):
            # The Judge has no output_schema (ADK #3969), so before its
            # after_agent_callback lands, `verdict` is raw model text. Unparsed text is
            # not a decision — keep looping.
            verdict = {}

        iteration = _as_int(state.get("iteration"), 0)
        reward = _as_int(verdict.get("reward"), -1)
        should_stop = (
            bool(verdict.get("stop", False))
            or reward >= WINNING_REWARD
            or iteration >= MAX_LOOP_ITERATIONS
        )

        state_delta: dict[str, Any] = {}
        if should_stop and not state.get("bandit_credited"):
            skill_id = str(state.get("selected_skill_id") or "")
            if skill_id:
                best_reward = _as_int(state.get("best_reward"), reward)
                result = await asyncio.to_thread(update_bandit_stats, skill_id, best_reward)
                if not result.get("updated"):
                    logger.warning("bandit credit failed for %s: %s", skill_id, result.get("error"))
                # Credited or not, do not retry: a second attempt on the next
                # invocation would double-count the pull if the first one landed.
                state_delta["bandit_credited"] = True

        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(escalate=should_stop, state_delta=state_delta),
        )


def build_escalation_checker() -> EscalationChecker:
    """Fresh instance. ADK binds `parent_agent` on assembly, so instances are not shared."""
    return EscalationChecker(
        name="EscalationChecker",
        description=(
            "Decides whether the refinement loop is finished. Reads the verdict from "
            "state and escalates; never calls a model."
        ),
    )
