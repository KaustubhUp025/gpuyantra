"""The Judge: runs the verifier and turns its numbers into a Verdict (spec 4.2).

The Judge has tools and NO `output_schema`. Combining the two is fragile in ADK
(#3969), so structure is recovered afterwards: `record_verdict` runs as an
`after_agent_callback`, parses the model's final text as JSON, validates it against the
`Verdict` model, and writes the result back over the raw string that `output_key`
left in state.

The reconciliation rule matters more than the parsing. Every *measured* field — reward,
correctness, both speedups, latencies, stderr — is taken from the `verify_kernel` tool
response found in the event log, never from the model's transcription of it. The model
contributes only `next_action`, the one concrete fix it wants the Coder to make. A
model that reports `"reward": 3` for a kernel the verifier scored -1 therefore changes
nothing (red line #3: never weaken the verifier).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext

from kernelsmith import config
from kernelsmith.agents.state_view import as_dict, render
from kernelsmith.memory.schemas import Verdict
from kernelsmith.tools.verifier_tool import verifier_tool

logger = logging.getLogger(__name__)

#: Reward floor is -1, so this sentinel loses every comparison on the first iteration.
_NO_BEST = -2

_INSTRUCTION = """\
You evaluate kernel candidates for correctness and performance. You never write kernels.

TASK:
{task_spec}

CANDIDATE KERNEL (from the Coder):
{kernel_draft}

PROTOCOL:
1. Call verifier_tool exactly once, with:
   - kernel_code: the candidate's `code` field, verbatim and complete
   - entrypoint: the candidate's `entrypoint` field
   - task_spec: the task above, as {{"op_name": ..., "hidden_size": ...}}
2. Read the reward JSON it returns. Its numbers are final — never restate them from
   memory and never adjust them.
3. Decide next_action from the reward:
   - reward >= 3: next_action="STOP", stop=true
   - reward == -1: read stderr_tail and failed_cases, then give ONE concrete fix
     (for example "add a mask for the tail elements", "cast to float32 before rsqrt").
     stop=false.
   - reward == 1 or 2: read latency_ms_by_shape, then suggest ONE performance change
     (for example "increase BLOCK_SIZE to 1024", "coalesce the load pattern").
     stop=false.
4. Reply with a single JSON object and no other text:
   {{"reward": int, "correctness_pass": bool, "speedup_vs_eager": float,
     "speedup_vs_compile": float, "next_action": str, "stop": bool,
     "stderr_tail": str, "latency_ms_by_shape": object}}

Give exactly one fix, the highest-leverage one. A list of five suggestions is worse
than one, because the Coder can only test one change per iteration.
"""


def build_instruction(ctx: ReadonlyContext) -> str:
    """Render the Judge prompt from session state."""
    state = ctx.state
    return _INSTRUCTION.format(
        task_spec=render(state.get("task_spec"), empty="(not set)"),
        kernel_draft=render(state.get("kernel_draft"), empty="(the Coder produced nothing)"),
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the verdict object out of the model's final message.

    The model is asked for bare JSON, but with tools in play the text turn often carries
    a fenced block or a sentence of preamble, and ADK's `output_key` accumulates every
    text segment of the turn. Scan for balanced `{...}` spans and prefer the last one
    that parses and looks like a verdict.
    """
    if not isinstance(text, str) or "{" not in text:
        return {}

    candidates: list[dict[str, Any]] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                except ValueError:
                    parsed = None
                if isinstance(parsed, dict):
                    candidates.append(parsed)
                start = -1
            elif depth < 0:
                depth = 0

    for candidate in reversed(candidates):
        if "reward" in candidate or "next_action" in candidate:
            return candidate
    return candidates[-1] if candidates else {}


def find_verifier_response(callback_context: CallbackContext) -> dict[str, Any]:
    """Return the most recent `verify_kernel` tool response in this invocation.

    This is the authoritative record of what was actually measured. Searching backwards
    picks up the current iteration's call; earlier iterations' calls sit further back in
    the same event log because a whole run is one invocation.
    """
    session = callback_context.session
    for event in reversed(getattr(session, "events", []) or []):
        if event.invocation_id and event.invocation_id != callback_context.invocation_id:
            continue
        for response in event.get_function_responses():
            if response.name != verifier_tool.name:
                continue
            payload = response.response
            if isinstance(payload, Mapping) and "reward" in payload:
                return dict(payload)
    return {}


def reconcile(parsed: Mapping[str, Any], measured: Mapping[str, Any]) -> Verdict:
    """Build the Verdict: measurements from the verifier, the fix instruction from the model."""
    if not measured:
        # No tool call landed. Nothing was verified, so nothing may be claimed.
        next_action = str(parsed.get("next_action", "")).strip() or (
            "No verifier result for this draft. Call verifier_tool with the candidate's "
            "code and entrypoint before returning a verdict."
        )
        return Verdict(
            reward=-1,
            correctness_pass=False,
            speedup_vs_eager=0.0,
            speedup_vs_compile=0.0,
            next_action=next_action,
            stop=False,
            stderr_tail=str(parsed.get("stderr_tail", ""))[-500:],
            latency_ms_by_shape={},
        )

    reward = int(measured.get("reward", -1))
    next_action = str(parsed.get("next_action", "")).strip()
    if not next_action:
        # The tool ships a usable default; an unparseable model turn must not lose it.
        next_action = str(measured.get("next_action", "")).strip() or "Retry with one concrete fix."

    latencies = measured.get("latency_ms_by_shape")
    return Verdict(
        reward=reward,
        correctness_pass=bool(measured.get("correctness_pass", False)),
        speedup_vs_eager=float(measured.get("speedup_vs_eager", 0.0) or 0.0),
        speedup_vs_compile=float(measured.get("speedup_vs_compile", 0.0) or 0.0),
        next_action=next_action,
        # The model may ask to stop early; it may never veto stopping on a +3.
        stop=bool(parsed.get("stop", False)) or reward >= 3,
        stderr_tail=str(measured.get("stderr_tail", ""))[-500:],
        latency_ms_by_shape=latencies if isinstance(latencies, dict) else {},
    )


def record_verdict(callback_context: CallbackContext) -> None:
    """after_agent_callback: replace the raw model text in `verdict` with a typed Verdict.

    Also advances `iteration` and keeps `best_reward`/`best_kernel` monotonic, so a
    final iteration that regresses cannot lose an earlier winner (spec 4.2).
    """
    state = callback_context.state
    parsed = extract_json_object(state.get("verdict") or "")
    measured = find_verifier_response(callback_context)
    verdict = reconcile(parsed, measured)

    if not measured:
        logger.warning("Judge produced no verifier_tool response; recording reward=-1.")

    state["verdict"] = verdict.model_dump()

    try:
        iteration = int(state.get("iteration", 0) or 0)
    except (TypeError, ValueError):
        iteration = 0
    state["iteration"] = iteration + 1

    try:
        best_reward = int(state.get("best_reward", _NO_BEST))
    except (TypeError, ValueError):
        best_reward = _NO_BEST

    if verdict.reward > best_reward:
        code = str(as_dict(state, "kernel_draft").get("code", ""))
        state["best_reward"] = verdict.reward
        if code:
            state["best_kernel"] = code
        # Not in the spec 4.1 table, but the Supervisor needs the winning iteration's
        # speedups to build a SkillRecord, and `verdict` is overwritten every loop.
        state["best_verdict"] = verdict.model_dump()

    return None


def build_judge_agent() -> LlmAgent:
    """Fresh Judge. No output_schema — the callback above supplies the structure."""
    return LlmAgent(
        name="Judge",
        model=config.PRIMARY_MODEL,
        description="Verifies a kernel draft in the sandbox and returns a scored verdict.",
        instruction=build_instruction,
        tools=[verifier_tool],
        output_key="verdict",
        after_agent_callback=record_verdict,
        # The loop order is Coder -> Judge -> EscalationChecker; transfers would break it.
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )
