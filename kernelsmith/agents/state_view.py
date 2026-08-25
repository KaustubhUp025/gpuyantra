"""Rendering `session.state` into prompt text (spec 4.1).

Every agent prompt in this tree is built by an `InstructionProvider` callable rather
than an f-string template with `{placeholders}`. Two reasons, both load-bearing:

1. ADK's regex instruction templating raises `KeyError` on a missing state key. On
   iteration 1 `bottleneck_fingerprint`, `retrieved_skills` and `verdict` do not exist
   yet, so a literal `{verdict}` template crashes the Coder's very first turn.
2. It does not resolve dotted paths — `{verdict.next_action}` is left in the prompt
   verbatim, so the Coder would read the placeholder instead of the Judge's feedback.

A provider bypasses ADK's injection entirely (`canonical_instruction` returns
`bypass_state_injection=True`), which puts truncation and missing-key handling here,
where they can be tested.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

#: Prior kernel sources are full Python files; a few of them would crowd out the task.
SKILL_SOURCE_CHARS = 1800
#: The Judge's stderr tail is already capped at 500 chars upstream; cap again defensively.
FEEDBACK_CHARS = 2000


def as_dict(state: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Read a dict-valued state key, tolerating absence and model-authored garbage."""
    value = state.get(key)
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def render(value: Any, *, empty: str, limit: int | None = None) -> str:
    """JSON-render a state value for a prompt, or `empty` if there is nothing to show."""
    if value is None or value == {} or value == [] or value == "":
        return empty
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, indent=2, default=str)
        except (TypeError, ValueError):
            text = str(value)
    if limit is not None and len(text) > limit:
        text = text[:limit] + f"\n... [truncated, {len(text)} chars total]"
    return text


def render_skills(skills: Any, *, empty: str) -> str:
    """Render retrieved skills as numbered blocks, truncating each kernel source.

    Retrieval is bottleneck-indexed (spec 6.4), so a hit may come from a different op
    than the one being optimized. The `fix_rule` is the transferable part and is shown
    in full; the source is a starting point and is capped.
    """
    if not isinstance(skills, list) or not skills:
        return empty

    blocks = []
    for i, skill in enumerate(skills, start=1):
        if not isinstance(skill, dict):
            continue
        source = str(skill.get("winning_kernel_source", ""))
        if len(source) > SKILL_SOURCE_CHARS:
            source = source[:SKILL_SOURCE_CHARS] + "\n# ... [truncated]"
        blocks.append(
            f"--- prior skill {i}: {skill.get('op_signature', 'unknown')} "
            f"({skill.get('op_family', 'unknown')} on {skill.get('hardware', 'L4')}) ---\n"
            f"fix_rule: {skill.get('fix_rule', '')}\n"
            f"speedup_vs_eager: {skill.get('speedup_vs_eager', 'unknown')}\n"
            f"source:\n{source}"
        )
    return "\n\n".join(blocks) if blocks else empty
