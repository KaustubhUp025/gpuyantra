"""Gemma 4 explains the winning kernel in English (spec 15, bonus +0.2).

The BYOF story this closes: "I cannot write GPU kernels" — the agent tree not only
writes one, it explains it back to the person who could not have written it. The
explanation is prose for a reader who knows C++ and not CUDA, which is the actual
audience for a hot-swapped Triton kernel on a team that has no kernel engineer.

Gemma is called through the managed Vertex AI MaaS endpoint (`gemma-4-26b-a4b-it`), so
there is nothing to deploy and nothing to keep warm. It is deliberately a DIFFERENT
model family from the gemini-3.7-flash agents: the explanation is a second opinion on
the kernel, not the Coder marking its own homework.

This tool touches no measurement and no decision. It runs after the verifier has scored
the kernel and the library has stored it, so a failed or slow explanation costs the run
nothing — every failure comes back as text, never as an exception.
"""

from __future__ import annotations

import google.genai as genai
from google.adk.tools import FunctionTool

from kernelsmith.config import GCP_LOCATION, GCP_PROJECT, GEMMA_MODEL

#: Gemma has no system-instruction slot on MaaS, so the framing rides in the prompt.
_PROMPT = """Explain what this Triton kernel does, why it's faster than the eager PyTorch
implementation, and what hardware feature it exploits. Write for an engineer who knows
C++ but not GPU programming.

```python
{kernel_source}
```"""

#: A kernel is a few hundred lines at most; anything longer is not a kernel, and
#: sending it would only spend tokens on a paste error.
MAX_SOURCE_CHARS = 20_000

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazily construct the Vertex AI genai client (ADC only — never a key)."""
    global _client
    if _client is None:
        _client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    return _client


def explain_kernel(kernel_source: str) -> str:
    """Explain a verified Triton kernel in plain English, for a non-GPU engineer.

    Call this once, on the winning kernel, after it has been verified and saved. The
    text is for the dashboard and the demo narration — it is never an input to any
    decision, and nothing in the system reads it back.

    Args:
        kernel_source: Complete Python source of the verified kernel.

    Returns:
        The explanation, or a string beginning "error:" if Gemma could not be reached.
        Never raises: an explanation is a bonus, and losing it must not fail a run whose
        kernel is already verified and live.
    """
    source = (kernel_source or "").strip()
    if not source:
        return "error: no kernel source to explain"
    if len(source) > MAX_SOURCE_CHARS:
        source = source[:MAX_SOURCE_CHARS] + "\n# ... [truncated]"

    try:
        response = _get_client().models.generate_content(
            model=GEMMA_MODEL,
            contents=_PROMPT.format(kernel_source=source),
        )
    except Exception as exc:  # noqa: BLE001 — the bonus path never breaks the main one
        return f"error: {type(exc).__name__}: {exc}"

    text = (getattr(response, "text", "") or "").strip()
    return text or "error: Gemma returned an empty explanation"


#: Registered on the Supervisor (spec 4.2), called after the winning skill is saved.
explainer_tool = FunctionTool(explain_kernel)
