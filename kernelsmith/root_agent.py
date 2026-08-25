"""Entry point. Exports `root_agent` so `adk web` and the Runner can discover it.

`build_supervisor()` constructs the entire tree — Supervisor, Profiler, RefinementLoop,
Coder, Judge, EscalationChecker — and ADK binds each sub-agent to exactly one parent, so
the tree is built once here and shared. Call `build_supervisor()` directly if you need
an independent tree (tests do).
"""

from google.adk.agents import LlmAgent

from kernelsmith.agents.supervisor import build_supervisor

root_agent: LlmAgent = build_supervisor()
