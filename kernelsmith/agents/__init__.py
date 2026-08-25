"""The ADK agent tree (spec 4).

Every agent is exposed as a `build_*()` factory rather than a module-level instance.
ADK sets `parent_agent` on each sub-agent when a parent is constructed and refuses to
re-parent one, so a shared singleton can only ever live in a single tree — and a second
`build_supervisor()` (a test, a second Runner) would fail on the first shared child.
Factories make the tree cheap to rebuild and safe to construct more than once.

Import from the module that owns the agent, e.g.
`from kernelsmith.agents.supervisor import build_supervisor`.
"""
