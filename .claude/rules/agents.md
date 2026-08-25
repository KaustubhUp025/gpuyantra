---
paths:
  - "kernelsmith/agents/**"
  - "kernelsmith/root_agent.py"
---

# Agent Rules

Read Section 4 of the kernelsmith-spec skill before editing.

- All agents use model="gemini-3.7-flash"
- State mutation only via event.actions.state_delta or output_key
- Coder: has output_schema=KernelDraft, NO tools, disallow_transfer_to_parent=True
- Judge: has tools=[verifier_tool], NO output_schema. Parse JSON in after_agent_callback.
- EscalationChecker: BaseAgent subclass, yields Event(actions=EventActions(escalate=should_stop))
- RefinementLoop: LoopAgent(max_iterations=6) with sub_agents=[Coder, Judge, EscalationChecker]
- Supervisor: LlmAgent with tools=[retrieval_tool, upsert_tool, hotswap_tool]
