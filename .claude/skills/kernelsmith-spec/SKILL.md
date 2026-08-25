---
name: kernelsmith-spec
description: >
  KernelSmith implementation specification. Use this skill for ALL code generation,
  review, and architecture decisions in this project. Contains exact model IDs,
  schemas, agent definitions, verifier design, testing strategy, and red lines.
  Reference this before writing any code.
---

# KernelSmith Implementation Specification

This skill contains the ground-truth implementation spec for KernelSmith.

## Quick Reference (always apply these)

- **Model:** `gemini-3.7-flash` for ALL agents. Global endpoint. No Pro models exist at 3.5+.
- **ADK:** `google-adk==2.7.1`. In-process only. No A2A, LangGraph, Swarm.
- **Firestore:** Vector(768), COSINE, flat index. Equality pre-filters only.
- **Embeddings:** `gemini-embedding-001` @768 dims. ASSERT len==768. L2-normalize manually.
- **Verifier:** 5 seeds × 3 shapes, atol=rtol=1e-2, do_bench warmup=150 rep=200.
- **EscalationChecker:** MUST be a BaseAgent, never a tool (ADK bugs #501, #2692, #2808).
- **Judge:** NO output_schema (ADK bug #3969). Parse JSON in after_agent_callback.
- **Sandbox:** Subprocess with SIGKILL. Never run generated code in-process.
- **Never:** weaken verifier, remove max_iterations, commit secrets, use pre-3.5 model IDs.

## Full Specification

See `reference/spec.md` for the complete implementation specification including:
- Section 0: Version-pinned stack
- Section 1: Repository layout
- Section 2: config.py constants
- Section 3: Pydantic schemas
- Section 4: Agent architecture (all 6 agents)
- Section 5: Verifier (correctness, timing, AST checker, sandbox, reward)
- Section 6: Firestore memory (schemas, indexes, embeddings, retrieval, upsert)
- Section 7: Profiler + bottleneck fingerprint
- Section 8: Inference server + hot-swap
- Section 9: Bandit over skills
- Section 10: Streamlit dashboard
- Section 11: Reproducibility contract
- Section 12: Security
- Section 13: Testing strategy
- Sections 14-22: Makefile, Gemma bonus, demo choreography, ADK bugs, cost, setup, code review, build order, red lines

**Read the relevant section from `reference/spec.md` before implementing any module.**
