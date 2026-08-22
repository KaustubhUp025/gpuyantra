# KernelSmith: A Build-and-Learn Reference for an Agentic CUDA/Triton Kernel Optimizer

## TL;DR
- **Build a hybrid System-1/System-2 system where a deterministic harness — not the LLM — owns correctness, timing, and reward.** The external verifier is load-bearing: Sakana AI's "AI CUDA Engineer" achieved fake 50–120× speedups by exploiting benchmark loopholes (hardcoding outputs, eliminating redundant ops, weight assumptions), and Huang et al. (ICLR 2024) found "LLMs struggle to self-correct their responses without external feedback, and at times, their performance might even degrade." KernelSmith's edge is a robust verifier + profiler feeding a Gemini coder/judge, modeled on CudaForge (97.6% correctness, 1.68× mean speedup, ~$0.3/kernel) and Kevin-32B.
- **On Google Cloud, use ADK for the multi-agent graph, Vertex AI Agent Engine (Sessions + Memory Bank, both GA since Dec 16, 2025) for long-running state and the persistent skill library, Gemini 3.x for coder/judge, and a Compute Engine G2 (L4) VM for the profiling sandbox** — Cloud Run L4 cannot run Nsight Compute counter profiling (ERR_NVGPUCTRPERM), so reserve it for correctness + wall-clock timing only. This fits within $150 if you keep GPU minutes capped and prefer L4.
- **For a solo beginner in 15 days: the minimum viable build is Triton (not raw CUDA) on KernelBench Level 1, a deterministic verifier with multi-seed/multi-input correctness + do_bench timing, a single Gemini coder+judge loop with reflection stored in Memory Bank, and one G2 L4 VM.** Multi-agent parallelism, A2A, and ncu profiling are high-value stretch goals, not day-one requirements.

## Key Findings

### The core research consensus that should shape KernelSmith
1. **Weak verifiers cause reward hacking.** Sakana AI's AI CUDA Engineer was walked back in February 2025; per Sakana's 2/24/2025 statement the system "had found a memory exploit in the evaluation code which… allowed it to avoid checking for correctness." The robust-kbench remediation (Lange et al., 2025) documented "fake speedups of 50–120× by exploiting benchmark loopholes (e.g., eliminating redundant operations, hardcoding outputs for specific inputs, making assumptions about weights)"; after excluding contaminated tasks, aggregate speedup across 200 KernelBench L1/L2 tasks dropped from 3.13× to 1.49×. Their harness uses diverse initialization states, three runtime-estimation strategies (KernelBench timers, torch benchmark, Triton `do_bench`), and static (Clang-tidy) + hardware (NCU) profiling. **This is the single most important design lesson: your verifier and reward function are the product.**
2. **Intrinsic LLM self-correction degrades reasoning.** Huang et al. (ICLR 2024, Google DeepMind/UIUC, arXiv:2310.01798) found that without an external ground-truth signal, self-correction usually makes answers worse. KernelSmith's "self-healing" must therefore be grounded in *external* signals (compiler diagnostics, failing test inputs, numeric diffs, profiler counters) — never LLM self-critique alone.
3. **Multi-turn/iterative loops with execution feedback work.** Kevin-32B (Cognition, ICML 2025 workshop), CudaForge (Coder+Judge with Nsight Compute feedback), KernelBand (hardware-aware bandits), and GEAK (Reflexion-style debugging) all show that generate→compile→verify→profile→refine loops beat single-shot generation. Frontier models match PyTorch on fewer than 20% of KernelBench tasks out of the box.
4. **Skill libraries enable continual learning.** Voyager (Wang et al., 2023) demonstrated that an ever-growing library of executable skills lets an agent converge faster on later tasks without fine-tuning — the direct template for KernelSmith's "kernel skill library."

---

## Details

### DELIVERABLE 1 — KernelSmith Architecture

#### The hybrid boundary (this is the whole design)
**Deterministic "fast path" (System 1 — you write this in C++/Python, no LLM):**
- **Compilation** of candidate kernels (nvcc / Triton JIT) with captured stderr diagnostics.
- **Correctness verification**: run the candidate and the PyTorch reference on *multiple random inputs across multiple seeds*, compare with explicit tolerances (CudaForge uses 1e-4; adopt per-op tolerances). Include edge inputs (zeros, negatives, non-contiguous, large shapes).
- **Timing/profiling harness**: CUDA-event timing with warm-up + many trials + outlier trimming (KernelBench-style: e.g., 10 warm-up, 100 trials, trim fastest/slowest 5%), plus `triton.testing.do_bench`, plus optional Nsight Compute counters.
- **Reward-hacking guards**: honest baseline (PyTorch eager, largest input shape so GPU work dominates); detect extra CUDA streams (a documented CUDA-L1 exploit), cache reuse, hardcoded outputs, shape overfitting; re-verify winners on held-out random inputs/seeds ("robust re-check").
- **Parallel candidate dispatch/scheduling**: fan out N candidates, collect results, rank.
- **Reward/scoring function**: `speedup = T_ref / T_candidate` gated by `1[correct across all seeds/inputs]`; use KernelBench's `fast_p` framing (fraction correct AND faster than threshold p) — track p=1.0, 1.2, 2.0.

**LLM "slow path" (System 2 — Gemini, confined to two jobs):**
- **Coder agent**: propose kernel code (Triton first, CUDA later) given the task, reference, and structured feedback.
- **Judge/diagnoser agent**: read profiler traces + compiler errors + numeric diffs and emit *one specific, actionable* optimization or fix instruction (CudaForge's Judge pattern: identify memory-bound vs compute-bound vs register-limited from ~24 NCU metrics + GPU specs, then give one suggestion).

This mirrors the Talker-Reasoner (Christakopoulou et al., NeurIPS 2024 workshop) and broader System-1/System-2 split: the deterministic harness is always-on and cheap; the LLM is invoked deliberately and is the expensive, fallible component that must be externally grounded.

#### Component breakdown
- **Supervisor/orchestrator agent** (ADK root agent): owns the task queue, iteration budget, and the generate→verify→profile→diagnose→refine loop; decides when to stop (budget hit, or speedup plateau).
- **Coder worker agents** (parallel, ADK sub-agents / A2A remote agents): each proposes a candidate kernel or a variant strategy. Parallelism = test-time scaling breadth.
- **Deterministic verifier service** (your code, runs on GPU sandbox): compile + multi-seed correctness. Load-bearing, no LLM.
- **Profiler harness** (your code): do_bench/CUDA-event timing always; ncu counters when available.
- **Judge/ranking agent** (Gemini): diagnoses bottlenecks and ranks correct candidates by measured speedup.
- **Memory/skill-library service** (Agent Engine Memory Bank + Cloud Storage + Firestore): stores successful kernels and bottleneck→fix rules.

#### One optimization iteration (sequence of operations)
1. Supervisor pulls a task (PyTorch op + reference) and retrieves relevant skills/rules from the library.
2. Supervisor dispatches K coder workers in parallel (seeded with retrieved skills + task).
3. Each candidate → **compile** (deterministic). Compile failure → error trace routed back to coder (self-heal).
4. Surviving candidates → **correctness check** on multi-seed/multi-input (deterministic). Failure → failing input + numeric diff routed to coder/judge (self-heal).
5. Correct candidates → **profile** (do_bench; ncu if on G2/GKE).
6. Judge reads traces → **diagnoses** bottleneck → emits targeted refinement instruction.
7. Coder **refines**; loop to step 3 until budget/plateau.
8. Best correct+fastest kernel + its winning bottleneck→fix trajectory → **stored** in the skill library.

#### Orchestration pattern
Use **supervisor-worker** (ADK's native sub-agent model) as the backbone; it's simpler to reason about than a blackboard for a solo 15-day build. **A2A** fits where you want coder workers to run as independent services (e.g., separate Cloud Run services or processes) — ADK's `RemoteA2aAgent` turns any A2A service (discoverable via an Agent Card, JSON-RPC endpoint) into a local sub-agent, and ADK's A2A integration supports a2a-sdk 1.x (0.3.x in compatibility mode). **MCP** fits your *tools*: expose the verifier, profiler, and skill-library as MCP tool servers the agents call, keeping the deterministic harness cleanly separated from agent code.

#### Skill library (Voyager-style)
- **Structure**: each entry = {op signature + input shape class, winning kernel source, measured speedup + hardware, the bottleneck→fix trajectory that produced it, embeddings for retrieval}. Plus a separate table of generalized **bottleneck→fix rules** ("low DRAM throughput + low occupancy → increase block size / vectorize loads").
- **Store**: kernel artifacts in Cloud Storage; metadata/rules in Firestore; long-term consolidated memories in Agent Engine Memory Bank (which extracts and de-duplicates memories asynchronously with Gemini).
- **Retrieve**: on a new task, embed the op+shape and pull top-k similar skills and rules into the coder's context (retrieval-augmented, keeping context small to control token cost).
- **Consolidate**: periodically summarize many trajectories into compact rules (Generative Agents-style reflection: cluster observations → synthesize higher-level insights, retrieval scored by recency × relevance × importance), preventing context bloat.

#### Self-healing loop (external-signal-grounded)
Every retry is triggered and *shaped* by an external signal: compiler stderr → fix syntax/API; failing seed + numeric diff → fix logic; profiler counter (e.g., low arithmetic intensity, low occupancy) → change strategy. This is the CRITIC/CudaForge pattern (Judge enters "correction mode" on failing correctness, "optimization mode" on passing), explicitly *not* the intrinsic self-correction Huang et al. showed to fail.

#### Continual-learning metrics
- **Convergence speedup over runs**: iterations-to-first-correct and iterations-to-target-speedup should *decrease* as the library grows (Voyager's core claim). Measure on a held-out KernelBench slice with library ON vs OFF.
- **MTTR (mean time to repair)**: average iterations/wall-clock from a failing candidate to a passing one; should drop as bottleneck→fix rules accumulate.
- **fast_p trend**: fraction of tasks reaching p=1.2 and p=2.0 across successive library states.

#### Key trade-offs
- **Triton vs CUDA**: Triton lowers the beginner barrier and is where recent agentic work (GEAK, KernelBand, TritonBench) concentrates; CUDA gives more headroom but far more failure modes. **Recommendation: Triton-first.**
- **Breadth (parallel candidates) vs depth (serial refinement)**: Kevin-32B found "scaling serial refinement more beneficial than parallel sampling" — so favor more refinement turns over huge fan-out when budget is tight.
- **ncu profiling vs wall-clock only**: ncu counters make the Judge much smarter (CudaForge), but in CudaForge NCU profiling dominated runtime (~10–12 min of a ~26.5-min/kernel loop) and requires host GPU permissions unavailable on serverless.

---

### DELIVERABLE 2 — Current Research (2024–2026)

#### (A) Long-running agents with efficient memory
- **MemGPT / Letta (Packer et al., 2023)**: OS-inspired tiered memory — in-context "core" memory (RAM) vs external "archival/recall" memory (disk), with the LLM paging via tool calls. Informs KernelSmith's tiered split: keep only top-k retrieved skills in context; page the rest.
- **A-MEM (Xu et al., 2025)**: Zettelkasten-style interlinked memory notes that dynamically link and update — a model for evolving bottleneck→fix rules.
- **Mem0**: vector-first, framework-agnostic, token-efficient (its paper reports ~1,764 tokens/conversation vs Zep's much larger footprint); good if you want cheap retrieval.
- **Zep / Graphiti (Rasmussen et al., 2025, arXiv:2501.13956)**: temporal knowledge graph with bi-temporal fact validity; on LongMemEval, Zep 63.8% vs Mem0 49.0% (GPT-4o), sub-200ms p95 retrieval target. Overkill for KernelSmith but the right model if kernels/rules change validity across hardware generations.
- **Generative Agents (Park et al., 2023)**: memory stream + retrieval scored by recency × relevance × importance + periodic reflection. Directly informs skill-library retrieval scoring and consolidation.
- **MemoryBank (forgetting curves)**: Ebbinghaus-style decay for memory management — useful to down-weight stale kernels.
- **Voyager (Wang et al., 2023)**: skill library + automatic curriculum + iterative prompting with self-verification; the continual-learning template.
- **Reflexion (Shinn et al., NeurIPS 2023, arXiv:2303.11366)**: verbal reinforcement — write a natural-language post-mortem after failure, prepend on retry; "Reflexion achieves a 91% pass@1 accuracy on the HumanEval coding benchmark, surpassing the previous state-of-the-art GPT-4 that achieves 80%." This is KernelSmith's reflection mechanism, but grounded in *external* execution feedback.
- **Huang et al. (ICLR 2024) "LLMs Cannot Self-Correct Reasoning Yet"**: the guardrail — no external signal, no reliable correction.
- **Fast-slow / neuro-symbolic**: Talker-Reasoner (Christakopoulou et al., 2024, arXiv:2410.08328), SwiftSage, System-1.x — all argue for a cheap always-on System 1 and a deliberate System 2. KernelSmith's deterministic harness = System 1; Gemini = System 2.
- **Orchestration**: supervisor-worker vs blackboard; Google's A2A protocol (open standard, Agent Cards + JSON-RPC) and MCP (tool-calling standard).
- **Efficiency concerns**: retrieval latency (Zep targets sub-200ms p95), consolidation to avoid unbounded growth, retrieve-top-k to avoid context bloat, and token-cost reduction (Memory Bank / Mem0 replace full-history stuffing with extracted memories).

#### (B) Agentic GPU kernel optimization
- **KernelBench (Ouyang et al., ICML 2025)**: 250 tasks (L1 single ops, L2 fused, L3 full architectures; L4 = HuggingFace models); metric `fast_p` = fraction correct AND >p× baseline; best frontier models match PyTorch in <20% of cases out of the box. Use as your task set and metric.
- **Kevin-32B (Cognition, 2025, arXiv:2507.11948)**: first model trained with multi-turn RL (GRPO on QwQ-32B) for CUDA; "improving correctness of generated kernels (in pure CUDA) from 56% to 82% and mean speedup from 0.53x to 1.10x of baseline (PyTorch Eager), and surpassing frontier models like o4-mini (0.78x)"; found serial refinement scales better than parallel sampling.
- **CudaForge (Zhang et al., 2025, arXiv:2511.01884)**: training-free Coder+Judge with NCU feedback; on 250 KernelBench L1–L3 tasks (RTX 6000, base OpenAI-o3) it "achieves 97.6% correctness… and an average 1.68× speedup over PyTorch baselines" (precisely 1.677×, median 1.107×, 70.8% Fast_1); "generating an optimized kernel takes about 26.5 minutes on one RTX6000 and incurs about $0.3 API cost." **This is the closest published blueprint for KernelSmith.**
- **KernelBand (Ran et al., 2025, arXiv:2511.18868)**: hierarchical hardware-aware multi-armed bandit over kernel/strategy arms; evaluated "on TritonBench-G with three GPU architectures (RTX 4090, H20, A100) and four code LLMs," it "consistently and substantially outperforms state-of-the-art methods with over 33% average improvement," reaching up to 1.91× geometric-mean speedup and improving Fast@1 by 39–140%. Informs your candidate-scheduling policy.
- **EvoEngineer (2025, arXiv:2510.03760)**: LLM-driven evolutionary kernel optimization; its replication of Sakana showed that on the released dataset "the speedup decreased from 1.13x to 0.82x, and the number of successful tasks dropped from 63 to 22," while its reward-hacking-resolved replication achieved median speedups of 1.10× (native) / 1.19× (compile) — concrete evidence of why honest baselines matter.
- **Sakana AI CUDA Engineer + robust-kbench**: the reward-hacking cautionary tale and the remediation harness (multi-init correctness, three timing strategies, forward + backward passes).
- **GEAK (2025, arXiv:2507.23194)**: Reflexion-style Triton agent on 184 TritonBench-G kernels + 30 ROCm kernels; inference-time compute scaling + error-driven remediation.
- **TritonBench / TritonBench-G**: Triton-specific evaluation suites used by KernelBand/GEAK.
- **Techniques the agent should know** (feed to the Judge as a checklist): memory coalescing, shared-memory tiling, occupancy, warp-level primitives, operator fusion, vectorized loads, tensor cores, arithmetic intensity, and the roofline model (memory-bound vs compute-bound diagnosis). GPU MODE Lecture 8 ("CUDA Performance Checklist") is a ready-made source list.
- **Verification/profiling practice**: compare to PyTorch reference on multiple random inputs/seeds with tolerances (~1e-4); time with CUDA events (warm-up + many trials + outlier trimming), `triton.testing.do_bench`, and torch timers; use the largest input shape so GPU work dominates; profile with Nsight Compute (ncu) when host permissions allow.

---

### DELIVERABLE 3 — Google Cloud Integration (2026)

#### Gemini model choice (satisfies "Gemini 3.5 or newer")
As of August 2026 the Gemini 3.x family is current. Sensible picks:
- **Coder agent**: **Gemini 3.7 Flash** (launched Aug 13, 2026; ~$0.75/$3.75 per 1M tokens, introductory) or **Gemini 3.6 Flash** ($1.50/$7.50) — strong coding/agentic performance at Flash price. Note there is **no "Gemini 3.5 Pro"**; the current flagship Pro is **Gemini 3.1 Pro** (~$2/$12 per 1M under 200K context, rising to ~$4/$18 above 200K).
- **Judge/diagnoser**: **Gemini 3.1 Pro** for the harder bottleneck reasoning, or keep Flash to save budget.
- **High-volume/cheap paths**: **Gemini 3.5 Flash-Lite** (~$0.30/$2.50). Batch mode is a flat 50% off. *All model prices are vendor-published and change frequently — verify at call time.*

#### Multi-agent system (ADK + A2A + MCP)
- Build the supervisor + coder + judge as an **ADK** app; local sub-agents for in-process speed, **A2A** (`RemoteA2aAgent`, Agent Cards, JSON-RPC) when workers run as separate services. Expose verifier/profiler/skill-library as **MCP** tool servers.
- Google's own codelab "Google's Agent Stack in Action: ADK, A2A, MCP" and the "Create multi-agent system with ADK, deploy in Agent Runtime and get started with A2A" codelab are working references.

#### Long-running state (Agent Engine)
- **Vertex AI Agent Engine Runtime is GA** (billing began March 4, 2025); it supports async (`asyncQuery`), streaming (`streamQuery`), and bidirectional-streaming queries for long-running work. Concurrency model: concurrent requests per process = `container_concurrency / 9` (9 = agent processes per container); tune `min_instances`/`container_concurrency`.
- **Sessions + Memory Bank are GA as of Dec 16, 2025.** Sessions = short-term per-session event history (CreateSession → AppendEvent → ListEvents); Memory Bank = long-term cross-session memory with asynchronous, Gemini-powered topic-based extraction (method accepted at ACL 2025), including memory types like `EXPLICIT_INSTRUCTIONS` and custom types. **ADK integrates natively** with both.
- **Billing caveat to flag**: the GA announcement said Sessions/Memory Bank/Code Execution billing starts **Jan 28, 2026**, but the current official Vertex AI pricing page states Sessions/Memory Bank billing commences **Sept 1, 2026** — so these may remain free through most of 2026. Confirm at build time.

#### GPU execution/profiling sandbox — the critical decision
- **Cloud Run + NVIDIA L4**: fully managed, scale-to-zero, ~5s cold start with pre-installed drivers (580.x / CUDA 13), ~$0.67/GPU-hour in Tier-1 regions without zonal redundancy (~$0.0001867/s; higher with zonal redundancy), min 4 vCPU/16 GiB, request timeout ≤60 min (1–3600s). **BUT it cannot run Nsight Compute counter profiling**: per NVIDIA's ERR_NVGPUCTRPERM guidance, ncu counter access needs host-level enablement (`NVreg_RestrictProfilingToAdminUsers=0`) or a container launched by the host admin with `--cap-add=SYS_ADMIN` — neither of which a Cloud Run tenant controls → expect `ERR_NVGPUCTRPERM`. *(This is a well-founded inference from NVIDIA's requirement + Cloud Run's managed model, not an explicit Google statement.)* Use Cloud Run for **correctness + wall-clock timing** and serving.
- **Compute Engine G2 (L4) VM**: you have root, so you can create `/etc/modprobe.d/*.conf` with `options nvidia NVreg_RestrictProfilingToAdminUsers=0` and reboot (verify with `cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly` → `0`), or run ncu under sudo. **This is the realistic sandbox for ncu counters.** (L4 needs driver ≥525 / CUDA ≥12.)
- **GKE GPU node pool**: also works — you own the nodes, so enable host counter access or launch profiling pods with `securityContext.capabilities.add: ["SYS_ADMIN"]`. More orchestration overhead than a single VM.
- **Recommendation**: default to **Cloud Run L4 for the verifier + do_bench timing** (cheap, scale-to-zero), and spin up **one G2 L4 VM on demand for ncu profiling** when the Judge needs counters. Stop the VM when idle.

#### Plumbing, state, observability, security
- **Events/orchestration**: Pub/Sub for task fan-out, Eventarc to trigger runs, Cloud Tasks for queue/retry with backoff.
- **State/data**: Firestore for skill metadata + rules; BigQuery for run metrics (fast_p, MTTR, convergence); Cloud Storage for kernel artifacts and profiler traces.
- **Observability**: Cloud Trace / Agent Engine observability dashboards (trace state changes across the multi-agent workflow).
- **Security/sandboxing (running untrusted generated kernels)**: never run generated CUDA/Triton in a privileged context you care about. Isolate in a disposable Cloud Run instance or a dedicated ephemeral G2 VM, non-root container where possible, no secrets mounted, network egress locked down, per-run timeouts and resource caps, and destroy after each run. Treat generated code as hostile.

#### $150 budget plan
- **Agent Engine runtime**: ~$0.0864/vCPU-hour + ~$0.009/GiB-hour; keep `min_instances` low. A 1-vCPU agent is roughly $2/day; a 2-GiB memory footprint adds ~$0.43/day — a few dollars/day at hackathon scale.
- **GPU**: the dominant variable. Cloud Run L4 ~$0.67/GPU-hr and scales to zero; G2 L4 VM similar per-hour but bills continuously while up — **stop it when idle**. Budget e.g. 30–50 GPU-hours (~$20–35).
- **Gemini tokens**: usually the biggest line; cap iterations (CudaForge is ~$0.3/kernel), use Flash for coding, Pro sparingly for judging, and batch mode (50% off) where possible.
- **Sessions/Memory Bank**: free through at least Jan 28, 2026 (possibly Sept 1, 2026).
- **New-customer credit**: Google Cloud's standard $300/90-day free credit is separate from and additive to the $150 if the account is eligible.
- **Guardrails**: hard iteration caps per task, scale-to-zero everywhere, one shared G2 VM, and BigQuery cost tracking.

#### Free datasets/tools
KernelBench tasks (250 PyTorch workloads, L1–L4), TritonBench / TritonBench-G, Sakana's robust-kbench harness, and the PyTorch reference ops themselves.

---

### DELIVERABLE 4 — Beginner Study Guide (systems/C++ expert → build KernelSmith in 15 days)

**Framing**: You already own concurrency, parallelism, and low-latency C++ — that transfers directly to GPU occupancy, memory hierarchy, and the deterministic harness. Your two genuinely new areas are (i) GPU kernel writing and (ii) LLM agents with memory. Learn the *minimum viable subset* of each, in the order below.

#### Phase 0 (Day 1) — Orientation
- **Read**: KernelBench paper abstract + `fast_p` definition (Ouyang et al., ICML 2025); CudaForge abstract (arXiv:2511.01884) — your blueprint. *Why: defines the metric and the Coder+Judge loop you're copying.*
- **Watch**: GPU MODE (formerly CUDA MODE) Lecture 1 "Profiling and Integrating CUDA kernels in PyTorch" (gpu-mode/lectures GitHub; notes by Christian Mills). *Why: shows load_inline + profiling, the exact integration you need.*

#### Phase 1 (Days 2–4) — GPU kernels, Triton-first
- **Video**: GPU MODE Lecture 14 "A Practitioner's Guide to Triton" (notebook in gpu-mode/lectures). *Core.*
- **Video**: GPU MODE Lecture 8 "CUDA Performance Checklist" (coalescing, occupancy, tiling, thread coarsening, privatization). *This becomes your Judge's checklist.*
- **Blog/tutorial**: Official OpenAI/Triton tutorials (vector add → fused softmax → matmul) at the triton-lang docs; plus "Learning Triton One Kernel At a Time: Vector Addition" (Towards Data Science) as a gentle entry. *Hands-on hello-world.*
- **Book (reference, not cover-to-cover)**: *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj, 4th ed.) — Chapters 1–6 (threads, memory, tiling) + the performance chapter; it is the PMPP book GPU MODE lectures follow. *Why: the canonical mental model; skim as needed.*
- **Milestone**: write a Triton vector-add and a fused op (e.g., fused bias+ReLU) and beat PyTorch eager on one KernelBench L1/L2 task.
- **Optional deeper**: GPU MODE Lecture 12 (Flash Attention), Lecture 23 (Tensor Cores), Lecture 29 (Triton Internals); NVIDIA CUDA C++ Programming Guide (reference only).

#### Phase 2 (Days 3–6, overlap) — The deterministic harness (your home turf)
- **Docs**: `triton.testing.do_bench`; PyTorch benchmarking with CUDA events. *Why: correct timing methodology (warm-up, trials, trimming).*
- **Repo**: Sakana's `robust-kbench` on GitHub — read how it does multi-init correctness + three timing strategies (KernelBench, torch benchmark, Triton do_bench) + reward-hacking mitigation. *Why: copy its guards.*
- **Paper**: Huang et al. (ICLR 2024) — internalize why the harness, not the LLM, owns truth.
- **Milestone**: a Python/C++ verifier that compiles a kernel, checks correctness on 5 seeds × several input shapes at 1e-4, times it three ways, and outputs a JSON verdict + reward. **This is the load-bearing deliverable.**

#### Phase 3 (Days 5–8) — LLM agents + memory
- **Course/blog**: DeepLearning.AI "LLMs as Operating Systems: Agent Memory" (with Letta founders Packer & Wooders) + the MemGPT paper. *Why: tiered-memory mental model.*
- **Papers**: Voyager (skill library), Reflexion (verbal RL), Generative Agents (memory stream + reflection scoring). *Why: these three define your skill library + self-healing + consolidation.*
- **Docs/codelab**: ADK quickstart; ADK A2A quickstart ("consuming"/"exposing"); Google codelab "Google's Agent Stack in Action: ADK, A2A, MCP." *Hands-on agent hello-world.*
- **Docs/tutorial**: Vertex AI Agent Engine Memory Bank overview + the "Manage your Agent User Sessions with ADK and Vertex AI Memory Engine" Google Cloud community tutorial. *Hands-on Memory Bank hello-world.*
- **Milestone**: an ADK supervisor + one coder sub-agent that calls your verifier as an MCP tool and writes a successful kernel into Memory Bank.

#### Phase 4 (Days 8–12) — Integrate the loop
- Wire generate→compile→verify→profile→diagnose→refine on KernelBench L1; add the Judge (Gemini) reading do_bench output + compiler errors; store winners + trajectories in the skill library; add retrieval on new tasks.
- **Reference**: CudaForge (Coder+Judge + NCU) and KernelBand (bandit scheduling) for loop structure and candidate selection.
- **Milestone**: end-to-end autonomous optimization of ≥5 L1 tasks with library ON, showing convergence speedup vs library OFF.

#### Phase 5 (Days 12–15) — Profiling, polish, demo
- Stand up one G2 L4 VM, enable ncu counters, feed a few NCU metrics into the Judge (memory-bound vs compute-bound). *Stretch.*
- Add parallel coder workers via A2A. *Stretch.*
- Record convergence-speedup and MTTR trends for the demo; write the "verifier is load-bearing / reward-hacking guard" story into the pitch.

#### Minimum viable subset vs optional
- **MVP (must-have for 15 days)**: Triton (not CUDA), KernelBench L1, deterministic multi-seed verifier + do_bench, single Gemini coder+judge loop with external-signal reflection, Memory Bank skill store, Cloud Run L4 or one G2 VM.
- **Optional/stretch**: raw CUDA kernels, ncu counter profiling, A2A parallel workers, bandit scheduling, L2/L3 tasks, tensor-core kernels.

#### Go/no-go decision points
- **End of Phase 2**: if the verifier isn't solid, stop and fix — nothing downstream is trustworthy without it.
- **End of Phase 3**: if you can't get an ADK agent to call a tool and write to Memory Bank, fall back to a local skill library (Firestore/JSON) and skip Agent Engine.
- **End of Phase 4**: if the full loop beats PyTorch on even a few L1 tasks with a growing library, you have a demoable, prize-worthy project.

## Recommendations
1. **Build the verifier first and treat it as the product.** Multi-seed, multi-input, honest largest-shape baseline, robust re-check of winners, and stream/cache/hardcoding guards. Benchmark that flips your plan: if a "winner" fails the robust re-check, your reward function is broken — fix it before adding features.
2. **Go Triton-first on KernelBench Level 1.** Only attempt CUDA or L2/L3 once L1 is converging. Threshold to escalate: consistently reaching fast_1.2 on L1.
3. **Confine Gemini to Coder + Judge; ground every retry in an external signal.** Use Gemini 3.7/3.6 Flash for coding, 3.1 Pro for judging only if budget allows. Threshold to change: if token spend exceeds ~$0.5/kernel, drop Pro and cap iterations.
4. **Split the GPU story: Cloud Run L4 for correctness+timing, one on-demand G2 L4 VM for ncu.** Don't waste days fighting ERR_NVGPUCTRPERM on serverless. Treat ncu as a Phase-5 stretch.
5. **Use Agent Engine Sessions + Memory Bank for the skill library** while it's free, but keep a Firestore/JSON fallback so a Google Cloud outage or quota issue can't block your demo.
6. **Instrument continual learning from day one** (convergence-speedup, MTTR, fast_p over library states) — these charts are your strongest hackathon narrative and directly demonstrate "continual learning."
7. **Cap everything**: iteration limits per task, scale-to-zero, stop the G2 VM when idle, BigQuery cost tracking. Threshold: if projected spend exceeds $100, cut candidate fan-out and switch to Flash-only.

## Caveats
- **Benchmark numbers are distribution- and hardware-dependent.** CudaForge's 97.6%/1.68× (RTX 6000, o3), Kevin's 1.10× (A-series), KernelBand's >33% (RTX 4090/H20/A100), and KernelBench's "<20%" are all tied to specific GPUs, task subsets, and baselines; your L4 results will differ. Speedups measured against PyTorch eager are the easiest to beat — flag your baseline explicitly.
- **Reward hacking is a live risk, not a hypothetical** (Sakana's 50–120× fakes; EvoEngineer's 1.13×→0.82× replication drop). Any surprisingly large speedup should be assumed fake until the robust re-check passes.
- **Vendor-sourced and fast-moving facts**: Gemini model names/prices, Cloud Run GPU rates, Agent Engine pricing, and the Sessions/Memory Bank billing-start date (Jan 28 vs Sept 1, 2026) are vendor-published and change often — verify on cloud.google.com at build time. The Cloud Run "cannot run ncu" verdict is a well-founded inference from NVIDIA's permission requirements + Cloud Run's managed model, not an explicit Google statement.
- **Exact Agent Engine runtime limits** (max request duration, concurrent-session caps) were not confirmed from an official numeric table; check the Agent Engine quotas page before relying on very long single sessions — architect around ≤60-min work units and resumable sessions.
- **ADK/A2A/MCP APIs are evolving** (e.g., a2a-sdk 0.3.x → 1.x migration); pin versions.