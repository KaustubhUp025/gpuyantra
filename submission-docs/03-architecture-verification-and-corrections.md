# Architecture Plan — Verification, Corrections & Corrected Stack

**Status:** Verification complete. Input to the architecture decision.
**Date:** 2026-08-17
**Subject:** The Gemini-authored "Taskmaster MLOps Agent" plan (memory / routing / scaffolding).
**Method:** Direct arXiv resolution by me, plus three research agents against primary sources
(Google Cloud docs, `adk.dev`, PyPI, the actual `google-adk` 2.7.0 wheel source, arXiv abs pages).

---

## 1. Headline verdict

**The research is real. The numbers attached to it are not. The stack is buildable. The models
and the budget are not.**

| Dimension | Verdict |
|---|---|
| arXiv citations (7 IDs) | **All 7 resolve to real papers.** Better than typical LLM output |
| Numeric claims about those papers | **3 of 4 checked were wrong** — invented or transplanted figures |
| One citation ID | **Mis-attributed** — ACRouter given the WRP paper's ID |
| ADK component names | Largely correct; **1 capitalization bug**, 1 behavior change, 1 conflation |
| Model choices | **NON-COMPLIANT — fails Stage One judging** |
| Cost model | **$243–$273 minimum against a $150 budget**; worst case >$1,000 |
| Corrected version | **~$25–$28 total.** Same architecture |

**Assessment of the source, from the verification agent:**

> "Three of four claims were wrong in a specific, characteristic way — right-ish concept,
> invented or transplanted numbers, wrong ID. That is the signature of an LLM-written secondary
> summary, not of a human misreading papers. Discount everything else from it."

It found a fifth instance of the same drift independently while searching.

**Working rule: treat the plan as a reading list, never as a source of figures.** Every number
that reaches the submission narrative must be re-read off the paper.

---

## 2. Citation corrections

### Verified real (all 7)

| ID | Actual title | Date |
|---|---|---|
| 2606.06448 | Agent Memory: Characterization and System Implications of Stateful Long-Horizon Workloads | 2026-06-04 |
| 2606.24775 | **Are We Ready For An Agent-Native Memory System?** | 2026-06-23 |
| 2604.01670 | Hierarchical Memory Orchestration for Personalized Persistent Agents | 2026-04-02 |
| 2603.21354 | The Workload-Router-Pool Architecture… **A Vision Paper** | 2026-03-22 |
| 2604.23577 | RouteNLP: Closed-Loop LLM Routing with Conformal Cascading and Distillation Co-Optimization | 2026-04-26 |
| 2604.11465 | Three Roles, One Model: Role Orchestration at Inference Time | 2026-04-13 |
| 2603.29493 | MemFactory: Unified Inference & Training Framework for Agent Memory | 2026-03-31 |

### Corrections

| Plan claims | Actual |
|---|---|
| "improves execution success rates by **up to 3x**" (2604.11465) | AppWorld **5.4% → 8.9%** (FP16), 3.0% → 5.9% (AWQ). The paper's own words: *"roughly doubling performance in both settings."* The only ~2.6× figure is a difficulty-1 slice, not the headline |
| "**62%** cost reduction, **96%+** accuracy" (RouteNLP) | 8-week pilot: **58% cost reduction at 91% acceptance**, p99 1,847ms → 387ms. Benchmark: 40–85% cost, 96–100% quality. The "62/96" is a **mashup of two different results** |
| MemFactory uses GRPO for memory ops | True but **oversold**. MemFactory is *infrastructure* — a LLaMA-Factory analogue. Its own gain is "up to 14.8% relative." The novel GRPO-for-memory idea is **Memory-R1 (arXiv:2508.19828)**, which MemFactory wraps |
| ACRouter = arXiv:2603.21354 | **Wrong ID.** ACRouter is **arXiv:2606.22902**, "Agent-as-a-Router: Agentic Model Routing for Coding Tasks" (2026-06-22). 2603.21354 is the unrelated WRP paper |
| WRP as an architecture to implement | It is an **explicitly-labeled vision paper** — a framing device, not a technique. Cite for vocabulary; implement RouteNLP or ACRouter |

---

## 3. Compliance — the disqualification risk

**The plan specifies Gemini 2.5 Flash and Gemini 2.5 Pro. The rules mandate Gemini 3.5 or
newer.** Stage One is pass/fail on properly applying required technologies. As written, this
fails before scoring begins.

### Models that satisfy the requirement (verified, stable)
- **`gemini-3.7-flash`** — newest, updated August 2026
- `gemini-3.6-flash`
- `gemini-3.5-flash`
- `gemini-3.5-flash-lite`

### The finding that reshapes the architecture

**There is no Pro-class model that satisfies "Gemini 3.5 or newer."** The newest Pro is
`gemini-3.1-pro-preview` — numerically *below* 3.5 **and** still Preview.

**Consequence:** every part of the plan that says "route the hard reasoning step to Gemini Pro"
is unbuildable in a compliant way. The tiering must be **within the Flash family**
(`3.7-flash` for judgment, `3.5-flash-lite` for triage/extraction), optionally with **Gemma 4
26B as a managed API** for bulk work. This is a simplification, not a loss.

**Gemma:** current generation is **Gemma 4** (E2B / E4B / 12B / 26B-A4B / 31B, first release
2026-03-31). **There is no 9B in Gemma 4.** Gemma 2 is no longer in the supported-tuning list.

---

## 4. Cost — the real failure

Budget: **$150 in credits.** All rates verified against Google's live pricing pages.

### As proposed

| Line | Cost |
|---|---|
| Cloud Run GPU dev+demo, ~30 GPU-h on the official RTX Pro 6000 config | $95.60 |
| One Gemma 3 12B tuning job (3.6M training tokens) | $6.55 |
| **Tuned-model Vertex endpoint, 5 days** | **$138.08** |
| Memory Bank + Sessions | $0 |
| Gemini API | $0–30 |
| Registry / Build / GCS | ~$3 |
| **Total** | **~$243–$273** |

**Short by $100–$125 in the optimistic case.** Leave a `--min-instances 1` GPU service or the
tuned endpoint running for 14 days and it lands between **$500 and $1,100**.

**Warm-instance rates:** L4 minimum config **$1.047/hr = $25.12/day**. RTX Pro 6000 — the config
Google's own Gemma 4 tutorial specifies — **$3.187/hr = $76.48/day**. $150 buys **47 hours** on
the tutorial config.

### The silent budget-killer
A tuned open model **must be deployed to a Vertex endpoint** to be used, and that endpoint
**does not scale to zero**: **$1.15069/hr = $27.62/day = $386.63 for 14 days**. This is the most
likely single way to lose the entire budget without noticing.

### Corrected version — same architecture

| Line | Cost |
|---|---|
| `gemini-3.7-flash` via Gemini Developer API **free tier** | **$0** |
| Memory Bank + Sessions — **unbilled until 2026-09-01** | **$0** |
| Cloud Run CPU-only agent service | $0–3 |
| Optional: ~12 GPU-h on **L4** (not RTX Pro 6000) for one live demo | $12.56 |
| Optional: one tuning job, endpoint **deleted within 2h of eval** | $8.85 |
| Misc | ~$3 |
| **Total** | **~$25–$28** |

### Two timing facts worth exploiting
1. **Memory Bank and Sessions billing commences 2026-09-01.** Our deadline is 2026-08-31.
   **They are free for the entire build window.** After that there is still a real monthly free
   tier (50 vCPU-h compute, 100 GiB-h memory, 1 GiB-month storage).
2. **`gemini-3.7-flash` is free of charge on the Gemini Developer API free tier** — input,
   output, *and* context caching. On Vertex it is also cheaper than 3.5 Flash
   ($0.75/$3.75 vs $1.50/$9.00 per 1M).

**Recommendation: drop self-hosted GPU serving entirely.** `Gemma 4 26B` is available as a
**managed API at $0.15/$0.60 per 1M tokens** — no GPU bill, no 4-minute cold start, no quota, no
region constraint. Self-hosting is the largest cost *and* the largest demo-failure risk in the
design, and buys almost nothing in a 14-day solo build.

---

## 5. Stack corrections

| Plan says | Reality |
|---|---|
| `VertexAISessionService` | **`VertexAiSessionService`** (lowercase `i`). The plan's spelling raises `ImportError` |
| `PreloadMemoryTool` injects into **system instructions** | In **ADK 2.x** it injects a transient **user-role** `<PAST_CONVERSATIONS>` block via `_insert_transient_user_content()`. The system-instruction behavior was **ADK 1.x** |
| "ADK 2.0 graph workflows (Sequential, ParallelAgent)" | **Two different features conflated.** Graph workflows = `google.adk.workflow` (`Workflow`, `Node`, `Edge`, `@node`). `SequentialAgent`/`ParallelAgent`/`LoopAgent` are *template workflows*, explicitly **superseded** by graph workflows in 2.0 (still functional, not deprecated) |
| `MCPToolset` | Deprecated alias — emits `DeprecationWarning`. Use **`McpToolset`** |
| "Vertex AI Agent Engine" | **Renamed** (Google Cloud Next, April 2026). Vertex AI → **Agent Platform**; Agent Engine → **Agent Runtime**. ADK *class* names unchanged |
| docs at `google.github.io/adk-docs` | 301-redirects to **`adk.dev`** |

**Confirmed correct:** `VertexAiMemoryBankService` (exact name, async LLM fact extraction),
`InMemorySessionService`, A2A protocol (v1.0 under Linux Foundation since 2026-04-09, 150+ orgs,
`RemoteA2aAgent` ships in ADK), "Agent Cards" (correct term; v1.0 added signed cards), and
first-party hosted Google Cloud MCP servers (`https://run.googleapis.com/mcp`, plus BigQuery,
Compute Engine, Cloud Logging, Cloud Monitoring, Composer — **hosted endpoints, not pip
packages**).

**Current version: `google-adk` 2.7.0, published 2026-08-13.** ADK 2.0.0 shipped 2026-05-19.

---

## 6. Blockers

1. **Model non-compliance** (§3) — fix first, it is a disqualification risk.
2. **No compliant Pro-class model** — any "escalate to Pro" design is unbuildable.
3. **GPU cold start is ~4 minutes, not 5 seconds.** The "~5s" figure is *GPU+driver attach only*;
   both official LLM tutorials set `initialDelaySeconds=240`. Scale-to-zero + GPU + a live
   judged demo is a bad combination.
4. **Tuned-model endpoints don't scale to zero** — $27.62/day, silently.
5. **Region trap.** In India: L4 in `asia-south1` (Mumbai) is **invitation-only**; `asia-south2`
   (Delhi) has RTX Pro 6000 only. Expect to deploy to `us-central1` or `asia-southeast1` and
   plan demo latency accordingly.
6. **ADK 2.x is a breaking change from 1.x.** `_run_async_impl()` overrides are *silently
   ignored*; manual event appending is gone; broad `try/except` disables automatic retry/HITL;
   sessions written by 2.0 are unreadable by ADK <1.28. **Any tutorial predating 2026-05-19 is
   1.x and will mislead.**
7. **`VertexAiSessionService` and `VertexAiMemoryBankService` both require an `agent_engine_id`**
   — a `reasoningEngines` resource. Agent Runtime must be provisioned even though the agent runs
   on Cloud Run. Prerequisite step, not a drop-in.
8. **Gemma 4 tuning has no published price** despite being documented as supported. Budgeting
   blind if tuning Gemma 4 is load-bearing.

**Not a blocker (good news):** **GPU quota is auto-granted** — 3 L4 GPUs on first deployment in
a region, no request, no wait. This was the biggest feared risk and it does not exist.

---

## 7. Unresolved conflict between agents — routing topology

**This needs a decision and the evidence points opposite ways.**

- **For cascading:** RouteNLP (2604.23577) — 58% cost reduction at 91% acceptance in production,
  with conformal prediction giving a distribution-free escalation threshold.
- **Against cascading:** arXiv:2605.06350 — *"Cascade performance is limited primarily by
  **structural cost**, since cascades pay the cheap model before any escalation decision."* A
  lightweight **pre-generation router beat the best cascade policy on 4 of 5 datasets.**

**Resolution I'd propose:** route, don't cascade. On a **request-per-day**-limited free tier the
argument is decisive — a cascade burns two requests on every hard query. And per ACRouter
(2606.22902), simply feeding the router a static table of per-task-type model performance is
worth **+15.3% relative** — that is a JSON file, not a model.

**Note this is further simplified by §3:** with no compliant Pro tier, there are fewer tiers to
route between anyway.

---

## 8. Second conflict — memory sophistication

The plan proposes elaborate tiered memory. Two 2026 papers push back:

- **arXiv:2606.04315** — 8 memory systems across 5 scenarios; the best cross-task performer was
  **an agentic harness over flat text files manipulated by tool calls**. *"Memory performance
  hinges on giving the agent active control over storage and retrieval rather than on a passive
  store behind a fixed pipeline."*
- **arXiv:2606.24775** — *"No single architecture dominates across all scenarios."*
- **arXiv:2604.01670 (HMO)**, the plan's tiering source, **publishes no numbers in its abstract.**

Also load-bearing for cost: **arXiv:2606.06448** shows **memory construction, not retrieval,
dominates cost** (construction energy exceeds total query-phase energy across 300 queries;
47× spread in energy-per-correct-answer between systems). **Writes will exhaust a free-tier
quota, not reads.**

---

## 9. Highest-value techniques (verified, ranked by impact ÷ effort)

| # | Technique | Effort | Evidence |
|---|---|---|---|
| 1 | **Constraint pinning** — keep constraints outside the compactable region, re-inject verbatim after every compaction | **~2 hours** | **arXiv:2606.22528** — violation rate **0% → 30% after compaction (59% worst model)**; when constraint survived: **0%**, when omitted: **38%**; pinning **restores 0%**. Largest measured effect found anywhere |
| 2 | **Deterministic pre-execution gates** — read-only checks before any write | ~0.5 day | arXiv:2607.07405 — 29.6% → 42.0%; **+19.2pp** where gates fired |
| 3 | **Cache-aware prompt layout** — static first, dynamic last, tool results *outside* the cached prefix | ~4 hours | arXiv:2601.06007 — 41–80% cost, 13–31% TTFT. **Warns naive full-context caching can *increase* latency** |
| 4 | **Read-back verification + idempotency keys** | ~1 day | arXiv:2608.02645 (2026-07-31). Directly answers OSWorld 2.0's "agents skip verification" |
| 5 | **Zero-token watcher** — deterministic differ in front of every LLM call | ~0.5 day | SentinelBench (2606.05342) — resource use is a scored axis |
| 6 | **Flat-file agent-controlled memory** | ~1 day | arXiv:2606.04315 — beat 8 purpose-built systems |
| 7 | **ID-addressable observation log** — replace old observations with citations, `recall(id)` to re-fetch | ~1 day | arXiv:2607.25066 — NIAH **99.40% vs 88.12%** |
| 8 | **Artifact-preserving summarizer** — preserve identifiers/credentials/API responses verbatim | ~1 day | arXiv:2604.11465 — the mechanism behind the AppWorld doubling |
| 9 | **Isolated correction pass** — artifact only, no conversation history | ~1 day | arXiv:2604.11465 — history-free isolation is what breaks failure loops |
| 10 | **Self-disagreement confidence gate** — run 3×, escalate on divergence | ~1 day | arXiv:2602.11619 — ≤2 unique paths → 82–87% accuracy; ≥4 → 41–65%. **No training, no logits** |
| 11 | **Pre-generation router** (not a cascade) | ~2–3 days | §7 |
| 12 | **Sleep-time compute** — precompute briefs during idle via Batch API | ~1.5 days | arXiv:2504.13171 — ~5× less test-time compute, 2.5× lower per-query cost |
| 13 | **Tool isolation via service-account boundaries** — drafting agent holds no send permission | ~0.5 day | Not from a paper — from the rubric's "tool isolation and credential scoping". Converts a prompt promise into an IAM guarantee |

### Do NOT attempt in 14 days
- **RL-trained memory policies** (Memory-R1 / MemFactory) — needs training infra, and has a known
  unfixed credit-assignment bug (**the memory-reward trap**, arXiv:2608.02508).
- **Speculative tool execution** — 43.5% latency win, but **spends extra requests to buy
  wall-clock, which is backwards on a request-quota-limited free tier**.
- **Mutation-time forgetting hooks** — best-in-class (91.7–93.2%) but a full bitemporal control
  plane at 2.3 s/case.
- **ACON optimized compression guidelines** — needs a paired success/failure trajectory corpus
  we won't have.

---

## 10. Things to measure rather than assume

1. **The Gemini cached-token discount rate.** The widely-cited "10% of input" figure appears
   only on third-party blogs; **Google's official caching doc does not state it.** There is also
   an unverified June-2026 report of implicit caching silently ceasing to discount on
   `gemini-3.1-flash-lite` for some projects. **Instrument `cachedContentTokenCount` on day 1.**
2. **Whether compaction is invalidating the cache and eating its own savings.** Compaction
   rewrites the prefix; caching wants a stable prefix. **Nobody has published this interaction** —
   the verification agent named it the most important unanswered practical question in the
   literature for a budget-constrained build.
3. **Current Gemini free-tier quotas** — third-party sources only; confirm against Google docs
   before planning a request budget.
4. Implicit caching needs a **4,096-token minimum prefix** on 3.x models. Check the static
   header is even long enough to qualify.

---

## 11. Corrected stack

- **Model:** `gemini-3.7-flash` (compliant, newest, **free on the Developer API free tier**),
  `gemini-3.5-flash-lite` for triage/extraction. **Gemma 4 26B managed API** ($0.15/$0.60 per 1M)
  for bulk work if a third tier is wanted. **No self-hosted GPU. No Pro tier — none is compliant.**
- **Framework:** `google-adk` 2.7.0. **Graph workflows** (`google.adk.workflow`), not the
  superseded template-workflow primitives. `McpToolset`, not `MCPToolset`.
- **State:** `VertexAiSessionService` + `VertexAiMemoryBankService` — **free until 2026-09-01**.
  Requires provisioning an Agent Runtime `agent_engine_id`. Firestore for domain state.
- **Compute:** Cloud Run **CPU-only**, scale to zero.
- **Estimated total: ~$25–$28** of the $150.

---

## 12. Open decisions

- [ ] **Domain.** The plan's MLOps framing sits in the most saturated category identified in
      `02-problem-research-findings.md` (dev tooling), and the 40% criterion asks how much
      *real-world friction* is removed — an agent that deploys models serves engineers who
      already have CI/CD. **The architecture is the good idea; the domain is the weak part.**
      These techniques score identically pointed at a non-developer workflow.
- [ ] Routing topology — router vs cascade (§7). Recommendation: router.
- [ ] Memory sophistication — flat files vs tiered (§8). Recommendation: start flat, measure.
- [ ] Confirm whether any tuning/distillation is worth it at all given §4 endpoint costs.
