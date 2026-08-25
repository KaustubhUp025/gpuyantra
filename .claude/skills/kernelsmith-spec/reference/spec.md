# KernelSmith — Implementation Specification for Claude Code

*Ground-truth reference document. Claude Code: follow this exactly. Do not deviate from locked decisions. Ask when ambiguous.*

**Project:** KernelSmith — an in-process Google ADK agent tree that autonomously profiles PyTorch model operations, generates optimized Triton GPU kernels, verifies them against correctness and anti-reward-hacking criteria, and hot-swaps them into a live Qwen2.5-1.5B inference server on a single NVIDIA L4 VM.

**Builder:** Solo (Kaustubh). C++ distributed-systems engineer. Expert in concurrency, testing, backend infra. Beginner in GPU kernels, Triton, LLM agents, ADK.

**Deadline:** August 31, 2026, 5:00 PM PDT.

**GCP Project:** `gpuyantra`

---

## 0. Version-Pinned Stack (VERIFIED August 2026 — do not change without explicit approval)

| Component | Exact Version / ID | Notes |
|---|---|---|
| Python | 3.11+ | VM ships with 3.11 via Deep Learning VM image |
| `google-adk` | `2.7.1` | Latest stable (supersedes 2.7.0 on Aug 17, 2026). No model-ID gating. |
| Primary LLM (all agents) | `gemini-3.7-flash` | GA Aug 13, 2026. Satisfies "Gemini 3.5 or newer." $0.75/$3.75 per M tokens. Global endpoint only. |
| Embeddings | `gemini-embedding-001` | $0.15/M input tokens. Truncate to 768 dims. Manual L2-normalize required. |
| `torch` | `2.12` (pin exact minor) | Verify on VM; pin whatever the DLVM ships. |
| `triton` | `3.7.1` (pin exact) | Ships with the DLVM's PyTorch. |
| `transformers` | Pin exact (e.g., `4.x.y`) | Pin whatever is current at build time. `Qwen2RMSNorm` internals can change between releases. |
| Firestore | Native mode, `us-central1` | Already created. Vector(768), COSINE, flat index. |
| Compute | `g2-standard-4` (1× NVIDIA L4) | Spot, us-central1-b, `--instance-termination-action=STOP` |
| Package manager | `uv` | All deps in `pyproject.toml`. |

**CRITICAL MODEL RULE:** There is NO `gemini-3.5-pro`. The Pro line stops at `gemini-3.1-pro-preview` which does NOT satisfy the hackathon's "3.5 or newer" rule. Use `gemini-3.7-flash` for ALL agents — Supervisor, Coder, Judge, Profiler. All models on the **global** Vertex endpoint (`location="global"`).

**Gemma (bonus +0.2):** `gemma-4-26b-a4b-it` is available as managed MaaS on Vertex AI (no self-deployment). Use for the kernel-explainer bonus agent.

---

## 1. Repository Layout

```
kernelsmith/
├── pyproject.toml              # uv-managed; all deps pinned
├── Makefile                    # `make demo`, `make test`, `make lint`
├── README.md                   # Setup + reproduction + spin-up for judges
├── .env.example                # Non-secret config template
├── .gitignore
├── kernelsmith/
│   ├── __init__.py
│   ├── root_agent.py           # Builds full agent tree; exports `root_agent` for `adk web`
│   ├── config.py               # Central constants (model IDs, thresholds, region, etc.)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py       # Supervisor LlmAgent
│   │   ├── profiler_agent.py   # Thin LlmAgent wrapping profiler_tool
│   │   ├── coder_agent.py      # Coder LlmAgent (emits KernelDraft JSON)
│   │   ├── judge_agent.py      # Judge LlmAgent (calls verifier_tool)
│   │   ├── escalation.py       # EscalationChecker BaseAgent
│   │   └── refinement_loop.py  # LoopAgent wiring Coder→Judge→EscalationChecker
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── profiler_tool.py    # Roofline fingerprint from do_bench + analytic FLOP/byte
│   │   ├── verifier_tool.py    # Spawns sandbox subprocess, returns reward JSON
│   │   ├── retrieval_tool.py   # Bottleneck-indexed find_nearest, top-3 skills
│   │   ├── upsert_tool.py      # Dedupe by skill_id, keep highest reward
│   │   └── hotswap_tool.py     # POST /swap to inference server, verify, rollback
│   ├── verifier/
│   │   ├── __init__.py
│   │   ├── correctness.py      # 5 seeds × 3 shapes, torch.allclose(atol=rtol=1e-2)
│   │   ├── timing.py           # do_bench(warmup=150, rep=200, return_mode="median")
│   │   ├── static_checker.py   # AST rules for 7 reward-hack patterns
│   │   ├── sandbox.py          # Subprocess-per-candidate, SIGKILL timeout, GPU health probe
│   │   └── reward.py           # CUDA-Agent milestone reward −1/+1/+2/+3
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── firestore_store.py  # Client, collection refs, CRUD, index bootstrap
│   │   ├── schemas.py          # Pydantic: SkillRecord, RunRecord, BottleneckFingerprint, etc.
│   │   └── embeddings.py       # gemini-embedding-001 @768 + L2-normalize + assert len==768
│   ├── inference_server/
│   │   ├── __init__.py
│   │   ├── server.py           # FastAPI: /generate /stats /swap /rollback
│   │   ├── patchable_ops.py    # Registry: name→(module class, patch fn, reference fn)
│   │   └── models.py           # Load Qwen2.5-1.5B, warmup, token/s meter
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── streamlit_app.py    # 3-column dashboard
│   │   └── event_stream.py     # Background thread + asyncio consumer → Queue
│   └── bench/
│       ├── __init__.py
│       ├── kernelbench_tasks.py # Curated L1/L2 task specs
│       └── run_bench.py        # Offline before/after harness (backup demo)
├── infra/
│   ├── create_index.sh         # gcloud firestore composite vector index command
│   ├── start_vm.sh
│   ├── stop_vm.sh
│   └── gpu_reset.sh            # nvidia-smi --gpu-reset / reboot fallback
├── scripts/
│   ├── seed_skill.py           # Insert one hand-written winning RMSNorm kernel skill
│   └── export_firestore.sh     # Daily backup
├── tests/
│   ├── __init__.py
│   ├── test_correctness.py     # Good kernels + planted reward-hacks
│   ├── test_static_checker.py  # 7 hostile AST snippets
│   ├── test_firestore.py       # CRUD + vector roundtrip
│   ├── test_embeddings.py      # Truncation to 768 + unit-norm assertion
│   ├── test_hotswap.py         # One op golden swap + rollback
│   ├── test_reward.py          # Milestone reward logic
│   └── test_sandbox.py         # SIGKILL on hang, GPU probe recovery
└── docs/
    └── design.md               # Architecture reference (this document, abridged)
```

Each file has ONE job (annotated above). Do not merge responsibilities.

---

## 2. Configuration (`config.py`)

```python
"""Central constants. Every magic number lives here. Import from here, never hardcode."""
import os

# --- GCP ---
GCP_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]  # "gpuyantra"
GCP_LOCATION = "global"  # All Gemini 3.x models require the global endpoint on Vertex AI
FIRESTORE_DATABASE = "(default)"
FIRESTORE_COLLECTION_SKILLS = "skills"
FIRESTORE_COLLECTION_RUNS = "runs"

# --- Models ---
PRIMARY_MODEL = "gemini-3.7-flash"       # All agents: Supervisor, Coder, Judge, Profiler
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768                       # MRL truncation; assert after every call
GEMMA_MODEL = "gemma-4-26b-a4b-it"       # Bonus kernel-explainer (MaaS, no self-deploy)

# --- Verifier ---
CORRECTNESS_SEEDS = 5
CORRECTNESS_SHAPES = [
    (1, 128),    # small: batch=1, seq=128
    (8, 512),    # medium
    (16, 2048),  # large
]
ATOL = 1e-2
RTOL = 1e-2
DO_BENCH_WARMUP = 150      # Default 25 underestimates by ~30% (Triton issue #2306)
DO_BENCH_REP = 200
SPEEDUP_THRESHOLD = 0.05   # 5% gate for +2/+3 milestones
SANDBOX_TIMEOUT_S = 60     # SIGKILL after this
GPU_HEALTH_PROBE_TIMEOUT_S = 10

# --- Agent Loop ---
MAX_LOOP_ITERATIONS = 6    # Circuit breaker; never remove
RETRIEVAL_TOP_K = 3

# --- Inference Server ---
SERVED_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
INFERENCE_HOST = "127.0.0.1"
INFERENCE_PORT = 8000

# --- Hardware (NVIDIA L4 constants) ---
L4_MEM_BW_GBPS = 300.1        # GB/s, GDDR6
L4_FP16_TFLOPS = 30.3         # Non-tensor FP16
L4_TENSOR_FP16_TFLOPS = 121.0 # Tensor Core FP16
L4_VRAM_GB = 24

# --- Reproducibility ---
GLOBAL_SEED = 42
DETERMINISTIC_CUDA = True  # torch.use_deterministic_algorithms(True)
CUBLAS_WORKSPACE = ":4096:8"  # CUBLAS_WORKSPACE_CONFIG env var
```

---

## 3. Pydantic Schemas (`memory/schemas.py`)

```python
"""Data models for the entire system. Every structured boundary uses these."""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime

class BottleneckFingerprint(BaseModel):
    """Roofline-derived fingerprint of WHY an op is slow."""
    op_family: str = Field(..., description="norm | rope | mlp | elementwise | reduction")
    hardware: str = Field(default="L4")
    memory_throughput_gbps: float
    achieved_occupancy: float = Field(ge=0.0, le=1.0)
    arithmetic_intensity: float = Field(ge=0.0)
    is_memory_bound: bool
    is_compute_bound: bool
    tile_size_hint: int = Field(ge=1)

    def to_embedding_text(self) -> str:
        """Convert to text for embedding. This IS the bottleneck-indexed retrieval key."""
        return (
            f"op={self.op_family} "
            f"mem_bound={self.is_memory_bound} "
            f"ai={self.arithmetic_intensity:.1f} "
            f"tile={self.tile_size_hint} "
            f"hw={self.hardware}"
        )


class KernelDraft(BaseModel):
    """Output of the Coder agent. One Triton kernel attempt."""
    code: str = Field(..., description="Complete Python source with @triton.jit kernel + wrapper")
    entrypoint: str = Field(..., description="Name of the callable wrapper function")
    block_sizes: dict = Field(default_factory=dict, description="BLOCK_SIZE and num_warps suggestions")
    rationale: str = Field(..., description="Why this kernel should be faster, referencing the fingerprint")


class Verdict(BaseModel):
    """Output of the Judge agent after calling verifier_tool."""
    reward: int = Field(..., ge=-1, le=3)
    correctness_pass: bool
    speedup_vs_eager: float
    speedup_vs_compile: float
    next_action: str = Field(..., description="STOP or a concrete fix instruction for the Coder")
    stop: bool
    stderr_tail: str = Field(default="", description="Last 500 chars of subprocess stderr if any")
    latency_ms_by_shape: dict = Field(default_factory=dict)


class SkillRecord(BaseModel):
    """A verified, reusable kernel skill stored in Firestore."""
    skill_id: str
    op_signature: str          # e.g., "rmsnorm_fp16_[B,S,H]"
    op_family: str             # "norm" | "rope" | "mlp" | "elementwise" | "reduction"
    hardware: str = "L4"
    bottleneck_fingerprint: BottleneckFingerprint
    winning_kernel_source: str # Complete Python source
    speedup_vs_eager: float
    speedup_vs_torch_compile: float
    fix_rule: str              # Human-readable description of the optimization applied
    embedding: list[float]     # 768-dim, L2-normalized
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # Bandit stats
    bandit_pulls: int = 0
    bandit_total_reward: float = 0.0

    @field_validator("embedding")
    @classmethod
    def validate_embedding_dim(cls, v):
        assert len(v) == 768, f"Embedding must be 768-dim, got {len(v)}"
        return v


class RunRecord(BaseModel):
    """Metadata for one complete optimization run."""
    run_id: str
    task_ref: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    final_reward: int = -1
    total_iterations: int = 0
    total_tokens_spent: int = 0
    cost_estimate_usd: float = 0.0
    status: str = "running"  # running | completed | failed


class TraceRecord(BaseModel):
    """One iteration's trace within a run (Firestore subcollection: runs/{run_id}/traces)."""
    iteration: int
    agent: str
    prompt_summary: str
    response_summary: str
    reward: int
    latency_ms_by_shape: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

---

## 4. Agent Architecture (ADK In-Process)

### 4.1 Shared `session.state` Keys

Agents communicate exclusively through `session.state`. Never use global variables, files, or direct function calls between agents.

| Key | Type | Writer | Reader(s) |
|---|---|---|---|
| `task_spec` | dict | Supervisor | All |
| `bottleneck_fingerprint` | dict (BottleneckFingerprint) | Profiler | Coder, Retrieval |
| `retrieved_skills` | list[dict] | Supervisor (via retrieval_tool) | Coder |
| `kernel_draft` | dict (KernelDraft) | Coder | Judge |
| `verdict` | dict (Verdict) | Judge | EscalationChecker, Coder (next iteration), Supervisor |
| `iteration` | int | Judge (after_agent_callback) | EscalationChecker |
| `best_reward` | int | Judge (after_agent_callback) | Supervisor |
| `best_kernel` | str | Judge (after_agent_callback) | Supervisor, Upsert |
| `supervisor_summary` | str | Supervisor (output_key) | UI |

**State mutation rule:** Always use `event.actions.state_delta` or `output_key`. Never directly assign to a fetched session object — this bypasses ADK's event tracking and causes lost data.

### 4.2 Agent Definitions

#### Supervisor (`agents/supervisor.py`)

- **Primitive:** `LlmAgent` (root agent)
- **Model:** `gemini-3.7-flash` on global endpoint
- **Tools:** `retrieval_tool`, `upsert_tool`, `hotswap_tool`
- **Sub-agents:** Profiler, RefinementLoop
- **output_key:** `supervisor_summary`
- **Instruction prompt (template):**

```
You are KernelSmith's supervisor. Given the task {task_spec}, follow this protocol:
1. Delegate to the Profiler to compute a bottleneck fingerprint.
2. Call retrieval_tool to fetch prior winning kernels matching the fingerprint.
3. Hand off to the RefinementLoop.
4. When the loop returns, read verdict from state:
   - If best_reward >= 3 OR iterations exhausted: call upsert_tool to save the best kernel.
   - If best_reward >= 2: call hotswap_tool to patch the live inference server.
5. STOP. Never write Triton code yourself.
```

#### Profiler (`agents/profiler_agent.py`)

- **Primitive:** `LlmAgent` (thin wrapper) or direct tool call from Supervisor
- **Model:** `gemini-3.7-flash`
- **Tools:** `profiler_tool`
- **output_key:** `bottleneck_fingerprint`
- **Purpose:** Exists so the fingerprint write appears as a visible ADK Event in the dashboard. If dashboard granularity is not needed, call `profiler_tool` directly from the Supervisor.

#### Coder (`agents/coder_agent.py`)

- **Primitive:** `LlmAgent`
- **Model:** `gemini-3.7-flash`
- **Tools:** NONE (ADK constraint: agents with `output_schema` cannot also call tools)
- **output_key:** `kernel_draft`
- **output_schema:** `KernelDraft` (pydantic)
- **`disallow_transfer_to_parent`:** True
- **`disallow_transfer_to_peers`:** True
- **Instruction prompt (template):**

```
You write ONE Triton kernel to optimize the operation described in {task_spec}.

Bottleneck analysis: {bottleneck_fingerprint}
Prior winning kernels for similar bottlenecks: {retrieved_skills}
Previous judge feedback: {verdict.next_action}

RULES:
- Return ONLY valid KernelDraft JSON.
- The kernel MUST use @triton.jit and tl.load/tl.store — never call torch.nn or F.*.
- Do NOT include try/except blocks.
- Do NOT use torch.empty for outputs — always write to the output via tl.store.
- Do NOT spawn extra CUDA streams or threads.
- Reuse self.weight from the original module (it is already on cuda with correct dtype).
- For RMSNorm: upcast to float32 for variance, compute rsqrt, multiply by weight, downcast.
- Target BLOCK_SIZE that keeps the working set in L4 SRAM (~48KB per SM).
```

#### Judge (`agents/judge_agent.py`)

- **Primitive:** `LlmAgent`
- **Model:** `gemini-3.7-flash`
- **Tools:** `verifier_tool`
- **output_key:** `verdict`
- **NO `output_schema`** — ADK's schema+tools combo is fragile (issue #3969). Instead, parse JSON from the final message and validate with pydantic in an `after_agent_callback`.
- **`after_agent_callback`:** Parse the Judge's text response as `Verdict` JSON. Track `best_reward`/`best_kernel` in state so the best-so-far survives a regressing final iteration. Increment `iteration` counter.
- **Instruction prompt (template):**

```
You evaluate kernel candidates for correctness and performance.

1. Call verifier_tool with the current kernel_draft and task_spec.
2. Read the reward JSON it returns.
3. Based on the results:
   - reward >= 3: set next_action="STOP", stop=true
   - reward == -1: analyze stderr_tail. Give ONE concrete fix (e.g., "add a mask for the
     tail elements", "cast to float32 before rsqrt"). Set stop=false.
   - reward == 1 or 2: analyze latency_ms_by_shape. Suggest ONE performance improvement
     (e.g., "increase BLOCK_SIZE to 1024", "coalesce the load pattern"). Set stop=false.
4. Return a Verdict JSON object.
```

#### EscalationChecker (`agents/escalation.py`)

- **Primitive:** Custom `BaseAgent` subclass (NOT a tool, NOT a callback)
- **Why:** The `escalate`-in-tool pattern is buggy in ADK (issues #501, #2692, #2808, #2988). Setting `tool_context.actions.escalate = True` inside a tool call or callback fails to terminate the LoopAgent cleanly, throws OTel context errors, or escalates all enclosing loops. The documented, robust pattern is a dedicated sub-agent that yields an Event with `actions.escalate`.

```python
from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions

class EscalationChecker(BaseAgent):
    """Reads verdict from state; escalates to exit the LoopAgent if done."""

    async def _run_async_impl(self, ctx):
        verdict = ctx.session.state.get("verdict", {})
        iteration = ctx.session.state.get("iteration", 0)
        should_stop = (
            verdict.get("stop", False)
            or verdict.get("reward", -1) >= 3
            or iteration >= 6  # Belt-and-suspenders; LoopAgent also has max_iterations
        )
        yield Event(
            author=self.name,
            actions=EventActions(escalate=should_stop),
        )
```

#### RefinementLoop (`agents/refinement_loop.py`)

- **Primitive:** `LoopAgent`
- **Sub-agents:** `[Coder, Judge, EscalationChecker]` (in this order)
- **`max_iterations`:** 6 — this is a circuit breaker. NEVER remove it. An infinite loop burns credits.
- **Exit conditions:** EscalationChecker escalates (reward ≥ 3 or stop=True) OR max_iterations hit.

```python
from google.adk.agents import LoopAgent

refinement_loop = LoopAgent(
    name="RefinementLoop",
    sub_agents=[coder_agent, judge_agent, escalation_checker],
    max_iterations=6,
)
```

**Fallback:** If `LoopAgent` misbehaves at runtime (e.g., does not exit on escalate, throws OTel errors), replace with a custom `BaseAgent` that implements a `while` loop manually and `yield`s events from each sub-agent.

### 4.3 Root Agent Assembly (`root_agent.py`)

```python
"""Entry point. Exports `root_agent` so `adk web` can discover it."""
from kernelsmith.agents.supervisor import build_supervisor

root_agent = build_supervisor()
# `adk web` picks up root_agent automatically from this module.
```

### 4.4 ADK + Vertex AI Auth

Environment variables (set in `.env`, read by ADK/google-genai automatically):

```bash
GOOGLE_CLOUD_PROJECT=gpuyantra
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_LOCATION=global
```

Auth uses Application Default Credentials (ADC) on the VM — the VM's attached service account. No API keys in code. `gcloud auth application-default login` for local dev.

---

## 5. Verifier (The Trust Anchor — highest-risk custom code)

**Design philosophy:** The verifier is what makes KernelSmith honest. Honesty is the entire pitch. Every component is informed by published failure modes (Sakana AI, KernelBench-Verified, CUDA Agent, CUDA-L1). Do not weaken any component.

### 5.1 Correctness (`verifier/correctness.py`)

For each of `CORRECTNESS_SEEDS` (5) seeds × `CORRECTNESS_SHAPES` (3 shape classes), generate random inputs, run the reference forward and the candidate forward inside the sandbox subprocess, and compare:

```python
# Pseudocode — implement this exactly
for seed in range(CORRECTNESS_SEEDS):
    torch.manual_seed(seed)
    for shape in CORRECTNESS_SHAPES:
        # shape = (batch, seq_len); hidden_size from the model config
        x = torch.randn(shape[0], shape[1], hidden_size, device="cuda", dtype=torch.float16)
        ref_out = reference_forward(x)
        cand_out = candidate_forward(x)

        # Guard: NaN/Inf
        assert torch.isfinite(ref_out).all(), "Reference produced NaN/Inf"
        assert torch.isfinite(cand_out).all(), "Candidate produced NaN/Inf"

        # Guard: shape/dtype match
        assert cand_out.shape == ref_out.shape
        assert cand_out.dtype == ref_out.dtype

        # Tolerance check (CUDA Agent uses exactly atol=rtol=1e-2)
        assert torch.allclose(cand_out, ref_out, atol=ATOL, rtol=RTOL)
```

All 15 checks (5 × 3) must pass. If any fail, `reward = -1` immediately.

### 5.2 Timing (`verifier/timing.py`)

```python
import triton.testing

def bench_kernel(fn, warmup=DO_BENCH_WARMUP, rep=DO_BENCH_REP):
    """Primary timer: triton.testing.do_bench with adequate warmup.
    Default warmup=25 underestimates by ~30% (Triton issue #2306).
    Returns median ms.
    """
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    return ms
```

Measure three baselines:

1. **PyTorch eager with TF32 enabled:** `torch.set_float32_matmul_precision('high')` then run the reference op. This is the honest baseline per KernelBench-Verified.
2. **torch.compile:** Compile the reference op with `torch.compile(mode="reduce-overhead")`, warm up, then bench. This enables the +3 milestone tier.
3. **Candidate kernel:** Bench the generated Triton kernel wrapper.

Compute speedups: `speedup_vs_eager = eager_ms / candidate_ms`, `speedup_vs_compile = compile_ms / candidate_ms`.

### 5.3 Static AST Checker (`verifier/static_checker.py`)

Walk `ast.parse(candidate_code)` and REJECT if any of these patterns match:

| # | Pattern | Detection | Literature |
|---|---|---|---|
| 1 | **`F.rms_norm` / `torch.nn.functional` fallback** | Any `ast.Attribute` referencing `torch.nn`, `torch.nn.functional`, or `F.` | Kevin assigns reward 0; CUDA Agent forbids fallback calls |
| 2 | **Identity output** (return input unchanged) | Output variable is the input variable with no reduction/write | KernelBench-Verified: GPT-5.5 ReLU identity shortcut, 374× fake speedup |
| 3 | **Decoy kernel** (defined but never called) | `@triton.jit` function defined but the returned tensor doesn't data-depend on a `tl.store` from it | Sakana: real-looking kernels that aren't on the compute path |
| 4 | **`torch.empty` stale memory** | `torch.empty` feeding the output without a complete write | Berkeley RDI: stale GPU memory contains the reference answer |
| 5 | **Hardcoded constants** | Literal tensor values in the output path | CUDA Agent data filter: outputs must differ across inputs |
| 6 | **`try`/`except` fallback** | Any `ast.Try` node in the candidate | Kevin assigns reward 0 to kernels with try/except |
| 7 | **Extra CUDA streams / threading** | `ast.Import`/`ast.ImportFrom` of `threading`, `multiprocessing`; `torch.cuda.Stream()` creation | CUDA-L1: asynchronous stream exploits fool timing |
| 8 | **Network imports** | `socket`, `urllib`, `requests`, `http` | No external communication allowed |

Implementation: walk `ast.parse(code)`, match `ast.Call`, `ast.Attribute`, `ast.Import`, `ast.ImportFrom`, `ast.Try` nodes. Return a list of `(rule_id, line_number, description)` violations. Any violation → reject the candidate (`reward = -1`).

### 5.4 Sandbox (`verifier/sandbox.py`)

Every candidate kernel executes in a **separate subprocess**. Never in the ADK/Streamlit/inference server process.

```python
import subprocess
import signal

def run_in_sandbox(script_path: str, timeout: int = SANDBOX_TIMEOUT_S) -> dict:
    """Run a verification script in a sandboxed subprocess.

    - Scrubbed env: only PATH, CUDA_VISIBLE_DEVICES, HOME, CUBLAS_WORKSPACE_CONFIG
    - Hard timeout → SIGKILL (not SIGTERM — Triton can ignore SIGTERM)
    - After every candidate, run GPU health probe
    """
    safe_env = {
        "PATH": os.environ["PATH"],
        "CUDA_VISIBLE_DEVICES": "0",
        "HOME": os.environ["HOME"],
        "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE,
    }
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True, text=True,
            timeout=timeout, env=safe_env,
            cwd="/tmp/kernelsmith_sandbox",  # throwaway dir
        )
        return parse_result(result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        # SIGKILL the process group
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        return {"reward": -1, "error": "timeout_sigkill"}
    finally:
        # GPU health probe after every candidate
        if not gpu_health_probe():
            run_gpu_reset()
```

**GPU health probe:** Run a trivial known-answer kernel (e.g., `a + b` where a=1, b=2, expect 3). If it fails or times out (`GPU_HEALTH_PROBE_TIMEOUT_S`), the GPU is wedged → invoke `scripts/gpu_reset.sh`.

**File permissions:** `chmod 444` the verifier scripts (`correctness.py`, `timing.py`, `static_checker.py`, `reward.py`) so generated code cannot rewrite them.

### 5.5 Reward (`verifier/reward.py`)

CUDA Agent milestone reward, gated on all-seed correctness:

```python
def compute_reward(correctness_pass: bool, speedup_vs_eager: float,
                   speedup_vs_compile: float) -> int:
    if not correctness_pass:
        return -1
    if speedup_vs_eager <= 1.0:
        return 1   # Correct but not faster
    if speedup_vs_eager > 1.0 + SPEEDUP_THRESHOLD:
        if speedup_vs_compile > 1.0 + SPEEDUP_THRESHOLD:
            return 3   # Beats BOTH eager and torch.compile by >5%
        return 2       # Beats eager by >5% only
    return 1           # Correct, marginal speedup (below threshold)
```

Return a JSON dict: `{reward, correctness_pass, speedup_vs_eager, speedup_vs_torch_compile, stderr_tail, latency_ms_by_shape}`.

---

## 6. Firestore Memory (`memory/`)

### 6.1 Schema

Collection `skills` (doc id = `skill_id`): fields match `SkillRecord` schema above plus the `embedding` field stored as `Vector(768)`.

Collection `runs` (doc id = `run_id`): fields match `RunRecord`.

Sub-collection `runs/{run_id}/traces`: fields match `TraceRecord`. Subcollection (not top-level) so a run's history is co-located.

### 6.2 Composite Vector Index

Create ONCE (takes minutes to build — never on demo day):

```bash
gcloud firestore indexes composite create \
  --project=gpuyantra \
  --collection-group=skills \
  --query-scope=COLLECTION \
  --field-config=field-path=op_family,order=ASCENDING \
  --field-config=field-path=hardware,order=ASCENDING \
  --field-config='vector-config={"dimension":"768","flat":"{}"},field-path=embedding' \
  --database="(default)"
```

The equality pre-filter fields (`op_family`, `hardware`) MUST precede the vector field in the index definition. Firestore vector search does NOT support inequality pre-filters — only equality.

### 6.3 Embeddings (`memory/embeddings.py`)

```python
import google.genai as genai
import numpy as np

def embed_768(text: str) -> list[float]:
    """Embed text to 768 dims using gemini-embedding-001. L2-normalize.

    TWO TRAPS:
    1. output_dimensionality=768 is silently ignored in some client paths.
       ASSERT the returned length.
    2. Sub-3072 vectors are NOT auto-normalized by gemini-embedding-001.
       Manual L2-norm is required.
    """
    client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={"output_dimensionality": EMBEDDING_DIM},
    )
    vec = result.embeddings[0].values

    # Trap 1: assert dimension
    if len(vec) != EMBEDDING_DIM:
        # Fallback: truncate manually (MRL guarantees first 768 dims are usable)
        vec = vec[:EMBEDDING_DIM]
    assert len(vec) == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM}, got {len(vec)}"

    # Trap 2: L2-normalize
    arr = np.array(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    assert norm > 0, "Zero-norm embedding"
    arr = arr / norm
    return arr.tolist()
```

### 6.4 Retrieval (`tools/retrieval_tool.py`)

```python
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

def retrieve_skills(op_family: str, hardware: str, fingerprint_text: str, k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """Bottleneck-indexed retrieval. The core novelty.

    Retrieves by WHY an op is slow (the fingerprint), not by the op's name.
    A skill learned on RMSNorm can surface for RoPE if the fingerprint matches.
    """
    query_vec = Vector(embed_768(fingerprint_text))
    query = (
        db.collection(FIRESTORE_COLLECTION_SKILLS)
        .where("op_family", "==", op_family)
        .where("hardware", "==", hardware)
        .find_nearest(
            vector_field="embedding",
            query_vector=query_vec,
            distance_measure=DistanceMeasure.COSINE,
            limit=k,
            distance_result_field="vector_distance",
        )
    )
    return [doc.to_dict() for doc in query.stream()]
```

### 6.5 Upsert (`tools/upsert_tool.py`)

Deduplicate by `skill_id`. Keep only the version with the highest `speedup_vs_eager`. Update bandit stats.

```python
def upsert_skill(rec: SkillRecord):
    ref = db.collection(FIRESTORE_COLLECTION_SKILLS).document(rec.skill_id)
    snap = ref.get()
    if snap.exists and snap.get("speedup_vs_eager") >= rec.speedup_vs_eager:
        return "kept_existing"
    ref.set(rec.model_dump() | {"embedding": Vector(rec.embedding)})
    return "upserted"
```

---

## 7. Profiler + Bottleneck Fingerprint (`tools/profiler_tool.py`)

**Avoid Nsight Compute (NCU)** — it requires elevated perf counters unreliable on virtualized VMs and is slow. Instead, compute a roofline-style fingerprint from `do_bench` timing + analytic FLOP/byte counts:

```python
def profile_op(reference_fn, input_shapes, hidden_size: int) -> BottleneckFingerprint:
    """Compute a roofline fingerprint for a PyTorch op on the L4."""
    # 1. Bench the reference op
    x = torch.randn(*input_shapes, hidden_size, device="cuda", dtype=torch.float16)
    median_ms = bench_kernel(lambda: reference_fn(x))

    # 2. Analytic FLOP and byte counts (op-specific)
    numel = x.numel()
    bytes_moved = numel * x.element_size() * 2  # read input + write output (minimum)
    flops = numel * 5  # approximate for RMSNorm: square, add, mean, rsqrt, multiply

    # 3. Derived metrics
    median_s = median_ms / 1000.0
    memory_throughput_gbps = bytes_moved / median_s / 1e9
    arithmetic_intensity = flops / bytes_moved

    # 4. Roofline classification
    ridge_point = L4_FP16_TFLOPS * 1e12 / (L4_MEM_BW_GBPS * 1e9)  # ~100 FLOP/byte
    is_memory_bound = arithmetic_intensity < ridge_point
    is_compute_bound = not is_memory_bound

    return BottleneckFingerprint(
        op_family=classify_op_family(reference_fn),  # "norm", "rope", "mlp", etc.
        hardware="L4",
        memory_throughput_gbps=memory_throughput_gbps,
        achieved_occupancy=estimate_occupancy(input_shapes),  # heuristic, labeled approximate
        arithmetic_intensity=arithmetic_intensity,
        is_memory_bound=is_memory_bound,
        is_compute_bound=is_compute_bound,
        tile_size_hint=compute_tile_hint(hidden_size),
    )
```

**Fallback:** If profiling times out or errors, default to `is_memory_bound=True`, `arithmetic_intensity=0.5`, and proceed. True for most norm/rope/elementwise ops.

---

## 8. Inference Server (`inference_server/`)

### 8.1 FastAPI Server (`server.py`)

Endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/generate` | POST | Generate text from prompt. Returns tokens + timing. |
| `/stats` | GET | `{tokens_per_s, tokens_total, active_kernel, last_swap_ts}` |
| `/swap` | POST | Hot-swap a kernel module. Body: `{op_name, kernel_source, entrypoint}` |
| `/rollback` | POST | Restore original forward for a given op_name. |
| `/health` | GET | Basic health check. |

### 8.2 Model Loading (`models.py`)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model():
    """Load Qwen2.5-1.5B-Instruct on the L4.

    At fp16: ~3 GB VRAM, leaving >20 GB for verifier subprocesses.
    Enable TF32 for the honest baseline.
    """
    torch.set_float32_matmul_precision('high')  # TF32 baseline (KernelBench-Verified)

    tokenizer = AutoTokenizer.from_pretrained(SERVED_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        SERVED_MODEL,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer
```

### 8.3 Hot-Swap Mechanism (`patchable_ops.py` + `hotswap_tool.py`)

**Patch targets (verified against `transformers` main for `qwen2` architecture):**

| Target | Module Path | Forward Signature | Priority |
|---|---|---|---|
| RMSNorm | `Qwen2RMSNorm.forward(self, hidden_states)` | FP32 upcast → rsqrt → downcast → weight multiply | **P0 (demo op)** |
| SwiGLU/MLP | `Qwen2MLP.forward(self, x)` | `down_proj(act_fn(gate_proj(x)) * up_proj(x))` | P1 (stretch) |
| RoPE | `apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)` | Module-level function, not instance method | P2 (stretch) |

**Hot-swap via `types.MethodType`:**

```python
import types

def swap_op(model, op_class_name: str, new_forward_fn):
    """Replace forward on all matching modules. Reuses original weights (zero copy).

    self.weight and self.variance_epsilon are the original nn.Parameters —
    already on cuda with correct dtype. The new forward must use them.
    """
    originals = {}
    for name, module in model.named_modules():
        if op_class_name in type(module).__name__:
            originals[name] = module.forward  # Save for rollback
            module.forward = types.MethodType(new_forward_fn, module)
    return originals  # Handle for rollback

def rollback_op(model, op_class_name: str, originals: dict):
    """Restore original forward methods."""
    for name, module in model.named_modules():
        if name in originals:
            module.forward = originals[name]
```

**Critical gotchas:**

1. **Do NOT `torch.compile` before patching.** Compiling bakes the old forward into the graph. If you compile, do it AFTER the swap and recapture graphs.
2. **Guard generation with `asyncio.Lock`.** Acquire, patch, run a 5-seed numeric parity check vs the saved original forward, release. If parity fails, auto-rollback.
3. **Reuse original `nn.Parameter` tensors.** Never re-initialize weights. The learned gain/bias already lives on the correct device and dtype.
4. **Match by class-name substring** (`"RMSNorm"` catches all instances across all decoder layers).

### 8.4 Token/s Meter

Wrap `model.generate` with a streamer that increments a counter. `/stats` returns the rolling average. Streamlit polls it ~every 1s.

---

## 9. Bandit Over Skills

UCB1 (simpler, implement first) with optional Thompson sampling upgrade.

```python
import math

def ucb1_select(skills: list[dict], total_pulls: int, c: float = 1.41) -> dict:
    """Select a skill using UCB1. Optimism under uncertainty.

    ucb = mean_reward + c * sqrt(ln(total_pulls) / n_pulls)
    Arms with zero pulls get infinite UCB → explored first.
    """
    best_ucb = -float("inf")
    best_skill = None
    for skill in skills:
        n = skill.get("bandit_pulls", 0)
        if n == 0:
            return skill  # Explore unpulled arms first
        mean = skill.get("bandit_total_reward", 0) / n
        exploration = c * math.sqrt(math.log(total_pulls) / n)
        ucb = mean + exploration
        if ucb > best_ucb:
            best_ucb = ucb
            best_skill = skill
    return best_skill
```

**Warm-start from replay buffer:** Pre-seed `bandit_pulls` and `bandit_total_reward` on skill documents from hand-tested results (e.g., the seed RMSNorm skill with 3 pulls and total_reward=9 → mean=3.0).

---

## 10. Streamlit Dashboard (`ui/`)

### 10.1 Layout (3 columns)

- **Left:** Agent panels (Supervisor, Profiler, Coder, Judge) — each shows last thought, last decision, last action.
- **Middle:** Large live tokens/sec metric + rolling latency line chart polled from `/stats`.
- **Right:** Skill-library summary (total count, top-3 by speedup) + run-history table.
- **Bottom banner:** Reward-hack rejection alert (red flash when AST checker fires).

### 10.2 Async Pattern (the safe approach)

Streamlit reruns the entire script top-to-bottom on every interaction. ADK's `Runner.run_async` needs a long-lived asyncio loop. The robust pattern:

```python
import streamlit as st
import threading
import asyncio
import queue

@st.cache_resource
def get_runner_and_queue():
    """Singleton: ADK Runner + background thread + thread-safe Queue.
    Created once, survives Streamlit reruns.
    """
    from kernelsmith.root_agent import root_agent
    from google.adk.runners import Runner

    runner = Runner(agent=root_agent, ...)
    event_queue = queue.Queue()

    def background_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_forever()

    bg_thread = threading.Thread(target=background_loop, daemon=True)
    bg_thread.start()

    return runner, event_queue, bg_thread

# On each Streamlit rerun:
runner, event_queue, _ = get_runner_and_queue()

# Drain events (non-blocking)
while not event_queue.empty():
    event = event_queue.get_nowait()
    st.session_state.setdefault("events", []).append(event)

# Render from st.session_state["events"]
```

**Fallback:** If Streamlit + async misbehaves during rehearsal, switch to a plain `st.empty()` log tail + matplotlib charts reading a JSONL file that the ADK runner writes.

---

## 11. Reproducibility Contract

Every item exists to inoculate against the field's credibility problem (Sakana 3.13×→1.49×, KernelBench-Verified 1.43×→0.88×).

| Item | Implementation | Why |
|---|---|---|
| Seed everything | `torch.manual_seed(GLOBAL_SEED)`, `numpy.random.seed(GLOBAL_SEED)`, `random.seed(GLOBAL_SEED)`, Gemini `temperature=0` on Judge | A speedup that only appears under one lucky seed is the Sakana failure mode |
| Deterministic CUDA | `torch.use_deterministic_algorithms(True)` | Forces deterministic kernel selection; otherwise different algorithms run-to-run |
| cuBLAS workspace | `CUBLAS_WORKSPACE_CONFIG=:4096:8` (env var) | Required for deterministic GEMMs; without it, `use_deterministic_algorithms` throws |
| Pin all versions | Exact pins in `pyproject.toml` for ADK, torch, triton, transformers, CUDA | Triton codegen and transformers internals change between releases |
| Firestore snapshot | `gcloud firestore export gs://<bucket>/snapshot` at demo time | Demo replays with known library and bandit state |
| TF32 baseline | `torch.set_float32_matmul_precision('high')` on EVERY baseline measurement | KernelBench-Verified lesson: un-TF32'd baseline manufactures fake speedups |
| `make demo` | One command, fresh L4, < $5 | A judge can reproduce the headline number |

---

## 12. Security & Sandboxing

- Generated Triton code NEVER runs in the ADK/Streamlit/inference process — only in the sandbox subprocess.
- Verifier scripts: `chmod 444` (read-only). Generated code cannot rewrite them.
- Subprocess env: scrubbed to a minimal allowlist. No proxy, no network env vars.
- Static checker: blocks `socket`, `urllib`, `requests`, `http` imports.
- Auth: Application Default Credentials on the VM (service account). No API keys in code. `.env` holds non-secret config only, `.gitignore`d.
- If Gemini API key is ever needed, store in Secret Manager, never in code or env files.

---

## 13. Testing Strategy

### 13.1 Unit Tests

| Test File | Tests | Assertion |
|---|---|---|
| `test_correctness.py` | 3 known-good kernels (hand-written RMSNorm variants) | Reward +1 to +3 |
| `test_correctness.py` | 3 planted reward-hacks: `F.rms_norm` fallback, identity output, decoy kernel | Reward −1 or deny |
| `test_static_checker.py` | 7 hostile AST snippets (one per pattern in §5.3) | Each rejected with correct rule_id |
| `test_firestore.py` | CRUD on SkillRecord + vector roundtrip (embed → store → retrieve → compare) | Retrieved skill matches stored skill |
| `test_embeddings.py` | Embed a string → assert len==768 → assert unit norm (‖v‖ ≈ 1.0 ± 1e-6) | Dimension and normalization correct |
| `test_reward.py` | Boundary cases: speedup exactly at threshold, NaN inputs, zero speedup | Correct milestone assigned |
| `test_hotswap.py` | Swap a known-good RMSNorm kernel → verify numeric parity → rollback → verify original restored | Parity holds both ways |
| `test_sandbox.py` | Submit a deliberately hanging kernel (infinite loop) → expect SIGKILL within timeout → GPU probe recovers | No GPU wedge persists |
| `test_schemas.py` | Pydantic validation: valid SkillRecord, invalid embedding dim, missing fields | Validation errors where expected |

### 13.2 Integration Tests

- End-to-end single task with a real `gemini-3.7-flash` call on one RMSNorm task. Assert: final reward ≥ +1, a skill row is written to Firestore, hotswap succeeds or is skipped based on reward.
- Budget-cap the integration test: `max_iterations=2` to limit token spend.

### 13.3 Chaos Tests

- Kill the sandbox subprocess mid-verify → expect graceful `reward = -1`, no hang.
- Submit a kernel that allocates 100 GB → expect OOM in subprocess, not in the main process.
- Simulate Firestore rate-limit → expect retry/backoff, not crash.

### 13.4 Golden Tests (per patchable op)

Before an op is EVER hot-swapped live, a torch reference with 5 seeds at `atol=1e-2` must pass. This is the gate for `/swap`.

### 13.5 Test Invocation

```bash
# Unit tests (no GPU required for most)
make test-unit    # pytest tests/ -k "not integration and not chaos"

# Integration tests (requires GPU + Vertex AI auth)
make test-int     # pytest tests/ -k "integration"

# All tests
make test         # pytest tests/ -v
```

---

## 14. `Makefile` Targets

```makefile
.PHONY: demo test test-unit test-int lint serve-inference serve-ui

demo:
	@echo "=== KernelSmith Demo ==="
	@echo "Seeding reproducibility..."
	CUBLAS_WORKSPACE_CONFIG=:4096:8 python -c "import kernelsmith; kernelsmith.run_demo()"

test-unit:
	pytest tests/ -k "not integration and not chaos" -v --tb=short

test-int:
	pytest tests/ -k "integration" -v --tb=short

test: test-unit test-int

lint:
	ruff check kernelsmith/ tests/
	ruff format --check kernelsmith/ tests/

serve-inference:
	uvicorn kernelsmith.inference_server.server:app --host 0.0.0.0 --port 8000

serve-ui:
	streamlit run kernelsmith/ui/streamlit_app.py --server.port 8501
```

---

## 15. Gemma Bonus Agent (+0.2)

After a kernel is verified with reward ≥ 2, pass it to Gemma 4 for explanation:

```python
# Use the managed MaaS endpoint — no self-deployment needed
explanation_prompt = f"""Explain what this Triton kernel does, why it's faster than the
eager PyTorch implementation, and what hardware feature it exploits.
Write for an engineer who knows C++ but not GPU programming.

```python
{verified_kernel_source}
```"""

# Call via google-genai with model=GEMMA_MODEL
```

Display the explanation in the Streamlit dashboard beside the kernel code. This earns the +0.2 bonus and maps to the BYOF story: "I can't write GPU kernels → the agent not only writes them but explains them to me."

---

## 16. Demo Choreography (4 minutes, 16 beats of ~15s)

| Beat | Time | On-Screen | Rubric Axis |
|---|---|---|---|
| 1 | 0:00–0:15 | Hook: "I'm a C++ systems engineer who can't write GPU kernels." | Innovation/story |
| 2 | 0:15–0:30 | Problem slide: KernelBench-Verified 1.43×→0.88× collapse | Innovation/credibility |
| 3 | 0:30–0:45 | Architecture diagram (Mermaid/draw.io visual) | Architecture |
| 4 | 0:45–1:00 | Live Streamlit: Qwen2.5-1.5B serving, baseline tok/s | Architecture + Demo |
| 5 | 1:00–1:15 | Profiler fingerprints RMSNorm as bandwidth-bound | Innovation |
| 6 | 1:15–1:30 | Retrieval: Firestore find_nearest + bandit selects arm | Innovation |
| 7 | 1:30–1:45 | Coder streams Triton kernel into panel | Innovation |
| 8 | 1:45–2:00 | Verifier runs: 5 seeds × 3 shapes, green checks | Demo/Production |
| 9 | 2:00–2:15 | **THE HOT-SWAP: tokens/sec jumps live** | Demo (money shot) |
| 10 | 2:15–2:30 | Honest speedup bar chart: eager vs TF32 vs compile vs KernelSmith | Innovation + credibility |
| 11 | 2:30–2:45 | **REWARD-HACK REJECTION:** red banner on planted F.rms_norm fallback | Innovation + Demo |
| 12 | 2:45–3:00 | Skill library grows (Firestore write, bandit stats update) | Innovation |
| 13 | 3:00–3:15 | Cross-op transfer: RMSNorm skill surfaces for RoPE by fingerprint | Innovation (novelty payoff) |
| 14 | 3:15–3:30 | `make demo` on fresh L4 + GCP Console proof (5-sec cutaway) | Production readiness |
| 15 | 3:30–3:45 | Backup: offline KernelBench L1/L2 before/after chart | Demo/Production |
| 16 | 3:45–4:00 | Close: pitch line, blog QR, `#AllThingsAgenticHackathon`, Gemma badge | Bonus + story |

**Record beats 4–11 as clean B-roll early** (Day 9). A live GPU hiccup on recording day must never cost the money shots.

---

## 17. Known ADK Bugs and Workarounds

| Bug | Issue # | Impact | Workaround |
|---|---|---|---|
| `escalate` in tool/callback fails or throws OTel context error | #501, #2692, #2808, #2988 | LoopAgent doesn't exit | EscalationChecker BaseAgent (§4.2) |
| Agent with `output_schema` + tools is fragile | #3969 | Judge can't use both schema and verifier_tool | Judge uses tools only; parse JSON in `after_agent_callback` |
| Nested LoopAgent escalation escapes all loops | #2692 | Breaks multi-level nesting | Only one LoopAgent level; Supervisor is not a loop |

**Fallback for LoopAgent:** If the above workarounds still fail, replace `LoopAgent` with a custom `BaseAgent` that implements a `while` loop with explicit `yield Event(...)` calls. Budget half a day for this swap.

---

## 18. Cost Budget

| Service | Estimated Cost | Credit Source |
|---|---|---|
| Compute Engine (~43 hrs Spot) | $27 | Free Trial ($271, expires Aug 27) |
| Gemini 3.7 Flash (~115 tasks) | $17 | Free Trial or GenAI App Builder ($1,148) |
| Embeddings | $0.03 | Free Trial |
| Firestore | $0 (free tier) | — |
| Cloud Trace | $0 (free tier) | — |
| Gemma bonus | $0.05 | Free Trial |
| **Total** | **~$44** | **$386 confirmed-usable credits** |

**VM discipline:** Stop the VM every evening. One overnight idle costs $5–10 for zero value. Use Spot for all dev work. Switch to on-demand ONLY for Aug 31 (demo day) to avoid preemption.

---

## 19. Environment Setup (for Claude Code reference)

### 19.1 Local Dev (Kaustubh's laptop)

```bash
# Auth
gcloud auth application-default login
gcloud auth application-default set-quota-project gpuyantra

# Clone and setup
cd ~/Project/gpuyantra
uv init kernelsmith  # or cd into existing repo
uv add google-adk==2.7.1 google-cloud-firestore google-genai \
       torch triton transformers fastapi uvicorn streamlit \
       pydantic pytest ruff numpy

# Env
cp .env.example .env
# Edit .env: GOOGLE_CLOUD_PROJECT=gpuyantra, GOOGLE_GENAI_USE_VERTEXAI=true, GOOGLE_CLOUD_LOCATION=global
```

### 19.2 VM Setup

```bash
# Create (already have the command; quota approved)
gcloud compute instances create kernelsmith-vm \
  --project=gpuyantra --zone=us-central1-b \
  --machine-type=g2-standard-4 --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB --boot-disk-type=pd-ssd \
  --metadata="install-nvidia-driver=True" \
  --scopes=cloud-platform --labels=project=kernelsmith

# Verify
gcloud compute ssh kernelsmith-vm --zone=us-central1-b -- nvidia-smi
# Should show NVIDIA L4, 24GB

# Shell aliases (add to ~/.bashrc locally)
alias stopvm='gcloud compute instances stop kernelsmith-vm --zone=us-central1-b --project=gpuyantra'
alias startvm='gcloud compute instances start kernelsmith-vm --zone=us-central1-b --project=gpuyantra'
alias sshvm='gcloud compute ssh kernelsmith-vm --zone=us-central1-b --project=gpuyantra'
```

---

## 20. Code Review Checklist (for every PR / commit)

Before merging any code, verify:

- [ ] No model IDs other than `gemini-3.7-flash`, `gemini-embedding-001`, or `gemma-4-26b-a4b-it`
- [ ] No `gemini-3-flash`, `gemini-3.1-pro`, `gemini-3.5-flash` (unless explicitly falling back)
- [ ] No API keys, secrets, or credentials in code or `.env` committed
- [ ] No direct `session.state` mutation — only `state_delta` or `output_key`
- [ ] Verifier tolerance is `atol=rtol=1e-2` — not looser
- [ ] `do_bench` warmup ≥ 150 — not the default 25
- [ ] `max_iterations` is set on any LoopAgent — no unbounded loops
- [ ] Sandbox subprocess used for all generated code execution — never in-process
- [ ] `assert len(embedding) == 768` after every embedding call
- [ ] Embedding is L2-normalized after truncation
- [ ] TF32 baseline enabled (`torch.set_float32_matmul_precision('high')`)
- [ ] Seeds are set before any random generation
- [ ] No `torch.compile` before monkey-patching
- [ ] No `try/except` in generated kernel candidates
- [ ] Static checker covers all 7 reward-hack patterns
- [ ] Tests exist for new functionality (TDD for verifier/memory)

---

## 21. Implementation Order (Build-Day Mapping)

This is the critical-path order. Each gate must pass before proceeding.

| Priority | Module | Gate | Day Target |
|---|---|---|---|
| 1 | `config.py` + `memory/schemas.py` + `pyproject.toml` | Schemas validate, deps install | Day 2 (Aug 22) |
| 2 | `memory/embeddings.py` + `memory/firestore_store.py` + `scripts/seed_skill.py` | `find_nearest` returns seeded skill | Day 2 |
| 3 | `verifier/correctness.py` + `verifier/sandbox.py` + `verifier/static_checker.py` + `verifier/timing.py` + `verifier/reward.py` | Good kernel → +3, planted hack → −1, SIGKILL works | Day 3 (Aug 23) |
| 4 | `tools/profiler_tool.py` + `tools/verifier_tool.py` | Fingerprint computed, reward JSON returned | Day 4 (Aug 24) |
| 5 | `agents/*` + `root_agent.py` | Loop runs, exits cleanly, writes best kernel to state | Day 4 |
| 6 | `inference_server/*` + `tools/hotswap_tool.py` | Hot-swap changes tok/s, parity holds, rollback works | Day 5 (Aug 25) |
| 7 | `ui/*` | Dashboard streams events without asyncio errors | Day 6 (Aug 26) |
| 8 | Bandit (`ucb1_select`) + `tools/retrieval_tool.py` + `tools/upsert_tool.py` | Bandit converges; skill transfers across ops by fingerprint | Day 7 (Aug 27) |
| 9 | Reproducibility (`make demo`) + architecture diagram + Gemma bonus | `make demo` reproduces headline on fresh L4 | Day 8 (Aug 28) |
| 10 | Demo capture + blog + social | Submittable video exists | Day 9–10 |

**If behind schedule:** Ship the backup demo (offline KernelBench L1/L2 before/after chart) instead of the live hot-swap. It exercises the same verifier + skill-library + reward loop and is far more robust. Cut the bandit and Gemma bonus before cutting the verifier or reproducibility.

---

## 22. Absolute Red Lines (Never Cross)

1. **Never weaken the verifier.** Fewer seeds, looser atol, dropped AST checks = Sev0.
2. **Never remove `max_iterations` from the LoopAgent.** Infinite loops burn credits.
3. **Never run generated code in the main process.** Sandbox subprocess only.
4. **Never commit secrets.** ADC + Secret Manager only.
5. **Never claim a speedup without a measured `do_bench` number or a paper citation.**
6. **Never use model IDs that don't satisfy "Gemini 3.5 or newer."**
7. **Never fine-tune any model.** Budget doesn't fit and the Gemini loop wins anyway.
8. **Never add A2A, LangGraph, Swarm, or separate services.** In-process ADK is locked.
9. **Never use Firestore inequality pre-filters with vector search.** Equality only.
10. **Never `torch.compile` before monkey-patching.** It bakes the old forward.

---

*End of implementation specification. This document is the ground truth. When in doubt, follow this. When this conflicts with earlier design docs, this wins (it incorporates the Aug 22 reconciliation of all prior artifacts). Flag any ambiguity rather than guessing.*
