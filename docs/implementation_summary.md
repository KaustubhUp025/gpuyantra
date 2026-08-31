# gpuyantra / KernelSmith — Implementation Summary

Reference document for the demo video script and the dev.to write-up.
**Naming:** `gpuyantra` is the project (judge-facing surfaces: dashboard header, explorer,
Cloud Run services). `KernelSmith` is the agent tree inside it, and `kernelsmith` is the
Python package.

Everything below was read out of the source on 2026-08-31. Where a number is quoted, the
file it came from is named. Nothing here was carried over from memory.

---

## 1. Architecture Overview

### The tree

`kernelsmith/root_agent.py` is three lines: `root_agent = build_supervisor()`. The whole
tree is built by one factory, because ADK binds `parent_agent` per instance and refuses
re-parenting — module-level agent singletons would raise on a second assembly.

```
Supervisor                (LlmAgent, root)
├── Profiler              (LlmAgent)                      sub_agent
└── RefinementLoop        (LoopAgent, max_iterations=6)   sub_agent
    ├── Coder             (LlmAgent, output_schema=KernelDraft, no tools)
    ├── Judge             (LlmAgent, tools=[verify_kernel], NO output_schema)
    └── EscalationChecker (BaseAgent — never a tool, never a callback)
```

**Model IDs.** Every LlmAgent is built with `model=config.PRIMARY_MODEL` —
`gemini-3.7-flash` on the `global` Vertex endpoint (`kernelsmith/config.py:14`). No agent
carries an inline override. The bonus explainer uses `GEMMA_MODEL =
"gemma-4-26b-a4b-it-maas"`; the `-maas` suffix is the Vertex MaaS serving name and the
unsuffixed id 404s in every region.

**Roles, one sentence each:**

| Agent | Role |
|---|---|
| Supervisor | Orchestrates a resumable 7-step protocol over `session.state`; never writes Triton itself. Tools: `retrieve_skills_for_agent`, `upsert_skill`, `hotswap_kernel`, `explain_kernel`. |
| Profiler | Calls `profile_op_by_name` once and records the roofline bottleneck fingerprint that is also the retrieval key. |
| Coder | Writes exactly one Triton kernel draft per iteration, plus the deployment contract. Has **no tools** and cannot transfer in either direction. |
| Judge | Calls the verifier once, then emits a `Verdict`; every measured field is taken from the tool response, never from its own prose. |
| EscalationChecker | Reads `verdict` from state, escalates to exit the loop, and credits the bandit arm exactly once per run. Never calls a model. |

**LoopAgent configuration** (`kernelsmith/agents/refinement_loop.py`):

```python
LoopAgent(
    name="RefinementLoop",
    sub_agents=[build_coder_agent(), build_judge_agent(), build_escalation_checker()],
    max_iterations=MAX_LOOP_ITERATIONS,  # 6 — NEVER remove: red line #4
)
```

Exactly one LoopAgent level exists: nested loops escalate through every enclosing loop at
once in ADK (#2692), which is why the Supervisor is an LlmAgent rather than an outer loop.

**EscalationChecker's exit rule** (`kernelsmith/agents/escalation.py`) — three conditions,
in order of authority:

```python
should_stop = (
    bool(verdict.get("stop", False))          # 1. the Judge asked to stop
    or reward >= WINNING_REWARD                # 2. reward >= 3
    or iteration >= MAX_LOOP_ITERATIONS        # 3. budget exhausted (belt-and-suspenders)
)
yield Event(..., actions=EventActions(escalate=should_stop, state_delta=state_delta))
```

It is a `BaseAgent` yielding one `Event` carrying `actions.escalate` — setting `escalate`
from a tool or callback is broken in ADK (#501/#2692/#2808). Unparsed model text for
`verdict` is treated as `{}` (not a decision) and the loop keeps going. The bandit credit
is written through `EventActions(state_delta=...)`, guarded by `bandit_credited`, because
direct state mutation inside `_run_async_impl` does not persist.

**The two-turn protocol.** An ADK `LlmAgent`'s turn ends when it delegates, and a
`LoopAgent` cannot transfer back — so the Supervisor's invocation ends when the loop
escalates, with the kernel scored but neither saved nor deployed. Steps 4–7 (upsert,
hot-swap, explain, summarize) run on a **second message into the same session**.
`run_demo` sends both turns; the dashboard's `drive_run()` fires turn 2 automatically.
Anything that sends only one message silently skips the hot-swap.

---

## 2. Tool Inventory

All six are ADK `FunctionTool`s. `kernelsmith/tools/__init__.py` deliberately re-exports
nothing — a `FunctionTool` instance named `retrieval_tool` at package level would shadow
the `tools/retrieval_tool` module.

| Tool (published name) | Signature | What it does | Returns | Used by |
|---|---|---|---|---|
| `profile_op_by_name` | `(op_name, batch, seq_len, hidden_size)` | Benches the reference op and places it on the L4 roofline. | `BottleneckFingerprint` dict + `fingerprint_text` + `ridge_point_flops_per_byte` | Profiler |
| `retrieve_skills_for_agent` | `(op_family, hardware, fingerprint_text, k=3)` | Firestore vector search by bottleneck, then UCB1 picks one arm. | `{skills, count, selected_skill_id, selected_skill}` | Supervisor |
| `verify_kernel` (callable `verify_kernel_for_agent`) | `(kernel_code, entrypoint, task_spec, tool_context)` | Static check → contract validation → sandbox → reward. | Verdict fields + `violations`, `failed_cases`, `baseline_ms` | Judge |
| `upsert_skill` | `(skill_data: dict)` | Writes a verified skill, embedding computed here from the fingerprint. | `"upserted"` / `"kept_existing"` / `"error: …"` | Supervisor |
| `hotswap_kernel` | `(kernel_source, entrypoint, op_name, adapter_mapping=None)` | POSTs to the live server's `/swap`. | `{success, modules_patched, stats, error, rolled_back}` | Supervisor |
| `explain_kernel` | `(kernel_source: str)` | Gemma 4 explains the winning kernel in English. | `str` (or `"error: …"`) | Supervisor |
| `audit_model` (callable `audit_model_for_agent`) | `(model_name_or_path, device="cpu")` | Whole-model roofline triage. | Audit report dict + `report_text` | CLI / dashboard Audit tab |

Two naming tricks that are load-bearing. `verify_kernel_for_agent.__name__ =
"verify_kernel"` and `audit_model_for_agent.__name__ = "audit_model"` pin the LLM-facing
tool names, because the Judge's prompt names `verify_kernel` and
`find_verifier_response` matches the event log on it.

**The Judge does not pass the deployment contract.** `verify_kernel_for_agent`'s
LLM-visible parameters are exactly `kernel_code`, `entrypoint`, `task_spec`; the mapping is
read from `state["kernel_draft"]` via `tool_context` (see §11, bug 2).

---

## 3. Verifier Details

Four defences, in order of cost (`kernelsmith/tools/verifier_tool.py`):
static AST check → adapter-contract validation → subprocess sandbox → reward recomputed
in-process.

### Correctness (`verifier/correctness.py`)

5 seeds × 3 shapes = **15 checks, all of which must pass**:

```python
CORRECTNESS_SEEDS = 5
CORRECTNESS_SHAPES = [(1, 128), (8, 512), (16, 2048)]   # (batch, seq); hidden appended
ATOL = RTOL = 1e-2                                       # CUDA Agent's tolerance
```

Inputs are `torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float16)`. Beyond
`allclose` there are three guards, each closing a known reward hack: **NaN/Inf** on either
side, **shape mismatch**, and **dtype mismatch** (an upcast to fp32 would beat the
tolerance dishonestly). A candidate that raises is one failed check, not a crash — the
caller needs all 15 verdicts.

### Timing (`verifier/timing.py`)

```python
DO_BENCH_WARMUP = 150   # default 25 underestimates by ~30% (Triton #2306)
DO_BENCH_REP = 200
triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
```

`bench_kernel` **raises** if `warmup < 150`. Median, not mean. Two baselines are measured:
eager with TF32 on (`set_float32_matmul_precision("high")`) and
`torch.compile(mode="reduce-overhead")` with three untimed warm calls to pay for dynamo,
inductor and the CUDA-graph capture. Timing runs only after all 15 correctness checks pass,
and only at the largest shape (`16x2048`), where memory traffic dominates.

### The AST checker (`verifier/static_checker.py`)

Purely syntactic, runs before any execution, costs nothing, cannot be fooled by runtime
behaviour. See §12 for the full rule list.

### The sandbox (`verifier/sandbox.py`)

```python
subprocess.run([sys.executable, script], timeout=60, env=_safe_env(),
               cwd="/tmp/kernelsmith_sandbox", start_new_session=True, check=False)
```

- **Scrubbed environment** — exactly four variables reach the child: `PATH`,
  `CUDA_VISIBLE_DEVICES`, `HOME`, `CUBLAS_WORKSPACE_CONFIG`.
- **SIGKILL, not SIGTERM** — `subprocess.run` kills on `TimeoutExpired` (60 s), and
  `start_new_session=True` puts the child in its own process group so the kill cannot
  reach back.
- **GPU health probe after every candidate**, pass or fail: a known-answer `1+2==3` kernel
  in *another* subprocess with a 10 s timeout (in-process would hang on a wedged GPU).
  Failure sets `gpu_wedged` and runs `scripts/gpu_reset.sh`.
- **Contract:** one JSON object as the last line of stdout. A non-zero exit is −1
  regardless of what the child printed.

### Reward (`verifier/reward.py`)

```python
def compute_reward(correctness_pass, speedup_vs_eager, speedup_vs_compile) -> int:
    if not correctness_pass:                     return -1
    if speedup_vs_eager <= 1.0:                  return +1
    if speedup_vs_eager > 1.0 + SPEEDUP_THRESHOLD:      # 1.05
        if speedup_vs_compile > 1.0 + SPEEDUP_THRESHOLD:
            return +3
        return +2
    return +1
```

Correctness is a **gate, not a term** — no speedup can lift a wrong kernel above −1. The
comparison is strictly greater than 1.05, so a speedup sitting exactly on the 5% line
scores +1: measurement noise must never be rewarded.

Critically, the reward is **recomputed in-process** from the parsed numbers. The
subprocess's own `reward` field is discarded — the candidate controls that stdout, so
trusting it is a trust-boundary violation. A kernel that prints `{"reward": 3}` and
nothing else still scores −1.

### What the TF32 baseline means

`torch.set_float32_matmul_precision("high")` is set before every baseline measurement (in
`measure_baselines`, in `reproducibility.seed_everything`, and in the server's
`load_model`). Without it, eager PyTorch runs fp32 matmuls at full precision on the L4's
slow path — and any Triton kernel would look ~2× faster on a matmul for free. This is the
KernelBench-Verified baseline: **the number to beat is PyTorch at its best, not PyTorch
handicapped.**

---

## 4. Memory / Skill Library

### Firestore layout (`memory/firestore_store.py`, `memory/schemas.py`)

```
skills/{skill_id}                 -> SkillRecord, embedding stored as Vector(768)
runs/{run_id}                     -> RunRecord
runs/{run_id}/traces/{auto_id}    -> TraceRecord
```

`SkillRecord` fields: `skill_id`, `op_signature` (e.g. `rmsnorm_fp16_[B,S,H]`), `op_family`,
`hardware`, `bottleneck_fingerprint`, `winning_kernel_source`, `speedup_vs_eager`,
`speedup_vs_torch_compile`, `fix_rule`, `embedding` (768 floats, validated by a Pydantic
field validator), `tags`, `created_at`, `bandit_pulls`, `bandit_total_reward`.

Auth is **ADC only** — never a service-account key file.

### Embeddings (`memory/embeddings.py`)

`gemini-embedding-001` at `output_dimensionality=768`, with two traps handled explicitly:

```python
# Trap 1: output_dimensionality is silently ignored on some client paths -> assert length
assert len(vec) == EMBEDDING_DIM
# Trap 2: sub-3072 vectors are NOT auto-normalized -> manual L2 norm (COSINE needs it)
arr = np.array(vec, dtype=np.float64); arr = arr / np.linalg.norm(arr)
```

### The bottleneck fingerprint

```python
def to_embedding_text(self) -> str:
    return (f"op={self.op_family} mem_bound={self.is_memory_bound} "
            f"ai={self.arithmetic_intensity:.1f} tile={self.tile_size_hint} hw={self.hardware}")
```

Example: `op=norm mem_bound=True ai=1.2 tile=1024 hw=L4`. **No model name and no op name
appear in it.** That is the entire mechanism behind cross-model transfer — retrieval is
keyed on *why* an op is slow, not on what it is called.

### Vector search

```python
skills_collection(db)
    .where("op_family", "==", op_family)      # equality pre-filter
    .where("hardware", "==", hardware)        # equality pre-filter
    .find_nearest(vector_field="embedding", query_vector=Vector(embed_768(text)),
                  distance_measure=DistanceMeasure.COSINE, limit=k,
                  distance_result_field="vector_distance")
```

Index type is **flat**, 768 dimensions, and the composite index must list the two equality
fields *before* the vector field (`INDEX_COMMAND` in `firestore_store.py`; created once by
`infra/create_index.sh`). Firestore permits equality pre-filters only — no inequalities.
The 768 floats are stripped from every returned skill: it keeps a top-3 response ~2 KB
instead of ~40 KB of digits.

### UCB1 bandit (`tools/retrieval_tool.py`)

```python
UCB1_C = 1.41                                     # sqrt(2), textbook
ucb = mean_reward + c * sqrt(ln(total_pulls) / n_pulls)
if n <= 0: return skill                           # unpulled arms explored first
```

Retrieval answers *which skills are relevant*; the bandit answers *which one do we
actually start from*. Nearest-by-distance alone would lock the library onto whichever
kernel was seeded first — an untried skill has no evidence against it, only none for it.
The bandit's pick is moved to the front of `skills` so the Coder reads it first.

Feedback closes through `update_bandit_stats(skill_id, reward)`, written as **two
Firestore `Increment`s in one update** so concurrent runs cannot lose a pull to a
read-modify-write race. The reward credited is the **verifier's**, never a model's
self-report, and it is credited **once per run** by the EscalationChecker (six iterations
are one experiment, not six pulls).

---

## 5. Inference Server & Hot-Swap

### Endpoints (`inference_server/server.py`)

| Method | Path | Request | Response |
|---|---|---|---|
| POST | `/generate` | `{prompt, max_tokens=128 (1..2048), temperature=0.7 (0..2)}` | `{text, tokens, time_ms}` |
| GET | `/stats` | — | `{tokens_per_s, tokens_total, active_kernel, last_swap_ts}` |
| POST | `/swap` | `{op_name, kernel_source, entrypoint, adapter_mapping}` | `{success, modules_patched, parity, stats, error, rolled_back}` |
| POST | `/rollback` | `{op_name}` | `{success, modules_restored, stats}` |
| GET | `/health` | — | `{status, model, model_loaded, active_kernel, gpu}` |

A failed swap is **HTTP 200 with `success: false`**, not an error status: a kernel that
fails parity is an expected answer from this endpoint. Only a server that cannot serve at
all (no model loaded) returns 503.

`/generate` and `/swap` share one `asyncio.Lock` — one lock, not two, because the whole
point is that they exclude each other, so no patch can land between two decode steps.
Generation runs through `asyncio.to_thread` or `/stats` would stop answering for its
duration.

### The swap mechanism (`inference_server/patchable_ops.py`)

```python
for name, module in model.named_modules():
    if op_class_name in type(module).__name__:
        originals[name] = module.forward            # bound method, callable as-is
        module.forward = types.MethodType(new_forward_fn, module)
```

Matching is by **class-name substring**, so one call reaches all 57 `Qwen2RMSNorm`
instances across every decoder layer. Nothing is copied and nothing is re-initialized:
`self.weight` and `self.variance_epsilon` stay the original parameters, on the right
device in the right dtype. An empty return dict means nothing matched and the model is
untouched — the caller **must** treat that as a failure, which is exactly why the
layernorm-on-Qwen2 run refuses with `no module whose class name contains 'LayerNorm'`
rather than reporting a successful zero-module patch.

`PATCHABLE_OPS`: `rmsnorm` → `Qwen2RMSNorm` (P0), `layernorm` → `LayerNorm` (P0),
`swiglu` → `Qwen2MLP` (P1), `rope` → `apply_rotary_pos_emb` (P2, unreachable — it is a
module-level function, not an `nn.Module`, so `swap_op` matches nothing and `/swap`
refuses explicitly rather than faking a speedup).

`build_forward` has three paths in precedence order: a wrapper whose first parameter is
`self` is used as-is; a declared `adapter_mapping` goes through the **generic adapter**;
otherwise the hard-coded per-op adapter (kept for seed kernels).

### Loading the kernel — the detail that made hot-swap work at all

`_load_entrypoint` writes the source to a real file in a process-lifetime temp dir and
imports it via `importlib.util.spec_from_file_location`. Two things are load-bearing: the
module is registered in `sys.modules` **before** `exec_module` (so `inspect` can resolve it
while the `@triton.jit` decorators are still running), and **the file is never deleted** —
Triton compiles lazily and re-reads the source when specializing for a new shape, so
cleanup would move the failure from swap time to first-token time. See §11, bug 1.

### Parity checking

```python
def check_parity(module, original_forward, seeds=5, shape=(1, 128)):
    for seed in range(seeds):
        torch.manual_seed(seed)
        x = torch.randn(batch, seq, param.shape[-1], device=param.device, dtype=param.dtype)
        reference, candidate = original_forward(x), module.forward(x)
```

Deliberately the same contract as the verifier's correctness gate — atol=rtol=1e-2, NaN/Inf,
shape and dtype guards — but on the **live weights** instead of synthetic ones. A kernel can
pass the sandbox and still be wrong against a real weight distribution.

Before any of that, `/swap` re-runs the static AST checker and re-validates the
`adapter_mapping` against the real module class, because `/swap` is an HTTP endpoint and a
kernel can arrive here without ever having gone through the verifier.

### Rollback

On parity failure, or on a parity check that raises, `rollback_op(model, originals)` runs
immediately and the swap is refused with `rolled_back: True`. Where the saved handle is
just the class's own method, the instance attribute is **deleted** rather than reassigned —
that leaves the module byte-for-byte as it was, with no lingering
`module.__dict__["forward"]` to confuse a later swap. `STATE.originals` uses
`setdefault`, never assignment, so a second swap of the same op cannot overwrite the
handles pointing at the *stock* forwards with handles pointing at the previous generated
one.

### The torch.compile constraint

**The served model is never `torch.compile`d.** Compiling bakes the current `forward` into
the graph; a later `types.MethodType` patch silently no-ops — the compiled graph keeps
running the old forward. The demo would report a successful swap and unchanged throughput.
`load_model()` says so in a comment; the only `torch.compile` in the repo is in
`measure_baselines()`, where the compiled *reference* is timed for comparison and then
discarded.

---

## 6. Profiler / Roofline Analysis

### Arithmetic intensity

```python
flops, bytes_moved = analytic_counts(family, x.numel(), hidden_size, x.element_size())
arithmetic_intensity = flops / bytes_moved
memory_throughput_gbps = bytes_moved / (median_ms / 1000.0) / 1e9
```

FLOPs and bytes are **analytic**, not measured; only the latency is measured. Byte counts
are the *minimum* traffic a fused kernel must move — one cold read of each input, one write
of the output — which keeps the intensity a property of the **op** rather than of the
implementation being replaced. Per family: norm 5 flop/elem over 2 tensors, elementwise
2 over 2, reduction 5 over 3, rope 6 over 4; MLP is three GEMMs where the weight matrices
dominate the byte count.

Nsight Compute is deliberately avoided: it needs elevated perf counters that are unreliable
on a virtualized VM, and it is far too slow to sit inside an agent loop.

### Bandwidth

The one measurement is `bench_kernel(lambda: reference_fn(x))` — `do_bench` at
warmup=150, rep=200, median. Achieved bandwidth is `bytes_moved / median_s`. In the whole-model
audit, `_measure_bandwidth_pct` expresses it as a percentage of the L4's 300.1 GB/s, one
`do_bench` per unique module **type** (57 identical RMSNorms would otherwise cost 57
identical measurements).

### The ridge point

```python
RIDGE_POINT = L4_FP16_TFLOPS * 1e12 / (L4_MEM_BW_GBPS * 1e9)   # 30.3e12 / 300.1e9 ≈ 101
is_memory_bound = arithmetic_intensity < RIDGE_POINT
```

**101 FLOP/byte** is where the L4 stops running out of bandwidth and starts running out of
math. RMSNorm measures ~1.25 FLOP/byte — roughly 81× below the ridge — so the chip spends
~81× longer moving bytes than computing on them. That gap is the headroom a fused kernel
recovers, and it is the one thing the Coder actually needs to know.

### Tile hint and occupancy

`compute_tile_hint` is the next power of two above `hidden_size`, clamped to [64, 1024]
(1536 → **2048 → clamped to 1024**; note the winning kernel chooses 2048 itself, which the
hint informs but does not dictate). `estimate_occupancy` is an explicit heuristic —
wave fill (`rows / (58 SMs × 4 blocks)`) times tile fill — and is documented as never a
measurement.

### Failure behaviour

`profile_op` **never raises**. Any failure — no GPU, a reference that throws, a timeout —
returns `fallback_fingerprint`: memory-bound, AI = 0.5, occupancy 0.5, and
`memory_throughput_gbps = 0.0`, which is the tell that nothing was actually measured.

### Fingerprint → retrieval

The tool returns `fingerprint.model_dump() | {"fingerprint_text": ..., "ridge_point_flops_per_byte": ...}`.
The Profiler agent stashes the **raw dict** under a `temp:` key and promotes it over
`output_key`'s prose in an `after_agent_callback` — because `retrieve_skills` needs the
exact `fingerprint_text` the library was written with, and prose does not round-trip.

**Trap worth naming in the write-up:** `classify_op_family` takes a **callable**;
`family_from_name` takes a **string**. Passing `"RMSNorm"` to the former silently returns
`"elementwise"` (it reads `type(x).__name__`, which is `str`), which would have made the
transfer demo query the wrong `op_family` pre-filter.

---

## 7. Demo Dashboard

`kernelsmith/ui/demo_dashboard.py` (3,146 lines) is a **separate** Streamlit app on :8502;
the operator dashboard `streamlit_app.py` on :8501 is untouched.

### Capture (`ui/event_capture.py`)

`EventLogger` appends one JSON object per line to `data/traces/{run_id}.jsonl`
(never `/tmp` — the VM is preemptible and `/tmp` does not survive a restart, which is the
one moment you reach for a recorded fallback). Each record:

```json
{"elapsed_s", "author", "event_type", "content_text", "function_calls",
 "function_responses", "state_delta", "transfer_to", "escalate", "partial", "is_final"}
```

`event_type` ∈ `function_call | function_response | transfer | escalate | text`. Three
properties are load-bearing: **every line is flushed** (a run that dies halfway still leaves
the trace you most want); **`log_event` never raises into the agent loop** (it is called from
`EventStreamConsumer`'s background thread inside the `async for` draining
`Runner.run_async`); and non-serializable ADK fields **degrade per value, not per event** —
`_jsonable` walks containers and falls back to `repr()` only at the leaf that failed.

### Replay (`ui/event_replay.py`)

File order is the event order — no re-sorting. Pacing is
`sleep((elapsed_s[n] - elapsed_s[n-1]) / speed)`, with four defences in `gap_seconds`: the
first event never waits, `speed <= 0` means instant, a negative gap (clock skew) is zero,
and a gap longer than `MAX_GAP_S = 10.0` is clamped so a pause in the original run does not
look like a hung dashboard. Recorded `elapsed_s` is **never rewritten**. A malformed line
is skipped with a warning — half a demo beats a traceback in front of an audience.

The rendering code is identical in both modes; it receives event dicts from different
sources. `default_mode()` probes `/health` once (cached 10 s) and opens in **Replay**
unless `--live` was passed or a server answers — which is exactly the Cloud Run case.

### The "Try the model" chat panel (Task 14/15)

Live mode only, second on the page (before the timeline — the demo flow is ask → run → ask
again, and a panel *below* a finished log is one nobody presses first, which leaves the
Tokens/s card with nothing to compare against).

```python
httpx.post(generate_url(), json={"prompt": prompt, "max_tokens": int(max_tokens),
                                 "temperature": 0.0}, timeout=120.0)
tokens_per_s = tokens / (elapsed_ms / 1000.0) if tokens and elapsed_ms > 0 else None
record["kernel"] = active_kernel_now()   # read from /stats right after the request
```

Four one-click presets in a 2×2 grid plus a free-text box. `CHAT_MAX_TOKENS = 128` (the
server's own default; it was 48, which is where "every answer is exactly 48 tokens" came
from) with a 32–512 sidebar slider — the ceiling is far below the server's 2048 because
generation is synchronous and holds the swap lock.

Three honesty rules: the throughput is the **server's own arithmetic** out of the
`/generate` response, not a stopwatch around the HTTP call; each exchange records which
forward was live **at the time**, so `chat_throughput_pair` never infers before/after from
list order (two requests on the same side of a swap produce no comparison); and a
zero-token response yields **no rate at all** rather than a division.

Two traps: `st.form` is unusable here — a form context created in this container is found
when the *sidebar's* Start Run button is created on the next run, killing the whole live
page with a traceback pointing at the sidebar. And the 1 Hz `st_autorefresh` is scoped by
`should_autorefresh()` to when the tree is working or turn 2 is owed, because a rerun
landing while a `/generate` is in flight kills the script waiting for it.

### Tokens/s

`tokens_metric()` has three sources in order of trustworthiness: the live `/stats` poll
(a reachable server reporting `0.0` shows **0.0**, not "—" — that distinction was the bug),
the `hotswap_kernel` response's `stats`, or nothing. When there is nothing the card says
*why*: a `--no-server` run shows "—" with a `no server` delta and help text stating the
kernel passed every test but was never deployed and that nothing is estimated in its place.
A live `/stats` outranks the swap-response snapshot, which is taken the instant
`TokenMeter` clears its rolling window — the least informative reading of the whole run.

### The speedup bar chart

```python
rows = [("PyTorch", 1.0)]                       # 1.0 by definition
if eager_ms and compile_ms: rows.append((COMPILER_LABEL, eager_ms / compile_ms))
elif vs_compile:            rows.append((COMPILER_LABEL, speedup / vs_compile))
rows.append((KERNEL_LABEL, speedup))            # speedup_vs_eager, as measured
```

Nothing is hardcoded. A verdict carrying neither the baseline timings nor both ratios gets
**two bars, not three with a guess in the middle**. Rendered with altair (a Streamlit
dependency, so it works in the slim replay container) into a slot that exists from the
first frame, so it fills in rather than shoving the page down.

### The agent graph

`build_agent_graph()` emits a **DOT string** — `st.graphviz_chart` renders it in the
browser, so no system graphviz binary and no import can fail mid-recording.
`highlight_active_agent(dot, agent)` rewrites one node's `[...]` attribute list by regex;
both it and the builder go through one `_node_line()`, so the two can never draw the same
agent differently. The active node gets an amber fill **and** a light outline, because at
video bitrates on a dark theme a fill change alone is easy to miss. Labels may contain no
`]`, or the regex would end early. An agent that is not a node returns the DOT unchanged —
turn-2 events name agents that are not in the graph, and that must not blank the diagram.

`extract_metrics` reads **tool responses only** — `verify_kernel` and `hotswap_kernel`. An
agent describing its kernel as "roughly seven times faster" cannot move the Speedup card.

---

## 8. Multi-Model Audit

### `MODEL_REGISTRY` (`config.py`)

| key | hf_id | family | norm | activation | hidden |
|---|---|---|---|---|---|
| `qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | decoder | RMSNorm | SiLU | 1536 |
| `gpt2` | `openai-community/gpt2` | decoder | LayerNorm | GELU | 768 |
| `resnet50` | `microsoft/resnet-50` | vision | BatchNorm | ReLU | 2048 |

All three MIT/Apache-2.0 and **ungated** — an audit that stops to ask for a HuggingFace
token is an audit nobody runs. ~3.4 GB FP16 total, loaded one at a time.

### `OP_REGISTRY` (`tools/profiler_tool.py`)

`op_name` → builder callable returning an `OpBinding(family, reference, bind)`:

| op | family | reference signature the verifier benches |
|---|---|---|
| `rmsnorm` | norm | `entry(x, weight, eps)`, eps 1e-6, fp32 reduction |
| `layernorm` | norm | `entry(x, weight, bias, eps)`, eps 1e-5 |
| `softmax` | reduction | `entry(x)` |
| `silu` | elementwise | `entry(x)` |
| `rope` | rope | `entry(x, cos, sin)` |
| `mlp` | mlp | `entry(x, w_gate, w_up, w_down)` |

`verify_kernel` resolves the reference **through this registry**, not through executable
source in the task spec — source in the spec would be a second, unchecked path into the
sandbox. Weights are seeded (`GLOBAL_SEED`) so parent and sandbox agree exactly.

### `audit_model(model_name_or_path, device="cpu")`

Walks `named_modules()`, groups by class, places every unique type on the L4 roofline from
its own declared shapes, and ranks them. Returns an `AuditReport` (`model_name`,
`total_modules`, `unique_types`, `module_entries`, `top_target`, `recommendation`, `device`,
`hidden_size`, `measured`, `weights_loaded`, `gpu_name`).

**On CPU it never downloads weights.** The tree is built from `config.json` on the
meta device:

```python
config = AutoConfig.from_pretrained(hf_id)      # ~1 KB
with torch.device("meta"):
    AutoModel.from_config(config).eval()        # 0.11 s, zero bytes allocated
```

The module tree is exact (369 modules, 57 `Qwen2RMSNorm`, weight `[1536]`), because the
audit reads `in_features`, `normalized_shape` and parameter *shapes* and never runs a
forward. On CUDA the weights are real, because `do_bench` cannot time a meta tensor.

Three honesty rules in the report: an unestimated module prints `—` and `n/a`, never a
fabricated 0%, and is forced to LOW priority (AI 0.0 is an absence of information, not a
bottleneck); `BW %` is a fraction of the **L4's** 300 GB/s, so measuring anywhere else
prints a warning naming the actual GPU; and container modules (ModuleList, Sequential, the
root) are excluded so `total_modules` agrees with the table.

`assign_priority`: HIGH for a memory-bound norm or activation, MEDIUM for anything else
memory-bound, LOW for compute-bound / skipped / unestimated.

### Cross-model skill transfer

`demonstrate_cross_model_transfer(source, target)` in `run_demo.py`: it audits the target,
builds the fingerprint its top op *would* produce (from the audit's own numbers — nothing
is measured; the target is not being optimized), and runs the **real** retrieval query
against the **real** library. It writes nothing.

The join is on `(op_family, hardware)` — exactly what `retrieve_skills` pre-filters on.
Joining on the model or the op name would make the table agree with itself while describing
something retrieval does not do. Since `family_from_name("LayerNorm") == "norm" ==
family_from_name("RMSNorm")`, GPT-2's LayerNorm fingerprint retrieves Qwen2.5's RMSNorm
skills. A name-keyed cache cannot make that jump; that is the point.

### CLI (`run_demo.py`)

`audit` (`--model`, `--device`, `--all`, `--output`), `optimize` (the default, so
`make demo` is unchanged), and `full` (audit → optimize → re-audit → transfer demo).
A single `audit` follows the GPU (the one case where a measurement is the point) and prints
a warning line before a multi-GB weight download; `audit --all` defaults to **CPU
regardless** — three models of real weights plus `do_bench` did not finish in 600 s, and the
comparison table has no bandwidth column anyway.

---

## 9. React Explorer

`web/kernelsmith_explorer.jsx` (1,578 lines), served by `web/index.html` + `web/Dockerfile`
(nginx on :8080) with **no build step**: React 18, ReactDOM and Babel standalone from a
pinned CDN, Tailwind Play CDN for the utility classes, and the page `fetch`es the JSX next
to it, strips the two module lines a browser cannot resolve, compiles and renders it.

It fetches rather than inlines **on purpose**: pasting the JSX into the HTML would put a
second copy of every measured number in the repo, and the shipped copy would be the one
nobody edits. Cost: `file://` cannot fetch, so opening it off disk shows an explanation.
The strip regexes are line-bound and name-specific — an earlier `[^;]+` pattern crossed
newlines and ate 53 lines out of the `KERNEL_SOURCE` template literal (which starts with
`import torch` at column 0). `tests/test_explorer_packaging.py` reads the real regexes out
of `index.html` and applies them to the real JSX.

### Embedded data (every value provenance-tagged in the file header)

| Constant | Contents |
|---|---|
| `HEADLINE` | 7.24×, 659 tests (641 unit + 18 integration), 3 models |
| `HARDWARE` | L4: 24 GB, 300.1 GB/s, 30.3 fp16 TFLOPS, ridge point 101 |
| `AUDIT_DATA` | Real `audit --device cuda` output for all three models, run on the L4 2026-08-30; AI analytic, `bw_pct` measured |
| `TRACE` | The agent run's steps |
| `RESULT` | reward +3, 7.24× / 1.39×, 1 of 6 iterations, 15/15 |
| `KERNEL_SOURCE` | The winning kernel verbatim — single-pass fused RMSNorm, BLOCK_SIZE = next pow2 above N (1536 → 2048) |
| `EXPLANATION` | `gemma-4-26b-a4b-it-maas` output, four paragraphs, the model's own sentences |
| `ADAPTER_BINDINGS` | `x` (implicit, never declared), `weight`→`weight`, `eps`→`variance_epsilon` |
| `TRANSFER` | Live Firestore query, 2026-08-30 |
| `VERIFICATION` | Three cards: Correctness, Baselines, Anti-hack (the 7 AST rules) |

The header states that the two arithmetic-intensity numbers in the file (audit's 0.83 vs
profiler's 1.25 for fp16 RMSNorm) **disagree on purpose** and must not be reconciled.

### Three-model comparison

`AuditSection` holds `useState(MODEL_ORDER[0])`; three `ModelCard`s act as a segmented
selector and `AuditTable` re-renders for the selected model — columns: Module type, Count,
Regime, AI (FLOP/B), BW util, Priority, with the model's `top_target` row highlighted.
`fmtAI` renders `null` as **`n/a`**, never 0.

### Interactive elements

Scroll-spy `Nav` driven by an `IntersectionObserver` over the six sections; the model
selector; `TraceStep` accordion with expand/collapse-all; copy-to-clipboard on the kernel
with a `useRef` timer for the confirmation; a ~40-line in-file Triton tokenizer
(`highlightTriton`) rather than an npm highlighter — one more package on the deploy path is
one more thing that can break on demo day.

### "Watch the agent run" → the hosted dashboard

Wired 2026-08-31. `DASHBOARD_URL` points at the Cloud Run service and is used twice: as a
secondary CTA in the hero, beside the existing in-page `See how it works ↓`, and as the
first entry in the footer's `LINKS`.

```js
const DASHBOARD_URL = "https://gpuyantra-dashboard-p6o5zbfooq-uc.a.run.app";
const REPO_URL = "https://github.com/KaustubhUp025/gpuyantra";

const LINKS = [
  { label: "Watch the agent run", href: DASHBOARD_URL },
  { label: "GitHub repository", href: REPO_URL },
  { label: "Demo video", href: "#", todo: true },        // does not exist yet
  { label: "Technical write-up", href: "#", todo: true },// does not exist yet
];
```

**What a judge sees at the other end is a Play button, not a live run** — the dashboard
container has no GPU and no inference server, so `default_mode()` probes the inference port,
finds nothing, and opens in Replay. That is the designed behaviour for the hosted copy; it
replays whatever was in `data/traces/` at image build time.

The two remaining `todo: true` entries render an amber dot beside the label
(`title="URL pending"`) rather than pointing somewhere plausible, and they deliberately do
**not** get `target="_blank"` — an `href="#"` opening a blank tab is worse than an
in-page no-op. Fill them once the video and the write-up exist; `grep TODO(vm)` finds them.

---

## 10. Key Numbers (verified from code)

### Model IDs — all confirmed

```python
PRIMARY_MODEL   = "gemini-3.7-flash"        # Supervisor, Profiler, Coder, Judge
EMBEDDING_MODEL = "gemini-embedding-001"
GEMMA_MODEL     = "gemma-4-26b-a4b-it-maas" # NOT the unsuffixed id — that 404s everywhere
```

`GCP_LOCATION = "global"` for all of them. No inline model id exists anywhere in the agent
tree. ⚠️ The `-maas` suffix is a deliberate documented departure from CLAUDE.md rule 1
(same Gemma 4 26B model, MaaS serving name); the unsuffixed id returns 404 in `global`,
`us-central1`, `us-east4` and `europe-west4`.

### Verifier parameters

| Constant | Value |
|---|---|
| `ATOL` / `RTOL` | 1e-2 / 1e-2 |
| `CORRECTNESS_SEEDS` | 5 |
| `CORRECTNESS_SHAPES` | `(1,128)`, `(8,512)`, `(16,2048)` — 15 checks |
| `DO_BENCH_WARMUP` / `DO_BENCH_REP` | 150 / 200, `return_mode="median"` |
| `SPEEDUP_THRESHOLD` | 0.05 (strictly greater than 1.05) |
| `SANDBOX_TIMEOUT_S` | 60 (SIGKILL) |
| `GPU_HEALTH_PROBE_TIMEOUT_S` | 10 |
| `MAX_LOOP_ITERATIONS` | **6** |
| `RETRIEVAL_TOP_K` | 3 |
| `EMBEDDING_DIM` | **768**, L2-normalized, asserted after every call |
| `GLOBAL_SEED` | 42 |
| `CUBLAS_WORKSPACE` | `:4096:8` |
| `LLM_TEMPERATURE` | 0.0 on **every** agent, `seed=42`, 5 retries on `[429,500,502,503,504]` |
| L4 constants | 300.1 GB/s, 30.3 fp16 TFLOPS, 24 GB, 58 SMs, 48 KB SRAM/SM → ridge point ≈ **101** |
| `SWAP_PARITY_SHAPE` | `(1, 128)`, 5 seeds |
| `HOTSWAP_TIMEOUT_S` | 120 |

### ⚠️ Hardcoded numbers, and where they disagree with the traces

`7.24×` and `1.39×` appear only as **data** in `web/kernelsmith_explorer.jsx` (`HEADLINE`,
`RESULT`, `VERIFICATION`) and as illustrative text in two `demo_dashboard.py` docstrings.
No production code path contains a hardcoded speedup — every displayed number is read from
a `verify_kernel` or `hotswap_kernel` response.

What the seven committed traces in `data/traces/` actually contain:

| Trace | op | reward | vs eager | vs compile | checks | hot-swap | tokens/s |
|---|---|---|---|---|---|---|---|
| `demo-20260830-100847` | rmsnorm | +3 | 7.227 | 1.400 | 15/15 | ❌ refused | — |
| `demo-20260830-101414` | rmsnorm | +3 | **7.243** | 1.403 | 15/15 | ❌ refused | — |
| `demo-20260831-065705` | rmsnorm | +3 | 7.228 | 1.393 | 15/15 | ✅ **57 modules** | 0.0 |
| `demo-20260831-081406` | rmsnorm | +3 | 7.224 | 1.393 | 15/15 | (no swap in trace) | — |
| `demo-20260831-081623` | **layernorm** | +3 | **7.454** | 1.385 | 15/15 | ❌ `no module whose class name contains 'LayerNorm'` | **28.07** |
| `demo-20260831-094045` | rmsnorm | +3 | 7.221 | 1.391 | 15/15 | ✅ **57 modules** | 0.0 (`tokens_total: 0`) |
| `sample_run.jsonl` | rmsnorm | +3 | 7.24 | 1.39 | 15/15 | ✅ 57 modules | 22.9 (hand-written fixture) |

Reading of the discrepancies, for the script:

- **7.24× is real and reproducible.** Four independent rmsnorm runs land at 7.224–7.243 —
  spread 0.3%, which is `do_bench` noise. The explorer's `7.24×` is trace `101414`'s
  `7.243` rounded. Quote 7.24× for rmsnorm without qualification.
- **7.45× is a *different op*, not a better run.** The newest trace optimizes **layernorm**,
  not rmsnorm. Do not present 7.45× as an improvement on 7.24×; they measure different
  reference implementations.
- **The layernorm run's hot-swap correctly refused.** Qwen2.5 has no `LayerNorm` modules —
  the class-name substring match found nothing and the server refused rather than reporting
  a zero-module success. That is the anti-fake-speedup guard firing in a real trace, and it
  is a *good* beat: the same trace still shows **28.07 tokens/s**, measured on the stock
  forwards.
- **Both successful hot-swap traces (`065705` and `094045`, 57 modules each) report
  `tokens_per_s: 0.0` with `tokens_total: 0`,** because `TokenMeter.record_swap` clears the
  rolling window at the swap instant and nothing generated afterwards inside the run.
- ⚠️ **The before/after throughput pair can never appear in a trace file, by design — do not
  plan the video around capturing one.** `EventLogger.log_event` is called from exactly one
  place, `EventStreamConsumer` (`event_stream.py:369`), on ADK events drained from
  `Runner.run_async`. The chat panel's `send_prompt` POSTs straight to `/generate` and
  `record_chat` writes only to `st.session_state` — no ADK event exists, so nothing reaches
  the JSONL. **The pair exists only on screen, in Live mode, on the L4.** Film it; a replay
  will never show it.
- **Test count, re-measured 2026-08-31 and now correct on the page:** pytest collects
  **716** in total — **698 unit** (the `def test_` functions across 27 files, expanded by
  parametrization, including the trace-ordering regression test added the same day) and
  **18 integration**, confirmed by `pytest -k integration --collect-only` reporting
  `18/716 tests collected`. The explorer's headline was 659 (641 + 18) and now reads 716
  with the note "698 unit (hermetic) + 18 integration on the L4". Quote **716 total /
  698 unit** — and re-count if any test is added before recording, because this number is
  a claim on a judge-facing page.

---

## 11. Three Integration Bugs That Passed Every Unit Test

All three were found by writing `tests/test_integration.py` and running the system end to
end against live Vertex AI, Firestore and a GPU (commit `b86acc2`, "Add the integration
test, fix three defects it surfaced"). Each had a full row of green checks while being
wrong. This is the strongest section of the story.

### Bug 1 — `/swap` could not load a Triton kernel at all

**Was:** `_load_entrypoint` used `exec(compile(source, "<hotswap-kernel>", "exec"))`.
`@triton.jit` calls `inspect.getsourcelines` on the decorated function at **decoration**
time, so with no file on disk it raises
`ValueError: @jit functions should be defined in a Python file`.

Every kernel this system produces is a Triton kernel, so **100% of hot-swaps failed** — and
failed as a refused `/swap` carrying a load error, which reads like a bad kernel rather than
a broken loader. The demo's money shot could never have fired.

**Why it hid:** every test in `test_hotswap.py` used a pure-torch stand-in kernel, which has
no `@triton.jit` and therefore no decoration-time source lookup.

**Fix** (`inference_server/server.py::_load_entrypoint`) — write to a real file, import by
spec, with two non-obvious details:

```python
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module      # BEFORE exec_module, so inspect can resolve it
spec.loader.exec_module(module)        # while the @triton.jit decorators are running
# and the file is NEVER deleted: Triton re-reads the source when it specializes for a
# new shape, so cleanup would move the failure from swap time to first-token time.
```

`tests/test_hotswap.py` now carries Triton-specific loader tests — they need no GPU, because
decoration is what fails.

### Bug 2 — `dict[str, str]` on an LLM boundary made structured output emit `{}`

**Was:** `KernelDraft.adapter_mapping: dict[str, str]`, and the Judge passed
`adapter_mapping` to `verify_kernel` as another free-form dict. Observed live: the Coder
emitted `{}` on every draft and the Judge passed `{}` on every call. **The generic adapter —
the novel contribution — never ran once.**

**Why it hid:** an empty contract is not an error anywhere. It silently falls back to the
hard-coded per-op adapter, i.e. the human-written bridge this project claims the agent writes
for itself. Every green check still passed while the demo beat ("No human wrote the adapter")
would have been false.

**Root cause:** `dict[str, str]` compiles to a JSON schema with **no named properties**, so
structured generation has nothing to anchor on. Measured directly against gemini-3.7-flash,
same prompt, 3 trials each:

| Schema | Filled correctly |
|---|---|
| `adapter_mapping: dict[str, str]` | **0/3** (`{}` every time) |
| `adapter_mapping: list[AdapterBinding]` | **3/3** |

Making the field merely *required* was not enough — `{}` satisfies a required dict.

**Fix, at both boundaries:**

```python
class AdapterBinding(BaseModel):          # two NAMED string fields
    kernel_param: str
    module_attr: str

class KernelDraft(BaseModel):
    adapter_mapping: list[AdapterBinding] = Field(...)   # a LIST, not a mapping
    def mapping_as_dict(self) -> dict[str, str]:
        return {b.kernel_param: b.module_attr for b in self.adapter_mapping}
```

And the Judge no longer passes it at all: `verify_kernel_for_agent`'s LLM-facing parameters
are exactly `kernel_code`, `entrypoint`, `task_spec`, and it reads the contract from
`state["kernel_draft"]` through `tool_context`. That removes the second free-form dict and
structurally enforces what the prompt could only *ask* for.

Confirmed live after the fix, the Coder emits exactly:

```json
[{"kernel_param": "weight", "module_attr": "weight"},
 {"kernel_param": "eps",    "module_attr": "variance_epsilon"}]
```

**General rule this established for the codebase: never put a free-form `dict[str, str]` on
a boundary an LLM has to fill. Use a list of objects with named fields.**

### Bug 3 — baseline fairness was documented but never implemented

**Was:** `measure_baselines()` timed both baselines with
`torch.use_deterministic_algorithms(True)` still on. That flag forces slower cuBLAS/cuDNN
codepaths and costs eager and torch.compile **~23%** — while leaving a Triton candidate
completely untouched, because Triton generates its own PTX and never consults it.

**The reported speedup was therefore inflated by a measurement artifact: 8.52× vs eager with
the flag on, 6.92× with it off.**

**Fix** (`verifier/timing.py`) — a context manager that saves the flag (and `warn_only`),
turns it off, and restores it in a `finally` so a benchmark that raises cannot leave
determinism off for the correctness gate:

```python
@contextmanager
def _nondeterministic_for_timing():
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)
```

The flag stays **on** everywhere else — correctness checks, the agent loop, the demo. Only
the timed comparison runs without it. `tests/test_timing.py` asserts the flag is off during
both baselines and restored after, including on exceptions.

This one is worth telling because it moves the headline number *down*. The honest 7.24× is
the number this project reports; 8.52× was available and was rejected.

---

## 12. Anti-Reward-Hacking

### The seven AST rules (`verifier/static_checker.py`, 560 lines)

Purely syntactic, runs before any execution, cannot be fooled by runtime behaviour. Rules
may be **added, never removed or loosened**. Rule 0 is "the candidate does not parse".

| # | Pattern | Caught how | Example it stops |
|---|---|---|---|
| 1 | `torch.nn` / `F.*` fallback | Resolves dotted chains through import aliases; `F` is assumed to mean `torch.nn.functional` even with no import | `return F.rms_norm(x, w)` — call the library and claim the speedup |
| 2 | Identity output | Follows the return expression through aliases and passthrough methods (`view`, `reshape`, `clone`, `to`, `contiguous`, `float`, …) back to its root; flags it if the root is a parameter | `return x.view(x.shape)` |
| 3 | Decoy kernel | A `@triton.jit` / `@triton.autotune` / `@triton.heuristics` function whose name never appears in call position (including `kernel[grid](...)` launch syntax) | A real-looking kernel that is never on the compute path (Sakana) |
| 4 | Stale `torch.empty` | `torch.empty` / `empty_like` / `empty_strided` returned without a complete write — no `out[...] =`, no in-place `foo_()`, no `out=`, no launch that also contains a `tl.store` | Returning uninitialised VRAM, which on a warm GPU often still holds the reference answer (Berkeley RDI) |
| 5 | Hardcoded constant | Literal returns; `torch.tensor/as_tensor/Tensor` of only literals; `zeros/ones/full(_like)` returned unwritten | An output that does not depend on the input at all (CUDA Agent data filter) |
| 6 | `try` / `except` | Any `ast.Try` or `ast.TryStar` | A silent fallback to the reference on failure (Kevin) |
| 7 | Unsafe runtime | Imports of `threading`, `multiprocessing`, `socket`, `urllib`, `requests`, `http`; calls to `torch.cuda.Stream` / `stream` / `ExternalStream` | Async-stream exploits that hide work from the timer (CUDA-L1); network egress |

Violations are returned as `(rule_id, line, description)`, sorted and de-duplicated, and any
non-empty list is **reward −1 with no execution at all**. The same checker runs again inside
`/swap`, because that endpoint is reachable without going through the verifier.

`make harden` `chmod 444`s the four verifier files (`correctness.py`, `timing.py`,
`static_checker.py`, `reward.py`). Generated code runs as a subprocess owned by the same
uid, so nothing at the OS level stops it from rewriting the checker about to judge it —
except the write bit. Not a sandbox; a cheap interlock that makes tampering deliberate.

### How the TF32 baseline prevents inflated speedups

`torch.set_float32_matmul_precision("high")` before every baseline. Without it, an
un-TF32'd eager baseline hands any candidate a free ~2× on any matmul — a speedup
manufactured out of a precision setting. Combined with the determinism fix (§11 bug 3),
these are the two ways the *baseline* could have been rigged, and both are closed.

### Honest +1 reporting

The reward ladder makes "correct but not faster" a **first-class, reportable outcome**
rather than a failure: `+1` is what a correct kernel scores when `speedup_vs_eager <= 1.0`
or sits within the 5% noise band. A compute-bound op that cuBLAS already saturates should
land there, and the system says so.

The dashboard carries this through: `extract_metrics` sets

```python
metrics["honest_plus_one"] = bool(payload.get("reward") == 1 and payload.get("correctness_pass"))
```

described in the source as "the honest case the demo is proud of: correct, verified, not
faster". The Supervisor's prompt reinforces it — *"Report the verifier's numbers as they
are. A kernel that did not beat torch.compile did not beat torch.compile"* — and a +1 is
never hot-swapped (only reward ≥ +2 reaches `/swap`).

The same instinct runs through the audit (`n/a`, never a fabricated 0%), the speedup chart
(two bars rather than three with a guess), the Tokens/s card (says *why* it is empty), and
the layernorm trace in §10 (a refused swap, recorded and shipped as-is).

---

## 13. Reproducibility Contract

### `make demo`

```make
demo:
	CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python -m kernelsmith.run_demo $(DEMO_ARGS)
```

`run_demo()` order is not negotiable:

```
seed_everything() → Firestore run record → inference server subprocess → Runner → two turns
```

It seeds first, starts (or attaches to) the inference server on :8000 as a **subprocess**
(reusing a healthy one and leaving it running; `--no-server` attaches without starting one),
creates an ADK `Runner` over `root_agent` with in-memory session and artifact services,
sends **turn 1** (profile → retrieve → refine) and then **turn 2** (upsert → hot-swap →
explain → summarize) into the same session, records the run to Firestore, and prints results.
Every number printed comes from `best_verdict` — never from the Supervisor's prose.

Related targets: `make setup` (uv sync --frozen → harden → create-index → seed-skill),
`make test-unit` / `test-int` / `test`, `make lint`, `make serve-inference` (:8000),
`make serve-ui` (:8501), `make serve-demo` (:8502, dark theme pinned via
`--theme.base dark` because `st.set_page_config` has no theme parameter),
`make demo-with-dashboard` (the recording take), `make audit` / `audit-all`,
`make deploy-dashboard` / `deploy-explorer`, `make export-firestore`.

### Seeds — where and why the order matters

`kernelsmith/reproducibility.py::seed_everything(seed=42)`:

```python
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # FIRST — read at cuBLAS handle creation
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.set_float32_matmul_precision("high")          # TF32 baseline
torch.use_deterministic_algorithms(True)            # LAST, once seeds are in place
```

Set `CUBLAS_WORKSPACE_CONFIG` after the handle exists and it is ignored — and
`use_deterministic_algorithms(True)` then throws on the first GEMM. It is called before
anything in `run_demo` imports torch; the Makefile exports it too, belt-and-braces. The
sandbox runner re-seeds identically inside the subprocess, and `tests/conftest.py` sets the
env for tests.

**The seeds are only half of it.** `kernelsmith/sampling.py::deterministic_config()` applies
`temperature=0` and `seed=42` to **every** agent — Supervisor, Profiler, Coder, Judge — and
to the Gemma call. Reseeding torch cannot make a *sampled* kernel come back the same: the
seeds pinned everything below the model while the model itself was free to write a different
kernel every run. Two tests walk the built tree and fail if an agent is added without a
config. The same object carries the retry policy (5 attempts, exponential backoff, on
`[429,500,502,503,504]` only — a 400 or 403 is a bug, and retrying it just spends the budget
more slowly).

**Measured reproducibility** (two back-to-back `--no-server` runs, same seed): reward +3
both times, 1 iteration both times, speedups **bit-identical** (7.04× / 1.39× on the dev
box's RTX A500), latency identical at two of three shapes and 0.03% apart at `16x2048`. The
one thing that legitimately differs is the **bandit arm** — run 1 upserted the kernel it
verified, and UCB1 gives a zero-pull arm an unbounded exploration bonus, so run 2 pulls the
new one. That is the memory working, and it means back-to-back runs are not a valid replay:
restore the Firestore snapshot (`make export-firestore` / `gcloud firestore import`) before
any run that must reproduce an earlier one.

### Environment variables

`.env` is loaded by `kernelsmith/__init__.py` with **`override=False`**, so a real
environment variable always beats the file and a stale `.env` can never silently redirect a
run to another GCP project. `config.py` reads `GOOGLE_CLOUD_PROJECT` with `os.environ[...]`
strictly — a missing project fails loudly.

| Variable | Value | Why |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | `gpuyantra` | Read strictly at import |
| `GOOGLE_CLOUD_LOCATION` | `global` | Gemini 3.x is global-endpoint only |
| `GOOGLE_GENAI_USE_VERTEXAI` | `true` | Route google-genai through Vertex. ADK 2.7.1 warns this name is deprecated in favour of `GOOGLE_GENAI_USE_ENTERPRISE`; both work, the new one wins. Set only one, or both to the same value |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Determinism; must be set before cuBLAS init |

**No API keys, ever.** ADC only; `.env` is `.gitignore`d and holds non-secret config.

### Cost of a fresh L4 reproduction (README §Cost)

| Line | Conservative | Optimistic |
|---|---|---|
| `g2-standard-4` (one L4), ~30 min | $0.35 on-demand | $0.11 Spot |
| `gemini-3.7-flash`, 4 agents × iterations | $0.25 (6 iters) | $0.05 (1 iter) |
| `gemini-embedding-001` | <$0.01 | <$0.01 |
| `gemma-4-26b-a4b-it-maas` (bonus) | ~$0.01 | ~$0.01 |
| Firestore, Cloud Trace | $0 (free tier) | $0 |
| **Total** | **~$0.62** | **~$0.18** |

Whole project ~$44 against credits; the dominant line is VM hours, not tokens.

### Version pins (`pyproject.toml`)

Every **direct** dependency is `==`-pinned; the full 127-package transitive closure is pinned
by the committed `uv.lock`, and `make setup` installs with `uv sync --frozen`, which refuses
to re-resolve.

```
google-adk==2.7.1              torch==2.12.1          streamlit==1.62.0
google-cloud-firestore==2.28.1 triton==3.7.1          streamlit-autorefresh==1.0.1
google-genai==2.19.0           transformers==4.57.6   graphviz==0.21
fastapi==0.141.1               accelerate==1.14.0     pydantic==2.13.4
uvicorn[standard]==0.52.4      httpx==0.28.1          numpy==2.4.6
python-dotenv==1.2.3
dev: pytest==9.1.1  pytest-asyncio==1.4.0  ruff==0.16.4  google-auth==2.56.3
```

`requires-python = ">=3.11"`. `transformers` is deliberately held at the last 4.x release:
`Qwen2RMSNorm.forward(self, hidden_states)` is identical in 5.x, but 5.x pulls
`tokenizers>=0.22`, `safetensors>=0.8.0` and a newer `huggingface-hub` — transitive risk on
the DLVM for no gain on the monkey-patching surface.

`Dockerfile.dashboard` deliberately does **not** `uv sync --frozen`: the locked closure pins
torch + triton + transformers + ADK, several GB of CUDA wheels, on an image whose only job
is to read a JSONL file. Replay mode imports none of them, so it installs four packages at
`pyproject`'s pinned versions — ~400 MB instead of ~6 GB. **The two lists are kept in step
by hand; there is no check that they agree.**

---

## Loose ends worth closing before recording

1. ~~`LINKS` in the explorer is three `href: "#"` TODOs.~~ **Done 2026-08-31** — the hosted
   dashboard and the GitHub repo are wired, and the dashboard also has a hero CTA (§9). The
   demo video and the technical write-up are still `todo: true`, because neither exists yet;
   fill them when they do. **The explorer must be re-deployed for this to be live.**
2. **The before/after throughput pair must be FILMED, not captured** (§10). Both
   successful-swap traces report `0.0` tokens/s, and no trace ever can do better: chat
   exchanges never become ADK events, so `EventLogger` never sees them. On the L4, in Live
   mode: ask a preset question, run the optimization, ask again — and record the screen.
3. ~~Pick the trace the hosted dashboard opens on.~~ **Fixed 2026-08-31.**
   `ordered_traces()` sorted by mtime, which `COPY data/traces/` restamps at image build
   time — so in the container the first trace was decided by whichever file the copy
   touched last, out of five sharing one checkout mtime. It now sorts on the timestamp in
   the filename, which is capture order and survives any copy. The default is
   `demo-20260831-094045`, captioned "23 steps · 114s · 7.22× faster than PyTorch · went
   live on 57 layers". Pinned by
   `test_the_newest_capture_opens_first_whatever_the_mtimes_say`, which inverts the mtimes
   so a regression to mtime ordering fails there rather than in the deployed container.
4. ~~Test count is understated on a judge-facing surface.~~ **Fixed 2026-08-31** — the
   explorer reads 716 total / 698 unit + 18 integration (§10). **Re-deploy the explorer for
   this to be live.** The 18 integration tests are a collection count taken here; a full
   `make test-int` run on the L4 would confirm they still all pass.
