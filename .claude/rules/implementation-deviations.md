# Implementation Deviations from spec.md

These override the corresponding spec sections. Claude Code must follow these
over anything conflicting in spec.md. Each entry documents what changed, why,
and which task introduced it.

---

## transformers version (overrides "pin whatever is current")

Pinned at `transformers==4.57.6` (last 4.x release). The `Qwen2RMSNorm.forward(self,
hidden_states)` signature is identical in 5.x (verified against v5.10.2 source). The
only addition in 5.x is a `@use_kernel_forward_from_hub("RMSNorm")` class decorator,
which does not affect the monkey-patching surface. Do NOT upgrade to 5.x — it pulls
`tokenizers>=0.22`, `safetensors>=0.8.0`, and a newer `huggingface-hub`, introducing
transitive dependency risk on the DLVM. The uv venv isolates from any system
transformers the DLVM may ship.

*Source: Task 1 review, verified against HuggingFace transformers v5.10.2 source.*

---

## Agent prompts — InstructionProvider pattern (overrides §4.2)

Use `InstructionProvider` callables, NOT `{template}` strings. ADK's regex-based
template injection raises `KeyError` on missing state keys (iteration 1 has no
`verdict`, `bottleneck_fingerprint`, etc.) and cannot resolve dotted paths like
`{verdict.next_action}`. Providers bypass injection entirely by reading
`session.state` directly.

See `kernelsmith/agents/state_view.py` for the implementation.

*Source: Task 4, Claude Code decision #1.*

---

## Supervisor flow — resumable state machine (overrides §4.1 step sequence)

The Supervisor is a resumable state machine over `session.state`. After the
LoopAgent escalates, the Supervisor's turn ends (ADK limitation — an LlmAgent's
turn ends when it delegates to a sub-agent, and LoopAgent cannot transfer back).
The demo/UI driver must send a **second follow-up message** after escalation to
trigger upsert + hot-swap. Each step is idempotent; the Supervisor executes the
first undone step on each invocation.

Single-turn end-to-end (profile → loop → upsert → swap) is NOT possible with
ADK's LoopAgent. Any code that assumes it will silently skip upsert and swap.

*Source: Task 4, Claude Code decision #5.*

---

## Agent singletons — factory pattern required (overrides §4.2 wiring)

ADK binds `parent_agent` per instance and refuses re-parenting. Agent objects
must be created via factory functions (e.g., `build_supervisor()`), never as
module-level singletons. The single shared tree lives in `root_agent.py`.

*Source: Task 4, Claude Code decision #4.*

---

## `__init__.py` re-export ban (overrides any import convenience patterns)

Do NOT re-export `FunctionTool` instances from `kernelsmith/tools/__init__.py`.
A `FunctionTool` instance named `retrieval_tool` at the package level shadows
the `tools/retrieval_tool` module, breaking imports. Always import from the
specific module file:

```python
# WRONG:
from kernelsmith.tools import retrieval_tool

# RIGHT:
from kernelsmith.tools.retrieval_tool import retrieve_skills_for_agent
```

*Source: Task 3, Claude Code decision #3.*

---

## OP_REGISTRY — no executable source in task_spec (overrides §7)

`verify_kernel` resolves `op_name` through the `OP_REGISTRY` dict to get a
reference implementation, NOT through executable source code in the task_spec.
If the task_spec carried Python source, it would create a second, unchecked path
into the sandbox — a code-injection vector. Do not add any mechanism for the
task_spec to carry executable Python into the sandbox.

*Source: Task 3, Claude Code decision #1.*

---

## Subprocess reward must be recomputed in-process (overrides sandbox stdout trust)

The subprocess's own `reward` field in stdout is discarded. The candidate
controls that stdout, so trusting its reported reward is a trust-boundary
violation. Reward is always recomputed in-process from the correctness and
timing results returned by the verifier.

*Source: Task 3, Claude Code decision #2.*

---

## Hot-swap adapter — build_forward() bridge (overrides §8.3)

The verifier tests kernels as `entry(x, weight, eps)` — explicit tensor
arguments, no `self`. But `types.MethodType(new_fn, module)` causes Python
to call `new_fn(self, hidden_states)` where `self` is the nn.Module instance.

`build_forward()` in `patchable_ops.py` creates per-op adapters that close
over `self.weight`/`self.variance_epsilon` to bridge this gap. Any new
patchable op needs a matching adapter.

Adapter signatures were verified against `transformers==4.57.6` source:
- `Qwen2RMSNorm.forward(self, hidden_states)` → adapter extracts `self.weight`, `self.variance_epsilon`
- `Qwen2MLP.forward(self, x)` → adapter extracts `self.gate_proj`, `self.up_proj`, `self.down_proj`, `self.act_fn`

*Source: Task 5, Claude Code decision #1.*

---

## RoPE is NOT swappable (overrides §8.2 patchable ops list)

`apply_rotary_pos_emb` is a module-level function, NOT an `nn.Module`. It
cannot be discovered via `named_modules()` and therefore cannot be swapped
by `swap_op()`. `swap_op` correctly returns an empty match set, and `/swap`
refuses with an explicit error message. A silent zero-module patch reported
as success would fake a speedup.

If RoPE optimization is needed, it requires a different mechanism
(module-attribute rebind), which is out of scope for the hackathon.

*Source: Task 5, Claude Code decision #2.*

---

## Inference model must NEVER be torch.compile'd (overrides any performance optimization attempts)

`torch.compile` bakes the current `forward` into a compiled graph.
`types.MethodType` patching after compilation silently no-ops — the compiled
graph ignores the new method and continues executing the old forward. The
model must stay in eager mode for hot-swapping to work.

The only place `torch.compile` appears is in `measure_baselines()` where
the compiled baseline is measured for comparison, then discarded.

*Source: Task 5, Claude Code decision #6.*

---

## Baseline measurement fairness (overrides §5.2 timing)

`torch.use_deterministic_algorithms(True)` penalizes eager baselines by ~23%
(forces slower cuBLAS/cuDNN codepaths) but does NOT affect Triton kernels
(they generate their own PTX). This inflates reported speedups:
- With flag ON: 8.52× vs eager
- With flag OFF: 6.9× vs eager (fair comparison)

Fix: `measure_baselines()` must call `torch.use_deterministic_algorithms(False)`
before timing eager and torch.compile baselines, then restore the previous
setting after. The deterministic flag stays ON for correctness checks and the
agent loop — only the timed comparison needs it off.

Report both numbers in the demo if possible: "6.9× vs eager (fair), 1.36× vs
torch.compile."

*Source: Task 3 review, Claude Code honesty flag on measurement numbers.*

---

## TokenMeter clears rolling window on swap (clarifies §8.2)

`TokenMeter.rolling_tokens_per_sec` clears its rolling window on a swap event.
Without this, `/stats` reports an average straddling the swap boundary, which
undermines the demo's before/after claim. The Streamlit dashboard should show
a visible discontinuity at the swap point — that's the demo money shot.

*Source: Task 5, Claude Code decision on TokenMeter.*

---

## Generic adapter mapping — THE NOVEL CONTRIBUTION (adds to §8.3, implement in Task 7)

### Research basis

The field has converged on declared contracts for kernel deployment bridges:
- HF `kernels` pure-layer model: exact attribute-name matching, no inference
- FlashInfer-Bench Definition schema: entry-point arg names must match I/O keys
- Kernel Contracts (arXiv:2604.22032): eight-part formal contract closing "claim-scope gaps"
- FastKernels (arXiv:2605.23215): sandbox-optimized kernels achieve only 0.94× vs
  production baselines due to interface incompatibilities

**All published systems use human-authored deployment bridges.** KernelSmith's novel
contribution: the agent *generates* the deployment contract (`adapter_mapping`) and
the verifier *validates* it deterministically. No published system does this.

### Architecture (three-layer safety model)

**Layer 1 — Declared contract (NEW):** The Coder outputs `adapter_mapping: dict[str, str]`
as part of `KernelDraft`. Maps kernel parameter names → module attribute names.
Example: `{"weight": "weight", "eps": "variance_epsilon"}`.
The forward input arg ("x", "hidden_states") is implicit, not in the mapping.

**Layer 2 — Deterministic validation (NEW):** `validate_adapter_mapping(op_name, mapping)`
checks that every `module_attr` in the mapping exists via `hasattr()` on the target
class from `OP_REGISTRY[op_name].module_cls`, and that mapped attrs are
Parameters/buffers, not methods. If validation fails → reward −1, skip sandbox.
This runs BEFORE the subprocess sandbox, like the static checker.

**Layer 3 — Numerical equivalence gate (EXISTING):** 5 seeds × 3 shapes, atol=rtol=1e-2,
auto-rollback. Already implemented in Task 2.

### Implementation details

- `KernelDraft.adapter_mapping: dict[str, str]` — Coder declares kernel_param → module_attr
- `validate_adapter_mapping(op_name, mapping) -> list[str]` — deterministic check
- `build_forward_from_mapping(kernel_entry_fn, adapter_mapping)` — generic adapter
  that uses the validated mapping instead of per-op closures
- Per-op hard-coded adapters REMAIN as fallback when `adapter_mapping` is None/empty
  (backward compatibility with seed kernels)

### Judge verification flow (updated order)

1. `check_static(kernel_code)` — AST reward-hack checker
2. `validate_adapter_mapping(op_name, adapter_mapping)` — NEW: attribute existence check
3. Subprocess sandbox — correctness + timing
4. `compute_reward()` — in-process reward computation

### Demo beat

"Watch the system optimize an op it's never seen — it discovers the parameter layout,
writes a verified deployment contract, and hot-swaps the kernel into a live inference
server. No human wrote the adapter."

### References

- HF `kernels` pure-layer: https://github.com/huggingface/kernels/blob/main/docs/kernel-requirements.md
- FlashInfer-Bench: arXiv:2601.00227
- Kernel Contracts: arXiv:2604.22032
- FastKernels: arXiv:2605.23215
- CUDA Agent: arXiv:2602.24286

### Fallback

If the generic adapter doesn't work reliably by Day 8, revert to hard-coded adapters.
The three-layer story still holds for the blog/video even if the generic path only
works for RMSNorm in the demo.

*Source: Research review Aug 25, 2026. Decision: implement in Task 7, validate in Task 8.*

---

## Always use `uv run python`, never system `python3`

System Python lacks venv dependencies and the `GOOGLE_CLOUD_PROJECT` env var.
Tests pass because `conftest.py` sets the env var, but manual gate checks need:

```bash
GOOGLE_CLOUD_PROJECT=gpuyantra uv run python -c "..."
```

*Source: Task 3 gate check failure.*

---

## `do_bench` parameters (reinforces §5.2)

`triton.testing.do_bench` must be called with `warmup=150, rep=200,
return_mode="median"`. The documented default `warmup=25` underestimates by
~30%. `bench_kernel()` enforces `warmup >= 150` and raises if violated.

*Source: Spec §5.2, verified in Task 2.*

---

## `streamlit-autorefresh` added to the pinned stack (adds to §0, implements §10.2)

Pinned at `streamlit-autorefresh==1.0.1`, resolved clean against `streamlit==1.62.0`.
Spec §10.2 leaves the refresh mechanism open ("`st_autorefresh` or `st.empty()` with a
polling loop"); the component reruns the *whole* script, which is what the dashboard
needs — `st.fragment(run_every=…)` reruns only its own fragment, so the agent panels
and the banner would freeze while the throughput metric ticked.

It is a 2021 custom component, so the import is guarded: `render_autorefresh()` falls
back to a manual "🔄 Refresh" button and a warning if the import or the component call
raises. The dashboard is fully usable on the fallback path.

*Source: Task 6, Claude Code decision.*

---

## Dashboard drives the two-message Supervisor flow automatically (implements §10.1)

The two-turn protocol above is real but must not be the operator's problem on demo day.
`streamlit_app.py`'s `drive_run()` watches `EventStreamConsumer.runs_completed` and
fires the follow-up message itself once turn 1 goes idle. One button press therefore
runs profile → loop → upsert → hot-swap → summary.

`EventStreamConsumer.start_run()` refuses overlapping runs (returns `False`) precisely
because the dashboard refreshes at 1 Hz: without that guard, a tick landing during a
run would launch a second Supervisor over the same session.

The latency chart inserts a `None` sample at each swap boundary rather than joining
across it, because `TokenMeter` clears its rolling window on swap — the two sides
describe different kernels, and a continuous line would smooth over the discontinuity
the demo is claiming.

*Source: Task 6, Claude Code decision.*

---

## accelerate added as required dependency (adds to §0 stack)

device_map='cuda' in AutoModelForCausalLM.from_pretrained() requires accelerate at import time. Missing from original spec §0 stack. Pinned at accelerate>=1.14.0 (VM smoke test installed 1.14.0 on Aug 26). Must be added to pyproject.toml dependencies.

Source: VM smoke test, Aug 26.

---

## torch_dtype → dtype parameter rename (affects §8 inference server)

transformers==4.57.6 emits a deprecation warning: "torch_dtype is deprecated! Use dtype instead!" All calls to AutoModelForCausalLM.from_pretrained() and any other HuggingFace API using torch_dtype= must be replaced with dtype=. This affects kernelsmith/inference_server/models.py and any test fixtures that load models. The parameter semantics are identical — only the name changed.

Source: VM smoke test, Aug 26.

---

## VM environment snapshot (reference for Task 9 version pins)

Recorded during VM smoke test, Aug 26:

DLVM image: pytorch-2-9-cu129-ubuntu-2204-nvidia-580
NVIDIA driver: 580.173.02, CUDA capability: 13.0
System Python: 3.10.12; uv venv Python: 3.14.7
torch==2.12.1+cu130, triton==3.7.1, transformers==4.57.6
accelerate==1.14.0, numpy==2.4.6
GPU: NVIDIA L4, 23034 MiB VRAM, model uses 3.09 GB at FP16
Qwen2RMSNorm: 57 modules, weight shape [1536], variance_epsilon=1e-06
CUBLAS_WORKSPACE_CONFIG not set in shell (dotenv loads it into Python); add export CUBLAS_WORKSPACE_CONFIG=:4096:8 to VM ~/.bashrc before Task 8.

*Source: VM smoke test, Aug 26.*

---

## Adapter-mapping validation lives in `verifier/`, and probes a meta-device instance (implements the generic-adapter section)

Two decisions the generic-adapter spec above left open.

**Where.** `validate_adapter_mapping` is `kernelsmith/verifier/adapter_mapping.py`, not
`patchable_ops.py`. It is a verifier gate (layer 2, run before the sandbox), and putting
it in `inference_server/` would make the verifier import the server package. The
generic adapter itself (`build_forward_from_mapping`) stays in `patchable_ops.py`, next
to the hard-coded adapters it supersedes.

**How the attributes are checked.** The spec says `hasattr()` on
`OP_REGISTRY[op_name].module_cls`. `hasattr(Qwen2RMSNorm, "weight")` is **False** —
`weight` and `variance_epsilon` are assigned in `__init__`, so they exist on instances,
not on the class. `OP_REGISTRY` also maps op names to *builder callables*, which carry
no `module_cls`. So:

- a separate `_OP_MODULES` table maps normalized op names to the transformers class
  (`rmsnorm` → `Qwen2RMSNorm`, `mlp`/`swiglu` → `Qwen2MLP`), and
- the check builds a real instance under `torch.device("meta")` — zero bytes allocated,
  no GPU — and runs `hasattr` against *that*.

Ops with no `nn.Module` in Qwen2 (`rope`, `softmax`, `silu`, `layernorm`) reject any
non-empty mapping: an unvalidatable contract is also an undeployable one.

If the probe cannot be built (transformers missing, a signature change upstream),
validation degrades to a declared name allowlist rather than blocking a verified
kernel — layers 3 (5x3 numerical gate) and the server's parity gate are still in front
of the live model.

Also: dotted paths (`"gate_proj.weight"`) are supported, `swiglu`/`mlp` and
`rms_norm`/`rmsnorm` are aliases, and mapping the implicit input arg (`x`,
`hidden_states`) is an error, not a no-op.

*Source: Task 7, Claude Code decision.*

---

## The bandit is credited by the EscalationChecker, once per run (implements §9)

One run is one pull. The reward that matters is `best_reward`, which is only final when
the RefinementLoop escalates — so the credit is written there, guarded by
`bandit_credited` in `session.state`, via `EventActions(state_delta=...)` (direct state
mutation inside `_run_async_impl` does not persist in ADK). Crediting per iteration
would record six pulls for one experiment; crediting from the Supervisor's upsert step
would silently skip every run that scored below +1, biasing every arm's mean upward.

This is the one side effect in an otherwise pure agent. It runs through
`asyncio.to_thread`, swallows its own failures, and marks the run credited either way —
a retry could double-count a write that actually landed.

`retrieve_skills_for_agent` returns `selected_skill_id` and reorders `skills` so the
bandit's pick leads; the Supervisor's `after_tool_callback` copies that id into state
for the checker to use.

*Source: Task 7, Claude Code decision.*

---

## `run_demo()` drives both turns itself and owns the inference server subprocess (implements §16)

`make demo` runs `python -m kernelsmith.run_demo`. The two-message Supervisor protocol
documented above is real for the CLI too: `run_demo` sends turn 1 (profile → retrieve →
refine) and then turn 2 (upsert → hot-swap → explain → summarize) into the same session.
A single-message demo silently skips the hot-swap and reports a run that never went live.

`seed_everything()` runs before anything imports torch, because `CUBLAS_WORKSPACE_CONFIG`
is read when the cuBLAS handle is created and ignored afterwards. The Makefile also
exports it, belt-and-braces.

The inference server is started as a `uvicorn` **subprocess** and stopped on exit; if a
healthy server is already on the port, it is reused and left running. `--no-server`
attaches without starting one. Every number printed comes from `best_verdict`, never
from the Supervisor's prose summary.

*Source: Task 7, Claude Code decision.*

---

## Baseline fairness fix is now IMPLEMENTED, not just documented (closes "Baseline measurement fairness")

The deviation above was written but never landed in code: `measure_baselines()` timed
both baselines with `torch.use_deterministic_algorithms(True)` still on. It now wraps
the timed region in `_nondeterministic_for_timing()`, which saves the flag (and
`warn_only`), turns it off, and restores it in a `finally` so a benchmark that raises
cannot leave determinism off for the correctness gate.

Measured on the seed RMSNorm kernel after the fix: **6.92x vs eager, 1.36x vs
torch.compile** — matching the "fair" numbers this file predicted, against the 8.52x the
flag was inflating them to. Locked down by `tests/test_timing.py`, which asserts the
flag is off during both baselines and restored after (including on exceptions).

*Source: Task 8.*

---

## `/swap` must import kernel source from a REAL FILE (fixes a 100% hot-swap failure)

`_load_entrypoint` used `exec(compile(source, "<hotswap-kernel>", "exec"))`. `@triton.jit`
calls `inspect.getsourcelines` on the decorated function at DECORATION time, so with no
file on disk it raises:

    ValueError: @jit functions should be defined in a Python file

Every kernel this system produces is a Triton kernel, so **every** hot-swap failed —
and failed as a refused `/swap` carrying a load error, which reads like a bad kernel
rather than a broken loader. The demo's money shot (beat 9) could never have fired.

It now writes the source into a process-lifetime temp dir and imports it via
`importlib.util.spec_from_file_location`, the same mechanism the verifier sandbox
already used. Two details that are load-bearing:

- the module is registered in `sys.modules` **before** `exec_module`, so `inspect` can
  resolve it while the `@triton.jit` decorators at module top level are still running;
- the file is **not** deleted afterwards. Triton compiles lazily and re-reads the source
  when it specializes for a new shape or dtype, so cleaning up at the end of the load
  would move the failure from swap time to first-token time.

The gap that hid this: every test in `test_hotswap.py` used a pure-torch stand-in
kernel. `tests/test_hotswap.py` now carries Triton-specific loader tests (they need no
GPU — decoration is what fails).

*Source: Task 8, found while validating the generic adapter on real Qwen2 modules.*

---

## The deployment contract must never be a free-form dict on an LLM boundary

Observed live: the Judge called `verify_kernel(..., adapter_mapping={})` on every run,
and the Coder's draft carried `{}` too. The generic adapter — the novel contribution —
therefore never ran once. Every green check still passed, because an empty contract is
not an error anywhere: it silently falls back to the hard-coded per-op adapter, i.e.
the human-written bridge this project claims the agent writes for itself. The demo beat
("No human wrote the adapter") would have been false with a full row of green ticks.

**Root cause.** `dict[str, str]` compiles to a JSON schema with no named properties, so
structured generation has nothing to anchor on and emits `{}`. Measured directly
against gemini-3.7-flash, same prompt, 3 trials each:

| Schema | Filled correctly |
|---|---|
| `adapter_mapping: dict[str, str]` | **0/3** (`{}` every time) |
| `adapter_mapping: list[AdapterBinding]` | **3/3** |

Making the field merely *required* was not enough — `{}` satisfies a required dict.

**Fix, at both LLM boundaries.**

1. `KernelDraft.adapter_mapping` is now `list[AdapterBinding]`, where `AdapterBinding`
   has two named string fields (`kernel_param`, `module_attr`). `KernelDraft.
   mapping_as_dict()` converts to the `{param: attr}` form every consumer downstream
   already takes, so the validator, `build_forward_from_mapping` and `/swap` are
   unchanged.
2. The Judge no longer passes the contract at all. `verifier_tool` is built from
   `verify_kernel_for_agent`, whose LLM-facing parameters are exactly
   `kernel_code`, `entrypoint`, `task_spec`; it reads the mapping from
   `state["kernel_draft"]` via `tool_context`. This removes the second free-form dict
   and structurally enforces the rule the prompt could only ask for — the Judge must
   never invent or "fix" the contract.

`adapter_mapping_from_draft()` accepts the list form, the legacy dict form, and a raw
JSON string, and degrades to `{}` on anything malformed rather than raising.

The tool's published name stays `verify_kernel` (`find_verifier_response` matches on it
and the Judge's prompt names it), pinned via `__name__` on the wrapper.

Confirmed live after the fix — the Coder now emits:

    [{"kernel_param": "weight", "module_attr": "weight"},
     {"kernel_param": "eps",    "module_attr": "variance_epsilon"}]

which is exactly the contract the real `Qwen2RMSNorm` requires.

**General rule for this codebase: never put a free-form `dict[str, str]` on a boundary
an LLM has to fill.** Use a list of objects with named fields.

*Source: Task 8, observed in a live dashboard run and confirmed by an A/B on the schema.*

---

## Task 8 validated on an RTX A500 (4 GB), not an L4 — what that does and does not cover

Task 8 ran on a laptop RTX A500 Laptop GPU (4096 MiB, driver 580.173.02, CUDA 13.0,
Python 3.12.12, torch 2.12.1+cu130), not the g2-standard-4 L4 the spec targets.

Fully validated here, and hardware-independent:

- the verifier end to end — 15/15 correctness checks, reward +3, 6.92x / 1.36x;
- the whole agent tree over live `gemini-3.7-flash` (16 integration tests);
- Firestore vector retrieval, the UCB1 bandit, and the composite index;
- the generic adapter on **real `Qwen2RMSNorm` modules**: contract validates, the
  generic path is taken rather than the fallback (asserted on `__qualname__`), parity
  holds at atol=1e-2, rollback is bitwise-exact, 6.90x on a 4-layer stack;
- the real Qwen2 architecture's attribute names, read from `AutoConfig` +
  `from_config` on a meta device: `weight` [1536], `variance_epsilon` 1e-06,
  **57 `Qwen2RMSNorm` modules** — matching the Aug 26 VM smoke test exactly;
- the dashboard: renders with no exceptions via `AppTest`, and all five agents
  (Supervisor, Profiler, Coder, Judge, EscalationChecker) stream events through
  `EventStreamConsumer`.

NOT covered here, and still owed on the L4:

- **live tokens/sec across a swap** — the 3.09 GB Qwen2.5-1.5B weights could not be
  fetched (the link ran at ~1.4 KB/s), and at 4 GB the card cannot hold the model plus
  a concurrent verifier subprocess anyway. Re-run `make demo` on the L4 for beat 9.
- absolute roofline numbers. `config.py`'s L4 constants (300.1 GB/s, 58 SMs) are wrong
  for an A500, so the profiler's `memory_throughput_gbps` and `achieved_occupancy` are
  not meaningful here. Speedup *ratios* are, since both sides are measured on the same
  card.

*Source: Task 8.*

---

## `GOOGLE_GENAI_USE_VERTEXAI` is deprecated in favour of `GOOGLE_GENAI_USE_ENTERPRISE`

ADK 2.7.1 warns on every run: "GOOGLE_GENAI_USE_VERTEXAI is deprecated, please use
GOOGLE_GENAI_USE_ENTERPRISE instead". Verified in
`google/adk/utils/env_utils.py::is_enterprise_mode_enabled` — `GOOGLE_GENAI_USE_ENTERPRISE`
is checked first and wins; the old name still works and only then warns.
`google/genai/_api_client.py` honours both the same way, and warns separately if the two
are set to *conflicting* values.

Currently set as the old name in `.env`, `.env.example` and `tests/conftest.py`. The
switch is safe whenever someone wants the warning gone — nothing in this repo depends on
the old name, because `embeddings.py` and `explainer_tool.py` both construct
`genai.Client(vertexai=True, ...)` explicitly rather than relying on the environment.

Not changed yet: the old flag still works, and `.env` lives only on the VM where it
cannot be edited from a dev box. If you switch, switch all three together, or leave both
set to the same value — mismatched values are the one case that actually misbehaves.

*Source: Task 8 integration warnings, verified against the ADK 2.7.1 source.*

---

## The integration credential guard must recognize the Compute Engine metadata server

`_credentials_are_available()` in `tests/test_integration.py` checked only
`GOOGLE_APPLICATION_CREDENTIALS` and the ADC file at
`~/.config/gcloud/application_default_credentials.json`. On a Compute Engine VM neither
exists — credentials come from the metadata server. The guard therefore skipped the
entire integration module on the one machine it was written to run on, and skipping
reads as green.

Fixed with a third check that falls through to `google.auth.default()` and treats
non-None credentials as available, wrapped in `try/except` so a machine with no
credentials at all still skips rather than errors.

*Source: VM integration run, Aug 27. Fix authored on the VM (commit 8a230db).*
