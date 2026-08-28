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

Ops with no `nn.Module` in Qwen2 (`rope`, `softmax`, `silu`) reject any
non-empty mapping: an unvalidatable contract is also an undeployable one.

Note: `layernorm` was previously in this reject list, but `torch.nn.LayerNorm`
IS an `nn.Module` — it was incorrectly lumped with non-module ops like
`apply_rotary_pos_emb`. LayerNorm is now a valid patchable op (see Task 10
addendum below).

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

## What Task 8 is validated on, and the one thing that still is not

The Task 8 work was authored on a dev box with an RTX A500 (4 GB) and re-run unchanged
on the L4 VM. Both runs collect and pass the **same 329 tests**, including all 18
integration tests against live `gemini-3.7-flash`, Firestore and the GPU — on Python
3.12.12 / torch 2.12.1+cu130 locally and Python 3.14.7 on the VM. Nothing in Task 8
turned out to be hardware-contingent, so this is a note about coverage, not a caveat on
the code.

What both runs cover:

- the verifier end to end — 15/15 correctness checks, reward +3, 6.92x vs eager and
  1.36x vs torch.compile after the fairness fix;
- the whole agent tree, budget-capped at `max_iterations=2`;
- Firestore vector retrieval, the composite index, and the UCB1 bandit;
- the generic adapter against **real `Qwen2RMSNorm` modules**: the declared contract
  validates, the generic path is taken rather than the per-op fallback (asserted on
  `__qualname__`), parity holds at atol=1e-2, rollback is bitwise-exact;
- the real Qwen2 attribute names — `weight` [1536], `variance_epsilon` 1e-06, 57
  `Qwen2RMSNorm` modules — matching the Aug 26 VM smoke test;
- the dashboard renders without exception, and all five agents stream events.

**Still not validated anywhere: live tokens/sec across a hot-swap (demo beat 9.)**
The pieces underneath it are individually verified (the swap mechanism, parity, rollback,
the adapter, `TokenMeter` clearing its window), but the end-to-end throughput jump the
demo is built around has not been observed on a served model.

*Updated Aug 28 (Task 9).* `run_demo` has now completed end to end, twice, on the dev box
with `--no-server`: the agent half runs, the verifier scores it, the skill is upserted and
a `runs` record lands. What is still missing is exactly the server half — those runs
report `hot-swap: not live (connection refused)` because a 4 GB RTX A500 cannot hold
Qwen2.5-1.5B (3.09 GB) and the verifier sandbox at the same time. Only an L4 can.

This remains the one open item before recording. Run `make demo` on the L4 — with the
server, no `--no-server` — and check that the `runs` record lands with a non-empty
`hotswap_result`.

*Source: Task 8, dev box Aug 27; VM re-run Aug 27 (18/18 integration, 329 total); Task 9
dev-box demo runs Aug 28.*

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

---

## The Gemma bonus model id needs the `-maas` suffix (overrides §15 and CLAUDE.md rule 1)

`GEMMA_MODEL` is `gemma-4-26b-a4b-it-maas`, not `gemma-4-26b-a4b-it`.

Observed in the first `make demo` that ever completed: the Gemma explainer returned
`404 NOT_FOUND` — "Publisher model `projects/gpuyantra/locations/global/publishers/
google/models/gemma-4-26b-a4b-it` was not found or your project does not have access".
Not a regional gap and not a permissions gap: the same id 404s in `global`,
`us-central1`, `us-east4` and `europe-west4`. Listing what the project can actually see
(`GET /v1beta1/publishers/google/models`) returns `gemma-4-26b-a4b-it-maas` — the
Model-as-a-Service serving name, which is how every Gemma MaaS model is published on
Vertex. Same model, different id.

Verified after the change: `gemma-4-26b-a4b-it-maas` on **`global`** returns a
completion. `us-central1` returns `400 FAILED_PRECONDITION` for this model, so the
explainer's client must stay on the global endpoint like everything else here.

This is a deliberate departure from CLAUDE.md critical rule 1, which names
`gemma-4-26b-a4b-it` literally. The rule exists to stop silent downgrades to an older
model family; this is the same Gemma 4 26B instruction-tuned model under the name Vertex
actually serves it as. The bonus agent was dead without it — it failed softly, returning
`error: ClientError: 404 …` into the demo output, because an explanation is a bonus and
must never fail a run whose kernel is already verified.

*Source: Task 9, first completed `make demo` run, Aug 28.*

---

## Every agent decodes greedily, and retries a rate limit (implements §11, adds to §4.2)

Spec §11 lists `temperature=0` **on the Judge** in the same row as the torch/NumPy/random
seeds. No agent set a `generate_content_config` at all, so all four — and the Gemma
explainer — ran at the Vertex default (~1.0).

That gap was larger than its one-line placement in the spec suggests. Reseeding torch
cannot make a *sampled* kernel come back the same: the seeds pinned everything below the
model while the model itself was free to write a different kernel every run. Two
`make demo` runs would have reported different rewards and different speedups from
identical inputs — the Sakana failure mode arriving through the one door the
reproducibility contract had left open.

`kernelsmith/sampling.py` now holds one `deterministic_config()` — `temperature=0`,
`seed=GLOBAL_SEED` — applied to Supervisor, Profiler, Coder, Judge and the Gemma call.
Applied to the Coder as well as the Judge on purpose: the kernel *is* the headline number.
The Coder does not stall on a repeated prompt, because its prompt carries the previous
verdict's `next_action` and `stderr_tail`, which change every iteration.

The same object carries the retry policy, found the hard way: two `make demo` runs back
to back exhausted the project's per-minute Vertex quota, and the second died five seconds
in on an unhandled `429 RESOURCE_EXHAUSTED` — forty lines of ADK traceback, exit code 2,
nothing salvaged. ADK surfaces the model error and stops; nothing below it retries. So
`http_options.retry_options` asks google-genai for 5 attempts with exponential backoff on
`[429, 500, 502, 503, 504]` only — a 400 or 403 is a bug or a permission gap, and
retrying it just spends the budget more slowly.

Locked down by `test_every_agent_decodes_greedily` and
`test_every_agent_backs_off_on_a_rate_limit`, both of which walk the built tree, so an
agent added later without a config fails the suite.

*Source: Task 9, spec §20 checklist audit + two live demo runs, Aug 28.*

---

## `.env` is loaded by `kernelsmith/__init__.py`, override=False (fixes `make demo` from a clean clone)

`config.py` reads `GOOGLE_CLOUD_PROJECT` with `os.environ[...]`, strictly, so a missing
project fails loudly instead of talking to the wrong one. Nothing in the repo loaded the
`.env` file the README tells you to create. ADK reads `.env` only through the `adk` CLI,
which neither `make demo` nor `make serve-ui` goes through, so the documented setup path —
`cp .env.example .env && make demo` — died on a `KeyError` one line into startup.

Invisible until now because it never fired anywhere it was being tested: the VM exports
the variables from the shell profile, and `tests/conftest.py` sets them itself. The
earlier deviations note claiming "dotenv loads it into Python" was describing something
that was not in this repo.

Loading it in the package `__init__` is the only hook guaranteed to run before any
`kernelsmith.*` import. `override=False` is the load-bearing argument: a real environment
variable — the VM's profile, a CI secret, `GOOGLE_CLOUD_PROJECT=... uv run ...` — always
beats the file, so a stale `.env` can never silently redirect a run to another GCP
project. `python-dotenv==1.2.3` is now a declared dependency rather than an accidental
transitive of ADK, with a minimal built-in parser as fallback.

*Source: Task 9, first attempt at `make demo` on a box without the variables exported.*

---

## `make demo` reproducibility, measured (closes the Task 9 gate)

Two `make demo --no-server` runs, back to back, same seed, dev box (RTX A500), Aug 28,
both with the greedy-decoding fix in place:

| | Run 1 | Run 2 |
|---|---|---|
| reward | +3 | +3 |
| iterations | 1 | 1 |
| speedup vs eager | 7.04x | 7.04x |
| speedup vs torch.compile | 1.39x | 1.39x |
| latency `1x128`, `8x512` | bit-identical | bit-identical |
| latency `16x2048` | 3.3178 ms | 3.3167 ms |
| bandit arm | `rmsnorm_fp16_l4_v1` | `rmsnorm_l4_single_pass_fused` |

Reward and both speedups reproduce exactly. The `16x2048` latency moves by 0.03%, which
is `do_bench` wall-clock noise on the largest shape, not a different kernel.

**The arm is supposed to differ.** Run 1 upserted the kernel it had just verified, and
UCB1 gives a zero-pull arm an unbounded exploration bonus, so run 2 pulls the new one.
This is the memory working. It also means back-to-back runs are not a valid replay:
restore the Firestore snapshot (`make export-firestore` /
`gcloud firestore import`) before any run that has to reproduce an earlier one. That is
what spec §11's "Firestore snapshot" row was always for; there was no script for it until
now (`scripts/export_firestore.sh`).

The winning kernels differ by exactly one line — a comment — for the same reason: a
different retrieved skill is a different prompt, and greedy decoding on a different
prompt may legitimately differ. The code was identical and measured identically.

Note the speedups here (7.04x / 1.39x) are the RTX A500's, not the L4's. The L4 numbers
from Task 8 are 6.92x / 1.36x. Do not quote these two interchangeably.

*Source: Task 9, Aug 28.*

---

## Multi-model audit — CPU-mode profiling (adds to §7, Task 10)

`audit_model()` can run on CPU with analytic FLOP/byte estimates (computed from
parameter shapes, not measured via `do_bench`). This means the audit tab in the
dashboard and `make audit` work without a GPU — only the actual kernel optimization
requires CUDA. The CPU-mode arithmetic intensity values are estimates; GPU-mode
values from `do_bench` are authoritative.

When running on CUDA, `audit_model()` uses `do_bench` on one representative instance
of each unique module type to get measured bandwidth utilization. On CPU, it uses
analytic estimates and labels the results "estimated" in the report.

*Source: Task 10.*

---

## CPU-mode audit builds the tree from `config.json` on the meta device, not from weights
(implements the CPU-mode note above)

The spec says "load model with `AutoModel.from_pretrained()`... for cpu use
`dtype=torch.float32`". Implemented differently on the CPU path, deliberately.

The audit reads `in_features`, `normalized_shape` and parameter SHAPES. It never runs a
forward pass on CPU, so it does not need the weights — and downloading 3.4 GB of them to
count 57 RMSNorms makes `make audit` unusable on a laptop and impossible offline. The
dev box proved the point: the HF cache held Qwen2.5-1.5B's `config.json` and an
`.incomplete` weight blob, and `from_pretrained` died with `AttributeError: 'NoneType'
object has no attribute 'endswith'`.

`_load_for_audit()` therefore does, on CPU:

    config = AutoConfig.from_pretrained(hf_id)      # ~1 KB
    with torch.device("meta"):
        AutoModel.from_config(config).eval()

0.11 s, zero bytes allocated, and the module tree is EXACT — 369 modules, 57
`Qwen2RMSNorm`, `weight` [1536], matching the Aug 26 VM smoke test. Same trick
`verifier/adapter_mapping._probe_instance` already uses. If config-only construction
fails for an architecture, it falls back to a real `from_pretrained` rather than giving
up on the audit.

On CUDA the weights ARE loaded, because `do_bench` cannot time a meta tensor.
`AuditReport.weights_loaded` records which path ran and the report prints it.

*Source: Task 10, dev box Aug 28.*

---

## Two FLOP/byte estimators now coexist, and they disagree on purpose

`analytic_counts` (existing, §7) counts the MINIMUM traffic a fused kernel must move:
read each input once, write the output once. Norm = 5 flop/elem over 2 tensors. It is the
retrieval fingerprint's estimator and must describe the OP, not an implementation.

`estimate_flops_and_bytes` (new, Task 10 Part B) counts traffic PER TENSOR TOUCHED,
including a weight and bias read per row and a Linear's whole weight matrix. Norm = 5 over
3 tensors, LayerNorm = 7 over 4. It is the audit's estimator, and it deliberately
describes what an UNFUSED eager implementation actually moves — which is the headroom a
kernel can recover, i.e. the number a triage table is asked for.

So the same RMSNorm reads AI 1.25 through one and 0.83 through the other (fp16). Both put
it two orders of magnitude below the ridge point, which is the only question either is
asked. **Where they differ numerically, `analytic_counts` is the one wired to Firestore.**
Do not "reconcile" them.

*Source: Task 10.*

---

## `classify_op_family` takes a CALLABLE; `family_from_name` takes a string

`classify_op_family("RMSNorm")` returns **"elementwise"**, silently. It reads names off
the object it is given, and for a `str` the only candidate is `type(x).__name__` == `str`,
which matches no keyword, so it falls through to the conservative default.

Two Task 10 call sites needed a family from a NAME rather than from a callable — the
audit's module class name (`"LayerNorm"`), and `MODEL_REGISTRY[...]["norm_type"]`. Both
got "elementwise", which would have made the dashboard's transfer-readiness table report
every model as a cold start and pointed `demonstrate_cross_model_transfer` at the wrong
`op_family` pre-filter: a demo that quietly claims transfer does not work.

The name-based classifier already existed as the private `_family_from_name`. It is now
public as **`family_from_name`**, with the trap documented on both functions. Anything
holding a name rather than a callable must use it.

Caught by `test_every_norm_type_classifies_into_the_retrieval_family_norm` in
`tests/test_model_registry.py`, which is why that test exists.

*Source: Task 10.*

---

## The audit reports `n/a`, never a fabricated zero, and names the GPU it measured on

Two honesty rules in `format_audit_report`, both instances of red line #3.

**Unestimated is not zero.** A module the estimator does not recognize (an embedding
gather, a dropout, a pooling layer) gets `(0, 0)` and `arithmetic_intensity = 0.0`.
`AuditEntry.bottleneck` still says `"memory"` — the conservative default, matching
`fallback_fingerprint` — but the table prints `—` in the Regime column and `n/a` for AI
and BW, because a regime nobody computed must not be displayed as one that was. Such
entries are also forced to priority LOW: AI 0.0 is an absence of information, not a
bottleneck, and without that rule every dropout in GPT-2 ranks as a MEDIUM target.

**`BW %` is a fraction of the L4's 300 GB/s.** Measured on any other GPU that denominator
is wrong. `AuditReport.gpu_name` records the device and the mode line appends
"⚠ measured on <GPU>, but BW % is against the L4's 300 GB/s — not comparable to an L4
run" whenever the name does not contain "L4". Observed on the dev box: GPT-2's LayerNorm
reported 39% on an RTX A500, which is ~104% of that card's actual bandwidth.

*Source: Task 10.*

---

## transformers' `Conv1D` is a Linear, and getting that wrong mis-called GPT-2's bottleneck

GPT-2's q/k/v and MLP projections are `transformers.pytorch_utils.Conv1D` — a Linear with
its weight stored transposed as `[in, out]`, not a convolution. Classified by name it
matched nothing, so all 48 of them returned `(0, 0)`, and the composite sum that gives
`GPT2Block` its intensity was left with only the LayerNorms and the GELU. The audit
therefore reported **GPT2Block and GPT2MLP as memory-bound** when GPT-2's arithmetic is
entirely in those 48 modules.

Fixed by recognizing the `nf` attribute (which only Conv1D carries) as "linear", and by
falling back to `weight.shape` in `_linear_counts` when `in_features`/`out_features` are
absent. The order does not matter: both `2*B*M*N` and `B*M + M*N + B*N` are symmetric in
(M, N), so one fallback covers Linear's `[out, in]` and Conv1D's `[in, out]`.

After: Conv1D 271, GPT2Attention 256, GPT2MLP 256, GPT2Block 210 FLOP/byte — all
compute-bound, with LayerNorm (0.88, 25 instances) the top target. Which is the correct
story for GPT-2.

*Source: Task 10, dev box Aug 28.*

---

## Container modules are excluded from the audit; composite blocks are not

`named_modules()` yields the root and every container. Neither `ModuleList` nor "the whole
model" is a swappable target, so the root and `_STRUCTURAL_CONTAINERS` (ModuleList,
ModuleDict, Sequential, ParameterList/Dict) are dropped, and `AuditReport.total_modules`
counts only what made the table — so the header agrees with the rows rather than with
`len(list(model.named_modules()))`.

Composite blocks (`Qwen2MLP`, `GPT2Block`, `ResNetStage`) ARE reported, estimated as the
sum of their children at each child's own derived probe shape. A fusable block is a real
target — `PATCHABLE_OPS` already knows how to swap `Qwen2MLP` — and without the sum they
would show AI 0.0 and rank LOW for the wrong reason. Known weakness, stated in the code: a
child whose width depends on a sibling's output (the activation between an MLP's up- and
down-projection runs at the intermediate width) is estimated at the parent's input shape
and understated. It does not move a composite across the ridge point.

One representative instance per CLASS, so a `Linear` row describes whichever projection
was met first, not an average over all 196. Right for triage (the regime is the same for
all of them), but the AI in that row is one instance's.

*Source: Task 10.*

---

## The dashboard auto-refresh ticks only on Optimize, and `drive_run` runs on every tab

Two halves of one constraint, and the second is load-bearing for the hot-swap.

The 1 Hz whole-script rerun would restart an in-progress audit forever, and the two new
read-only tabs have nothing to poll. So `render_autorefresh()` is called only when the
Optimize tab is showing **or a run is in flight** — the second clause matters, because a
run still needs ticks to reach turn 2 while the operator is looking at another tab.

`ingest_events()` and `drive_run()` are called BEFORE the tab dispatch returns, on every
tab. Moving them below the `return`s would break the two-message protocol — turn 2 never
sent, upsert and hot-swap silently skipped — while leaving every panel looking healthy.
`test_the_run_driver_is_called_on_every_tab_not_only_on_optimize` in
`tests/test_dashboard_tabs.py` asserts the ordering in the source, because reproducing it
for real needs Vertex, a GPU and a live server.

`tests/test_dashboard_tabs.py` is also the first automated check that the dashboard
renders at all (via `streamlit.testing.v1.AppTest`); until now that was verified by
opening it.

*Source: Task 10.*

---

## Transfer readiness joins on `(op_family, hardware)`, never on the model

The Skill Library tab's readiness table, and `demonstrate_cross_model_transfer`, both join
exactly the way `retrieve_skills` pre-filters: on `op_family` and `hardware`. Joining on
the model or the op name would make the table agree with itself while describing
something retrieval does not do.

Confirmed live against the real library on Aug 28: GPT-2's LayerNorm fingerprint
(`op=norm mem_bound=True ai=0.4 tile=1024 hw=L4`) retrieved all three Qwen2.5 RMSNorm
skills, nearest at vector distance 0.0128. `tests/test_cross_model.py` locks the query
shape down with a fake that HONOURS its recorded pre-filters, so a dropped filter cannot
pass as a match, and asserts that no model or op name leaks into the embedded text.

*Source: Task 10, live Firestore Aug 28.*

---

## LayerNorm added to OP_REGISTRY and PATCHABLE_OPS (extends §8.3, Task 10)

`torch.nn.LayerNorm` IS an `nn.Module` and CAN be discovered via `named_modules()`.
It was previously incorrectly listed in the adapter_mapping reject list alongside
actual non-module ops (rope, softmax, silu). LayerNorm is now in `PATCHABLE_OPS`,
`_OP_MODULES`, and `OP_REGISTRY` with a proper adapter.

LayerNorm adapter extracts: `self.weight`, `self.bias`, `self.eps`,
`self.normalized_shape`. Note `bias` (LayerNorm has bias; RMSNorm does not) —
the generic adapter path via `build_forward_from_mapping()` handles this correctly
because the Coder declares the full mapping.

Constructor for meta-device validation:
`torch.nn.LayerNorm(normalized_shape=hidden_size, device=torch.device("meta"))`

*Source: Task 10.*

---

## What LayerNorm actually needed, as implemented (corrects the section above)

`OP_REGISTRY` **already had** `layernorm` (`_build_layernorm`, binding
`entry(x, weight, bias, eps)`, eps 1e-5, reduction in fp32). Three things were genuinely
missing, and one line of the section above is wrong:

1. `PATCHABLE_OPS["layernorm"] = {"class_name": "LayerNorm", "priority": 0}` — same
   priority as `rmsnorm`, because whichever norm the architecture uses is the P0 target.
   Matched as a class-name substring like every other entry, so a `FusedLayerNorm`
   subclass matches too. On Qwen2 it correctly matches nothing (`Qwen2RMSNorm` does not
   contain "LayerNorm") and `/swap` refuses with "no module whose class name contains
   'LayerNorm'".
2. `_OP_MODULES["layernorm"]` in `verifier/adapter_mapping.py`, which is what takes it
   off the reject list. `_probe_instance` used to branch on `class_name == "Qwen2RMSNorm"`
   and otherwise assume a `Qwen2Config`; each entry now carries its own `build` callable
   instead, so adding an op no longer means editing that function.
3. `_layernorm_adapter` as the hard-coded fallback.

**The fallback adapter passes `(weight, bias, eps)` and NOT `normalized_shape`**, contrary
to the section above. It has to match the signature the verifier benched the kernel
against — `entry(x, weight, bias, eps)` — and adding a fourth argument the verified
wrapper never accepted would break every seed kernel. `normalized_shape` validates fine
as a declared binding and reaches a kernel that asks for it through the generic adapter;
`test_the_fallback_adapter_does_not_pass_normalized_shape` pins the arity.

`eps` vs `variance_epsilon` is the concrete reason one shared norm adapter was never
going to work, and `validate_adapter_mapping("layernorm", {"eps": "variance_epsilon"})`
now rejects it at validation time rather than as an `AttributeError` inside a hot forward.

*Source: Task 10.*

---

## New config constants for the audit (adds to §0, Task 10)

Beyond the specified `AUDIT_REPORT_WIDTH = 80`: `DEFAULT_AUDIT_MODEL = "qwen2.5-1.5b"`
(asserted equal to `SERVED_MODEL`, so `make audit` audits the model the server runs), and
the probe shapes the analytic estimates are taken at — `AUDIT_PROBE_BATCH = 1`,
`AUDIT_PROBE_SEQ = 512`, `AUDIT_PROBE_SPATIAL = 56` (ResNet-50's stage-1 feature map).
They place a module on the roofline; they are not the shapes the model is served at, and
in CPU mode nothing is allocated from them. CLAUDE.md: every magic number lives in
config.py.

The table renders at exactly `AUDIT_REPORT_WIDTH` — the first column absorbs the
remainder after the borders and the five fixed columns, so changing the constant cannot
break the box-drawing alignment. Asserted in `tests/test_audit.py`.

*Source: Task 10.*

---

## MODEL_REGISTRY — supported models for audit (adds to §0, Task 10)

Three architecturally diverse, ungated models are registered in `config.py`:
- `qwen2.5-1.5b`: Qwen/Qwen2.5-1.5B-Instruct (RMSNorm, SiLU, decoder)
- `gpt2`: openai-community/gpt2 (LayerNorm, GELU, decoder)
- `resnet50`: microsoft/resnet-50 (BatchNorm, ReLU, vision)

All MIT or Apache-2.0 licensed, no gating. Total FP16 footprint ~3.4 GB.
Load one at a time for audit profiling — do not load all simultaneously
during optimization (waste of VRAM).

*Source: Task 10.*

---

## run_demo.py CLI subcommands (extends §14, Task 10)

`run_demo.py` now uses argparse with subcommands: `audit`, `optimize`, `full`.
- `audit`: profiles a model, prints the audit report. Accepts `--model`, `--device`, `--all`.
- `optimize`: existing behavior (optimize RMSNorm on Qwen2.5). Backward compatible.
- `full`: audit → optimize → re-audit → cross-model transfer demo.
- Default (no subcommand): `optimize` — so `make demo` is backward compatible.

*Source: Task 10.*

---

## `audit --all` defaults to CPU; a single `audit` follows the GPU (and says so)

`make audit-all` passes no `--device`, so it resolved through `default_audit_device()` to
`cuda` on any box with a GPU — three models of REAL weights (~3.4 GB) plus `do_bench` at
warmup=150 over every unique module type. **It did not finish inside 600 s on the dev
box.** `run_audit_all` now defaults to `"cpu"` regardless of the GPU, and takes 4 s.

That is not only a speed concession. The comparison table has an AI column and no
bandwidth column, AI is analytic either way, and the bandwidth CUDA would buy is a
fraction of the L4's 300 GB/s — which `format_audit_report` itself warns is not
comparable when measured anywhere else. A cross-architecture sweep is precisely the case
where measuring adds nothing. `--device cuda` still forces it.

A SINGLE-model `audit` keeps the spec's stated default ("cuda if available"), because
that is the one case where a measurement is the point. It now prints
`[audit] loading <id> weights for do_bench (CUDA mode; --device cpu skips this)` before
`from_pretrained`: on a cold HuggingFace cache that line stands in front of a multi-GB
download, and a silent multi-minute pause reads as a hang. Measured on the dev box with
a cold cache: still downloading at 233 s.

*Source: Task 10, dev box Aug 28.*

---

## What a CUDA audit actually costs: 8 s of bench, and however long the weights take

Measured on the dev box, Aug 28. The bench is free; every slow audit was a download.

| | weights | download @1.75 MB/s | audit incl. load+bench |
|---|---|---|---|
| GPT-2 | 523 MB | (cached) | **8 s** |
| ResNet-50 | 102 MB | ~1 min | **62 s** |
| Qwen2.5-1.5B | 3,087 MB | ~29 min | not run locally |

`do_bench` over every unique module type of a model costs single-digit seconds, so
`--device cuda` is cheap wherever the weights are already local — which is the L4 VM for
Qwen2.5-1.5B, since the inference server serves that exact checkpoint. Pulling 3.1 GB to
a laptop to measure a column whose denominator is the L4's bandwidth is the one case not
worth it.

The `ResNet*` composite rows show `n/a` for BW: their forwards do not accept a bare
synthetic probe, so `_measure_bandwidth_pct` returns 0.0 and the table blanks the cell.
Working as intended — an unbenchable module is a blank cell, never an exception, and never
a fabricated 0%.

**A conv can cross the ridge point between fp32 and fp16.** ResNet-50's representative
`Conv2d` reads 53 FLOP/byte (memory-bound, MEDIUM) in CPU/fp32 mode and 107 FLOP/byte
(compute-bound, LOW) in CUDA/fp16 — halving the bytes doubles the intensity, and 107 is
just past the L4's ridge of 101. Not a bug in either mode: the byte count is real and the
op genuinely sits on the ridge at this probe shape. It does mean a conv's REGIME is not
dtype-independent the way a norm's is (a norm is two orders of magnitude clear), so quote
the mode alongside the number.

*Source: Task 10, dev box Aug 28.*

---

## What Task 10 is validated on

Dev box (RTX A500, Python 3.12.12, torch 2.12.1+cu130), Aug 28. **449 unit tests pass**
(up from 317) plus the 18 integration tests unchanged; `make lint` clean.

Verified by running it, not by inspection:

- `make audit-all` (4 s, CPU) and `make audit AUDIT_ARGS="--device cpu"` (4 s) on all
  three registered models. Top target
  is the model's own normalization every time — `Qwen2RMSNorm` (57), `LayerNorm` (25),
  `BatchNorm2d` (53) — and the comparison table puts the three side by side. The Qwen2
  numbers match the Aug 26 VM smoke test exactly (369 modules, 57 RMSNorms, weight [1536]).
- `audit --device cuda` with real weights on **GPT-2** (LayerNorm 0.88 F/B, 39% BW, top
  target) and **ResNet-50** (BatchNorm2d 0.88, 19%; ReLU 0.25, 25%; top target
  BatchNorm2d) — `do_bench` measured, off-L4 warning printed in both.
- `audit --output json` round-trips through `json.load`.
- Cross-model transfer against the **live** Firestore library: GPT-2's LayerNorm
  fingerprint retrieved all three Qwen2.5 RMSNorm skills, nearest at distance 0.0128.
- All three dashboard tabs via `AppTest`, including a real audit run from the Audit tab
  and the live skill library and transfer-readiness tables in the Library tab.
- `make demo`'s argv (`--no-server`, `--op`, bare) still parses to `optimize`.

**Not validated:** `run_full` end to end (its `optimize` leg is a full Vertex + GPU run;
that leg is unchanged from Task 9 and covered by the Task 9 note above); and **the CUDA
audit of Qwen2.5-1.5B — the served model — has never been run**, because its 3.1 GB is a
~29-minute download to the dev box and is already cached on the VM. Two of three
architectures are measured on CUDA; the third is the cheap one to do on the L4.

No audit has been run on the L4 itself. The CPU-mode numbers are hardware-independent by
construction, but **the CUDA-mode `BW %` column has only ever been measured on an RTX
A500, where its L4 denominator is wrong** — every such run says so in its own header.
`make audit` on the L4 is the one thing that makes that column mean what it says.

*Source: Task 10, dev box Aug 28.*
