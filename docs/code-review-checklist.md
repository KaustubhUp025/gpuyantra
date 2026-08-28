# Spec §20 code review checklist — audit of 2026-08-28

Run against the whole tree at commit `79cc2c5` + the Task 9 hardening changes.

**One checklist item failed** (#12) and is fixed. Two further defects were found while
executing the reproducibility task around it — neither is a §20 line item, and both would
have reached demo day: `.env` was never loaded, and the Gemma bonus model id 404s. All
three are written up below. Re-run the greps in the "evidence" column before any commit
that touches the relevant module.

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | No model IDs other than `gemini-3.7-flash`, `gemini-embedding-001`, `gemma-4-26b-a4b-it` | PASS, with the Gemma id corrected to its `-maas` serving name | Only three literals exist, all in `config.py`. Nothing hardcodes a model elsewhere. The Gemma id now carries the `-maas` suffix Vertex actually publishes it under — same model; see below. |
| 2 | No `gemini-3-flash` / `gemini-3.1-pro` / `gemini-3.5-flash` | PASS | No matches anywhere. |
| 3 | No secrets committed | PASS | `.env` is `.gitignore`d and untracked; no key/secret/credential literal in any tracked file; auth is ADC only. |
| 4 | No direct `session.state` mutation — `state_delta` or `output_key` only | PASS | The only `BaseAgent`, `EscalationChecker`, writes through `EventActions(state_delta=…)` (`escalation.py:79-94`). Every other write is `tool_context.state` / `callback_context.state`, which ADK accumulates into the event delta. `streamlit_app.py` hits are Streamlit's own `st.session_state`. |
| 5 | Verifier tolerance `atol=rtol=1e-2` | PASS | `config.ATOL`/`config.RTOL` = 1e-2, used in `correctness.py:99`. No looser literal anywhere. |
| 6 | `do_bench` warmup ≥ 150 | PASS | `config.DO_BENCH_WARMUP = 150`; `bench_kernel` *raises* below it (`timing.py:29-33`) rather than trusting the caller. |
| 7 | `max_iterations` set on every LoopAgent | PASS | One LoopAgent exists; `refinement_loop.py:39` sets `max_iterations=MAX_LOOP_ITERATIONS` (6). |
| 8 | Sandbox subprocess for all generated-code execution | PASS (one documented exception) | `verify_kernel` never executes candidate source in-process. The exception is `/swap`'s `_load_entrypoint`, which imports a kernel into the server *after* it has cleared all three gates — deployment, not verification. See `implementation-deviations.md`. |
| 9 | `assert len(embedding) == 768` | PASS | `embeddings.py:43`, after a truncation fallback. |
| 10 | Embedding L2-normalized after truncation | PASS | `embeddings.py:45-49`, with a zero-norm assert. |
| 11 | TF32 on for every baseline | PASS | `set_float32_matmul_precision("high")` in `reproducibility.py`, `timing.py:76`, `verifier_tool.py:85`, `models.py:45` — every path that measures. |
| 12 | Seeds set before any random generation | **FAIL → fixed** | The torch/NumPy/random half was correct: `seed_everything()` is the first statement in `run_demo`, the sandbox reseeds inside the subprocess, `correctness.py` reseeds per shape. The model half was missing entirely — no agent set a temperature. See below. |
| 13 | No `torch.compile` before monkey-patching | PASS | The served model is never compiled (`models.py:54`). The single `torch.compile` call is the discarded baseline in `timing.py:81`. |
| 14 | No `try/except` in generated kernel candidates | PASS | Static checker rule 6 (`static_checker.py:299-312`) rejects `ast.Try`/`ast.TryStar`. |
| 15 | Static checker covers all 7 reward-hack patterns | PASS | Rules 1–7 all present (`static_checker.py:26-32`), plus rule 0 for a candidate that does not parse. |
| 16 | Tests exist for new functionality | PASS | 317 unit tests; all three fixes below shipped with regression tests. |

## The three defects, and what was done

### Item 12 — every agent sampled at the Vertex default temperature

Spec §11 lists `temperature=0` on the Judge in the same row as the torch/NumPy/random
seeds. No agent set a `generate_content_config` at all, so all four ran at the model
default (~1.0) — and so did the Gemma explainer.

This mattered more than the item's placement suggests. Reseeding torch cannot make a
*sampled* kernel come back the same, so the seeds pinned everything below the model
while the model itself was free to write a different kernel each run. Two `make demo`
runs would have reported different rewards and different speedups from identical
inputs, which is exactly the credibility failure the reproducibility contract exists to
prevent — and it would have moved without anything in the repo changing.

Fixed in `kernelsmith/sampling.py`: one `deterministic_config()`
(`temperature=0.0`, `seed=GLOBAL_SEED`) applied to the Supervisor, Profiler, Coder and
Judge, and to the Gemma call in `explainer_tool.py`. Locked down by
`test_every_agent_decodes_greedily`, which walks the built tree, so an agent added later
without a config fails the suite.

### Not a §20 item — `.env` was never loaded, so `make demo` could not start from a clean clone

`config.py` reads `GOOGLE_CLOUD_PROJECT` with `os.environ[...]` at import time,
deliberately strictly. Nothing in the repo loaded the `.env` file the README tells you
to create: ADK reads `.env` only through the `adk` CLI, which neither `make demo` nor
`make serve-ui` goes through. The documented setup path — `cp .env.example .env && make
demo` — therefore died on a `KeyError` one line into startup. It was invisible on the
VM, where the variables are exported from the shell profile, and in tests, where
`conftest.py` sets them.

Fixed in `kernelsmith/__init__.py`, the only hook guaranteed to run before any
`kernelsmith.*` import. `override=False` is load-bearing: a real shell variable always
beats the file, so a stale `.env` can never silently redirect a run to another GCP
project.

### Not a §20 item — the Gemma bonus agent was dead

The first `make demo` that ever completed printed, where the explanation should be:

    error: ClientError: 404 NOT_FOUND. Publisher model
    `projects/gpuyantra/locations/global/publishers/google/models/gemma-4-26b-a4b-it`
    was not found or your project does not have access to it.

Not regional and not a permissions gap — the same id 404s in `global`, `us-central1`,
`us-east4` and `europe-west4`. Listing the publisher models the project can actually see
returns `gemma-4-26b-a4b-it-maas`: the Model-as-a-Service serving name. Same model,
different id. Verified working on `global` after the change (`us-central1` returns
`400 FAILED_PRECONDITION` for it, so the explainer stays on the global endpoint).

It hid because it fails softly by design: an explanation is a bonus and must never fail a
run whose kernel is already verified, so the 404 string was returned as the explanation
and everything downstream stayed green. `test_the_gemma_model_id_keeps_the_maas_suffix`
now pins it.

## Not a violation, but worth recording

- **`LoopAgent` is deprecated in ADK 2.7.1** ("in favor of Workflow"). The warning also
  says Workflow "cannot yet be used as an LlmAgent sub-agent", which is exactly how the
  RefinementLoop is wired. No action; the pin is 2.7.1 and the replacement does not fit.
- **`GOOGLE_GENAI_USE_VERTEXAI` is deprecated** in favour of `GOOGLE_GENAI_USE_ENTERPRISE`.
  Already analyzed in `implementation-deviations.md`; both names work.
