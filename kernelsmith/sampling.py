"""One sampling policy for every LLM call in the tree (spec 11, reproducibility).

Spec 11 lists `temperature=0` on the Judge alongside the torch/numpy/random seeds,
under a single heading: "a speedup that only appears under one lucky seed is the Sakana
failure mode". The seeds pin everything below the model. Without this file, the model
itself was still sampling at the Vertex default (~1.0), which is the largest source of
run-to-run variance in `make demo` by a wide margin — reseeding torch cannot make a
sampled kernel come back the same.

Applied to every agent, not only the Judge:

- **Judge** — spec-mandated. A scorer that wobbles turns the reward into a coin flip.
- **Coder** — the kernel *is* the headline number. Sampling it means two `make demo`
  runs report different speedups from identical inputs, which is precisely the
  credibility failure this project exists to avoid.
- **Profiler / Supervisor** — procedural: they read tool output and route. There is
  nothing here that creative sampling improves.

`seed` is belt-and-braces. At temperature 0 decoding is greedy and the seed should not
matter; it costs nothing and covers backends that treat 0 as "very small" rather than
"argmax".

This buys determinism *given the same inputs*. It does not make a served LLM a pure
function — batching and model-version rollovers on Vertex can still move an output. The
demo's numbers come from the verifier, not the model, so a drifted kernel shows up as a
different measured speedup rather than an unnoticed one.

The same object also carries the retry policy, for a reason found the hard way: two
`make demo` runs back to back exhausted the project's per-minute quota, and the second
died on an unhandled `429 RESOURCE_EXHAUSTED` five seconds in — mid-run, with a forty
line traceback and exit code 2. ADK surfaces the error and stops; nothing between the
agent and Vertex was retrying. On demo day that is a recording lost to a transient.
`google-genai` will do the backoff itself if asked, so it is asked here.
"""

from __future__ import annotations

from google.genai import types

from kernelsmith import config


def deterministic_config() -> types.GenerateContentConfig:
    """The generation config every agent in this tree is built with.

    A fresh object per call: ADK stores it on the agent, and a shared mutable singleton
    would let one agent's later edit reach every other agent.
    """
    return types.GenerateContentConfig(
        temperature=config.LLM_TEMPERATURE,
        seed=config.GLOBAL_SEED,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=config.LLM_RETRY_ATTEMPTS,
                initial_delay=config.LLM_RETRY_INITIAL_DELAY_S,
                max_delay=config.LLM_RETRY_MAX_DELAY_S,
                exp_base=2.0,
                jitter=1.0,
                # 429 is the one that matters here; the 5xx family is included because a
                # transient upstream failure should not end a run either. Nothing else is
                # retried: a 400 or a 403 is a bug or a permission gap, and retrying it
                # just spends the budget more slowly.
                http_status_codes=[429, 500, 502, 503, 504],
            ),
        ),
    )
