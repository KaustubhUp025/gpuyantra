# KernelSmith

An in-process [Google ADK](https://google.github.io/adk-docs/) agent tree that autonomously
profiles PyTorch model operations, generates optimized Triton GPU kernels, verifies them
against correctness and anti-reward-hacking criteria, and hot-swaps them into a live
Qwen2.5-1.5B inference server on a single NVIDIA L4.

The claim is narrow and checkable: **the agent writes its own deployment contract, and a
deterministic verifier decides whether it is true.** Every published kernel-generation
system uses a human-authored bridge between the kernel and the model it is deployed into.
Here the Coder declares that contract (`adapter_mapping`), the verifier validates it
against the real module's attributes, and only then does it reach the served model.

See [`docs/architecture.md`](docs/architecture.md) for the full design, and the
`kernelsmith-spec` skill for the implementation specification.

---

## Prerequisites

### 1. Hardware

| | Requirement |
|---|---|
| GPU | One NVIDIA L4 (24 GB). The served model needs ~3.1 GB at FP16; the verifier sandbox runs alongside it. |
| Image | `pytorch-latest-gpu` DLVM (validated on `pytorch-2-9-cu129-ubuntu-2204-nvidia-580`, driver 580.173.02) |
| Disk | 100 GB pd-ssd |

A smaller card runs the agent loop and the verifier but cannot host the inference
server at the same time; use `make demo DEMO_ARGS=--no-server` there.

### 2. Google Cloud project

A project with billing enabled (this repo assumes `gpuyantra`) and these APIs on:

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  firestore.googleapis.com \
  compute.googleapis.com \
  --project=gpuyantra
```

Plus:

- **A Firestore database in Native mode**, named `(default)`. Vector search requires
  Native mode; Datastore mode will not work.
- **Vertex AI quota** for `gemini-3.7-flash` and `gemini-embedding-001` on the
  **global** endpoint. All Gemini 3.x models are global-endpoint only — a regional
  location will 404.
- **GPU quota**: `NVIDIA_L4_GPUS` ≥ 1 in your zone (`us-central1-b` here).
- Optional, for `make export-firestore`: a GCS bucket, default `gs://gpuyantra-backups`.

### 3. Authentication — ADC only

No API keys, ever. Nothing in this repo reads one, and `.env` holds non-secret config only.

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project gpuyantra
```

On a Compute Engine VM there is nothing to do: credentials come from the metadata
server and `google.auth.default()` finds them.

### 4. Tooling

[`uv`](https://docs.astral.sh/uv/) is the package manager — the version pins below are
enforced through `uv.lock`, so `pip install` is not a supported path.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Environment variables

Copy the template and edit if your project differs:

```bash
cp .env.example .env
```

| Variable | Value | Why |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | `gpuyantra` | Read strictly at import (`config.py`); a missing value fails loudly rather than talking to the wrong project. |
| `GOOGLE_CLOUD_LOCATION` | `global` | Gemini 3.x is global-endpoint only. |
| `GOOGLE_GENAI_USE_VERTEXAI` | `true` | Route google-genai through Vertex, not the public API. ADK 2.7.1 warns on every run that this name is deprecated in favour of `GOOGLE_GENAI_USE_ENTERPRISE`. Both work; the new name is checked first and wins. Set only one, or set both to the same value — conflicting values are the one case that misbehaves. |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Required for deterministic GEMMs. cuBLAS reads it when its handle is created and ignores it afterwards, so it is also set by `seed_everything()` and exported by the Makefile. |

`.env` is `.gitignore`d and is loaded by `kernelsmith/__init__.py` before any config is
read. **A real shell variable always wins over the file** (`override=False`), so
`GOOGLE_CLOUD_PROJECT=other uv run ...` cannot be silently redirected by a stale `.env`.

---

## Quick start

```bash
git clone <repo> && cd gpuyantra
cp .env.example .env
gcloud auth application-default login

make setup     # uv sync --frozen + chmod 444 verifier + create index + seed skill
make demo      # profile -> generate -> verify -> save -> hot-swap
```

`make setup` is one command and is safe to re-run. It:

1. installs the exact pinned dependency closure (`uv sync --frozen` — refuses to re-resolve);
2. re-applies `chmod 444` to the four verifier scripts (git does not track that bit);
3. creates the Firestore composite vector index — **several minutes to build**, and
   already-exists is tolerated. Never run this on demo day;
4. seeds the hand-written RMSNorm skill so the first run has something to retrieve.

Then `make demo` prints the measured reward, both speedups, the bandit arm it pulled,
the winning kernel, and whether the hot-swap went live. Every number comes from the
verifier's verdict, never from the model's prose summary.

---

## Commands

```bash
make setup            # one-command setup (above)
make demo             # full reproducible demo; DEMO_ARGS=--no-server to skip the server
make test-unit        # unit tests (no GPU, no network) — 317 tests
make test-int         # integration tests (GPU + live Vertex AI + Firestore)
make test             # both
make lint             # ruff check + format --check
make format           # ruff format
make serve-inference  # FastAPI on :8000
make serve-ui         # Streamlit dashboard on :8501
make create-index     # ONCE: composite vector index on `skills` (minutes to build)
make seed-skill       # insert the hand-written RMSNorm seed skill
make harden           # chmod 444 the verifier scripts
make check-harden     # verify those modes; fails if any is writable
make unharden         # chmod 644 — take the write bit back deliberately to edit one
make export-firestore # snapshot skills + bandit state to GCS before recording a demo
```

---

## Reproducibility

The field has a credibility problem — Sakana's 3.13× became 1.49×, KernelBench-Verified's
1.43× became 0.88× — and every collapse was a seeding or baseline artifact. So:

| Pinned | How |
|---|---|
| RNGs | `seed_everything()`: `CUBLAS_WORKSPACE_CONFIG` first, then Python/NumPy/torch CPU + CUDA, then `use_deterministic_algorithms(True)` last |
| The model | **`temperature=0` and a fixed decode seed on every agent** and on the Gemma explainer. Reseeding torch cannot make a *sampled* kernel come back the same |
| Baselines | TF32 on (`set_float32_matmul_precision("high")`) for every baseline measurement; determinism turned off *only* around the timed comparison, because it costs eager ~23% while leaving Triton untouched |
| Versions | Every direct dependency `==`-pinned; the full 127-package closure pinned by the committed `uv.lock`, installed with `uv sync --frozen` |
| Memory | `make export-firestore` before recording |

Measured back to back on 2026-08-28 (RTX A500, `--no-server`), same seed:

| | Run 1 | Run 2 |
|---|---|---|
| reward | +3 | +3 |
| iterations | 1 | 1 |
| speedup vs eager | 7.04x | 7.04x |
| speedup vs `torch.compile` | 1.39x | 1.39x |
| latency `1x128` / `8x512` | identical to every digit | identical to every digit |
| latency `16x2048` | 3.3178 ms | 3.3167 ms (0.03%, wall-clock noise) |
| bandit arm | `rmsnorm_fp16_l4_v1` | `rmsnorm_l4_single_pass_fused` |
| winning kernel | — | one comment line differs; code identical |

Every number reproduces. The two things that moved are both explained, and neither is
the model wandering.

**The bandit arm differs by design.** Run 1 upserted the kernel it had just verified, and
UCB1 gives an arm with zero pulls an unbounded exploration bonus — so run 2 was *supposed*
to try the new one. That is the memory working, not drift. Two runs are only comparable
against the same skill library. To replay a run exactly, snapshot first and restore
before the replay:

```bash
make export-firestore    # before the run you want to be able to reproduce
# ... later, to replay:
gcloud firestore import gs://gpuyantra-backups/snapshot-YYYYmmdd-HHMMSS \
  --project=gpuyantra --database='(default)'
```

The one-line difference in the kernel follows from the same cause: a different retrieved
skill is a different prompt, and greedy decoding on a different prompt is allowed to
differ. It landed in a comment, and both kernels measured the same.

Determinism given the same inputs is what this buys. A served LLM is still not a pure
function — batching and model-version rollovers can move an output. That is survivable
here because the headline number comes from the verifier: a drifted kernel shows up as a
*different measured speedup*, not as an unnoticed one.

---

## Cost

Budget **~$0.62 for one `make demo` run on a fresh L4** — the conservative case: an
on-demand VM and the refinement loop going the full six iterations.

| Item | Conservative | Typical |
|---|---|---|
| `g2-standard-4` (one L4), ~30 min | $0.35 on-demand | $0.11 Spot |
| `gemini-3.7-flash` — 4 agents × iterations | $0.25 (6 iterations) | $0.05 (1 iteration) |
| `gemini-embedding-001` — one fingerprint, one upsert | <$0.01 | <$0.01 |
| `gemma-4-26b-a4b-it-maas` explanation (bonus) | ~$0.01 | ~$0.01 |
| Firestore, Cloud Trace | $0 (free tier) | $0 (free tier) |
| **Total** | **~$0.62** | **~$0.18** |

The typical column is not optimistic hand-waving: the run recorded on 2026-08-28
converged to reward +3 on **iteration 1**, so the loop never spent its budget. Plan for
the conservative number and expect the typical one.

The whole project budgets ~$44 against confirmed credits (spec §18); the dominant line is
VM hours, not tokens. **Stop the VM every evening** — one idle night costs $5–10 for
nothing:

```bash
gcloud compute instances stop kernelsmith-vm --zone=us-central1-b --project=gpuyantra
```

Two runs back to back can exhaust the project's per-minute Vertex quota. Every agent now
retries `429` with exponential backoff (5 attempts), but if you are rehearsing, leave a
few minutes between runs rather than relying on it.

---

## Architecture (locked)

| Piece | Choice |
|---|---|
| Framework | `google-adk==2.7.1`, in-process only |
| Model (all agents) | `gemini-3.7-flash`, global Vertex endpoint, `temperature=0` |
| Embeddings | `gemini-embedding-001` @768, manually L2-normalized |
| Memory | Firestore Native, `Vector(768)`, COSINE, composite equality pre-filter |
| Served model | `Qwen/Qwen2.5-1.5B-Instruct` on one NVIDIA L4, eager mode |
| Package manager | `uv` |

## Security

- Generated Triton code never runs in the agent, dashboard, or server process — only in
  a sandbox subprocess with a scrubbed environment and a SIGKILL timeout.
- The four verifier scripts are `chmod 444`: the sandbox runs as the same uid, so the
  write bit is the interlock that makes tampering deliberate rather than accidental.
  Git does not track that mode — `make setup` re-applies it and `make check-harden` verifies.
- The static checker rejects `socket` / `urllib` / `requests` / `http` imports, and six
  other reward-hack patterns, before anything executes.
- ADC only. No API keys in code or `.env`; `.env` is `.gitignore`d and non-secret.

See `CLAUDE.md` for the critical rules and `.claude/rules/implementation-deviations.md`
for every documented departure from the spec.
