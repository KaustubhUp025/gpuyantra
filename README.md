# KernelSmith

An in-process [Google ADK](https://google.github.io/adk-docs/) agent tree that autonomously
profiles PyTorch model operations, generates optimized Triton GPU kernels, verifies them
against correctness and anti-reward-hacking criteria, and hot-swaps them into a live
Qwen2.5-1.5B inference server on a single NVIDIA L4.

## Status

Foundation layer only: config, schemas, embeddings, Firestore store, seed script, index
bootstrap. Agents, verifier, and inference server are not implemented yet.

## Setup

```bash
cp .env.example .env         # GOOGLE_CLOUD_PROJECT, Vertex AI flags
gcloud auth application-default login
uv sync
```

## Commands

```bash
make test-unit        # unit tests (no GPU, no network)
make test-int         # integration tests (GPU + Vertex AI)
make lint             # ruff check + format --check
make create-index     # ONCE: composite vector index on `skills` (minutes to build)
make seed-skill       # insert the hand-written RMSNorm seed skill
make serve-inference  # FastAPI on :8000
make serve-ui         # Streamlit on :8501
make demo             # full reproducible demo
```

## Architecture (locked)

| Piece | Choice |
|---|---|
| Framework | `google-adk==2.7.1`, in-process only |
| Model (all agents) | `gemini-3.7-flash`, global Vertex endpoint |
| Embeddings | `gemini-embedding-001` @768, manually L2-normalized |
| Memory | Firestore Native, `Vector(768)`, COSINE, composite pre-filter |
| Served model | `Qwen/Qwen2.5-1.5B-Instruct` on one NVIDIA L4 |
| Package manager | `uv` |

See `CLAUDE.md` for the critical rules and the `kernelsmith-spec` skill for the full
specification.
