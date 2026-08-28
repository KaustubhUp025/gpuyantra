# KernelSmith

An in-process Google ADK agent tree that autonomously generates, verifies, and
hot-swaps Triton GPU kernels into a live Qwen2.5-1.5B inference server.

## Build & Test Commands
uv sync # Install deps
make test-unit # Unit tests (no GPU)
make test-int # Integration tests (GPU + Vertex AI)
make test # All tests
make lint # ruff check + format
make serve-inference # FastAPI on :8000
make serve-ui # Streamlit on :8501
make demo # Full reproducible demo
make audit # Audit default model (Qwen2.5-1.5B)
make audit-all # Audit all registered models
## Architecture (locked — do not change)

- **Framework:** google-adk==2.7.1, in-process only
- **Model (ALL agents):** gemini-3.7-flash on global Vertex endpoint
- **Memory:** Firestore Native, Vector(768), COSINE, composite pre-filter
- **Served model:** Qwen2.5-1.5B-Instruct on single NVIDIA L4
- **Package manager:** uv
- **Audit targets:** MODEL_REGISTRY in config.py (Qwen2.5-1.5B, GPT-2, ResNet-50)
- **Patchable ops:** RMSNorm (P0), SwiGLU/MLP (P1), LayerNorm (P0 on GPT-2)
- **Audit mode:** Works on CPU (analytic FLOP/byte) or CUDA (measured via do_bench)

## Critical Rules

1. Never use model IDs other than gemini-3.7-flash, gemini-embedding-001,
   gemma-4-26b-a4b-it-maas (the MaaS serving name; the unsuffixed id 404s in every
   region — see `.claude/rules/implementation-deviations.md`)
2. Never run generated Triton code in the main process — subprocess sandbox only
3. Never weaken the verifier (fewer seeds, looser tolerance, dropped AST checks)
4. Never remove max_iterations from any LoopAgent
5. Never commit secrets — ADC only
6. Never use output_schema on the Judge agent (ADK bug #3969)
7. EscalationChecker must be a BaseAgent, never a tool/callback (ADK bugs #501/#2692/#2808)
8. Always assert len(embedding)==768 and L2-normalize after every embedding call
9. Always set do_bench warmup>=150 (default 25 underestimates by ~30%)
10. Never torch.compile before monkey-patching
11. LayerNorm IS an nn.Module — do NOT reject it from adapter_mapping validation
12. audit_model() must work on CPU (no GPU) using analytic FLOP/byte estimates
13. MODEL_REGISTRY models must be ungated (no license acceptance required to download)

## Code Style

- Python 3.11+, type hints on all public functions
- Pydantic for all structured boundaries
- pytest for all tests, TDD for verifier and memory modules
- One responsibility per file (see repo layout in spec)

## Spec Reference

The full implementation specification is in the kernelsmith-spec skill.
Read the relevant section before implementing any module.
