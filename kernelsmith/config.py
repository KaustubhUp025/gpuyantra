"""Central constants. Every magic number lives here. Import from here, never hardcode."""

import os

# --- GCP ---
GCP_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]  # "gpuyantra"
GCP_LOCATION = "global"  # All Gemini 3.x models require the global endpoint on Vertex AI
FIRESTORE_DATABASE = "(default)"
FIRESTORE_COLLECTION_SKILLS = "skills"
FIRESTORE_COLLECTION_RUNS = "runs"
FIRESTORE_SUBCOLLECTION_TRACES = "traces"

# --- Models ---
PRIMARY_MODEL = "gemini-3.7-flash"  # All agents: Supervisor, Coder, Judge, Profiler
LLM_TEMPERATURE = 0.0  # Greedy decoding on EVERY agent; spec 11 mandates it on the Judge
LLM_RETRY_ATTEMPTS = 5  # A 429 mid-demo must back off, not end the run
LLM_RETRY_INITIAL_DELAY_S = 2.0
LLM_RETRY_MAX_DELAY_S = 60.0
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # MRL truncation; assert after every call
GEMMA_MODEL = "gemma-4-26b-a4b-it-maas"  # Bonus kernel-explainer; MaaS ids carry the suffix
# ^ The spec writes this without "-maas". That id 404s in every region: the Vertex
#   publisher model is literally named `gemma-4-26b-a4b-it-maas`. Same model, MaaS
#   serving name. See .claude/rules/implementation-deviations.md.

# --- Verifier ---
CORRECTNESS_SEEDS = 5
CORRECTNESS_SHAPES = [
    (1, 128),  # small: batch=1, seq=128
    (8, 512),  # medium
    (16, 2048),  # large
]
ATOL = 1e-2
RTOL = 1e-2
DO_BENCH_WARMUP = 150  # Default 25 underestimates by ~30% (Triton issue #2306)
DO_BENCH_REP = 200
SPEEDUP_THRESHOLD = 0.05  # 5% gate for +2/+3 milestones
SANDBOX_TIMEOUT_S = 60  # SIGKILL after this
GPU_HEALTH_PROBE_TIMEOUT_S = 10

# --- Agent Loop ---
MAX_LOOP_ITERATIONS = 6  # Circuit breaker; never remove
RETRIEVAL_TOP_K = 3

# --- Inference Server ---
SERVED_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
INFERENCE_HOST = "127.0.0.1"
INFERENCE_PORT = 8000
HOTSWAP_TIMEOUT_S = 120  # /swap parity-checks against the live model before it answers
SWAP_PARITY_SHAPE = (1, 128)  # (batch, seq) probe shape for the pre-swap parity gate
GPU_PROBE_TIMEOUT_S = 5  # nvidia-smi behind /health; a wedged GPU must not hang the API

# --- Hardware (NVIDIA L4 constants) ---
L4_MEM_BW_GBPS = 300.1  # GB/s, GDDR6
L4_FP16_TFLOPS = 30.3  # Non-tensor FP16
L4_TENSOR_FP16_TFLOPS = 121.0  # Tensor Core FP16
L4_VRAM_GB = 24
L4_SM_COUNT = 58  # AD104: 58 streaming multiprocessors
L4_SRAM_KB_PER_SM = 48  # Usable shared memory per SM; a Triton tile must fit inside it

# --- Profiler heuristics (roofline fingerprint, spec 7) ---
PROFILER_MIN_TILE = 64  # Below one warp-row of fp16 there is nothing left to coalesce
PROFILER_MAX_TILE = 1024  # 1024 fp16 lanes = 2 KB/row; wider tiles spill registers on the L4
PROFILER_BLOCKS_PER_SM = 4  # Target resident blocks per SM for the occupancy heuristic
PROFILER_FALLBACK_AI = 0.5  # Fallback arithmetic intensity when profiling fails: memory-bound
PROFILER_FALLBACK_OCCUPANCY = 0.5  # Fallback occupancy, deliberately uninformative

# --- Reproducibility ---
GLOBAL_SEED = 42
DETERMINISTIC_CUDA = True  # torch.use_deterministic_algorithms(True)
CUBLAS_WORKSPACE = ":4096:8"  # CUBLAS_WORKSPACE_CONFIG env var

# --- Multi-model audit (spec 7, Task 10) ------------------------------------
# The architecture was always general — the fingerprint is a bottleneck, not an op name,
# and the deployment contract is generated per module. MODEL_REGISTRY is what makes that
# generality checkable: three architecturally different models, so "it works on RMSNorm"
# cannot be mistaken for "it works on Qwen2".
#
# All three are ungated (MIT / Apache-2.0, no license click-through), because an audit
# that stops to ask for a HuggingFace token is an audit nobody runs. Total FP16 footprint
# is ~3.4 GB, but they are loaded ONE AT A TIME: profiling needs the module tree, not
# three models resident at once.
MODEL_REGISTRY: dict[str, dict[str, object]] = {
    "qwen2.5-1.5b": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "family": "decoder",
        "norm_type": "RMSNorm",
        "activation": "SiLU",
        "hidden_size": 1536,
        "description": "Modern decoder (GQA, RoPE, SwiGLU, RMSNorm)",
    },
    "gpt2": {
        "hf_id": "openai-community/gpt2",
        "family": "decoder",
        "norm_type": "LayerNorm",
        "activation": "GELU",
        "hidden_size": 768,
        "description": "Classic decoder (MHA, learned pos, GELU, LayerNorm)",
    },
    "resnet50": {
        "hf_id": "microsoft/resnet-50",
        "family": "vision",
        "norm_type": "BatchNorm",
        "activation": "ReLU",
        "hidden_size": 2048,
        "description": "Vision classifier (BatchNorm, ReLU, residual blocks)",
    },
}

DEFAULT_AUDIT_MODEL = "qwen2.5-1.5b"  # The served model; `make audit` with no --model
AUDIT_REPORT_WIDTH = 80  # Total width of the ASCII audit table, borders included

# Probe shapes for the audit's analytic FLOP/byte estimates. They place a module on the
# roofline; they are not the shapes the model is served at, and nothing is allocated
# from them in CPU mode.
AUDIT_PROBE_BATCH = 1
AUDIT_PROBE_SEQ = 512  # tokens per sequence for sequence-shaped modules
AUDIT_PROBE_SPATIAL = 56  # HxW for conv / BatchNorm2d; ResNet-50's stage-1 feature map
