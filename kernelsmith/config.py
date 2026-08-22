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
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # MRL truncation; assert after every call
GEMMA_MODEL = "gemma-4-26b-a4b-it"  # Bonus kernel-explainer (MaaS, no self-deploy)

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

# --- Hardware (NVIDIA L4 constants) ---
L4_MEM_BW_GBPS = 300.1  # GB/s, GDDR6
L4_FP16_TFLOPS = 30.3  # Non-tensor FP16
L4_TENSOR_FP16_TFLOPS = 121.0  # Tensor Core FP16
L4_VRAM_GB = 24

# --- Reproducibility ---
GLOBAL_SEED = 42
DETERMINISTIC_CUDA = True  # torch.use_deterministic_algorithms(True)
CUBLAS_WORKSPACE = ":4096:8"  # CUBLAS_WORKSPACE_CONFIG env var
