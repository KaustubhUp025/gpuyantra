# KernelSmith VM Session Results — August 30, 2026

Use this document to update all placeholder/stale values in:
- `web/kernelsmith_explorer.jsx` (React explorer for Cloud Run)
- `kernelsmith/ui/demo_dashboard.py` (demo dashboard)
- `data/traces/sample_run.jsonl` (sample fixture — update reward/speedup values)
- `README.md`
- Any other file referencing speedup, test count, or audit data

---

## Authoritative Numbers (all measured on NVIDIA L4, August 30 2026)

### Speedup
- **speedup_vs_eager: 7.24×** (most recent measurement; earlier runs gave 7.21×)
- **speedup_vs_compile: 1.39×**
- **reward: +3** (max possible)
- **correctness: PASS** (15/15 checks — 5 seeds × 3 shapes, atol=rtol=1e-2)
- **iterations: 1** (converged on first loop iteration)
- **bandit arm: rmsnorm_l4_single_pass_register_fused**

### End-to-End Inference (live hot-swap proof)
- **Before swap:** 529ms for 3 tokens (~176ms/token) — baseline eager PyTorch
- **After swap:** 87.5ms for 2 tokens (~44ms/token) — with Triton kernel
- **End-to-end latency improvement: ~4× faster**
- **57 modules patched** across all Qwen2RMSNorm layers
- **Output coherent** after swap (verified manually)

### Latency by Shape (do_bench measured)
- 1×128: 0.0123ms
- 8×512: 0.129ms
- 16×2048: 1.246ms (headline shape)

### Test Count
- **572 unit tests passed** (18 deselected, 12 warnings)
- **make lint: clean** (69 files formatted)

---

## CUDA Audit Data (do_bench measured on L4)

### Qwen2.5-1.5B-Instruct
```
Model: Qwen/Qwen2.5-1.5B-Instruct
Modules scanned: 367 profiled, 8 unique types, hidden_size=1536
Mode: cuda — intensity analytic, bandwidth MEASURED (do_bench)

Module Type          | Count | Regime  | AI (F/B) | BW %  | Priority
Qwen2RMSNorm        |  57   | memory  |   0.83   |  23%  | ★★★ HIGH
SiLUActivation       |  28   | memory  |   0.75   |  68%  | ★★★ HIGH
Linear              | 196   | compute |   307    |  51%  | ☆ LOW
Qwen2Attention      |  28   | compute |   269    |  n/a  | ☆ LOW
Qwen2DecoderLayer   |  28   | compute |   323    |  n/a  | ☆ LOW
Qwen2MLP            |  28   | compute |   358    |  47%  | ☆ LOW
Embedding           |   1   |   —     |   n/a    |  n/a  | ☆ LOW
Qwen2RotaryEmbedding|   1   |   —     |   n/a    |  n/a  | ☆ LOW

Top target: Qwen2RMSNorm
Recommendation: 57 memory-bound instances at 0.83 FLOP/byte, 121× below ridge
```

### GPT-2 (openai-community/gpt2)
```
Model: openai-community/gpt2
Modules scanned: 160 profiled, 8 unique types, hidden_size=768
Mode: cuda — intensity analytic, bandwidth MEASURED (do_bench)

Module Type          | Count | Regime  | AI (F/B) | BW %  | Priority
LayerNorm           |  25   | memory  |   0.88   |  93%  | ★★★ HIGH
NewGELUActivation   |  12   | memory  |   0.75   |  16%  | ★★★ HIGH
Conv1D              |  48   | compute |   271    |  42%  | ☆ LOW
Dropout             |  37   |   —     |   n/a    |  n/a  | ☆ LOW
GPT2Attention       |  12   | compute |   256    |  36%  | ☆ LOW
GPT2Block           |  12   | compute |   210    |  53%  | ☆ LOW
GPT2MLP             |  12   | compute |   256    |  42%  | ☆ LOW
Embedding           |   2   |   —     |   n/a    |  n/a  | ☆ LOW

Top target: LayerNorm
Recommendation: 25 memory-bound instances at 0.88 FLOP/byte, 115× below ridge
```

### ResNet-50 (microsoft/resnet-50)
```
Model: microsoft/resnet-50
Modules scanned: 260 profiled, 12 unique types, hidden_size=2048
Mode: cuda — intensity analytic, bandwidth MEASURED (do_bench)

Module Type          | Count | Regime  | AI (F/B) | BW %  | Priority
BatchNorm2d         |  53   | memory  |   0.88   |  33%  | ★★★ HIGH
ReLU                |  49   | memory  |   0.25   |  57%  | ★★★ HIGH
ResNetConvLayer     |  49   | memory  |   2.90   |  n/a  | ★★ MEDIUM
ResNetBottleNeckLayer|  16  | memory  |  13.94   |  n/a  | ★★ MEDIUM
ResNetShortCut      |   4   | memory  |  12.81   |  n/a  | ★★ MEDIUM
ResNetStage         |   4   | memory  |  15.51   |  n/a  | ★★ MEDIUM
ResNetEmbeddings    |   1   | memory  |   2.90   |  n/a  | ★★ MEDIUM
ResNetEncoder       |   1   | memory  |  24.06   |  n/a  | ★★ MEDIUM
Conv2d              |  53   | compute |   107    |   2%  | ☆ LOW
Identity            |  28   |   —     |   n/a    |  n/a  | ☆ LOW
AdaptiveAvgPool2d   |   1   |   —     |   n/a    |  n/a  | ☆ LOW
MaxPool2d           |   1   |   —     |   n/a    |  n/a  | ☆ LOW

Top target: BatchNorm2d
Recommendation: 53 memory-bound instances at 0.88 FLOP/byte, 115× below ridge
```

### Cross-Architecture Comparison
```
Model            Norm       Top target        Count   AI      Priority
qwen2.5-1.5b    RMSNorm    Qwen2RMSNorm       57    0.83     HIGH
gpt2            LayerNorm  LayerNorm           25    0.88     HIGH
resnet50        BatchNorm  BatchNorm2d         53    0.88     HIGH

Same roofline analysis, three architectures: the norm the model uses
is the target in every one.
```

---

## Cross-Model Skill Transfer (proven on live Firestore)

```
Source: qwen2.5-1.5b (RMSNorm)
Target: gpt2 (LayerNorm)
Query fingerprint: op=norm mem_bound=True ai=0.4 tile=1024 hw=L4

Retrieved skills (3):
  rmsnorm_fp16_l4_v1                    distance=0.0128
  rmsnorm_l4_single_pass_fused          distance=0.0154
  rmsnorm_l4_single_pass_register_fused distance=0.0154

Verdict: "Cross-model transfer: skill from qwen2.5-1.5b retrieved for gpt2
— fingerprints match on op_family (norm) and hardware (L4)"
```

---

## Winning Kernel Source (verbatim from L4 run)

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd(
    X, W, Y,
    stride_x_row,
    stride_y_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_ptrs = X + row * stride_x_row + cols
    y_ptrs = Y + row * stride_y_row + cols
    w_ptrs = W + cols

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rsqrt = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)
    y = (x * rsqrt * w).to(Y.dtype.element_ty)
    tl.store(y_ptrs, y, mask=mask)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, N = x2d.shape
    y = torch.empty_like(x2d)

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 8 if BLOCK_SIZE >= 2048 else (4 if BLOCK_SIZE >= 512 else 2)

    _rmsnorm_fwd[(M,)](
        x2d, weight.contiguous(), y,
        x2d.stride(0),
        y.stride(0),
        N,
        float(eps),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y.reshape(orig_shape)
```

---

## Gemma Explanation (verbatim from L4 run — use for the explorer)

### Overview: What this kernel does

This code implements **RMSNorm (Root Mean Square Layer Normalization)**, a popular normalization technique used in modern LLMs (like Llama).

Mathematically, for a vector $x$ of length $N$, RMSNorm performs:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}} \cdot w$$
where $w$ is a learnable scaling weight.

**The Logic Flow:**
1.  **Parallelization Strategy:** The kernel treats each row of the input matrix as an independent task. It launches $M$ instances of the function (where $M$ is the number of rows), one for each `row = tl.program_id(0)`.
2.  **Loading:** For a specific row, it loads all $N$ elements into high-speed on-chip memory (SRAM).
3.  **Reduction:** It calculates the mean of the squares ($\sum x^2 / N$).
4.  **Normalization:** It computes the reciprocal square root ($1/\sqrt{\text{var} + \epsilon}$).
5.  **Scaling & Store:** It multiplies the original values by the reciprocal square root and the weight vector $w$, then writes the result back to global memory.

### Why it is faster than Eager PyTorch

In "Eager" PyTorch, an operation like `y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w` is executed as a sequence of discrete kernels.

#### 1. Kernel Fusion (The "Memory Wall" Problem)
In Eager PyTorch, the execution looks like this:
1.  **Kernel A:** Compute $x^2 \rightarrow$ Write result to VRAM.
2.  **Kernel B:** Compute `mean` $\rightarrow$ Read $x^2$ from VRAM, Write result to VRAM.
3.  **Kernel C:** Compute `rsqrt` $\rightarrow$ Read `mean` from VRAM, Write result to VRAM.
4.  **Kernel D:** Compute multiplication $\rightarrow$ Read $x$, `rsqrt`, and $w$ from VRAM, Write result to VRAM.

Modern GPUs are significantly faster at math than they are at moving data (the "Memory Wall"). Eager PyTorch is **memory-bandwidth bound**: the GPU spends most of its time waiting for data to travel from the slow Global Memory (VRAM) to the fast compute cores.

**The Triton advantage:** This kernel is **fused**. It loads the data into the chip **once**, performs all mathematical operations (square, sum, rsqrt, multiply) while the data is sitting in fast local registers/SRAM, and writes the final result **once**. We have replaced multiple heavy memory round-trips with a single pass.

#### 2. Reduced Overhead
Every time PyTorch launches a kernel, there is a small amount of CPU-side overhead (driver calls, scheduling). By collapsing four operations into one, we reduce this overhead by 75%.

### Hardware Feature Exploited: SRAM and Register Files

This kernel exploits **On-Chip Data Reuse**.

When `tl.load` is called, the data is moved from the "hard drive" (VRAM) into the "cache/registers" (SRAM/Registers). In the Eager implementation, the data is "evicted" back to the hard drive after every single sub-operation.

In this Triton kernel, the variable `x` stays in the high-speed on-chip memory for the entire duration of the function. The `tl.sum(x * x)` operation happens entirely within the compute unit's local memory space. We are maximizing the **Arithmetic Intensity** — the ratio of floating-point operations performed per byte of data moved from main memory.

### Summary for the C++ Engineer

| Feature | Eager PyTorch | Triton Kernel |
| :--- | :--- | :--- |
| **Complexity** | Multiple loops over data | Single loop over data |
| **Memory Access** | O(Ops × N) reads/writes | O(N) reads/writes |
| **Bottleneck** | Memory Bandwidth (I/O bound) | Compute (ALU bound) |
| **Analogy** | Reading a book, closing it, then reopening it to highlight a sentence, then closing it again to write a note. | Opening the book once, reading, highlighting, and writing notes all in one sitting. |

---

## Supervisor Summary (verbatim)

- **Operation**: `rmsnorm` (hidden size 1536)
- **Bottleneck Fingerprint**: Memory-bound (is_memory_bound=True, arithmetic intensity 1.25, memory throughput 40.66 GB/s, tile size hint 1024)
- **Refinement Verdict & Reward**: Reward +3 (correctness passed across 15 checks, beat both eager baseline and torch.compile)
- **Measured Speedups**:
  - vs. Eager PyTorch: **7.24×** (latency reduced from 9.021 ms to 1.246 ms on headline shape 16×2048)
  - vs. torch.compile: **1.39×** (latency reduced from 1.727 ms to 1.246 ms on headline shape 16×2048)
- **Skill Library Status**: Upserted successfully as rmsnorm_l4_single_pass_register_fused
- **Live Server Deployment**: Hot-swap succeeded across 57 Qwen2RMSNorm modules (parity check passed over 5 seeds at atol=0.01, rtol=0.01, rollback: false)

---

## Agent Trace (from terminal — for updating sample_run.jsonl with real data)

Turn 1/2: profile → retrieve → refine
  [Supervisor] → transfer_to_agent()     # delegates to Profiler
  [Profiler]   → profile_op_by_name()    # profiles RMSNorm
  [Profiler]   → transfer_to_agent()     # returns to Supervisor
  [Supervisor] → retrieve_skills_for_agent()  # queries Firestore
  [Supervisor] → transfer_to_agent()     # delegates to RefinementLoop
  [Judge]      → verify_kernel()         # reward=3, eager=7.24×, compile=1.39×

Turn 2/2: upsert → hot-swap → explain
  [Supervisor] → upsert_skill()          # saves to Firestore
  [Supervisor] → hotswap_kernel()        # patches 57 modules
  [Supervisor] → explain_kernel()        # Gemma generates explanation

---

## Values to Find-and-Replace Everywhere

| Old value | New value | Reason |
|-----------|-----------|--------|
| 6.90× | 7.24× | Latest L4 measurement |
| 6.92× | 7.24× | Task 11 used stale Task 8 number |
| 7.04× | 7.24× | Task 9 number, superseded |
| 7.21× | 7.24× | Phase 4 run, superseded by Phase 5 |
| 329 tests | 572 tests | Updated count after Tasks 10-12 |
| 486 tests | 572 tests | Task 11 used stale count |
| 468 tests | 572 tests | Task 10 count, superseded |
| 43% BW | 23% BW | Qwen2.5 RMSNorm measured (was analytic estimate) |
| bw_pct: null | bw_pct: 23 | Qwen2.5 RMSNorm measured on L4 |
| "placeholder" | real Gemma text | See Gemma explanation section above |
| measured: false | measured: true | CUDA audit mode confirmed |

---

## Files That Need Updating

1. **web/kernelsmith_explorer.jsx** — speedup, test count, BW%, Gemma explanation, audit data
2. **data/traces/sample_run.jsonl** — update reward and speedup values to match real data
3. **README.md** — speedup figure
4. **kernelsmith/ui/demo_dashboard.py** — if any hardcoded values exist
5. **docs/architecture.md** — if speedup is mentioned
