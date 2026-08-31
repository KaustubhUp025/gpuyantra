/**
 * gpuyantra — Interactive Results Explorer
 * ----------------------------------------
 * Single-page product walkthrough for the All Things Agentic Hackathon submission.
 *
 * NAMING. `gpuyantra` is the project; `KernelSmith` is the agent tree inside it that
 * does the work. Judge-facing chrome — the page title, the nav mark, the hero — says
 * gpuyantra; anything describing who profiles, writes, verifies or deploys says
 * KernelSmith, because that is the thing being described. The Python package, the
 * module names and the Firestore collections stay `kernelsmith` and are not renamed:
 * the code is not a judge-facing surface.
 *
 * Requires: react >= 18, tailwindcss >= 3. No other runtime dependencies —
 * the Triton syntax highlighting is a ~40-line tokenizer in this file, on purpose:
 * one more npm package on the deploy path is one more thing that can break on
 * demo day.
 *
 * DATA PROVENANCE — read before editing any number.
 * Every value below is either MEASURED (with the machine it was measured on
 * recorded next to it) or explicitly marked TODO. Red line #3 of this project is
 * that no number is printed as measured unless it was. The UI honours that: a
 * null renders as "n/a", never as 0 and never as a plausible-looking guess.
 *
 *   REAL — captured 2026-08-30 on the L4 VM (`vm_session_results.md`):
 *     AUDIT_DATA          `run_demo audit --model <id> --device cuda`, all three models.
 *                         AI analytic, `bw_pct` MEASURED with do_bench against the L4's
 *                         own 300.1 GB/s — the one machine where that denominator is right.
 *     KERNEL_SOURCE       the kernel the Coder wrote in that run, verbatim; the one the
 *                         Judge scored +3 and the server hot-swapped across 57 modules
 *     RESULT / TRACE      `verify_kernel` verdict of that run (reward +3, 7.24x / 1.39x)
 *     EXPLANATION         gemma-4-26b-a4b-it-maas output from that run, condensed to four
 *                         paragraphs; the sentences are the model's own
 *     ADAPTER_BINDINGS    live Coder output, Task 8, unchanged in this run
 *     TRANSFER            live Firestore query, 2026-08-30
 *     HEADLINE.tests      641 unit + 18 integration, `make test-unit` / `make test-int`
 *
 *   Two arithmetic-intensity numbers appear in this file and they disagree on purpose:
 *   the audit's estimator counts traffic PER TENSOR TOUCHED by an unfused eager
 *   implementation (RMSNorm 0.83 in fp16), the profiler's counts the MINIMUM a fused
 *   kernel must move (1.25). The profiler's is the one Firestore is keyed on. Both put
 *   the op two orders of magnitude below the ridge point. Do not "reconcile" them.
 *
 *   TODO — still open (search this file for "TODO(vm)"):
 *     3. LINKS                       — repo / video / blog URLs
 */

import React, { useEffect, useMemo, useRef, useState } from "react";

/* ══════════════════════════════════════════════════════════════════════════
   DATA
   ══════════════════════════════════════════════════════════════════════════ */

const HEADLINE = {
  speedup: "7.24×",
  speedupNote: "vs eager PyTorch · NVIDIA L4 · 1.39× vs torch.compile",
  tests: "659",
  testsNote: "641 unit (hermetic) + 18 integration on the L4",
  models: "3",
  modelsNote: "decoder, classic decoder, vision — one analysis",
};

const HARDWARE = {
  name: "NVIDIA L4",
  vram_gb: 24,
  mem_bw_gbps: 300.1,
  fp16_tflops: 30.3,
  ridge_point: 101, // FLOP/byte — where memory-bound becomes compute-bound
};

const BUILT_WITH = [
  "Agent Development Kit 2.7.1",
  "Gemini 3.7 Flash",
  "Gemma 4 26B",
  "Firestore Vector Search",
  "Vertex AI",
  "Compute Engine · L4",
  "Cloud Run",
];

/**
 * Real output of `run_demo audit --model <id> --device cuda`, run ON the L4 VM,
 * 2026-08-30. This replaces the earlier CPU/meta-device sweep.
 *
 * mode "cuda" means: the module tree and the arithmetic intensity are analytic (exactly
 * as in CPU mode — AI is computed from shapes, never measured), and `bw_pct` is MEASURED
 * with `triton.testing.do_bench` on one representative instance of each unique module
 * type. The denominator is the L4's own 300.1 GB/s, so these percentages mean what they
 * say for the first time — measured on any other card they would not.
 *
 * The AI values therefore differ from the CPU sweep this file used to carry (RMSNorm
 * 0.42 -> 0.83, LayerNorm 0.44 -> 0.88). Not a correction of a bug: CPU mode estimates
 * at fp32 and CUDA mode at fp16, and halving the bytes doubles the intensity. Both put
 * every norm two orders of magnitude below the ridge point, which is the only question
 * the triage table asks.
 */
const AUDIT_DATA = {
  "qwen2.5-1.5b": {
    model_name: "Qwen/Qwen2.5-1.5B-Instruct",
    label: "Qwen2.5-1.5B",
    family: "Modern decoder",
    detail: "GQA · RoPE · SwiGLU · RMSNorm",
    norm_type: "RMSNorm",
    activation: "SiLU",
    params: "1.54B",
    hidden_size: 1536,
    total_modules: 367,
    unique_types: 8,
    mode: "cuda",
    measured: true,
    served: true,
    entries: [
      { type: "Qwen2RMSNorm", count: 57, regime: "memory", ai: 0.83, bw_pct: 23, priority: "HIGH" },
      { type: "SiLUActivation", count: 28, regime: "memory", ai: 0.75, bw_pct: 68, priority: "HIGH" },
      { type: "Linear", count: 196, regime: "compute", ai: 307, bw_pct: 51, priority: "LOW" },
      { type: "Qwen2Attention", count: 28, regime: "compute", ai: 269, bw_pct: null, priority: "LOW" },
      { type: "Qwen2DecoderLayer", count: 28, regime: "compute", ai: 323, bw_pct: null, priority: "LOW" },
      { type: "Qwen2MLP", count: 28, regime: "compute", ai: 358, bw_pct: 47, priority: "LOW" },
      { type: "Embedding", count: 1, regime: null, ai: null, bw_pct: null, priority: "LOW" },
      { type: "Qwen2RotaryEmbedding", count: 1, regime: null, ai: null, bw_pct: null, priority: "LOW" },
    ],
    top_target: "Qwen2RMSNorm",
    recommendation:
      "Start with Qwen2RMSNorm: 57 memory-bound instances at 0.83 FLOP/byte, 121× below the L4 ridge point of 101, and measured at 23% of the card's bandwidth — a fused single-pass Triton kernel recovers the traffic the unfused version wastes.",
  },
  gpt2: {
    model_name: "openai-community/gpt2",
    label: "GPT-2",
    family: "Classic decoder",
    detail: "MHA · learned positions · GELU · LayerNorm",
    norm_type: "LayerNorm",
    activation: "GELU",
    params: "124M",
    hidden_size: 768,
    total_modules: 160,
    unique_types: 8,
    mode: "cuda",
    measured: true,
    served: false,
    entries: [
      { type: "LayerNorm", count: 25, regime: "memory", ai: 0.88, bw_pct: 93, priority: "HIGH" },
      { type: "NewGELUActivation", count: 12, regime: "memory", ai: 0.75, bw_pct: 16, priority: "HIGH" },
      { type: "Conv1D", count: 48, regime: "compute", ai: 271, bw_pct: 42, priority: "LOW" },
      { type: "GPT2Attention", count: 12, regime: "compute", ai: 256, bw_pct: 36, priority: "LOW" },
      { type: "GPT2MLP", count: 12, regime: "compute", ai: 256, bw_pct: 42, priority: "LOW" },
      { type: "GPT2Block", count: 12, regime: "compute", ai: 210, bw_pct: 53, priority: "LOW" },
      { type: "Dropout", count: 37, regime: null, ai: null, bw_pct: null, priority: "LOW" },
      { type: "Embedding", count: 2, regime: null, ai: null, bw_pct: null, priority: "LOW" },
    ],
    top_target: "LayerNorm",
    recommendation:
      "Start with LayerNorm: 25 memory-bound instances at 0.88 FLOP/byte, 115× below the L4 ridge point of 101 — a fused single-pass Triton kernel recovers the traffic the unfused version wastes.",
  },
  resnet50: {
    model_name: "microsoft/resnet-50",
    label: "ResNet-50",
    family: "Vision classifier",
    detail: "BatchNorm · ReLU · residual bottlenecks",
    norm_type: "BatchNorm",
    activation: "ReLU",
    params: "25.6M",
    hidden_size: 2048,
    total_modules: 260,
    unique_types: 12,
    mode: "cuda",
    measured: true,
    served: false,
    entries: [
      { type: "BatchNorm2d", count: 53, regime: "memory", ai: 0.88, bw_pct: 33, priority: "HIGH" },
      { type: "ReLU", count: 49, regime: "memory", ai: 0.25, bw_pct: 57, priority: "HIGH" },
      { type: "ResNetEncoder", count: 1, regime: "memory", ai: 24.06, bw_pct: null, priority: "MEDIUM" },
      { type: "ResNetStage", count: 4, regime: "memory", ai: 15.51, bw_pct: null, priority: "MEDIUM" },
      { type: "ResNetBottleNeckLayer", count: 16, regime: "memory", ai: 13.94, bw_pct: null, priority: "MEDIUM" },
      { type: "ResNetShortCut", count: 4, regime: "memory", ai: 12.81, bw_pct: null, priority: "MEDIUM" },
      { type: "ResNetConvLayer", count: 49, regime: "memory", ai: 2.9, bw_pct: null, priority: "MEDIUM" },
      { type: "ResNetEmbeddings", count: 1, regime: "memory", ai: 2.9, bw_pct: null, priority: "MEDIUM" },
      { type: "Conv2d", count: 53, regime: "compute", ai: 107, bw_pct: 2, priority: "LOW" },
      { type: "Identity", count: 28, regime: null, ai: null, bw_pct: null, priority: "LOW" },
      { type: "AdaptiveAvgPool2d", count: 1, regime: null, ai: null, bw_pct: null, priority: "LOW" },
      { type: "MaxPool2d", count: 1, regime: null, ai: null, bw_pct: null, priority: "LOW" },
    ],
    top_target: "BatchNorm2d",
    recommendation:
      "Start with BatchNorm2d: 53 memory-bound instances at 0.88 FLOP/byte, 115× below the L4 ridge point of 101 — a fused single-pass Triton kernel recovers the traffic the unfused version wastes.",
  },
};

const MODEL_ORDER = ["qwen2.5-1.5b", "gpt2", "resnet50"];

/** The agent trace, in the order the tree actually executes it. */
const TRACE = [
  {
    agent: "Supervisor",
    kind: "LlmAgent",
    action: "Receives the goal, calls the profiler tool",
    summary: "Resumable state machine — executes the first undone step on each invocation.",
    body:
      "The Supervisor does not do the work; it sequences it. Its whole state lives in " +
      "session.state, and each step is idempotent, so an interrupted run resumes rather " +
      "than restarts. Target selected from the audit: Qwen2RMSNorm, 57 instances.",
    output: [
      ["op_name", "rmsnorm"],
      ["target", "Qwen2RMSNorm × 57"],
      ["next_step", "profile"],
    ],
  },
  {
    agent: "Profiler",
    kind: "LlmAgent + tool",
    action: "Measures the op and emits a bottleneck fingerprint",
    summary: "Roofline analysis, not a name lookup. The fingerprint IS the retrieval key.",
    body:
      "Arithmetic intensity 1.25 FLOP/byte against an L4 ridge point of 101 — memory-bound " +
      "by two orders of magnitude, moving 40.7 GB/s of the card's 300. The fingerprint " +
      "deliberately carries no operator name and no model name: it describes WHY the op is " +
      "slow, which is the only property that transfers between architectures. (The audit " +
      "table above reads 0.83 for the same op: a different estimator, counting what an " +
      "unfused eager implementation moves rather than the minimum a fused kernel must.)",
    fingerprint: "op=norm mem_bound=True ai=1.2 tile=1024 hw=L4",
    output: [
      ["op_family", "norm"],
      ["arithmetic_intensity", "1.25 FLOP/byte"],
      ["memory_throughput", "40.7 GB/s"],
      ["is_memory_bound", "true"],
      ["tile_size_hint", "1024"],
      ["hardware", "L4"],
    ],
  },
  {
    agent: "Retrieval",
    kind: "FunctionTool → Firestore",
    action: "Vector search over the skill library, then a UCB1 bandit picks the arm",
    summary: "768-dim COSINE search with a composite pre-filter on (op_family, hardware).",
    body:
      "The pre-filter is the part that matters: the query joins on op_family and hardware, " +
      "never on the model or the operator name. On a cold library this returns nothing and " +
      "the Coder writes from first principles; once seeded, UCB1 balances the arm with the " +
      "best mean reward against the one nobody has tried.",
    output: [
      ["pre_filter", "op_family=norm AND hardware=L4"],
      ["top_k", "3"],
      ["retrieved", "3 · nearest at distance 0.0"],
      ["selected_arm", "rmsnorm_l4_single_pass_fused"],
      ["arm_stats", "1 pull · mean reward 3.0"],
    ],
  },
  {
    agent: "Coder",
    kind: "LlmAgent · temperature 0",
    action: "Writes the Triton kernel and declares the deployment contract",
    summary: "Emits kernel source AND the adapter_mapping that binds it to the live module.",
    body:
      "Greedy decoding, seeded — the kernel is the headline number, so it must be " +
      "reproducible. The novel part is the second output: the Coder declares how its " +
      "kernel's parameters map onto the target module's attributes. Every published system " +
      "has a human write that bridge.",
    output: [
      ["entrypoint", "rmsnorm_triton"],
      ["strategy", "single-pass row reduction, x held in registers, fp32 accumulate"],
      ["BLOCK_SIZE", "next_pow2(N) · num_warps 8 at 2048"],
      ["adapter_mapping", "weight→weight, eps→variance_epsilon"],
    ],
  },
  {
    agent: "Judge",
    kind: "LlmAgent + verifier tool",
    action: "Static check → contract validation → sandboxed correctness & timing",
    summary: "Four gates, in order. The candidate never runs in the main process.",
    body:
      "The Judge cannot see or edit the deployment contract — the verifier tool reads it " +
      "from state, so the model that wrote the kernel cannot also grade its bridge. Reward " +
      "is recomputed in-process from the sandbox's correctness and timing results; the " +
      "subprocess's own reported reward is discarded, because the candidate controls that " +
      "stdout.",
    output: [
      ["static_ast_check", "pass — 0 of 7 hack patterns"],
      ["adapter_validation", "pass — attrs exist on Qwen2RMSNorm"],
      ["correctness", "15/15 · 5 seeds × 3 shapes · atol=rtol=1e-2"],
      ["speedup_vs_eager", "7.24×"],
      ["speedup_vs_torch_compile", "1.39×"],
      ["headline_shape", "16×2048 · 1.246 ms vs 9.021 ms eager"],
      ["reward", "+3"],
    ],
  },
  {
    agent: "EscalationChecker",
    kind: "BaseAgent — deterministic",
    action: "Decides whether to iterate again, and credits the bandit",
    summary: "Not an LLM. Not a callback. A BaseAgent, because ADK loops only escalate from one.",
    body:
      "Reward +3 clears the bar on the first iteration, so the loop escalates rather than " +
      "spending five more. One run is one bandit pull, credited here and nowhere else — " +
      "crediting per iteration would record six experiments for one, and crediting from the " +
      "Supervisor's upsert step would silently drop every run that scored below +1, biasing " +
      "every arm's mean upward.",
    output: [
      ["iteration", "1 of 6"],
      ["decision", "escalate — target met"],
      ["bandit_credit", "rmsnorm_l4_single_pass_fused ← +3"],
    ],
  },
  {
    agent: "Supervisor",
    kind: "turn 2",
    action: "Upserts the skill, then hot-swaps into the live server",
    summary: "types.MethodType patch across all 57 modules, with a parity gate and rollback.",
    body:
      "The model is never torch.compile'd — a compiled graph ignores a patched forward and " +
      "would report a swap that silently did nothing. Before the patch is kept, the server " +
      "re-checks parity against the unpatched forward; a failure rolls back bitwise-exactly. " +
      "TokenMeter clears its rolling window at the swap boundary so the throughput chart " +
      "shows the discontinuity instead of averaging across it.",
    output: [
      ["skill_upserted", "skills/rmsnorm_l4_single_pass_register_fused"],
      ["modules_patched", "57 / 57"],
      ["adapter_path", "generic (declared contract)"],
      ["parity_gate", "pass · 5 seeds on the live weights"],
      ["end_to_end", "529 ms / 3 tokens → 87.5 ms / 2 tokens"],
    ],
  },
];

const RESULT = {
  reward: "+3",
  vs_eager: "7.24×",
  vs_compile: "1.39×",
  iterations: "1 of 6",
  correctness: "15/15",
};

/**
 * Verbatim from the 2026-08-30 L4 run: the kernel the Coder wrote, the Judge scored +3,
 * and the server hot-swapped across all 57 Qwen2RMSNorm modules.
 *
 * Single pass, not two: the row is loaded once and kept in registers for both the
 * variance reduction and the scaling, which is the whole point on an op that is
 * bandwidth-bound. BLOCK_SIZE is the next power of two above N (1536 -> 2048), so the
 * row fits one tile and there is no loop to write.
 */
const KERNEL_SOURCE = `import torch
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

    # Single-pass load keeps x in registers, avoiding redundant DRAM round-trips
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
    return y.reshape(orig_shape)`;

/**
 * The bonus agent's real output from the 2026-08-30 L4 run, condensed from its markdown
 * to four paragraphs. The sentences are Gemma's; nothing here is a paraphrase written by
 * a human, and nothing is a placeholder.
 */
const EXPLANATION = {
  model: "gemma-4-26b-a4b-it-maas",
  placeholder: false,
  paragraphs: [
    "This code implements RMSNorm (Root Mean Square Layer Normalization), a popular normalization technique used in modern LLMs. The kernel treats each row of the input matrix as an independent task: it launches M instances of the function, one per row. For a specific row it loads all N elements into high-speed on-chip memory, calculates the mean of the squares, computes the reciprocal square root, then multiplies the original values by that and by the weight vector before writing the result back to global memory.",
    "In eager PyTorch, `y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w` is executed as a sequence of discrete kernels: compute x² and write it to VRAM; read that back to compute the mean and write it; read that to compute rsqrt and write it; read x, rsqrt and w to multiply and write y. Modern GPUs are significantly faster at math than they are at moving data — the memory wall — so eager PyTorch here is memory-bandwidth bound, spending most of its time waiting for data to travel from slow global memory to the fast compute cores.",
    "This kernel is fused. It loads the data into the chip once, performs all the mathematical operations — square, sum, rsqrt, multiply — while the data is sitting in fast local registers and SRAM, and writes the final result once. Multiple heavy memory round-trips are replaced with a single pass. There is a second, smaller win: every kernel launch carries CPU-side overhead for driver calls and scheduling, and collapsing four operations into one removes 75% of it.",
    "The hardware feature being exploited is on-chip data reuse. When `tl.load` is called, the data moves from VRAM into registers and SRAM. In the eager implementation it is evicted back after every single sub-operation; in this kernel `x` stays in high-speed on-chip memory for the entire duration of the function, and `tl.sum(x * x)` happens entirely within the compute unit's local memory space. That is what maximizing arithmetic intensity — floating-point operations performed per byte moved from main memory — means in practice.",
  ],
  takeaway:
    "Eager PyTorch is like reading a book, closing it, reopening it to highlight a sentence, then closing it again to write a note. This kernel opens the book once.",
};

const ADAPTER_BINDINGS = [
  {
    kernel_param: "x",
    module_attr: "— (implicit forward input)",
    implicit: true,
    note: "Never declared. The forward argument is positional and mapping it is an error, not a no-op.",
  },
  {
    kernel_param: "weight",
    module_attr: "weight",
    note: "nn.Parameter, shape [1536] — verified to exist on a meta-device Qwen2RMSNorm.",
  },
  {
    kernel_param: "eps",
    module_attr: "variance_epsilon",
    note:
      "1e-06. Note the rename: torch.nn.LayerNorm calls this eps; Qwen2RMSNorm calls it variance_epsilon. One shared norm adapter was never going to work.",
  },
];

const TRANSFER = {
  source: {
    model: "Qwen2.5-1.5B",
    op: "Qwen2RMSNorm",
    count: 57,
    fingerprint: "op=norm mem_bound=True ai=1.2 tile=1024 hw=L4",
    fields: [
      ["op_family", "norm"],
      ["hardware", "L4"],
      ["is_memory_bound", "true"],
      ["arithmetic_intensity", "1.25"],
      ["tile_size_hint", "1024"],
    ],
  },
  target: {
    model: "GPT-2",
    op: "LayerNorm",
    count: 25,
    fingerprint: "op=norm mem_bound=True ai=0.4 tile=1024 hw=L4",
    fields: [
      ["op_family", "norm"],
      ["hardware", "L4"],
      ["is_memory_bound", "true"],
      ["arithmetic_intensity", "0.4"],
      ["tile_size_hint", "1024"],
    ],
  },
  distance: 0.0128,
  similarity: "0.987",
  retrieved: 3,
  measuredOn: "live Firestore skill library, 2026-08-30",
};

const PRIOR_WORK = [
  { name: "HF kernels", note: "exact attribute-name matching, no inference" },
  { name: "FlashInfer-Bench", note: "entry-point arg names must match declared I/O keys" },
  { name: "Kernel Contracts", note: "eight-part formal contract, hand-written" },
  { name: "FastKernels", note: "0.94× vs production — interface incompatibility" },
];

const HACK_PATTERNS = [
  "torch.nn / F.* fallback in the output path",
  "identity output (returns an input unchanged)",
  "decoy kernel (@triton.jit defined, never launched)",
  "torch.empty returned unwritten (stale VRAM)",
  "hardcoded constants in the output path",
  "try/except fallback to the reference",
  "extra streams, threading, or network imports",
];

const VERIFICATION = [
  {
    title: "Correctness",
    headline: "15 / 15",
    sub: "5 seeds × 3 shapes",
    lines: [
      ["shapes", "(1,128) · (8,512) · (16,2048)"],
      ["tolerance", "atol = rtol = 1e-2"],
      ["determinism", "torch.use_deterministic_algorithms(True)"],
      ["isolation", "subprocess sandbox, SIGKILL at 60 s"],
    ],
    note: "A failure anywhere in the grid rolls the patch back bitwise-exactly. The candidate never executes in the agent's process.",
  },
  {
    title: "Baselines",
    headline: "2 of them",
    sub: "eager and torch.compile",
    lines: [
      ["warmup", "150 iters (the default 25 underestimates ~30%)"],
      ["rep", "200 · median, not mean"],
      ["vs eager", "7.24×"],
      ["vs torch.compile", "1.39×"],
    ],
    note: "Determinism is switched OFF for the timed region only — it penalizes eager by ~23% and does not touch Triton. With the flag left on, the same comparison reported 8.52×. The honest number for this run is 7.24×.",
  },
  {
    title: "Anti-hack",
    headline: "7 rules",
    sub: "AST, before execution",
    list: HACK_PATTERNS,
    note: "Purely syntactic, so it costs nothing and cannot be fooled by runtime behaviour. Rules may be added, never removed or loosened.",
  },
];

/* TODO(vm) #3 — real URLs. */
const LINKS = [
  { label: "GitHub repository", href: "#", todo: true },
  { label: "Demo video", href: "#", todo: true },
  { label: "Technical write-up", href: "#", todo: true },
];

const SECTIONS = [
  { id: "top", label: "Overview" },
  { id: "audit", label: "Audit" },
  { id: "trace", label: "Agent trace" },
  { id: "kernel", label: "Kernel" },
  { id: "transfer", label: "Transfer" },
  { id: "verification", label: "Verification" },
];

/* ══════════════════════════════════════════════════════════════════════════
   THEME
   ══════════════════════════════════════════════════════════════════════════ */

const THEME_CSS = `
:root {
  --ks-bg:        #08090b;
  --ks-surface:   #0e1014;
  --ks-raised:    #14171d;
  --ks-border:    #1e2230;
  --ks-border-hi: #2b3142;
  --ks-text:      #e8eaf0;
  --ks-muted:     #8b93a7;
  --ks-faint:     #5b6478;
  --ks-accent:    #7c9cff;
  --ks-high:      #f0a742;
  --ks-mem:       #4fd1c5;
  --ks-comp:      #a78bfa;
  --ks-ok:        #55d18e;
}
html { scroll-behavior: smooth; }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
body { background: var(--ks-bg); }
.ks-root { background: var(--ks-bg); color: var(--ks-text); }
.ks-anchor { scroll-margin-top: 5.5rem; }
.ks-grid-bg {
  background-image:
    linear-gradient(to right, rgba(124,156,255,.05) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(124,156,255,.05) 1px, transparent 1px);
  background-size: 56px 56px;
  mask-image: radial-gradient(ellipse 90% 60% at 50% 0%, #000 20%, transparent 75%);
  -webkit-mask-image: radial-gradient(ellipse 90% 60% at 50% 0%, #000 20%, transparent 75%);
}
.ks-num { font-variant-numeric: tabular-nums; }
.ks-scroll::-webkit-scrollbar { height: 8px; width: 8px; }
.ks-scroll::-webkit-scrollbar-thumb { background: var(--ks-border-hi); border-radius: 4px; }
.ks-scroll::-webkit-scrollbar-track { background: transparent; }
/* Triton / Python token colours */
.t-kw  { color: #c792ea; }
.t-def { color: #7c9cff; }
.t-str { color: #98c379; }
.t-com { color: #5b6478; font-style: italic; }
.t-num { color: #e5a06a; }
.t-dec { color: #f0a742; }
.t-ns  { color: #4fd1c5; }
.t-op  { color: #8b93a7; }
`;

/* ══════════════════════════════════════════════════════════════════════════
   SYNTAX HIGHLIGHTING
   Deliberately dependency-free. One master regex, alternation ordered so that
   comments and strings win before anything inside them can be tokenized.
   ══════════════════════════════════════════════════════════════════════════ */

const PY_KEYWORDS = new Set([
  "import", "from", "def", "return", "for", "in", "if", "else", "elif", "while",
  "class", "with", "as", "and", "or", "not", "None", "True", "False", "lambda",
  "yield", "pass", "break", "continue", "try", "except", "finally", "raise",
  "assert", "global", "nonlocal", "del", "is", "await", "async",
]);

const NAMESPACES = new Set(["tl", "triton", "torch"]);

const TOKEN_RE = new RegExp(
  [
    "(#[^\\n]*)",                                   // 1 comment
    '("""[\\s\\S]*?"""|\'\'\'[\\s\\S]*?\'\'\')',    // 2 triple-quoted string
    "(\"[^\"\\n]*\"|'[^'\\n]*')",                   // 3 string
    "(@[A-Za-z_][\\w.]*)",                          // 4 decorator
    "\\b(\\d+\\.?\\d*(?:e[-+]?\\d+)?)\\b",          // 5 number
    "([A-Za-z_]\\w*)",                              // 6 identifier
  ].join("|"),
  "g"
);

function highlightTriton(source) {
  const out = [];
  let last = 0;
  let key = 0;
  let m;
  TOKEN_RE.lastIndex = 0;
  while ((m = TOKEN_RE.exec(source)) !== null) {
    if (m.index > last) out.push(source.slice(last, m.index));
    const [full, comment, triple, str, decorator, num, ident] = m;
    let cls = null;
    if (comment) cls = "t-com";
    else if (triple || str) cls = "t-str";
    else if (decorator) cls = "t-dec";
    else if (num) cls = "t-num";
    else if (ident) {
      if (PY_KEYWORDS.has(ident)) cls = "t-kw";
      else if (NAMESPACES.has(ident)) cls = "t-ns";
      else if (source[m.index - 1] === "." && source[m.index - 2] !== ".") cls = null;
      else {
        // A name directly after `def ` is a definition.
        const before = source.slice(Math.max(0, m.index - 5), m.index);
        if (/\bdef\s$/.test(before)) cls = "t-def";
      }
    }
    out.push(cls ? <span key={key++} className={cls}>{full}</span> : full);
    last = m.index + full.length;
  }
  if (last < source.length) out.push(source.slice(last));
  return out;
}

/* ══════════════════════════════════════════════════════════════════════════
   PRIMITIVES
   ══════════════════════════════════════════════════════════════════════════ */

function Eyebrow({ children }) {
  return (
    <div className="font-mono text-[11px] uppercase tracking-[0.22em] text-[color:var(--ks-accent)]">
      {children}
    </div>
  );
}

function SectionHeading({ eyebrow, title, children }) {
  return (
    <header className="mb-10 max-w-3xl">
      {eyebrow ? <Eyebrow>{eyebrow}</Eyebrow> : null}
      <h2 className="mt-3 text-3xl font-semibold tracking-tight text-[color:var(--ks-text)] sm:text-4xl">
        {title}
      </h2>
      {children ? (
        <p className="mt-4 text-[15px] leading-relaxed text-[color:var(--ks-muted)]">{children}</p>
      ) : null}
    </header>
  );
}

function Section({ id, children, className = "" }) {
  return (
    <section
      id={id}
      className={`ks-anchor border-t border-[color:var(--ks-border)] px-6 py-20 sm:py-24 ${className}`}
    >
      <div className="mx-auto w-full max-w-6xl">{children}</div>
    </section>
  );
}

function Callout({ tone = "accent", label, children }) {
  const tones = {
    accent: "border-[color:var(--ks-accent)]/35 bg-[color:var(--ks-accent)]/[0.07]",
    warn: "border-[color:var(--ks-high)]/35 bg-[color:var(--ks-high)]/[0.07]",
    ok: "border-[color:var(--ks-ok)]/30 bg-[color:var(--ks-ok)]/[0.06]",
  };
  const dot = { accent: "var(--ks-accent)", warn: "var(--ks-high)", ok: "var(--ks-ok)" }[tone];
  return (
    <div className={`rounded-lg border ${tones[tone]} p-5 sm:p-6`}>
      {label ? (
        <div className="mb-2 flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full" style={{ background: dot }} />
          <span className="font-mono text-[11px] uppercase tracking-[0.18em]" style={{ color: dot }}>
            {label}
          </span>
        </div>
      ) : null}
      <div className="text-[14.5px] leading-relaxed text-[color:var(--ks-text)]/90">{children}</div>
    </div>
  );
}

function KV({ rows, className = "" }) {
  return (
    <dl className={`grid gap-x-6 gap-y-2 font-mono text-[12.5px] ${className}`}>
      {rows.map(([k, v]) => (
        <div key={k} className="flex flex-wrap items-baseline gap-x-3 border-b border-dashed border-[color:var(--ks-border)] pb-2">
          <dt className="shrink-0 text-[color:var(--ks-faint)]">{k}</dt>
          <dd className="ks-num ml-auto text-right text-[color:var(--ks-text)]">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function KVStacked({ rows }) {
  return (
    <dl className="space-y-2.5 font-mono text-[12.5px]">
      {rows.map(([k, v]) => (
        <div key={k} className="border-b border-dashed border-[color:var(--ks-border)] pb-2 last:border-0">
          <dt className="text-[color:var(--ks-faint)]">{k}</dt>
          <dd className="ks-num mt-0.5 break-words text-[color:var(--ks-text)]">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function PriorityBadge({ priority }) {
  const map = {
    HIGH: { fg: "var(--ks-high)", glyph: "★★★", ring: "rgba(240,167,66,.3)", bg: "rgba(240,167,66,.10)" },
    MEDIUM: { fg: "#9aa3b8", glyph: "★★", ring: "rgba(154,163,184,.25)", bg: "rgba(154,163,184,.08)" },
    LOW: { fg: "var(--ks-faint)", glyph: "☆", ring: "transparent", bg: "transparent" },
  };
  const s = map[priority] || map.LOW;
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded px-2 py-0.5 font-mono text-[11px] tracking-wide"
      style={{ color: s.fg, background: s.bg, boxShadow: `inset 0 0 0 1px ${s.ring}` }}
    >
      <span aria-hidden="true">{s.glyph}</span>
      {priority}
    </span>
  );
}

function RegimePill({ regime }) {
  if (!regime) {
    return <span className="font-mono text-[12px] text-[color:var(--ks-faint)]">—</span>;
  }
  const color = regime === "memory" ? "var(--ks-mem)" : "var(--ks-comp)";
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-[12px]" style={{ color }}>
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
      {regime}
    </span>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   NAV
   ══════════════════════════════════════════════════════════════════════════ */

function Nav() {
  const [active, setActive] = useState("top");
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const els = SECTIONS.map((s) => document.getElementById(s.id)).filter(Boolean);
    if (!els.length || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActive(visible[0].target.id);
      },
      { rootMargin: "-45% 0px -50% 0px", threshold: 0 }
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);

  return (
    <nav
      className={`sticky top-0 z-50 border-b transition-colors ${
        scrolled
          ? "border-[color:var(--ks-border)] bg-[color:var(--ks-bg)]/85 backdrop-blur-xl"
          : "border-transparent bg-transparent"
      }`}
    >
      <div className="mx-auto flex h-14 w-full max-w-6xl items-center gap-6 px-6">
        <a href="#top" className="flex items-center gap-2.5 shrink-0">
          <span
            className="grid h-6 w-6 place-items-center rounded font-mono text-[12px] font-bold"
            style={{ background: "var(--ks-accent)", color: "#08090b" }}
          >
            G
          </span>
          <span className="font-mono text-[13px] font-semibold tracking-tight">gpuyantra</span>
        </a>
        <div className="ks-scroll ml-auto flex items-center gap-1 overflow-x-auto">
          {SECTIONS.slice(1).map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              className="rounded px-2.5 py-1.5 font-mono text-[12px] whitespace-nowrap transition-colors"
              style={{
                color: active === s.id ? "var(--ks-text)" : "var(--ks-faint)",
                background: active === s.id ? "var(--ks-raised)" : "transparent",
              }}
            >
              {s.label}
            </a>
          ))}
        </div>
      </div>
    </nav>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   SECTION 1 — HERO
   ══════════════════════════════════════════════════════════════════════════ */

function StatCard({ value, label, note }) {
  return (
    <div className="rounded-lg border border-[color:var(--ks-border)] bg-[color:var(--ks-surface)] p-5">
      <div className="ks-num font-mono text-4xl font-semibold tracking-tight text-[color:var(--ks-text)]">
        {value}
      </div>
      <div className="mt-2 text-[13.5px] font-medium text-[color:var(--ks-text)]/85">{label}</div>
      <div className="mt-1 font-mono text-[11.5px] leading-relaxed text-[color:var(--ks-faint)]">
        {note}
      </div>
    </div>
  );
}

function Hero() {
  return (
    <header id="top" className="ks-anchor relative overflow-hidden px-6 pb-20 pt-16 sm:pt-24">
      <div className="ks-grid-bg pointer-events-none absolute inset-0" aria-hidden="true" />
      <div className="relative mx-auto w-full max-w-6xl">
        <div className="inline-flex items-center gap-2 rounded-full border border-[color:var(--ks-border-hi)] bg-[color:var(--ks-surface)] px-3 py-1">
          <span className="h-1.5 w-1.5 rounded-full" style={{ background: "var(--ks-ok)" }} />
          <span className="font-mono text-[11px] tracking-wide text-[color:var(--ks-muted)]">
            All Things Agentic Hackathon · Google ADK
          </span>
        </div>

        <h1 className="mt-7 text-5xl font-semibold tracking-[-0.03em] sm:text-7xl">gpuyantra</h1>
        <p className="mt-3 text-xl text-[color:var(--ks-muted)] sm:text-2xl">
          Your on-call GPU kernel engineer.
        </p>

        <p className="mt-7 max-w-2xl text-[16px] leading-relaxed text-[color:var(--ks-text)]/85">
          Point it at any HuggingFace model. The <strong>KernelSmith</strong> agent tree profiles
          every layer, writes verified Triton kernels, and hot-swaps them into your running
          inference server. Skills learned on one model transfer to the next.
        </p>

        <div className="mt-10 grid gap-4 sm:grid-cols-3">
          <StatCard value={HEADLINE.speedup} label="Speedup vs eager PyTorch" note={HEADLINE.speedupNote} />
          <StatCard value={HEADLINE.tests} label="Tests passing" note={HEADLINE.testsNote} />
          <StatCard value={`${HEADLINE.models} models`} label="Audited end to end" note={HEADLINE.modelsNote} />
        </div>

        <a
          href="#audit"
          className="mt-10 inline-flex items-center gap-2 rounded-md border border-[color:var(--ks-border-hi)] bg-[color:var(--ks-raised)] px-4 py-2.5 font-mono text-[13px] text-[color:var(--ks-text)] transition-colors hover:border-[color:var(--ks-accent)]/50"
        >
          See how it works
          <span aria-hidden="true">↓</span>
        </a>

        <div className="mt-16 border-t border-[color:var(--ks-border)] pt-6">
          <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-[color:var(--ks-faint)]">
            Built with
          </div>
          <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2">
            {BUILT_WITH.map((t) => (
              <span key={t} className="font-mono text-[12.5px] text-[color:var(--ks-muted)]">
                {t}
              </span>
            ))}
          </div>
        </div>
      </div>
    </header>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   SECTION 2 — AUDIT EXPLORER
   ══════════════════════════════════════════════════════════════════════════ */

function ModelCard({ id, data, selected, onSelect }) {
  return (
    <button
      type="button"
      onClick={() => onSelect(id)}
      aria-pressed={selected}
      className="group rounded-lg border p-4 text-left transition-colors"
      style={{
        borderColor: selected ? "var(--ks-accent)" : "var(--ks-border)",
        background: selected ? "rgba(124,156,255,.07)" : "var(--ks-surface)",
      }}
    >
      <div className="flex items-baseline justify-between gap-3">
        <span className="font-mono text-[14px] font-semibold text-[color:var(--ks-text)]">
          {data.label}
        </span>
        <span className="ks-num font-mono text-[12px] text-[color:var(--ks-muted)]">{data.params}</span>
      </div>
      <div className="mt-1.5 text-[12.5px] text-[color:var(--ks-muted)]">{data.family}</div>
      <div className="mt-3 flex flex-wrap gap-1.5 font-mono text-[11px]">
        <span className="rounded bg-[color:var(--ks-raised)] px-1.5 py-0.5 text-[color:var(--ks-mem)]">
          {data.norm_type}
        </span>
        <span className="rounded bg-[color:var(--ks-raised)] px-1.5 py-0.5 text-[color:var(--ks-muted)]">
          {data.activation}
        </span>
        {data.served ? (
          <span className="rounded bg-[color:var(--ks-raised)] px-1.5 py-0.5 text-[color:var(--ks-ok)]">
            served
          </span>
        ) : null}
      </div>
      <div className="mt-3 font-mono text-[11px] text-[color:var(--ks-faint)]">{data.detail}</div>
    </button>
  );
}

function AuditTable({ data }) {
  const fmtAI = (ai) => (ai == null ? "n/a" : ai >= 100 ? String(Math.round(ai)) : ai.toFixed(2));
  return (
    <div className="ks-scroll overflow-x-auto rounded-lg border border-[color:var(--ks-border)] bg-[color:var(--ks-surface)]">
      <table className="w-full min-w-[680px] border-collapse text-left">
        <thead>
          <tr className="border-b border-[color:var(--ks-border)]">
            {["Module type", "Count", "Regime", "AI (FLOP/B)", "BW util", "Priority"].map((h, i) => (
              <th
                key={h}
                className={`px-4 py-3 font-mono text-[11px] uppercase tracking-[0.14em] text-[color:var(--ks-faint)] ${
                  i > 0 ? "text-right" : ""
                }`}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.entries.map((e) => {
            const isTop = e.type === data.top_target;
            return (
              <tr
                key={e.type}
                className="border-b border-[color:var(--ks-border)]/60 last:border-0"
                style={{ background: isTop ? "rgba(240,167,66,.05)" : "transparent" }}
              >
                <td className="px-4 py-2.5 font-mono text-[13px] text-[color:var(--ks-text)]">
                  {isTop ? (
                    <span className="mr-2 text-[color:var(--ks-high)]" aria-label="top target">
                      ▸
                    </span>
                  ) : (
                    <span className="mr-2 opacity-0" aria-hidden="true">▸</span>
                  )}
                  {e.type}
                </td>
                <td className="ks-num px-4 py-2.5 text-right font-mono text-[13px] text-[color:var(--ks-muted)]">
                  {e.count}
                </td>
                <td className="px-4 py-2.5 text-right">
                  <RegimePill regime={e.regime} />
                </td>
                <td className="ks-num px-4 py-2.5 text-right font-mono text-[13px]"
                    style={{ color: e.ai == null ? "var(--ks-faint)" : "var(--ks-text)" }}>
                  {fmtAI(e.ai)}
                </td>
                <td className="ks-num px-4 py-2.5 text-right font-mono text-[13px] text-[color:var(--ks-faint)]">
                  {e.bw_pct == null ? "n/a" : `${e.bw_pct}%`}
                </td>
                <td className="px-4 py-2.5 text-right">
                  <PriorityBadge priority={e.priority} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function AuditSection() {
  const [selected, setSelected] = useState(MODEL_ORDER[0]);
  const data = AUDIT_DATA[selected];

  return (
    <Section id="audit">
      <SectionHeading eyebrow="Step 01" title="Audit your model">
        KernelSmith profiles every module type in your model, classifies each one against the
        L4's roofline, and tells you what to optimize first. The whole sweep runs on CPU — the
        module tree is built on the meta device from <span className="font-mono text-[13px]">config.json</span>,
        so nothing is downloaded and nothing is allocated.
      </SectionHeading>

      <div className="grid gap-4 sm:grid-cols-3">
        {MODEL_ORDER.map((id) => (
          <ModelCard key={id} id={id} data={AUDIT_DATA[id]} selected={selected === id} onSelect={setSelected} />
        ))}
      </div>

      <div className="mt-8 rounded-lg border border-[color:var(--ks-border)] bg-[color:var(--ks-surface)]/50 p-4 font-mono text-[12px] leading-relaxed">
        <div className="text-[color:var(--ks-text)]">
          <span className="text-[color:var(--ks-faint)]">$</span> kernelsmith audit --model {selected}
        </div>
        <div className="mt-2 text-[color:var(--ks-muted)]">
          {data.model_name} · {data.total_modules} modules profiled · {data.unique_types} unique
          types · hidden_size={data.hidden_size}
        </div>
        <div className="text-[color:var(--ks-muted)]">
          {HARDWARE.name} ({HARDWARE.vram_gb} GB, {HARDWARE.mem_bw_gbps} GB/s,{" "}
          {HARDWARE.fp16_tflops} TFLOPS FP16) · ridge {HARDWARE.ridge_point} FLOP/byte
        </div>
        <div className="mt-1" style={{ color: data.measured ? "var(--ks-ok)" : "var(--ks-high)" }}>
          {data.measured
            ? `Mode: cuda — intensity analytic, bandwidth MEASURED (do_bench)`
            : `Mode: cpu — all values ESTIMATED analytically (shapes only, meta device)`}
        </div>
      </div>

      <div className="mt-4">
        <AuditTable data={data} />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-[1.35fr_1fr]">
        <Callout tone="warn" label="Recommendation">
          {data.recommendation}
        </Callout>
        <Callout tone="accent" label="Why this is the target">
          The norm is memory-bound by <span className="font-mono">two orders of magnitude</span> — it
          moves bytes the arithmetic never needed. Everything above the ridge point is already
          saturating Tensor Cores, and no kernel rewrite reclaims what cuBLAS is already getting.
        </Callout>
      </div>

      <p className="mt-6 font-mono text-[11.5px] leading-relaxed text-[color:var(--ks-faint)]">
        BW util is <span className="text-[color:var(--ks-muted)]">do_bench</span>-measured on the L4
        itself, against that card's own 300.1 GB/s — measured anywhere else the percentage would be
        wrong. It reads <span className="text-[color:var(--ks-muted)]">n/a</span> for modules whose
        forward will not take a bare synthetic probe (the ResNet composites) and for ops the
        estimator does not recognize. An unmeasured number is never rendered as a zero.
      </p>
    </Section>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   SECTION 3 — AGENT TRACE
   ══════════════════════════════════════════════════════════════════════════ */

function TraceStep({ step, index, open, onToggle }) {
  const panelId = `trace-panel-${index}`;
  return (
    <li className="relative pl-10">
      {/* rail */}
      <span
        className="absolute left-[13px] top-8 bottom-0 w-px"
        style={{ background: "var(--ks-border)" }}
        aria-hidden="true"
      />
      <span
        className="absolute left-0 top-3 grid h-[27px] w-[27px] place-items-center rounded-full border font-mono text-[11px] transition-colors"
        style={{
          borderColor: open ? "var(--ks-accent)" : "var(--ks-border-hi)",
          background: open ? "var(--ks-accent)" : "var(--ks-surface)",
          color: open ? "#08090b" : "var(--ks-muted)",
        }}
        aria-hidden="true"
      >
        {index + 1}
      </span>

      <div className="pb-4">
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={open}
          aria-controls={panelId}
          className="group w-full rounded-lg border px-4 py-3 text-left transition-colors"
          style={{
            borderColor: open ? "var(--ks-border-hi)" : "var(--ks-border)",
            background: open ? "var(--ks-raised)" : "var(--ks-surface)",
          }}
        >
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="font-mono text-[13.5px] font-semibold text-[color:var(--ks-accent)]">
              {step.agent}
            </span>
            <span className="rounded bg-[color:var(--ks-bg)] px-1.5 py-0.5 font-mono text-[10.5px] text-[color:var(--ks-faint)]">
              {step.kind}
            </span>
            <span
              className="ml-auto font-mono text-[11px] text-[color:var(--ks-faint)] transition-transform"
              style={{ transform: open ? "rotate(90deg)" : "none" }}
              aria-hidden="true"
            >
              ▸
            </span>
          </div>
          <div className="mt-1.5 text-[14px] text-[color:var(--ks-text)]">{step.action}</div>
          <div className="mt-1 text-[12.5px] text-[color:var(--ks-muted)]">{step.summary}</div>
        </button>

        {open ? (
          <div
            id={panelId}
            className="mt-2 rounded-lg border border-[color:var(--ks-border)] bg-[color:var(--ks-surface)] p-4"
          >
            <p className="text-[13.5px] leading-relaxed text-[color:var(--ks-text)]/85">{step.body}</p>
            {step.fingerprint ? (
              <div className="mt-3 rounded border border-[color:var(--ks-accent)]/30 bg-[color:var(--ks-accent)]/[0.06] px-3 py-2">
                <div className="font-mono text-[10.5px] uppercase tracking-[0.16em] text-[color:var(--ks-accent)]">
                  bottleneck fingerprint
                </div>
                <div className="ks-scroll mt-1 overflow-x-auto font-mono text-[12.5px] text-[color:var(--ks-text)]">
                  {step.fingerprint}
                </div>
              </div>
            ) : null}
            <div className="mt-4">
              <KV rows={step.output} />
            </div>
          </div>
        ) : null}
      </div>
    </li>
  );
}

function TraceSection() {
  const [open, setOpen] = useState(() => new Set([0]));
  const toggle = (i) =>
    setOpen((prev) => {
      const next = new Set(prev);
      next.has(i) ? next.delete(i) : next.add(i);
      return next;
    });
  const allOpen = open.size === TRACE.length;

  return (
    <Section id="trace">
      <SectionHeading eyebrow="Step 02" title="The agent optimizes">
        Five agents, one shared session state. The Supervisor sequences; the Profiler measures;
        retrieval decides what prior work to reuse; the Coder writes; the Judge refuses to take
        anyone's word for it. Expand a step to see what it actually emitted.
      </SectionHeading>

      <div className="mb-5 flex justify-end">
        <button
          type="button"
          onClick={() => setOpen(allOpen ? new Set() : new Set(TRACE.map((_, i) => i)))}
          className="rounded-md border border-[color:var(--ks-border-hi)] bg-[color:var(--ks-surface)] px-3 py-1.5 font-mono text-[12px] text-[color:var(--ks-muted)] transition-colors hover:text-[color:var(--ks-text)]"
        >
          {allOpen ? "Collapse all" : "Expand all"}
        </button>
      </div>

      <ol className="relative">
        {TRACE.map((step, i) => (
          <TraceStep key={`${step.agent}-${i}`} step={step} index={i} open={open.has(i)} onToggle={() => toggle(i)} />
        ))}
      </ol>

      <div className="mt-8 rounded-lg border border-[color:var(--ks-ok)]/30 bg-[color:var(--ks-ok)]/[0.05] p-5 sm:p-6">
        <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-[color:var(--ks-ok)]">
          Run result
        </div>
        <div className="mt-4 grid gap-5 sm:grid-cols-5">
          {[
            ["reward", RESULT.reward],
            ["vs eager", RESULT.vs_eager],
            ["vs torch.compile", RESULT.vs_compile],
            ["correctness", RESULT.correctness],
            ["iterations", RESULT.iterations],
          ].map(([k, v]) => (
            <div key={k}>
              <div className="ks-num font-mono text-2xl font-semibold text-[color:var(--ks-text)]">{v}</div>
              <div className="mt-0.5 font-mono text-[11px] text-[color:var(--ks-faint)]">{k}</div>
            </div>
          ))}
        </div>
      </div>

      <p className="mt-4 font-mono text-[11.5px] leading-relaxed text-[color:var(--ks-faint)]">
        Measured on the L4 VM. Every agent decodes at temperature 0 with a fixed seed, so the same
        inputs produce the same kernel — a sampled kernel cannot be reproduced no matter how many
        torch seeds you pin below it.
      </p>
    </Section>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   SECTION 4 — THE KERNEL
   ══════════════════════════════════════════════════════════════════════════ */

function KernelSection() {
  const highlighted = useMemo(() => highlightTriton(KERNEL_SOURCE), []);
  const [copied, setCopied] = useState(false);
  const timer = useRef(null);

  useEffect(() => () => clearTimeout(timer.current), []);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(KERNEL_SOURCE);
      setCopied(true);
      clearTimeout(timer.current);
      timer.current = setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard unavailable — the source is selectable either way */
    }
  };

  return (
    <Section id="kernel">
      <SectionHeading eyebrow="Step 03" title="Inspect the output">
        The kernel, and the deployment contract that binds it to a live module. Both are agent
        output; neither was written by hand.
      </SectionHeading>

      <div className="grid gap-6 lg:grid-cols-[1.15fr_1fr]">
        <div className="flex flex-col overflow-hidden rounded-lg border border-[color:var(--ks-border)] bg-[color:var(--ks-surface)]">
          <div className="flex shrink-0 items-center gap-3 border-b border-[color:var(--ks-border)] px-4 py-2.5">
            <div className="flex gap-1.5" aria-hidden="true">
              <span className="h-2.5 w-2.5 rounded-full bg-[color:var(--ks-border-hi)]" />
              <span className="h-2.5 w-2.5 rounded-full bg-[color:var(--ks-border-hi)]" />
              <span className="h-2.5 w-2.5 rounded-full bg-[color:var(--ks-border-hi)]" />
            </div>
            <span className="font-mono text-[12px] text-[color:var(--ks-muted)]">rmsnorm_triton.py</span>
            <button
              type="button"
              onClick={copy}
              className="ml-auto rounded border border-[color:var(--ks-border-hi)] px-2 py-1 font-mono text-[11px] text-[color:var(--ks-muted)] transition-colors hover:text-[color:var(--ks-text)]"
            >
              {copied ? "copied" : "copy"}
            </button>
          </div>
          <pre className="ks-scroll min-h-0 flex-1 max-h-[560px] overflow-auto px-4 py-4 lg:max-h-none font-mono text-[12px] leading-[1.65] text-[color:var(--ks-text)]">
            <code>{highlighted}</code>
          </pre>
        </div>

        <div>
          <div className="rounded-lg border border-[color:var(--ks-border)] bg-[color:var(--ks-surface)] p-5">
            <div className="flex flex-wrap items-baseline gap-2">
              <h3 className="text-[15px] font-semibold">What this kernel does</h3>
              <span className="font-mono text-[11px] text-[color:var(--ks-faint)]">
                explained by {EXPLANATION.model}
              </span>
            </div>
            {EXPLANATION.placeholder ? (
              <div className="mt-3 rounded border border-[color:var(--ks-high)]/30 bg-[color:var(--ks-high)]/[0.07] px-3 py-2 font-mono text-[11px] text-[color:var(--ks-high)]">
                placeholder — replace with the explanation from a real run
              </div>
            ) : null}
            <div className="mt-4 space-y-3.5">
              {EXPLANATION.paragraphs.map((p, i) => (
                <p key={i} className="text-[13.5px] leading-relaxed text-[color:var(--ks-text)]/85">
                  {p}
                </p>
              ))}
            </div>
            <p className="mt-4 border-l-2 pl-3 text-[13.5px] italic leading-relaxed text-[color:var(--ks-text)]"
               style={{ borderColor: "var(--ks-accent)" }}>
              {EXPLANATION.takeaway}
            </p>
          </div>

          <div className="mt-5 rounded-lg border border-[color:var(--ks-border)] bg-[color:var(--ks-surface)] p-5">
            <h3 className="text-[15px] font-semibold">Deployment contract</h3>
            <p className="mt-1 font-mono text-[11.5px] text-[color:var(--ks-faint)]">
              adapter_mapping · kernel_param → module_attr
            </p>
            <div className="mt-4 space-y-3">
              {ADAPTER_BINDINGS.map((b) => (
                <div key={b.kernel_param} className="border-b border-dashed border-[color:var(--ks-border)] pb-3 last:border-0 last:pb-0">
                  <div className="flex flex-wrap items-center gap-2 font-mono text-[12.5px]">
                    <span className="rounded bg-[color:var(--ks-raised)] px-1.5 py-0.5 text-[color:var(--ks-accent)]">
                      {b.kernel_param}
                    </span>
                    <span className="text-[color:var(--ks-faint)]" aria-hidden="true">→</span>
                    <span
                      className="rounded bg-[color:var(--ks-raised)] px-1.5 py-0.5"
                      style={{ color: b.implicit ? "var(--ks-faint)" : "var(--ks-mem)" }}
                    >
                      {b.module_attr}
                    </span>
                  </div>
                  <p className="mt-1.5 text-[12.5px] leading-relaxed text-[color:var(--ks-muted)]">{b.note}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-[1.35fr_1fr]">
        <Callout tone="accent" label="The novel contribution">
          The agent wrote this deployment contract. A deterministic verifier validated it — every
          declared attribute is checked for existence against a meta-device instance of the real
          target class before a single byte reaches the sandbox. <strong>No human specified the
          parameter mapping.</strong>
        </Callout>
        <Callout tone="warn" label="A schema lesson, paid for in a live run">
          Declared as <span className="font-mono">dict[str, str]</span>, the model returned{" "}
          <span className="font-mono">{"{}"}</span> on 3 of 3 trials — and an empty contract is not
          an error anywhere, it just falls back to the hand-written bridge. Green ticks all the way
          down, claim quietly false. As a{" "}
          <span className="font-mono">list[AdapterBinding]</span> with named fields: 3 of 3 correct.
        </Callout>
      </div>
    </Section>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   SECTION 5 — CROSS-MODEL TRANSFER
   ══════════════════════════════════════════════════════════════════════════ */

function FingerprintCard({ side, data, tone }) {
  return (
    <div
      className="rounded-lg border bg-[color:var(--ks-surface)] p-5"
      style={{ borderColor: tone }}
    >
      <div className="flex items-baseline justify-between gap-3">
        <span className="font-mono text-[11px] uppercase tracking-[0.18em]" style={{ color: tone }}>
          {side}
        </span>
        <span className="ks-num font-mono text-[11.5px] text-[color:var(--ks-faint)]">
          {data.count} instances
        </span>
      </div>
      <div className="mt-2 text-[17px] font-semibold">{data.model}</div>
      <div className="font-mono text-[13px] text-[color:var(--ks-muted)]">{data.op}</div>
      <div className="ks-scroll mt-4 overflow-x-auto rounded border border-[color:var(--ks-border)] bg-[color:var(--ks-bg)] px-3 py-2 font-mono text-[12px] text-[color:var(--ks-text)]">
        {data.fingerprint}
      </div>
      <div className="mt-4">
        <KV rows={data.fields} />
      </div>
    </div>
  );
}

function TransferSection() {
  return (
    <Section id="transfer">
      <SectionHeading eyebrow="Step 04" title="Skills transfer across models">
        The library is indexed by the bottleneck fingerprint, not by the operator's name and not by
        the model it came from. Two different norms, two different architectures, one retrieval hit.
      </SectionHeading>

      <div className="grid items-stretch gap-4 lg:grid-cols-[1fr_auto_1fr]">
        <FingerprintCard side="Learned on" data={TRANSFER.source} tone="var(--ks-mem)" />

        <div className="flex flex-row items-center justify-center gap-3 lg:flex-col">
          <div className="h-px w-16 lg:h-16 lg:w-px" style={{ background: "var(--ks-border-hi)" }} aria-hidden="true" />
          <div className="rounded-md border border-[color:var(--ks-accent)]/40 bg-[color:var(--ks-accent)]/[0.08] px-3 py-2 text-center">
            <div className="ks-num font-mono text-[15px] font-semibold text-[color:var(--ks-accent)]">
              {TRANSFER.similarity}
            </div>
            <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-[color:var(--ks-faint)]">
              cosine similarity
            </div>
            <div className="ks-num mt-1 font-mono text-[10.5px] text-[color:var(--ks-faint)]">
              distance {TRANSFER.distance}
            </div>
          </div>
          <div className="h-px w-16 lg:h-16 lg:w-px" style={{ background: "var(--ks-border-hi)" }} aria-hidden="true" />
        </div>

        <FingerprintCard side="Retrieved for" data={TRANSFER.target} tone="var(--ks-comp)" />
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <Callout tone="accent" label="Why it works">
          A kernel learned on Qwen2.5's RMSNorm surfaces when the agent later targets GPT-2's
          LayerNorm, because both are memory-bound normalizations on the same hardware. The
          retrieval query pre-filters on{" "}
          <span className="font-mono">(op_family, hardware)</span> and nothing else — no model name,
          no operator name, ever enters the embedded text. A name-keyed cache cannot make that jump;
          that is the entire point.
        </Callout>
        <div className="rounded-lg border border-[color:var(--ks-border)] bg-[color:var(--ks-surface)] p-5">
          <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-[color:var(--ks-faint)]">
            Published systems
          </div>
          <div className="mt-3 space-y-2.5">
            {PRIOR_WORK.map((w) => (
              <div key={w.name} className="flex flex-wrap items-baseline gap-x-3 border-b border-dashed border-[color:var(--ks-border)] pb-2 last:border-0">
                <span className="font-mono text-[12.5px] text-[color:var(--ks-text)]">{w.name}</span>
                <span className="text-[12.5px] text-[color:var(--ks-muted)]">{w.note}</span>
              </div>
            ))}
          </div>
          <p className="mt-4 text-[13.5px] leading-relaxed text-[color:var(--ks-text)]/85">
            All four use a <strong>human-authored deployment bridge</strong>. KernelSmith is the
            first where the agent writes the bridge and a deterministic verifier checks it.
          </p>
        </div>
      </div>

      <p className="mt-6 font-mono text-[11.5px] text-[color:var(--ks-faint)]">
        Measured against the {TRANSFER.measuredOn} — the GPT-2 fingerprint retrieved all{" "}
        {TRANSFER.retrieved} Qwen2.5 RMSNorm skills, nearest at vector distance {TRANSFER.distance}.
      </p>
    </Section>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   SECTION 6 — VERIFICATION
   ══════════════════════════════════════════════════════════════════════════ */

function VerificationCard({ card }) {
  return (
    <div className="flex flex-col rounded-lg border border-[color:var(--ks-border)] bg-[color:var(--ks-surface)] p-5">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="text-[15px] font-semibold">{card.title}</h3>
        <span className="font-mono text-[11px] text-[color:var(--ks-faint)]">{card.sub}</span>
      </div>
      <div className="ks-num mt-2 font-mono text-3xl font-semibold text-[color:var(--ks-ok)]">
        {card.headline}
      </div>
      <div className="mt-4 flex-1">
        {card.list ? (
          <ol className="space-y-2 font-mono text-[12.5px]">
            {card.list.map((rule, i) => (
              <li key={rule} className="flex gap-2.5 border-b border-dashed border-[color:var(--ks-border)] pb-2 last:border-0">
                <span className="ks-num shrink-0 text-[color:var(--ks-faint)]">{i + 1}</span>
                <span className="text-[color:var(--ks-text)]">{rule}</span>
              </li>
            ))}
          </ol>
        ) : (
          <KVStacked rows={card.lines} />
        )}
      </div>
      <p className="mt-4 text-[12.5px] leading-relaxed text-[color:var(--ks-muted)]">{card.note}</p>
    </div>
  );
}

function VerificationSection() {
  return (
    <Section id="verification">
      <SectionHeading eyebrow="Step 05" title="Every number survives scrutiny">
        A kernel agent that grades its own homework is a random number generator with good prose.
        Three gates stand between a generated kernel and a claim, and none of them is an LLM.
      </SectionHeading>

      <div className="grid gap-4 lg:grid-cols-3">
        {VERIFICATION.map((c) => (
          <VerificationCard key={c.title} card={c} />
        ))}
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <Callout tone="warn" label="What about compute-bound ops?">
          Nothing. The MLP layers sit at 358 FLOP/byte — well past the L4's ridge point of 101 —
          and are dominated by GEMM on Tensor Cores that cuBLAS already saturates. Pointed at one,
          the agent returns <span className="font-mono">reward +1: correct, not faster</span> and
          the kernel is never deployed. KernelSmith does not claim speedups it does not have.
        </Callout>
        <Callout tone="ok" label="Closed on 2026-08-30: the hot-swap is live">
          The end-to-end jump has now been recorded on the served model. The same prompt through{" "}
          <span className="font-mono">/generate</span> took 529 ms for 3 tokens before the swap and
          87.5 ms for 2 tokens after it, across 57 patched{" "}
          <span className="font-mono">Qwen2RMSNorm</span> modules, with coherent output and no
          restart. What is still not recorded is a sustained tokens/sec curve under load — the
          number above is two single requests, and it is quoted as exactly that.
        </Callout>
      </div>
    </Section>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   FOOTER
   ══════════════════════════════════════════════════════════════════════════ */

function Footer() {
  return (
    <footer className="border-t border-[color:var(--ks-border)] px-6 py-14">
      <div className="mx-auto w-full max-w-6xl">
        <div className="grid gap-10 sm:grid-cols-[1fr_auto]">
          <div>
            <div className="flex items-center gap-2.5">
              <span
                className="grid h-6 w-6 place-items-center rounded font-mono text-[12px] font-bold"
                style={{ background: "var(--ks-accent)", color: "#08090b" }}
              >
                G
              </span>
              <span className="font-mono text-[13px] font-semibold">gpuyantra</span>
            </div>
            <p className="mt-3 max-w-md text-[13.5px] leading-relaxed text-[color:var(--ks-muted)]">
              Built by Kaustubh Upadhyay for the All Things Agentic Hackathon.
            </p>
            <div className="mt-5 flex flex-wrap gap-x-5 gap-y-2">
              {LINKS.map((l) => (
                <a
                  key={l.label}
                  href={l.href}
                  className="font-mono text-[12.5px] text-[color:var(--ks-accent)] underline-offset-4 hover:underline"
                >
                  {l.label}
                  {l.todo ? (
                    <span className="ml-1.5 text-[color:var(--ks-high)]" title="URL pending">
                      ·
                    </span>
                  ) : null}
                  <span aria-hidden="true"> ↗</span>
                </a>
              ))}
            </div>
          </div>

          <div className="sm:text-right">
            <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-[color:var(--ks-faint)]">
              Stack
            </div>
            <ul className="mt-3 space-y-1.5">
              {BUILT_WITH.map((t) => (
                <li key={t} className="font-mono text-[12.5px] text-[color:var(--ks-muted)]">
                  {t}
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="mt-10 border-t border-[color:var(--ks-border)] pt-5 font-mono text-[11.5px] text-[color:var(--ks-faint)]">
          Audit figures captured 2026-08-30 on the L4 in CUDA mode — intensity analytic, bandwidth
          do_bench-measured. Speedups measured on the same NVIDIA L4. Unmeasured values render as{" "}
          <span className="text-[color:var(--ks-muted)]">n/a</span>, never as zero.
        </div>
      </div>
    </footer>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   ROOT
   ══════════════════════════════════════════════════════════════════════════ */

export default function KernelSmithExplorer() {
  return (
    <div className="ks-root min-h-screen antialiased">
      <style>{THEME_CSS}</style>
      <Nav />
      <main>
        <Hero />
        <AuditSection />
        <TraceSection />
        <KernelSection />
        <TransferSection />
        <VerificationSection />
      </main>
      <Footer />
    </div>
  );
}
