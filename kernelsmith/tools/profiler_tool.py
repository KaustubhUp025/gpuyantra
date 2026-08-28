"""Roofline bottleneck fingerprint for a PyTorch op on the L4 (spec 7).

Nsight Compute is deliberately avoided: it needs elevated perf counters that are
unreliable on a virtualized VM, and it is far too slow to sit inside an agent loop.
Instead the fingerprint is roofline-style — one honest `do_bench` measurement plus
ANALYTIC FLOP and byte counts — which is enough to answer the only question the Coder
actually needs answered: is this op starved of bandwidth or of math?

The fingerprint is also the retrieval key: `BottleneckFingerprint.to_embedding_text()`
is what gets embedded, so a skill learned on RMSNorm can surface for RoPE when the
bottleneck matches. Retrieval is by WHY an op is slow, not by its name.

Every FLOP/byte count here is an ESTIMATE with a stated assumption, and the occupancy
number is an explicit heuristic — see `estimate_occupancy`. They are used to place the
op relative to the L4 ridge point (~100 FLOP/byte), not to report to a user as truth.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from google.adk.tools import FunctionTool

from kernelsmith.config import (
    AUDIT_PROBE_BATCH,
    AUDIT_PROBE_SEQ,
    AUDIT_PROBE_SPATIAL,
    AUDIT_REPORT_WIDTH,
    GLOBAL_SEED,
    L4_FP16_TFLOPS,
    L4_MEM_BW_GBPS,
    L4_SM_COUNT,
    L4_VRAM_GB,
    MODEL_REGISTRY,
    PROFILER_BLOCKS_PER_SM,
    PROFILER_FALLBACK_AI,
    PROFILER_FALLBACK_OCCUPANCY,
    PROFILER_MAX_TILE,
    PROFILER_MIN_TILE,
)
from kernelsmith.memory.schemas import BottleneckFingerprint
from kernelsmith.verifier.timing import bench_kernel

TensorFn = Callable[[torch.Tensor], torch.Tensor]

#: FLOP/byte above which the L4 runs out of math before it runs out of bandwidth.
#: 30.3 TFLOP/s / 300.1 GB/s ~= 101 FLOP/byte.
RIDGE_POINT = L4_FP16_TFLOPS * 1e12 / (L4_MEM_BW_GBPS * 1e9)

#: Op families, as stored in Firestore and used as an equality pre-filter.
OP_FAMILIES = ("norm", "rope", "mlp", "elementwise", "reduction")

#: Substring -> family. Longest match wins, so "rmsnorm" beats "norm" and
#: "softmax" is a reduction rather than an elementwise op.
_FAMILY_KEYWORDS: dict[str, str] = {
    "rmsnorm": "norm",
    "layernorm": "norm",
    "layer_norm": "norm",
    "groupnorm": "norm",
    "batchnorm": "norm",
    "norm": "norm",
    "rope": "rope",
    "rotary": "rope",
    "embedding": "rope",
    "mlp": "mlp",
    "matmul": "mlp",
    "gemm": "mlp",
    "linear": "mlp",
    "proj": "mlp",
    "attention": "mlp",
    "softmax": "reduction",
    "logsumexp": "reduction",
    "reduce": "reduction",
    "sum": "reduction",
    "mean": "reduction",
    "max": "reduction",
    "argmax": "reduction",
    "silu": "elementwise",
    "gelu": "elementwise",
    "relu": "elementwise",
    "swiglu": "elementwise",
    "sigmoid": "elementwise",
    "elementwise": "elementwise",
    "add": "elementwise",
    "mul": "elementwise",
}

#: An MLP block's intermediate width, as a multiple of hidden. Qwen2.5-1.5B is
#: 1536 -> 8960 (5.8x); 4x is the conventional estimate and is only used to place
#: the op on the roofline, never to allocate anything.
_MLP_EXPANSION = 4


# --------------------------------------------------------------------------- #
# Op family classification
# --------------------------------------------------------------------------- #


def classify_op_family(fn: Any) -> str:
    """Classify a reference op into one of the five families by name.

    Looks at the callable's `__name__`, its class name (for `nn.Module`s), and its
    qualified name. A `functools.partial` or lambda wrapping a named op is unwrapped
    first. Anything unrecognised is "elementwise": the most conservative guess, since
    it implies memory-bound, which is true of most ops we patch.
    """
    for text in _name_candidates(fn):
        lowered = text.lower()
        matches = [kw for kw in _FAMILY_KEYWORDS if kw in lowered]
        if matches:
            return _FAMILY_KEYWORDS[max(matches, key=len)]
    return "elementwise"


def _name_candidates(fn: Any) -> list[str]:
    """Every name that might identify `fn`, most specific first."""
    target = getattr(fn, "func", fn)  # unwrap functools.partial
    names = [
        str(getattr(target, "op_name", "")),
        getattr(target, "__name__", ""),
        type(target).__name__,  # nn.Module instances: "Qwen2RMSNorm"
        getattr(target, "__qualname__", ""),  # a lambda's enclosing scope often names the op
    ]
    # "function"/"method" are the type names of plain callables and identify nothing.
    generic = {"function", "method", "builtin_function_or_method", "partial", "<lambda>"}
    return [n for n in names if n and n not in generic]


# --------------------------------------------------------------------------- #
# Analytic FLOP / byte counts
# --------------------------------------------------------------------------- #


def analytic_counts(
    op_family: str,
    numel: int,
    hidden_size: int,
    element_size: int,
) -> tuple[float, float]:
    """Estimated (FLOPs, bytes moved) for one forward pass.

    Byte counts are the MINIMUM traffic a fused kernel must move (a cold read of each
    input plus one write of the output) — not what eager PyTorch actually moves, which
    is several times higher because every intermediate round-trips through HBM. Using
    the minimum keeps the arithmetic intensity a property of the OP, not of the
    implementation we are trying to replace.

    Assumptions per family:
      norm         read x, write y; 5 flops/elem (square, add, mean, rsqrt, multiply)
      elementwise  read x, write y; 2 flops/elem
      reduction    read x twice (reduce pass + normalize pass), write y; 5 flops/elem
      rope         read x + cos + sin, write y; 6 flops/elem (2 muls + 1 add per half)
      mlp          3 GEMMs of [rows,H]x[H,4H]; weights dominate the byte count
    """
    rows = max(1, numel // max(1, hidden_size))

    if op_family == "mlp":
        intermediate = hidden_size * _MLP_EXPANSION
        # gate + up + down, 2 flops per MAC.
        flops = 3.0 * 2.0 * rows * hidden_size * intermediate
        bytes_moved = float(element_size) * (
            2.0 * rows * hidden_size  # read x, write y
            + 3.0 * hidden_size * intermediate  # three weight matrices
        )
        return flops, bytes_moved

    flops_per_elem, tensors_touched = {
        "norm": (5.0, 2.0),
        "reduction": (5.0, 3.0),
        "rope": (6.0, 4.0),
        "elementwise": (2.0, 2.0),
    }.get(op_family, (2.0, 2.0))

    return flops_per_elem * numel, tensors_touched * numel * element_size


# --------------------------------------------------------------------------- #
# Heuristics
# --------------------------------------------------------------------------- #


def compute_tile_hint(hidden_size: int) -> int:
    """Suggested Triton BLOCK_SIZE for a row of `hidden_size` elements.

    Next power of two, clamped to [64, 1024]. A row-per-program kernel wants the whole
    row resident where possible; past 1024 fp16 lanes the L4 starts spilling registers,
    so wider rows get a strided loop over 1024-wide tiles instead.
    """
    if hidden_size <= 0:
        return PROFILER_MIN_TILE
    pow2 = 1 << (hidden_size - 1).bit_length()
    return max(PROFILER_MIN_TILE, min(pow2, PROFILER_MAX_TILE))


def estimate_occupancy(input_shape: Sequence[int], hidden_size: int) -> float:
    """APPROXIMATE achieved occupancy in [0, 1]. A heuristic, never a measurement.

    Two effects, multiplied:
      1. Wave fill — a row-per-program launch with fewer rows than
         `L4_SM_COUNT * PROFILER_BLOCKS_PER_SM` (232) leaves SMs idle outright.
      2. Tile fill — a row that does not fill its tile wastes the lanes past its end.

    Real occupancy also depends on register and shared-memory pressure, which we cannot
    know without running the compiled kernel. Anything derived from this number must be
    labelled approximate.
    """
    rows = 1
    for dim in input_shape:
        rows *= max(1, int(dim))

    wave_fill = min(1.0, rows / float(L4_SM_COUNT * PROFILER_BLOCKS_PER_SM))
    tile = compute_tile_hint(hidden_size)
    tile_fill = min(1.0, hidden_size / float(tile)) if hidden_size > 0 else 1.0

    return max(0.0, min(1.0, wave_fill * tile_fill))


# --------------------------------------------------------------------------- #
# The profiler proper
# --------------------------------------------------------------------------- #


def profile_op(
    reference_fn: TensorFn,
    input_shape: Sequence[int],
    hidden_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    op_family: str | None = None,
) -> BottleneckFingerprint:
    """Compute a roofline fingerprint for a PyTorch op on the L4.

    Args:
        reference_fn: The op to profile, `x -> y`, with any weights already bound.
        input_shape: Leading dims of the probe input, e.g. `(8, 512)` for batch x seq.
            `hidden_size` is appended as the last axis.
        hidden_size: The model's hidden dimension.
        device: Where to allocate the probe input. "cuda" in production.
        dtype: Probe dtype. fp16 matches the served Qwen2.5-1.5B.
        op_family: Override the name-based family classification.

    Never raises. If benching or classification fails for any reason — no GPU, a
    reference that throws, a timeout — it returns the memory-bound fallback
    fingerprint (spec 7) so the loop can proceed with a conservative guess.
    """
    family = op_family or _safe_classify(reference_fn)

    try:
        x = torch.randn(*input_shape, hidden_size, device=device, dtype=dtype)
        median_ms = bench_kernel(lambda: reference_fn(x))
        if median_ms <= 0:
            raise ValueError(f"non-positive median from do_bench: {median_ms}")

        flops, bytes_moved = analytic_counts(family, x.numel(), hidden_size, x.element_size())
        if bytes_moved <= 0:
            raise ValueError("analytic byte count is zero")

        median_s = median_ms / 1000.0
        memory_throughput_gbps = bytes_moved / median_s / 1e9
        arithmetic_intensity = flops / bytes_moved
    except Exception:  # noqa: BLE001 — a failed probe must never stall the agent loop
        return fallback_fingerprint(family, hidden_size)

    is_memory_bound = arithmetic_intensity < RIDGE_POINT

    return BottleneckFingerprint(
        op_family=family,
        hardware="L4",
        memory_throughput_gbps=memory_throughput_gbps,
        achieved_occupancy=estimate_occupancy(input_shape, hidden_size),
        arithmetic_intensity=arithmetic_intensity,
        is_memory_bound=is_memory_bound,
        is_compute_bound=not is_memory_bound,
        tile_size_hint=compute_tile_hint(hidden_size),
    )


def fallback_fingerprint(op_family: str, hidden_size: int) -> BottleneckFingerprint:
    """The fingerprint used when profiling errors out (spec 7).

    Defaults to memory-bound with AI = 0.5, which is true of essentially every
    norm / rope / elementwise op we patch — a conservative guess, not a measurement.
    `memory_throughput_gbps = 0.0` is the tell that nothing was actually measured.
    """
    return BottleneckFingerprint(
        op_family=op_family if op_family in OP_FAMILIES else "elementwise",
        hardware="L4",
        memory_throughput_gbps=0.0,
        achieved_occupancy=PROFILER_FALLBACK_OCCUPANCY,
        arithmetic_intensity=PROFILER_FALLBACK_AI,
        is_memory_bound=True,
        is_compute_bound=False,
        tile_size_hint=compute_tile_hint(hidden_size),
    )


def _safe_classify(fn: Any) -> str:
    try:
        return classify_op_family(fn)
    except Exception:  # noqa: BLE001 — classification is a nicety, never a failure mode
        return "elementwise"


# --------------------------------------------------------------------------- #
# Reference op registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OpBinding:
    """A reference forward plus the adapter that gives a candidate the same signature.

    `reference(x) -> y` is what correctness compares against. `bind(entrypoint)`
    wraps a generated kernel's wrapper — which may take weights and an eps — into the
    same one-argument shape, using the SAME weights the reference saw.
    """

    family: str
    reference: TensorFn
    bind: Callable[[Callable[..., torch.Tensor]], TensorFn]


def _weights(shape: tuple[int, ...], device: str, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Deterministic weights, seeded so the parent and the sandbox agree exactly."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(*shape, generator=gen, dtype=torch.float32) * 0.02 + 1.0
    return w.to(device=device, dtype=dtype)


def _build_rmsnorm(hidden_size: int, device: str, dtype: torch.dtype) -> OpBinding:
    weight = _weights((hidden_size,), device, dtype, GLOBAL_SEED)
    eps = 1e-6

    def reference(x: torch.Tensor) -> torch.Tensor:
        # Qwen2RMSNorm: the reduction runs in fp32, the output returns to x's dtype.
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return (x.to(torch.float32) * torch.rsqrt(var + eps)).to(x.dtype) * weight

    return OpBinding("norm", reference, lambda entry: lambda x: entry(x, weight, eps))


def _build_layernorm(hidden_size: int, device: str, dtype: torch.dtype) -> OpBinding:
    weight = _weights((hidden_size,), device, dtype, GLOBAL_SEED)
    bias = _weights((hidden_size,), device, dtype, GLOBAL_SEED + 1) - 1.0
    eps = 1e-5

    def reference(x: torch.Tensor) -> torch.Tensor:
        f = x.to(torch.float32)
        normed = (f - f.mean(-1, keepdim=True)) * torch.rsqrt(
            f.var(-1, keepdim=True, unbiased=False) + eps
        )
        return (normed.to(x.dtype) * weight) + bias

    return OpBinding("norm", reference, lambda entry: lambda x: entry(x, weight, bias, eps))


def _build_softmax(hidden_size: int, device: str, dtype: torch.dtype) -> OpBinding:
    del hidden_size, device, dtype
    return OpBinding("reduction", lambda x: torch.softmax(x, dim=-1), lambda entry: entry)


def _build_silu(hidden_size: int, device: str, dtype: torch.dtype) -> OpBinding:
    del hidden_size, device, dtype
    return OpBinding(
        "elementwise", lambda x: x * torch.sigmoid(x.to(torch.float32)).to(x.dtype), lambda e: e
    )


def _build_rope(hidden_size: int, device: str, dtype: torch.dtype) -> OpBinding:
    """RoPE over the last axis, treating `hidden_size` as the head dimension."""
    half = hidden_size // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))

    def tables(seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(device=device, dtype=dtype), emb.sin().to(device=device, dtype=dtype)

    def rotate_half(t: torch.Tensor) -> torch.Tensor:
        return torch.cat((-t[..., half:], t[..., :half]), dim=-1)

    def reference(x: torch.Tensor) -> torch.Tensor:
        cos, sin = tables(x.shape[-2])
        return x * cos + rotate_half(x) * sin

    def bind(entry: Callable[..., torch.Tensor]) -> TensorFn:
        def candidate(x: torch.Tensor) -> torch.Tensor:
            cos, sin = tables(x.shape[-2])
            return entry(x, cos, sin)

        return candidate

    return OpBinding("rope", reference, bind)


def _build_mlp(hidden_size: int, device: str, dtype: torch.dtype) -> OpBinding:
    """SwiGLU MLP: `down(silu(gate(x)) * up(x))`, weights stored as [out, in]."""
    intermediate = hidden_size * _MLP_EXPANSION
    w_gate = _weights((intermediate, hidden_size), device, dtype, GLOBAL_SEED) * 0.05
    w_up = _weights((intermediate, hidden_size), device, dtype, GLOBAL_SEED + 1) * 0.05
    w_down = _weights((hidden_size, intermediate), device, dtype, GLOBAL_SEED + 2) * 0.05

    def reference(x: torch.Tensor) -> torch.Tensor:
        gate = x @ w_gate.T
        up = x @ w_up.T
        act = gate * torch.sigmoid(gate.to(torch.float32)).to(gate.dtype)
        return (act * up) @ w_down.T

    return OpBinding("mlp", reference, lambda e: lambda x: e(x, w_gate, w_up, w_down))


#: op_name -> builder. `verify_kernel` resolves references through this registry so a
#: task spec names an op instead of shipping executable source (spec 12).
OP_REGISTRY: dict[str, Callable[[int, str, torch.dtype], OpBinding]] = {
    "rmsnorm": _build_rmsnorm,
    "layernorm": _build_layernorm,
    "softmax": _build_softmax,
    "silu": _build_silu,
    "rope": _build_rope,
    "mlp": _build_mlp,
}


def build_op(
    op_name: str,
    hidden_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> OpBinding:
    """Look up a reference op by name. Raises KeyError for an unknown op."""
    if op_name not in OP_REGISTRY:
        raise KeyError(f"unknown op {op_name!r}; known ops: {sorted(OP_REGISTRY)}")
    return OP_REGISTRY[op_name](hidden_size, device, dtype)


# --------------------------------------------------------------------------- #
# ADK tool surface
# --------------------------------------------------------------------------- #


def profile_op_by_name(
    op_name: str,
    batch: int,
    seq_len: int,
    hidden_size: int,
) -> dict[str, Any]:
    """Profile a PyTorch operation on the L4 and return its bottleneck fingerprint.

    Benches the reference implementation and combines the measurement with analytic
    FLOP and byte counts to place the op on the L4 roofline. Call this BEFORE writing
    a kernel: the fingerprint says whether to optimize for bandwidth (fuse, coalesce,
    read once) or for math (bigger tiles, tensor cores).

    Args:
        op_name: One of "rmsnorm", "layernorm", "softmax", "silu", "rope", "mlp".
        batch: Batch size of the probe input.
        seq_len: Sequence length of the probe input.
        hidden_size: Hidden dimension of the served model (1536 for Qwen2.5-1.5B).

    Returns:
        A BottleneckFingerprint as a dict, plus "fingerprint_text" (the string used as
        the retrieval key) and "ridge_point_flops_per_byte". On failure the dict also
        carries "error", and the fingerprint is the conservative memory-bound fallback.
    """
    try:
        binding = build_op(op_name, hidden_size)
    except Exception as exc:  # noqa: BLE001 — an unknown op is a fallback, not a crash
        fingerprint = fallback_fingerprint(family_from_name(op_name), hidden_size)
        return _fingerprint_payload(fingerprint) | {"error": str(exc)}

    fingerprint = profile_op(
        binding.reference,
        (batch, seq_len),
        hidden_size,
        op_family=binding.family,
    )
    return _fingerprint_payload(fingerprint)


def family_from_name(op_name: str) -> str:
    """Classify a retrieval `op_family` from an op or CLASS NAME, longest keyword wins.

    The string-taking sibling of `classify_op_family`, which takes a CALLABLE and reads
    names off it. Passing a bare string to that one silently yields "elementwise" —
    `type("RMSNorm").__name__` is `str` — so anything holding a name rather than a
    callable (a module class name from the audit, a `norm_type` from MODEL_REGISTRY)
    must come through here.

    `"RMSNorm"`, `"LayerNorm"` and `"BatchNorm2d"` all return "norm", which is the whole
    mechanism behind cross-model retrieval: the pre-filter is the family, not the name.
    """
    for kw, family in sorted(_FAMILY_KEYWORDS.items(), key=lambda kv: -len(kv[0])):
        if kw in op_name.lower():
            return family
    return "elementwise"


def _fingerprint_payload(fingerprint: BottleneckFingerprint) -> dict[str, Any]:
    return fingerprint.model_dump() | {
        "fingerprint_text": fingerprint.to_embedding_text(),
        "ridge_point_flops_per_byte": RIDGE_POINT,
    }


#: Registered on the Profiler agent (spec 4.2), which writes the result to
#: `session.state["bottleneck_fingerprint"]` via its output_key.
profiler_tool = FunctionTool(profile_op_by_name)


# =========================================================================== #
# Whole-model audit (spec 7, Task 10)
# =========================================================================== #
#
# Everything above answers "why is THIS op slow". The audit answers the question that
# comes first: "which ops in this model are worth a kernel at all". It walks
# `named_modules()`, places every unique module type on the L4 roofline from its own
# declared shapes, and ranks them.
#
# It runs on CPU by design. Loading a model tree and reading `in_features` needs no GPU,
# so `make audit` and the dashboard's audit tab work on a laptop; only the kernel
# generation that follows needs CUDA. On CPU the arithmetic intensities are ANALYTIC
# ESTIMATES and the report says so in as many words. On CUDA one representative instance
# of each type is additionally benched with `do_bench`, which turns the bandwidth column
# from blank into a measurement.
#
# Two estimators now coexist and they do not agree, on purpose:
#
#   `analytic_counts` above counts the MINIMUM traffic a fused kernel must move (read
#   each input once, write the output once). It is the retrieval fingerprint's estimator
#   and it must describe the op, not an implementation.
#
#   `estimate_flops_and_bytes` below counts traffic PER TENSOR TOUCHED, including a
#   weight and bias read per row and the full weight matrix of a Linear. It is the
#   audit's estimator, and it deliberately describes what an unfused eager
#   implementation actually moves — which is what makes a norm look as starved as it is.
#
# Both place a norm far below the ridge point and an MLP far above it, which is the only
# question either one is asked. Where they differ numerically, `analytic_counts` is the
# one wired to Firestore.


#: Module-based op families. DISJOINT from `OP_FAMILIES` above, which is the retrieval
#: taxonomy (norm | rope | mlp | elementwise | reduction) and is keyed off op NAMES.
#: This one is keyed off the module INSTANCE and answers a different question — what
#: kind of arithmetic does this layer do — so it has `linear`, `conv` and `embedding`
#: where the retrieval taxonomy has `mlp` and `rope`. "norm" means the same thing in
#: both, which is the whole reason a skill learned on Qwen2's RMSNorm can be retrieved
#: for GPT-2's LayerNorm.
AUDIT_OP_FAMILIES = ("norm", "linear", "conv", "embedding", "activation", "dropout", "other")

#: Families with nothing to optimize: a lookup, and an identity at eval time. They are
#: reported (so the audit is a complete inventory) but never recommended.
_SKIPPED_FAMILIES = frozenset({"embedding", "dropout"})

#: Families whose memory-bound instances are the highest-value kernel targets: a norm or
#: an activation moves every byte of the activation tensor to do a handful of flops.
_HIGH_VALUE_FAMILIES = frozenset({"norm", "activation"})

#: Pure plumbing — no arithmetic, no parameters of their own. Excluded from the table so
#: an inventory of TARGETS is not half full of ModuleList.
_STRUCTURAL_CONTAINERS: tuple[str, ...] = (
    "ModuleList",
    "ModuleDict",
    "Sequential",
    "ParameterList",
    "ParameterDict",
)

#: FLOPs per element for the activations we recognize. ReLU is a single max; SiLU and
#: GELU are ~3 (exp/tanh approximated as one op each) — they are estimates, and the
#: conclusion they support (elementwise activations are bandwidth-starved) is robust to
#: being wrong by a factor of three.
_ACTIVATION_FLOPS_PER_ELEM: dict[str, int] = {
    "ReLU": 1,
    "ReLU6": 1,
    "LeakyReLU": 1,
    "SiLU": 3,
    "GELU": 3,
    "GELUActivation": 3,
    "NewGELUActivation": 3,
    "Mish": 3,
    "Sigmoid": 2,
    "Tanh": 2,
    # transformers wraps its activations in its own classes rather than using torch.nn:
    # Qwen2MLP.act_fn is a SiLUActivation, GPT-2's is a NewGELUActivation.
    "SiLUActivation": 3,
    "PytorchGELUTanh": 3,
}

#: How deep a composite module is summed before we give up. A decoder layer is 3 levels
#: from its Linears; the cap exists so a pathological tree cannot hang an audit.
_COMPOSITE_MAX_DEPTH = 6


@dataclass(frozen=True)
class AuditEntry:
    """One unique module type in the audited model, placed on the L4 roofline.

    `arithmetic_intensity` is FLOP/byte for ONE instance at the probe shape, and it is
    an estimate unless `measured` is True. `bandwidth_utilization_pct` is 0.0 whenever
    nothing was benched — CPU mode, a family we skip, or a bench that raised — and
    `format_audit_report` prints "n/a" rather than a fabricated 0%.
    """

    module_type: str
    count: int
    op_family: str
    bottleneck: str  # "memory" or "compute"
    arithmetic_intensity: float
    bandwidth_utilization_pct: float
    priority: str  # "HIGH" | "MEDIUM" | "LOW"
    param_shapes: dict[str, list[int]]
    probe_shape: tuple[int, ...] = ()
    measured: bool = False


@dataclass(frozen=True)
class AuditReport:
    """The ranked inventory of one model's optimizable module types.

    `total_modules` counts the instances that made it into `module_entries`, so it
    agrees with the table rather than with `len(list(model.named_modules()))` — the
    difference is the root module and the structural containers, none of which is a
    kernel target.
    """

    model_name: str
    total_modules: int
    unique_types: int
    module_entries: list[AuditEntry]
    top_target: str
    recommendation: str
    device: str = "cpu"
    hidden_size: int = 0
    measured: bool = False
    weights_loaded: bool = False
    gpu_name: str = ""  # the GPU the bandwidth was actually measured on, if any


# --------------------------------------------------------------------------- #
# Module-based classification
# --------------------------------------------------------------------------- #


def classify_op_family_from_module(module: Any) -> str:
    """Classify an `nn.Module` INSTANCE into one of `AUDIT_OP_FAMILIES`.

    The instance-based counterpart to `classify_op_family`, which classifies a callable
    by name into the retrieval taxonomy. Both exist because they answer different
    questions and neither can answer the other's: a name tells you which prior skills
    are relevant, an instance tells you what arithmetic the layer actually does.

    `isinstance` first, so a subclass of `nn.LayerNorm` is still a norm, then the class
    name, which is how `Qwen2RMSNorm` and `NewGELUActivation` — neither of which
    subclasses anything in torch.nn — are recognized.
    """
    from torch import nn

    if isinstance(
        module,
        nn.LayerNorm
        | nn.GroupNorm
        | nn.BatchNorm1d
        | nn.BatchNorm2d
        | nn.BatchNorm3d
        | nn.InstanceNorm1d
        | nn.InstanceNorm2d
        | nn.InstanceNorm3d,
    ):
        return "norm"
    if getattr(nn, "RMSNorm", None) is not None and isinstance(module, nn.RMSNorm):
        return "norm"
    if isinstance(module, nn.Linear | nn.Bilinear | nn.LazyLinear):
        return "linear"
    # transformers' own `Conv1D` (GPT-2's q/k/v and MLP projections) is a Linear with a
    # transposed weight, not a convolution. `nf` is the attribute that identifies it, and
    # getting this wrong left ALL of GPT-2's matmuls unestimated — which in turn reported
    # GPT2Block as memory-bound when its arithmetic is entirely in these 48 modules.
    if hasattr(module, "nf") and getattr(module, "weight", None) is not None:
        return "linear"
    if isinstance(module, nn.Conv1d | nn.Conv2d | nn.Conv3d | nn.ConvTranspose2d):
        return "conv"
    if isinstance(module, nn.Embedding | nn.EmbeddingBag):
        return "embedding"
    if isinstance(module, nn.Dropout | nn.Dropout1d | nn.Dropout2d | nn.Dropout3d):
        return "dropout"
    if isinstance(
        module, nn.SiLU | nn.GELU | nn.ReLU | nn.ReLU6 | nn.LeakyReLU | nn.Sigmoid | nn.Tanh
    ):
        return "activation"

    name = type(module).__name__
    lowered = name.lower()
    if "norm" in lowered:
        return "norm"
    if name in _ACTIVATION_FLOPS_PER_ELEM or "activation" in lowered:
        return "activation"
    if "dropout" in lowered:
        return "dropout"
    # "rotary"/"rope" excluded deliberately: Qwen2RotaryEmbedding is not a lookup
    # table, and calling it one would file it under a family we skip for the wrong
    # reason. It IS unpatchable, but because it has no forward to rebind usefully —
    # see the RoPE note in inference_server/patchable_ops.py.
    if "embedding" in lowered and not _has_children(module):
        if "rotary" in lowered or "rope" in lowered:
            return "other"
        return "embedding"
    return "other"


def _has_children(module: Any) -> bool:
    return next(iter(module.children()), None) is not None


# --------------------------------------------------------------------------- #
# Per-op-family FLOP / byte estimation (spec Task 10 Part B)
# --------------------------------------------------------------------------- #


def estimate_flops_and_bytes(
    module: Any,
    input_shape: tuple[int, ...],
    element_size: int | None = None,
) -> tuple[int, int]:
    """Estimated (FLOPs, bytes moved) for one forward pass of `module` at `input_shape`.

    Bytes are counted PER TENSOR TOUCHED by an unfused implementation — a norm reads its
    input, its weight (and bias, if it has one) and writes its output, so four passes
    over an [N, H] tensor, not two. That is deliberately not the minimum a fused kernel
    would move (`analytic_counts` computes that): the audit's job is to show how much
    traffic is currently being spent, which is the headroom a kernel can recover.

    Rules, one per family:

      RMSNorm    (N rows x H)      5 flop/elem,  3 tensors
      LayerNorm  (N rows x H)      7 flop/elem,  4 tensors (the extra one is `bias`)
      BatchNorm  (B x C x S)       7 flop/elem,  4 tensors
      Linear     (B x M -> B x N)  2*B*M*N flops; B*M + M*N + B*N elements
      Conv2d                       2*B*C_out*S*S*(C_in/groups)*K*K flops;
                                   input + weight + output elements
      activation (N elems)         1 flop/elem for ReLU, ~3 for SiLU/GELU, 2 tensors
      embedding, dropout           skipped — a gather, and an identity at eval time

    A module with children and no rule of its own is summed over its children, each at
    its own derived probe shape. That is what gives a `Qwen2MLP` or a decoder layer a
    real arithmetic intensity instead of a zero.

    Args:
        module: The `nn.Module` to estimate. Never called, only inspected.
        input_shape: Probe input shape, e.g. `(1, 512, 1536)`.
        element_size: Bytes per element. Defaults to the module's own parameter dtype,
            falling back to 2 (fp16) when it has no parameters.

    Returns:
        `(flops, bytes)`, both non-negative ints. `(0, 0)` means "nothing to estimate" —
        a skipped family, or a leaf this function does not recognize — and callers must
        treat it as unknown, not as free.
    """
    from torch import nn

    shape = tuple(int(d) for d in input_shape if int(d) > 0)
    elem = int(element_size) if element_size else _module_element_size(module)
    family = classify_op_family_from_module(module)

    if family in _SKIPPED_FAMILIES:
        return 0, 0

    if family == "norm":
        return _norm_counts(module, shape, elem)
    if family == "linear":
        return _linear_counts(module, shape, elem)
    if family == "conv" and isinstance(module, nn.Conv1d | nn.Conv2d | nn.Conv3d):
        return _conv_counts(module, shape, elem)
    if family == "activation":
        numel = _numel(shape)
        per_elem = _ACTIVATION_FLOPS_PER_ELEM.get(type(module).__name__, 3)
        return per_elem * numel, 2 * numel * elem

    return _composite_counts(module, shape, elem, depth=0)


def _norm_counts(module: Any, shape: tuple[int, ...], elem: int) -> tuple[int, int]:
    """RMSNorm: 5 flop/elem over 3 tensors. LayerNorm / BatchNorm: 7 over 4.

    The split is the bias: RMSNorm has none, so there is one fewer tensor to read and
    no mean-subtraction to do. Which is exactly why they need different adapters, and
    why a single hard-coded norm bridge was never going to cover both.
    """
    from torch import nn

    numel = _numel(shape)
    if numel == 0:
        return 0, 0
    has_bias = getattr(module, "bias", None) is not None
    is_rms = not has_bias and not isinstance(
        module, nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d
    )
    flops_per_elem, tensors = (5, 3) if is_rms else (7, 4)
    return flops_per_elem * numel, tensors * numel * elem


def _linear_counts(module: Any, shape: tuple[int, ...], elem: int) -> tuple[int, int]:
    """2*B*M*N flops for the matmul; B*M + M*N + B*N elements of traffic.

    The `M*N` term is the weight matrix, read once. It is what pushes a Linear's
    arithmetic intensity above the ridge point at any realistic batch: the weights are
    amortized over every row.
    """
    in_features = int(getattr(module, "in_features", 0) or 0)
    out_features = int(getattr(module, "out_features", 0) or 0)
    if in_features <= 0 or out_features <= 0:
        # transformers' Conv1D declares neither, and stores its weight as [in, out] —
        # the transpose of nn.Linear's [out, in]. Both formulas below are symmetric in
        # (M, N), so the order does not matter and one fallback covers both.
        weight = getattr(module, "weight", None)
        if weight is None or weight.dim() != 2:
            return 0, 0
        in_features, out_features = (int(d) for d in weight.shape)
    rows = max(1, _numel(shape[:-1]) if len(shape) > 1 else 1)
    flops = 2 * rows * in_features * out_features
    elements = rows * in_features + in_features * out_features + rows * out_features
    return flops, elements * elem


def _conv_counts(module: Any, shape: tuple[int, ...], elem: int) -> tuple[int, int]:
    """2 flops per MAC over the output volume; input + weight + output of traffic.

    `in_channels` is divided by `groups`, which is a no-op for ResNet-50 (every conv is
    dense) and simply correct for a depthwise conv, where the spec's ungrouped formula
    would overstate the FLOPs by the group count.
    """
    weight = getattr(module, "weight", None)
    if weight is None or len(shape) < 3:
        return 0, 0

    batch = max(1, shape[0])
    out_channels = int(getattr(module, "out_channels", weight.shape[0]))
    in_channels = int(getattr(module, "in_channels", shape[1]))
    groups = max(1, int(getattr(module, "groups", 1) or 1))
    kernel = tuple(int(k) for k in getattr(module, "kernel_size", ()) or ())
    if not kernel:
        return 0, 0

    spatial_out = _conv_spatial_out(module, shape[2:], kernel)
    out_volume = _numel(spatial_out)
    if out_volume == 0:
        return 0, 0

    macs_per_output = (in_channels // groups) * _numel(kernel)
    flops = 2 * batch * out_channels * out_volume * macs_per_output
    elements = (
        _numel(shape)  # input
        + out_channels * (in_channels // groups) * _numel(kernel)  # weight
        + batch * out_channels * out_volume  # output
    )
    return flops, elements * elem


def _conv_spatial_out(
    module: Any, spatial_in: tuple[int, ...], kernel: tuple[int, ...]
) -> tuple[int, ...]:
    """Output spatial dims from the standard convolution geometry."""
    stride = _as_tuple(getattr(module, "stride", 1), len(spatial_in))
    padding = _as_tuple(getattr(module, "padding", 0), len(spatial_in))
    dilation = _as_tuple(getattr(module, "dilation", 1), len(spatial_in))
    kernel = _as_tuple(kernel, len(spatial_in))

    out: list[int] = []
    for i, size in enumerate(spatial_in):
        effective = dilation[i] * (kernel[i] - 1) + 1
        out.append(max(0, (size + 2 * padding[i] - effective) // max(1, stride[i]) + 1))
    return tuple(out)


def _as_tuple(value: Any, length: int) -> tuple[int, ...]:
    """Normalize a conv hyperparameter (int, tuple, or "same") to a tuple of ints."""
    if isinstance(value, str):
        return (0,) * length
    if isinstance(value, int):
        return (value,) * length
    items = tuple(int(v) for v in value)
    if len(items) < length:
        items = items + (items[-1] if items else 0,) * (length - len(items))
    return items[:length]


def _composite_counts(
    module: Any, shape: tuple[int, ...], elem: int, depth: int
) -> tuple[int, int]:
    """Sum a composite module's immediate children, each at its own derived shape.

    An approximation with one known weakness: a child whose shape depends on a sibling's
    output — the activation between an MLP's up- and down-projection runs at the
    intermediate width, not the hidden width — is estimated at the parent's input shape
    and therefore understated. It does not move a composite across the ridge point,
    because the Linears dominate both sides of the ratio.
    """
    if depth >= _COMPOSITE_MAX_DEPTH:
        return 0, 0

    total_flops = 0
    total_bytes = 0
    for child in module.children():
        child_shape = _child_probe_shape(child, shape)
        family = classify_op_family_from_module(child)
        if family in _SKIPPED_FAMILIES:
            continue
        if family == "other" or (family in {"linear", "conv"} and _has_children(child)):
            flops, byte_count = _composite_counts(child, child_shape, elem, depth + 1)
        else:
            flops, byte_count = estimate_flops_and_bytes(child, child_shape, elem)
        total_flops += flops
        total_bytes += byte_count
    return total_flops, total_bytes


def _child_probe_shape(child: Any, parent_shape: tuple[int, ...]) -> tuple[int, ...]:
    """A child's own declared shape where it has one, else the parent's input shape."""
    hidden = parent_shape[-1] if parent_shape else 0
    derived = representative_input_shape(
        child,
        hidden_size=hidden,
        batch=parent_shape[0] if parent_shape else AUDIT_PROBE_BATCH,
        seq_len=parent_shape[1] if len(parent_shape) > 2 else AUDIT_PROBE_SEQ,
    )
    return derived or parent_shape


def _module_element_size(module: Any) -> int:
    """Bytes per element of the module's own parameters; 2 (fp16) when it has none."""
    for param in module.parameters(recurse=True):
        try:
            return int(param.element_size())
        except Exception:  # noqa: BLE001 — a meta/fake tensor still has a dtype below
            break
    return 2


def _numel(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= max(0, int(dim))
    return total if shape else 0


# --------------------------------------------------------------------------- #
# Probe shapes
# --------------------------------------------------------------------------- #


def representative_input_shape(
    module: Any,
    hidden_size: int,
    batch: int = AUDIT_PROBE_BATCH,
    seq_len: int = AUDIT_PROBE_SEQ,
    spatial: int = AUDIT_PROBE_SPATIAL,
) -> tuple[int, ...]:
    """One plausible input shape for `module`, taken from its own declared dimensions.

    Read from the module wherever it declares them (`in_features`, `num_features`,
    `normalized_shape`), so the estimate is about this model's real widths and not about
    a guessed hidden size. Shape-less modules — an activation, a dropout — fall back to
    `(batch, seq_len, hidden_size)`, which is the only thing there is to go on.
    """
    from torch import nn

    hidden = max(1, int(hidden_size or 1))

    if isinstance(module, nn.Linear | nn.LazyLinear):
        in_features = int(getattr(module, "in_features", 0) or hidden)
        return (batch, seq_len, max(1, in_features))

    if isinstance(module, nn.Conv2d | nn.ConvTranspose2d):
        return (batch, max(1, int(module.in_channels)), spatial, spatial)
    if isinstance(module, nn.Conv1d):
        return (batch, max(1, int(module.in_channels)), seq_len)
    if isinstance(module, nn.Conv3d):
        depth = max(1, spatial // 4)
        return (batch, max(1, int(module.in_channels)), depth, spatial, spatial)

    if isinstance(module, nn.BatchNorm2d | nn.InstanceNorm2d | nn.GroupNorm):
        channels = int(getattr(module, "num_features", 0) or getattr(module, "num_channels", 0))
        return (batch, max(1, channels or hidden), spatial, spatial)
    if isinstance(module, nn.BatchNorm1d | nn.InstanceNorm1d):
        return (batch, max(1, int(module.num_features)), seq_len)
    if isinstance(module, nn.BatchNorm3d | nn.InstanceNorm3d):
        depth = max(1, spatial // 4)
        return (batch, max(1, int(module.num_features)), depth, spatial, spatial)

    normalized = getattr(module, "normalized_shape", None)
    if normalized is not None:
        dims = (
            (int(normalized),) if isinstance(normalized, int) else tuple(int(d) for d in normalized)
        )
        return (batch, seq_len, *dims)

    if isinstance(module, nn.Embedding | nn.EmbeddingBag):
        return (batch, seq_len)

    return (batch, seq_len, hidden)


# --------------------------------------------------------------------------- #
# The audit proper
# --------------------------------------------------------------------------- #


def audit_model(model_name_or_path: str, device: str = "cpu") -> AuditReport:
    """Walk a HuggingFace model and rank its module types as kernel targets.

    Args:
        model_name_or_path: A `MODEL_REGISTRY` key ("gpt2", "qwen2.5-1.5b",
            "resnet50") or any HuggingFace model id / local path.
        device: "cpu" for analytic estimates only — no GPU required, fp32 weights — or
            "cuda" to additionally bench one representative instance of each module type
            with `do_bench` and report measured bandwidth utilization.

    Returns:
        An `AuditReport`, sorted HIGH priority first and by instance count within a
        priority. `report.measured` says whether the bandwidth column means anything.

    Raises:
        Whatever loading raises — a model that cannot be loaded at all is a real error
        and must not be reported as an empty audit.
    """
    import torch

    hf_id = resolve_model_id(model_name_or_path)
    on_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    dtype = torch.float16 if on_cuda else torch.float32

    model, weights_loaded = _load_for_audit(hf_id, on_cuda=on_cuda, dtype=dtype)
    hidden_size = _model_hidden_size(model, model_name_or_path)

    entries = _audit_entries(model, hidden_size, on_cuda=on_cuda, dtype=dtype)
    top_target = entries[0].module_type if entries else ""

    return AuditReport(
        model_name=hf_id,
        total_modules=sum(entry.count for entry in entries),
        unique_types=len(entries),
        module_entries=entries,
        top_target=top_target,
        recommendation=build_recommendation(entries, hf_id),
        device="cuda" if on_cuda else "cpu",
        hidden_size=hidden_size,
        measured=any(entry.measured for entry in entries),
        weights_loaded=weights_loaded,
        gpu_name=_gpu_name() if on_cuda else "",
    )


def _gpu_name() -> str:
    try:
        import torch

        return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001 — an unnameable GPU is still a GPU
        return "unknown CUDA device"


def _load_for_audit(hf_id: str, on_cuda: bool, dtype: Any) -> tuple[Any, bool]:
    """Get a module tree to walk. Returns `(model, weights_loaded)`.

    On CPU the audit reads `in_features`, `normalized_shape` and parameter SHAPES — it
    never runs a forward pass — so it does not need the weights, and downloading 3.4 GB
    of them to count 57 RMSNorms would make `make audit` unusable on a laptop and
    impossible offline. The tree is therefore built from `config.json` alone under
    `torch.device("meta")`: exact classes, exact shapes, zero bytes allocated. Same trick
    `verifier/adapter_mapping._probe_instance` already uses on Qwen2RMSNorm.

    On CUDA the weights are real, because `do_bench` cannot time a meta tensor.

    If config-only construction fails (a model whose `__init__` needs storage, an
    architecture `AutoModel` cannot map), this falls back to a real `from_pretrained` on
    the CPU rather than giving up on the audit.
    """
    import torch
    from transformers import AutoConfig, AutoModel

    if on_cuda:
        model = AutoModel.from_pretrained(hf_id, dtype=dtype).to("cuda").eval()
        return model, True

    try:
        config = AutoConfig.from_pretrained(hf_id)
        with torch.device("meta"):
            return AutoModel.from_config(config).eval(), False
    except Exception:  # noqa: BLE001 — fall through to the real thing, stated below
        return AutoModel.from_pretrained(hf_id, dtype=dtype).eval(), True


def resolve_model_id(model_name_or_path: str) -> str:
    """`MODEL_REGISTRY` key -> HuggingFace id. Anything unknown is passed through."""
    key = str(model_name_or_path or "").strip()
    entry = MODEL_REGISTRY.get(key.lower())
    return str(entry["hf_id"]) if entry else key


def _model_hidden_size(model: Any, requested: str) -> int:
    """The model's hidden width, from its config, the registry, or a last-resort guess.

    ResNet's config has `hidden_sizes` (a list per stage) and no `hidden_size`, which is
    why the registry value is consulted second rather than trusted first.
    """
    config = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    registry = MODEL_REGISTRY.get(str(requested or "").strip().lower())
    if registry:
        return int(registry["hidden_size"])  # type: ignore[arg-type]
    sizes = getattr(config, "hidden_sizes", None)
    if isinstance(sizes, list | tuple) and sizes:
        return int(sizes[-1])
    return AUDIT_PROBE_SEQ


def _audit_entries(
    model: Any,
    hidden_size: int,
    on_cuda: bool,
    dtype: Any,
) -> list[AuditEntry]:
    """One entry per unique module class, ranked. The root and containers are excluded.

    The root is excluded because "the whole model" is not a swappable target, and the
    structural containers because `ModuleList` does no arithmetic. Everything else is
    reported, including composites like `Qwen2MLP` — a fusable block IS a target, and
    `PATCHABLE_OPS` already knows how to swap that one.
    """
    # One representative per class, so a `Linear` row describes whichever projection was
    # encountered first — not an average over all 196 of them. That is what the spec
    # asks for and it is the right call for a triage table (the regime is the same for
    # all of them), but the AI figure in a `Linear` row is one instance's, not the type's.
    grouped: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not name:  # the root module
            continue
        type_name = type(module).__name__
        if type_name in _STRUCTURAL_CONTAINERS:
            continue
        record = grouped.setdefault(type_name, {"count": 0, "module": module})
        record["count"] += 1

    entries = [
        _build_entry(
            type_name,
            record["module"],
            record["count"],
            hidden_size,
            on_cuda=on_cuda,
            dtype=dtype,
        )
        for type_name, record in grouped.items()
    ]
    return sorted(entries, key=_entry_sort_key)


_PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def _entry_sort_key(entry: AuditEntry) -> tuple[int, int, str]:
    """HIGH first, then most instances first, then by name so the order is stable."""
    return (_PRIORITY_ORDER.get(entry.priority, 3), -entry.count, entry.module_type)


def _build_entry(
    type_name: str,
    module: Any,
    count: int,
    hidden_size: int,
    on_cuda: bool,
    dtype: Any,
) -> AuditEntry:
    """Place one representative module on the roofline."""
    family = classify_op_family_from_module(module)
    shape = representative_input_shape(module, hidden_size)
    element_size = _dtype_element_size(dtype)
    flops, byte_count = estimate_flops_and_bytes(module, shape, element_size)

    has_estimate = byte_count > 0
    intensity = (flops / byte_count) if has_estimate else 0.0
    is_memory_bound = intensity < RIDGE_POINT

    bandwidth_pct = 0.0
    measured = False
    if on_cuda and has_estimate and family not in _SKIPPED_FAMILIES:
        bandwidth_pct = _measure_bandwidth_pct(module, shape, byte_count, dtype)
        measured = bandwidth_pct > 0.0

    return AuditEntry(
        module_type=type_name,
        count=count,
        op_family=family,
        bottleneck="memory" if is_memory_bound else "compute",
        arithmetic_intensity=intensity,
        bandwidth_utilization_pct=bandwidth_pct,
        priority=assign_priority(family, is_memory_bound, has_estimate),
        param_shapes=_param_shapes(module),
        probe_shape=shape,
        measured=measured,
    )


def assign_priority(op_family: str, is_memory_bound: bool, has_estimate: bool = True) -> str:
    """HIGH for a memory-bound norm or activation, MEDIUM for anything else
    memory-bound, LOW for compute-bound.

    Two refinements on top of that rule, both because "memory-bound" is the DEFAULT
    answer when there is no estimate and it would otherwise promote things that cannot
    be optimized at all:

    - a family we skip (an embedding gather, a dropout that is the identity at eval) is
      always LOW, however bandwidth-bound it looks;
    - a module we could not estimate (`has_estimate=False`, i.e. zero bytes) is LOW,
      because arithmetic intensity 0.0 is an absence of information, not a bottleneck.
    """
    if not has_estimate or op_family in _SKIPPED_FAMILIES:
        return "LOW"
    if not is_memory_bound:
        return "LOW"
    return "HIGH" if op_family in _HIGH_VALUE_FAMILIES else "MEDIUM"


def _param_shapes(module: Any) -> dict[str, list[int]]:
    """Direct parameters and buffers of the representative module, name -> shape."""
    shapes: dict[str, list[int]] = {}
    for name, param in module.named_parameters(recurse=False):
        shapes[name] = [int(d) for d in param.shape]
    for name, buffer in module.named_buffers(recurse=False):
        if buffer is not None:
            shapes.setdefault(name, [int(d) for d in buffer.shape])
    return shapes


def _dtype_element_size(dtype: Any) -> int:
    try:
        return int(torch.empty(0, dtype=dtype).element_size())
    except Exception:  # noqa: BLE001 — an unusable dtype falls back to fp16
        return 2


def _measure_bandwidth_pct(
    module: Any,
    shape: tuple[int, ...],
    byte_count: int,
    dtype: Any,
) -> float:
    """Benched bandwidth as a percentage of the L4's 300 GB/s. 0.0 if it cannot be run.

    One `do_bench` per unique module TYPE, not per instance: 57 identical Qwen2RMSNorms
    would cost 57 identical measurements. Warmup and rep come from `bench_kernel`, which
    enforces warmup >= 150 — the documented default of 25 underestimates by ~30%.

    Any failure here is a blank cell, never an exception: a module that will not accept
    a synthetic probe (an attention block wanting a mask, a head wanting labels) must not
    take the rest of the audit down with it.
    """
    try:
        probe = torch.randn(*shape, device="cuda", dtype=dtype)
        with torch.no_grad():
            median_ms = bench_kernel(lambda: module(probe))
        if median_ms <= 0:
            return 0.0
        gbps = byte_count / (median_ms / 1000.0) / 1e9
        return max(0.0, min(100.0, gbps / L4_MEM_BW_GBPS * 100.0))
    except Exception:  # noqa: BLE001 — an unbenchable module is a blank cell
        return 0.0


def build_recommendation(entries: Sequence[AuditEntry], model_name: str) -> str:
    """One sentence naming the target and the reason, or saying there isn't one."""
    if not entries:
        return f"No profilable modules found in {model_name}."

    top = entries[0]
    if top.priority == "LOW":
        return (
            f"Nothing in {model_name} is memory-bound at the probe shapes: the heaviest "
            f"type is {top.module_type} at {top.arithmetic_intensity:.0f} FLOP/byte, "
            f"above the L4 ridge point of {RIDGE_POINT:.0f} — tile and tensor-core work, "
            "not fusion."
        )

    return (
        f"Start with {top.module_type}: {top.count} memory-bound instance"
        f"{'s' if top.count != 1 else ''} at {top.arithmetic_intensity:.2f} FLOP/byte, "
        f"{RIDGE_POINT / max(top.arithmetic_intensity, 1e-9):.0f}x below the L4 ridge "
        f"point of {RIDGE_POINT:.0f} — a fused single-pass Triton kernel recovers the "
        "traffic the unfused version wastes."
    )


# --------------------------------------------------------------------------- #
# Report formatting
# --------------------------------------------------------------------------- #

#: (heading, width). The first column absorbs whatever is left of AUDIT_REPORT_WIDTH
#: after the borders and the fixed columns, so changing that constant cannot break the
#: box-drawing alignment.
_AUDIT_COLUMNS: tuple[tuple[str, int], ...] = (
    ("Count", 7),
    ("Regime", 9),
    ("AI (F/B)", 11),
    ("BW %", 7),
    ("Priority", 12),
)

_PRIORITY_GLYPH = {"HIGH": "★★★ HIGH", "MEDIUM": "★★ MEDIUM", "LOW": "☆ LOW"}


def format_audit_report(report: AuditReport) -> str:
    """Render an `AuditReport` as a fixed-width ASCII table.

    The header states the device and, on CPU, that every arithmetic intensity is an
    ANALYTIC ESTIMATE. That line is not decoration: a bandwidth number nobody measured
    reads exactly like one somebody did, and this project's whole claim rests on the
    difference (red line #3).
    """
    fixed = sum(width for _, width in _AUDIT_COLUMNS)
    name_width = max(12, AUDIT_REPORT_WIDTH - fixed - (len(_AUDIT_COLUMNS) + 2))
    widths = (name_width, *[width for _, width in _AUDIT_COLUMNS])
    headings = ("Module Type", *[name for name, _ in _AUDIT_COLUMNS])

    title = f" KernelSmith Model Audit: {report.model_name} "
    lines = [
        "═══" + title + "═" * max(0, AUDIT_REPORT_WIDTH - len(title) - 3),
        f"Modules scanned: {report.total_modules} profiled, "
        f"{report.unique_types} unique types"
        + (f", hidden_size={report.hidden_size}" if report.hidden_size else ""),
        f"Hardware: NVIDIA L4 ({L4_VRAM_GB}GB, {L4_MEM_BW_GBPS:.0f} GB/s, "
        f"{L4_FP16_TFLOPS} TFLOPS FP16) · ridge {RIDGE_POINT:.0f} FLOP/byte",
        _mode_line(report),
        _rule(widths, "┌", "┬", "┐"),
        _row(headings, widths),
        _rule(widths, "├", "┼", "┤"),
    ]

    for entry in report.module_entries:
        lines.append(
            _row(
                (
                    entry.module_type,
                    str(entry.count),
                    # An entry with no estimate has `bottleneck == "memory"` by
                    # conservative default (as `fallback_fingerprint` does), but the
                    # table must not print a regime nothing was computed from.
                    entry.bottleneck if entry.arithmetic_intensity > 0 else "—",
                    _fmt_intensity(entry.arithmetic_intensity),
                    _fmt_bandwidth(entry.bandwidth_utilization_pct),
                    _PRIORITY_GLYPH.get(entry.priority, entry.priority),
                ),
                widths,
            )
        )
    if not report.module_entries:
        lines.append(_row(("(no profilable modules)", "", "", "", "", ""), widths))

    lines.append(_rule(widths, "└", "┴", "┘"))
    lines.append(f"Top target: {report.top_target or '(none)'}")
    lines.append(f"Recommendation: {report.recommendation}")
    return "\n".join(lines)


def _mode_line(report: AuditReport) -> str:
    """Says, in one line, whether the numbers below were measured or estimated."""
    if report.measured:
        line = f"Mode: {report.device} — intensity analytic, bandwidth MEASURED (do_bench)"
        # The BW% column is a fraction of the L4's 300 GB/s. Measured anywhere else that
        # denominator is wrong, and a percentage nobody can source is exactly the kind of
        # number this project refuses to print without a caveat (red line #3).
        if report.gpu_name and "L4" not in report.gpu_name:
            line += (
                f"\n  ⚠ measured on {report.gpu_name}, but BW % is against the L4's "
                f"{L4_MEM_BW_GBPS:.0f} GB/s — not comparable to an L4 run"
            )
        return line
    weights = "real weights" if report.weights_loaded else "shapes only, meta device"
    return f"Mode: {report.device} — all values ESTIMATED analytically ({weights})"


def _rule(widths: Sequence[int], left: str, mid: str, right: str) -> str:
    return left + mid.join("─" * width for width in widths) + right


def _row(cells: Sequence[str], widths: Sequence[int]) -> str:
    parts = []
    for cell, width in zip(cells, widths, strict=True):
        text = str(cell)
        if len(text) > width - 2:
            text = text[: max(0, width - 3)] + "…"
        parts.append(text.center(width) if len(text) < width else text[:width])
    return "│" + "│".join(parts) + "│"


def _fmt_intensity(value: float) -> str:
    if value <= 0:
        return "n/a"
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _fmt_bandwidth(value: float) -> str:
    return "n/a" if value <= 0 else f"{value:.0f}%"


# --------------------------------------------------------------------------- #
# ADK tool surface for the audit
# --------------------------------------------------------------------------- #


def audit_model_for_agent(model_name_or_path: str, device: str = "cpu") -> dict[str, Any]:
    """Audit a HuggingFace model to identify all module types, classify bottlenecks via
    roofline analysis, and recommend optimization targets.

    Walks every module in the model, groups them by class, and places each class on the
    L4 roofline from its own declared shapes. Call this BEFORE choosing an op to
    optimize: it says which module types are memory-bound (fuse them) and which are
    compute-bound (leave them to cuBLAS), ranked by how much is on the table.

    Args:
        model_name_or_path: "qwen2.5-1.5b", "gpt2", "resnet50", or any HuggingFace
            model id.
        device: "cpu" for analytic estimates (no GPU needed), "cuda" to also measure
            achieved bandwidth with do_bench.

    Returns:
        {"model_name", "total_modules", "unique_types", "top_target", "recommendation",
        "device", "measured", "module_entries": [...], "report_text"} — or "error" with
        the reason if the model could not be loaded.
    """
    from dataclasses import asdict

    try:
        report = audit_model(model_name_or_path, device=device)
    except Exception as exc:  # noqa: BLE001 — an unloadable model is a message, not a crash
        return {
            "model_name": model_name_or_path,
            "total_modules": 0,
            "unique_types": 0,
            "module_entries": [],
            "top_target": "",
            "recommendation": "",
            "error": f"{type(exc).__name__}: {exc}",
        }

    payload = asdict(report)
    payload["module_entries"] = [
        {k: v for k, v in entry.items() if k != "probe_shape"}
        | {"probe_shape": list(entry.get("probe_shape") or ())}
        for entry in payload["module_entries"]
    ]
    payload["report_text"] = format_audit_report(report)
    return payload


#: Published as `audit_model` so the agent's prompt and this tool agree on one name,
#: while the Python callable stays a dict-returning wrapper — an ADK FunctionTool
#: cannot return a dataclass.
audit_model_for_agent.__name__ = "audit_model"
audit_tool = FunctionTool(audit_model_for_agent)
