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
    GLOBAL_SEED,
    L4_FP16_TFLOPS,
    L4_MEM_BW_GBPS,
    L4_SM_COUNT,
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
        fingerprint = fallback_fingerprint(_family_from_name(op_name), hidden_size)
        return _fingerprint_payload(fingerprint) | {"error": str(exc)}

    fingerprint = profile_op(
        binding.reference,
        (batch, seq_len),
        hidden_size,
        op_family=binding.family,
    )
    return _fingerprint_payload(fingerprint)


def _family_from_name(op_name: str) -> str:
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
