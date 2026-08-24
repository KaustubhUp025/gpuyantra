"""Roofline fingerprint with do_bench mocked (spec 7 / 13.1).

No GPU here: `do_bench` is replaced with a known median, the probe tensors live on the
CPU, and the reference op is never actually executed. What is under test is the
ARITHMETIC — the analytic FLOP/byte counts, the ridge-point comparison, and the
fallback — not the timer, which has its own contract in verifier/timing.py.
"""

import pytest
import torch
import triton.testing

from kernelsmith.config import (
    L4_FP16_TFLOPS,
    L4_MEM_BW_GBPS,
    PROFILER_FALLBACK_AI,
    PROFILER_MAX_TILE,
    PROFILER_MIN_TILE,
)
from kernelsmith.memory.schemas import BottleneckFingerprint
from kernelsmith.tools.profiler_tool import (
    RIDGE_POINT,
    analytic_counts,
    build_op,
    classify_op_family,
    compute_tile_hint,
    estimate_occupancy,
    fallback_fingerprint,
    profile_op,
    profile_op_by_name,
)

MEDIAN_MS = 0.5
SHAPE = (8, 512)
HIDDEN = 1024


@pytest.fixture
def mock_bench(monkeypatch):
    """do_bench always reports MEDIAN_MS; record the calls so we can assert warmup."""
    calls = []

    def fake_do_bench(fn, warmup=25, rep=100, return_mode="mean", **kwargs):
        calls.append({"warmup": warmup, "rep": rep, "return_mode": return_mode})
        return MEDIAN_MS

    monkeypatch.setattr(triton.testing, "do_bench", fake_do_bench)
    return calls


def cpu_op(name: str):
    """A reference binding on the CPU, so no GPU is needed to build the probe."""
    return build_op(name, HIDDEN, "cpu", torch.float16)


# --------------------------------------------------------------------------- #
# Ridge point
# --------------------------------------------------------------------------- #


def test_ridge_point_is_the_l4_flop_per_byte_crossover():
    """~101 FLOP/byte: 30.3 TFLOP/s of fp16 math against 300.1 GB/s of bandwidth."""
    assert pytest.approx(L4_FP16_TFLOPS * 1e12 / (L4_MEM_BW_GBPS * 1e9)) == RIDGE_POINT
    assert 90 < RIDGE_POINT < 110


# --------------------------------------------------------------------------- #
# Classification: low AI -> memory-bound, high AI -> compute-bound
# --------------------------------------------------------------------------- #


def test_low_arithmetic_intensity_op_is_memory_bound(mock_bench):
    """RMSNorm moves 4 bytes per element to do 5 flops: AI ~1.25, far below the ridge."""
    fingerprint = profile_op(cpu_op("rmsnorm").reference, SHAPE, HIDDEN, device="cpu")

    assert isinstance(fingerprint, BottleneckFingerprint)
    assert fingerprint.op_family == "norm"
    assert fingerprint.arithmetic_intensity == pytest.approx(1.25)
    assert fingerprint.arithmetic_intensity < RIDGE_POINT
    assert fingerprint.is_memory_bound is True
    assert fingerprint.is_compute_bound is False


def test_high_arithmetic_intensity_op_is_compute_bound(mock_bench):
    """An MLP block reloads its weights once but does rows x H x I flops with them."""
    fingerprint = profile_op(cpu_op("mlp").reference, SHAPE, HIDDEN, device="cpu")

    assert fingerprint.op_family == "mlp"
    assert fingerprint.arithmetic_intensity > RIDGE_POINT
    assert fingerprint.is_compute_bound is True
    assert fingerprint.is_memory_bound is False


def test_memory_and_compute_bound_are_never_both_true(mock_bench):
    for op_name in ("rmsnorm", "softmax", "silu", "rope", "mlp"):
        fingerprint = profile_op(cpu_op(op_name).reference, SHAPE, HIDDEN, device="cpu")
        assert fingerprint.is_memory_bound != fingerprint.is_compute_bound


# --------------------------------------------------------------------------- #
# Derived metrics
# --------------------------------------------------------------------------- #


def test_memory_throughput_is_bytes_moved_over_the_measured_median(mock_bench):
    fingerprint = profile_op(cpu_op("rmsnorm").reference, SHAPE, HIDDEN, device="cpu")

    numel = SHAPE[0] * SHAPE[1] * HIDDEN
    expected_gbps = (numel * 2 * 2) / (MEDIAN_MS / 1000.0) / 1e9
    assert fingerprint.memory_throughput_gbps == pytest.approx(expected_gbps)


def test_bench_runs_with_the_mandated_warmup(mock_bench):
    """Red line #9: warmup 25 underestimates by ~30%, which would inflate every speedup."""
    profile_op(cpu_op("rmsnorm").reference, SHAPE, HIDDEN, device="cpu")
    assert mock_bench, "do_bench was never called"
    assert all(call["warmup"] >= 150 for call in mock_bench)
    assert all(call["return_mode"] == "median" for call in mock_bench)


def test_analytic_counts_use_minimum_traffic_not_eager_traffic():
    """The byte count is a property of the op, not of the implementation we replace."""
    flops, bytes_moved = analytic_counts("norm", numel=1000, hidden_size=100, element_size=2)
    assert flops == 5000
    assert bytes_moved == 4000  # read x + write y, fp16


# --------------------------------------------------------------------------- #
# Fallback (spec 7)
# --------------------------------------------------------------------------- #


def test_fallback_fingerprint_when_bench_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(triton.testing, "do_bench", boom)
    fingerprint = profile_op(cpu_op("rmsnorm").reference, SHAPE, HIDDEN, device="cpu")

    assert fingerprint.is_memory_bound is True
    assert fingerprint.is_compute_bound is False
    assert fingerprint.arithmetic_intensity == PROFILER_FALLBACK_AI
    # 0.0 GB/s is the tell that nothing was measured.
    assert fingerprint.memory_throughput_gbps == 0.0
    assert fingerprint.op_family == "norm"


def test_fallback_fingerprint_when_the_probe_cannot_be_allocated(monkeypatch):
    """No GPU, no VRAM, wrong dtype — all the same conservative answer."""
    fingerprint = profile_op(cpu_op("rmsnorm").reference, SHAPE, HIDDEN, device="cuda:7")
    assert fingerprint == fallback_fingerprint("norm", HIDDEN)


def test_fallback_fingerprint_when_do_bench_returns_zero(monkeypatch):
    """A zero median is a broken measurement, not an infinitely fast op."""
    monkeypatch.setattr(triton.testing, "do_bench", lambda *a, **k: 0.0)
    fingerprint = profile_op(cpu_op("silu").reference, SHAPE, HIDDEN, device="cpu")
    assert fingerprint.arithmetic_intensity == PROFILER_FALLBACK_AI
    assert fingerprint.memory_throughput_gbps == 0.0


def test_unknown_op_name_falls_back_instead_of_raising():
    payload = profile_op_by_name("attention_sdpa_v3", batch=1, seq_len=128, hidden_size=HIDDEN)
    assert payload["is_memory_bound"] is True
    assert payload["arithmetic_intensity"] == PROFILER_FALLBACK_AI
    assert "error" in payload


# --------------------------------------------------------------------------- #
# Heuristics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("hidden_size", "expected"),
    [
        (1, PROFILER_MIN_TILE),
        (128, 128),
        (1024, 1024),
        (1536, PROFILER_MAX_TILE),  # Qwen2.5-1.5B: wider than one tile, so loop
        (8960, PROFILER_MAX_TILE),
    ],
)
def test_tile_hint_is_a_clamped_power_of_two(hidden_size, expected):
    assert compute_tile_hint(hidden_size) == expected


def test_occupancy_is_a_fraction_and_rises_with_row_count():
    few = estimate_occupancy((1, 8), HIDDEN)
    many = estimate_occupancy((16, 2048), HIDDEN)
    assert 0.0 <= few < many <= 1.0
    assert many == 1.0  # 32768 rows saturates 58 SMs several times over


def test_occupancy_penalizes_a_row_that_does_not_fill_its_tile():
    """A 96-wide row in a 128-wide tile wastes a quarter of the lanes."""
    assert estimate_occupancy((16, 2048), 96) == pytest.approx(96 / 128)


# --------------------------------------------------------------------------- #
# Family classification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("op_name", "family"),
    [
        ("rmsnorm", "norm"),
        ("layernorm", "norm"),
        ("softmax", "reduction"),
        ("silu", "elementwise"),
        ("rope", "rope"),
        ("mlp", "mlp"),
    ],
)
def test_classify_op_family_from_the_registry(op_name, family):
    binding = cpu_op(op_name)
    assert binding.family == family
    assert classify_op_family(binding.reference) == family


def test_classify_op_family_reads_module_class_names():
    class Qwen2RMSNorm:
        def __call__(self, x):
            return x

    assert classify_op_family(Qwen2RMSNorm()) == "norm"


def test_classify_op_family_defaults_to_elementwise():
    """Unrecognised means memory-bound-by-default: the conservative guess."""

    def mystery_op(x):
        return x

    assert classify_op_family(mystery_op) == "elementwise"
