"""The timing contract: adequate warmup, and baselines that are not handicapped.

Two separate honesty guarantees live in `verifier/timing.py`, and both are the kind
that fail silently — a wrong number, never an exception:

1. Red line #9: `do_bench` warmup >= 150. The documented default of 25 underestimates
   latency by ~30% (Triton issue #2306), which inflates every speedup we report.
2. Baseline fairness: `torch.use_deterministic_algorithms(True)` costs the eager and
   torch.compile baselines ~23% while leaving a Triton candidate untouched, so the
   timed comparison runs with the flag off and restores it afterwards. With the flag
   left on we measured 8.52x vs eager; with it off, 6.9x. Only one of those is real.

These run without a GPU: `measure_baselines` is exercised against stubbed
`bench_kernel`/`torch.compile`, because what is under test is the flag discipline
around the measurement, not the measurement itself.
"""

from __future__ import annotations

import pytest
import torch

from kernelsmith.config import DO_BENCH_WARMUP
from kernelsmith.verifier import timing


@pytest.fixture(autouse=True)
def restore_deterministic_flag():
    """Every test here moves a global torch flag; put it back regardless of outcome."""
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)


# --------------------------------------------------------------------------- #
# Red line #9: warmup floor
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("warmup", [0, 1, 25, DO_BENCH_WARMUP - 1])
def test_bench_kernel_refuses_short_warmup(warmup):
    """25 is Triton's default and is exactly the value that produces the ~30% lie."""
    with pytest.raises(ValueError, match="below the required minimum"):
        timing.bench_kernel(lambda: None, warmup=warmup)


def test_bench_kernel_accepts_the_floor(monkeypatch):
    """The floor itself is allowed — the guard is `<`, not `<=`."""
    captured = {}

    def fake_do_bench(fn, warmup, rep, return_mode):
        captured.update(warmup=warmup, rep=rep, return_mode=return_mode)
        return 1.25

    monkeypatch.setattr(timing.triton.testing, "do_bench", fake_do_bench)

    assert timing.bench_kernel(lambda: None, warmup=DO_BENCH_WARMUP) == 1.25
    assert captured["warmup"] == DO_BENCH_WARMUP
    assert captured["return_mode"] == "median"


# --------------------------------------------------------------------------- #
# Baseline fairness
# --------------------------------------------------------------------------- #


def _stub_timing(monkeypatch, observed: list[bool]) -> None:
    """Record the deterministic flag as each baseline is timed, and skip the GPU."""

    def fake_bench_kernel(fn, warmup=DO_BENCH_WARMUP, rep=0):
        observed.append(torch.are_deterministic_algorithms_enabled())
        return 2.0

    monkeypatch.setattr(timing, "bench_kernel", fake_bench_kernel)
    monkeypatch.setattr(timing.torch, "compile", lambda fn, mode=None: fn)
    monkeypatch.setattr(timing.torch.cuda, "synchronize", lambda *a, **k: None)


def test_baselines_are_timed_without_deterministic_algorithms(monkeypatch):
    """Both baselines must be measured with the flag off, or the speedup is inflated."""
    torch.use_deterministic_algorithms(True)
    observed: list[bool] = []
    _stub_timing(monkeypatch, observed)

    result = timing.measure_baselines(lambda x: x, torch.zeros(2))

    assert observed == [False, False], "eager and torch.compile were timed under the flag"
    assert result == {"eager_ms": 2.0, "compile_ms": 2.0}


def test_deterministic_flag_is_restored_after_timing(monkeypatch):
    """The flag is off only inside the timed region; the agent loop still runs under it."""
    torch.use_deterministic_algorithms(True)
    _stub_timing(monkeypatch, [])

    timing.measure_baselines(lambda x: x, torch.zeros(2))

    assert torch.are_deterministic_algorithms_enabled() is True


def test_deterministic_flag_is_restored_even_when_timing_raises(monkeypatch):
    """A benchmark that throws must not leave determinism off for the correctness gate."""
    torch.use_deterministic_algorithms(True)

    def exploding_bench(fn, warmup=DO_BENCH_WARMUP, rep=0):
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(timing, "bench_kernel", exploding_bench)

    with pytest.raises(RuntimeError):
        timing.measure_baselines(lambda x: x, torch.zeros(2))

    assert torch.are_deterministic_algorithms_enabled() is True


def test_warn_only_setting_survives_the_round_trip(monkeypatch):
    """Restoring must carry `warn_only` too, not collapse it to the default."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    _stub_timing(monkeypatch, [])

    timing.measure_baselines(lambda x: x, torch.zeros(2))

    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.is_deterministic_algorithms_warn_only_enabled() is True


def test_a_flag_that_was_already_off_stays_off(monkeypatch):
    """The context manager restores the previous state, it does not force the flag on."""
    torch.use_deterministic_algorithms(False)
    _stub_timing(monkeypatch, [])

    timing.measure_baselines(lambda x: x, torch.zeros(2))

    assert torch.are_deterministic_algorithms_enabled() is False


# --------------------------------------------------------------------------- #
# Speedup arithmetic
# --------------------------------------------------------------------------- #


def test_speedups_are_ratios_over_the_candidate():
    result = timing.compute_speedups(eager_ms=10.0, compile_ms=5.0, candidate_ms=2.0)
    assert result == {"speedup_vs_eager": 5.0, "speedup_vs_compile": 2.5}


@pytest.mark.parametrize("candidate_ms", [0.0, -1.0])
def test_a_non_positive_candidate_time_scores_zero(candidate_ms):
    """A kernel that reports zero time never ran; it must not report infinite speedup."""
    result = timing.compute_speedups(10.0, 5.0, candidate_ms)
    assert result == {"speedup_vs_eager": 0.0, "speedup_vs_compile": 0.0}
