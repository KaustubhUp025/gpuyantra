"""The reproducibility contract (spec 11 / 13.1).

These tests exist because of a specific failure mode in this field, not for coverage.
Sakana's reported 3.13x became 1.49x and KernelBench-Verified's 1.43x became 0.88x when
re-run — seeding and baseline artifacts, both of them. The claim KernelSmith makes is
that `make demo` reproduces its headline number on a fresh L4, and that claim is only
as good as `seed_everything`.

Order is asserted as well as effect: `CUBLAS_WORKSPACE_CONFIG` must be in the
environment, because `torch.use_deterministic_algorithms(True)` throws on the first
deterministic GEMM without it.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from kernelsmith import config
from kernelsmith.reproducibility import seed_everything


@pytest.fixture(autouse=True)
def restore_torch_state():
    """Leave the process as we found it — this module mutates global torch state."""
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    yield
    torch.use_deterministic_algorithms(was_deterministic)
    if workspace is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


def test_seed_everything_sets_every_rng():
    """One call must pin Python, NumPy and torch — a missed one is a moving number."""
    seed_everything(config.GLOBAL_SEED)
    first = (random.random(), float(np.random.rand()), torch.randn(4).tolist())

    seed_everything(config.GLOBAL_SEED)
    second = (random.random(), float(np.random.rand()), torch.randn(4).tolist())

    assert first == second


def test_same_seed_reproduces_identical_torch_randn():
    """The verifier's correctness probes are torch.randn; identical seeds, identical input."""
    seed_everything(1234)
    first = torch.randn(64, 128)
    seed_everything(1234)
    second = torch.randn(64, 128)

    assert torch.equal(first, second)


def test_different_seeds_produce_different_draws():
    """A guard on the guard: a seeder that pins everything to a constant would pass above."""
    seed_everything(1)
    first = torch.randn(32)
    seed_everything(2)
    second = torch.randn(32)

    assert not torch.equal(first, second)


def test_cublas_workspace_config_is_exported():
    """Without it, use_deterministic_algorithms(True) raises on the first cuBLAS GEMM."""
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    seed_everything()

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == config.CUBLAS_WORKSPACE


def test_existing_workspace_config_is_overwritten():
    """A stale value from the shell is worse than none: it silently weakens determinism."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    seed_everything()

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == config.CUBLAS_WORKSPACE


def test_deterministic_algorithms_are_enabled():
    """Red line: deterministic kernel selection stays on outside the timed baselines."""
    torch.use_deterministic_algorithms(False)
    seed_everything()

    assert torch.are_deterministic_algorithms_enabled() is config.DETERMINISTIC_CUDA


def test_tf32_is_enabled_for_honest_baselines():
    """KernelBench-Verified's lesson: an un-TF32'd eager baseline invents a speedup."""
    torch.set_float32_matmul_precision("highest")
    seed_everything()

    assert torch.get_float32_matmul_precision() == "high"


def test_seed_everything_reports_what_it_applied():
    """The demo logs this, so a recording says what state the numbers were produced under."""
    applied = seed_everything(7)

    assert applied["seed"] == 7
    assert applied["cublas_workspace_config"] == config.CUBLAS_WORKSPACE
    assert applied["deterministic_algorithms"] is config.DETERMINISTIC_CUDA
    assert applied["cuda_devices_seeded"] >= 0


def test_default_seed_is_the_configured_one():
    """Every entry point must start from the same seed, not from its own default."""
    seed_everything()
    from_default = torch.randn(8)
    seed_everything(config.GLOBAL_SEED)
    from_explicit = torch.randn(8)

    assert torch.equal(from_default, from_explicit)


# --------------------------------------------------------------------------- #
# .env loading (the setup path the README documents)
# --------------------------------------------------------------------------- #


def test_the_package_loads_the_env_file_without_overriding_the_real_environment(tmp_path):
    """A shell variable must always beat the file.

    `config.py` reads GOOGLE_CLOUD_PROJECT strictly at import time, so `.env` has to be
    loaded by `kernelsmith/__init__.py` before anything else. The danger in doing that
    is a stale `.env` silently redirecting a run to the wrong GCP project behind an
    explicit `GOOGLE_CLOUD_PROJECT=... uv run ...`, so the loader is override=False.
    """
    import os
    from unittest import mock

    import kernelsmith

    env_file = tmp_path / ".env"
    env_file.write_text("KS_TEST_ONLY_UNSET=from-file\nKS_TEST_ONLY_SET=from-file\n")

    with (
        mock.patch.object(kernelsmith, "_ENV_FILE", env_file),
        mock.patch.dict(os.environ, {"KS_TEST_ONLY_SET": "from-shell"}, clear=False),
    ):
        os.environ.pop("KS_TEST_ONLY_UNSET", None)
        try:
            kernelsmith._load_env_file()
            assert os.environ["KS_TEST_ONLY_UNSET"] == "from-file"
            assert os.environ["KS_TEST_ONLY_SET"] == "from-shell"
        finally:
            os.environ.pop("KS_TEST_ONLY_UNSET", None)


def test_a_missing_env_file_is_not_an_error(tmp_path):
    """The VM exports these in the shell profile and has no repo-root .env."""
    from unittest import mock

    import kernelsmith

    with mock.patch.object(kernelsmith, "_ENV_FILE", tmp_path / "nope.env"):
        kernelsmith._load_env_file()  # must not raise
