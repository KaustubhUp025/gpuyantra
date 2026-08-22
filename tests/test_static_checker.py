"""One hostile snippet per AST rejection rule, plus a clean kernel (spec 5.3 / 13.1).

The clean-kernel test is the important one: a checker that rejects everything is not
a verifier, it is a wall. Every rule must fire on its hack and stay silent on honest
Triton code.
"""

import textwrap

import pytest

from kernelsmith.verifier.static_checker import (
    RULE_DECOY_KERNEL,
    RULE_HARDCODED_CONSTANT,
    RULE_IDENTITY_OUTPUT,
    RULE_STALE_EMPTY,
    RULE_SYNTAX_ERROR,
    RULE_TORCH_NN_FALLBACK,
    RULE_TRY_EXCEPT,
    RULE_UNSAFE_RUNTIME,
    check_static,
)

# A real, honest fused RMSNorm: allocates with torch.empty_like, launches a jit kernel
# that tl.stores into it, returns the written buffer. NOTHING here may be flagged.
CLEAN_KERNEL = textwrap.dedent(
    """
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _rms_norm_fwd(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x * rstd) * w
        tl.store(out_ptr + row * n_cols + cols, y.to(tl.float16), mask=mask)

    def rms_norm_forward(hidden_states, weight, eps=1e-6):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        out = torch.empty_like(x)
        n_rows, n_cols = x.shape
        BLOCK = triton.next_power_of_2(n_cols)
        _rms_norm_fwd[(n_rows,)](x, weight, out, n_cols, eps, BLOCK=BLOCK, num_warps=8)
        return out.view(shape)
    """
)


def rule_ids(code: str) -> list[int]:
    return [rule for rule, _line, _desc in check_static(code)]


def test_clean_kernel_has_no_violations():
    """An honest fused Triton RMSNorm must pass untouched."""
    assert check_static(CLEAN_KERNEL) == []


def test_rule_1_torch_nn_functional_fallback():
    code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F

        def rms_norm_forward(hidden_states, weight, eps=1e-6):
            return F.rms_norm(hidden_states, (hidden_states.shape[-1],), weight, eps)
        """
    )
    violations = check_static(code)
    assert RULE_TORCH_NN_FALLBACK in rule_ids(code)
    rule, line, desc = next(v for v in violations if v[0] == RULE_TORCH_NN_FALLBACK)
    assert line == 6, "violation must point at the offending call site"
    assert "torch.nn.functional.rms_norm" in desc


@pytest.mark.parametrize(
    "call",
    [
        "torch.nn.functional.rms_norm(x, (4,))",
        "torch.nn.LayerNorm(4).cuda()(x)",
        "F.layer_norm(x, (4,))",
    ],
)
def test_rule_1_catches_every_torch_nn_spelling(call):
    code = f"import torch\n\ndef fwd(x):\n    y = {call}\n    return y\n"
    assert RULE_TORCH_NN_FALLBACK in rule_ids(code)


def test_rule_2_identity_output():
    """KernelBench-Verified: the ReLU identity shortcut that faked a 374x speedup."""
    code = textwrap.dedent(
        """
        import torch

        def rms_norm_forward(hidden_states, weight, eps=1e-6):
            return hidden_states
        """
    )
    assert RULE_IDENTITY_OUTPUT in rule_ids(code)


def test_rule_2_identity_through_alias_and_reshape():
    """Laundering the input through an alias and a view is still the identity."""
    code = textwrap.dedent(
        """
        import torch

        def rms_norm_forward(hidden_states, weight, eps=1e-6):
            y = hidden_states
            return y.view(hidden_states.shape)
        """
    )
    assert RULE_IDENTITY_OUTPUT in rule_ids(code)


def test_rule_3_decoy_kernel():
    """Sakana: a real-looking @triton.jit kernel that is never on the compute path."""
    code = textwrap.dedent(
        """
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def _decoy_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            row = tl.program_id(0)
            cols = tl.arange(0, BLOCK)
            x = tl.load(x_ptr + row * n + cols)
            tl.store(out_ptr + row * n + cols, x)

        def rms_norm_forward(hidden_states, weight, eps=1e-6):
            var = hidden_states.float().pow(2).mean(-1, keepdim=True)
            out = hidden_states * torch.rsqrt(var + eps)
            return (out * weight).to(hidden_states.dtype)
        """
    )
    violations = check_static(code)
    assert RULE_DECOY_KERNEL in rule_ids(code)
    _rule, _line, desc = next(v for v in violations if v[0] == RULE_DECOY_KERNEL)
    assert "_decoy_kernel" in desc


def test_rule_3_silent_when_kernel_is_launched():
    """The launch is `kernel[grid](...)` — a Subscript call, not a plain one."""
    assert RULE_DECOY_KERNEL not in rule_ids(CLEAN_KERNEL)


def test_rule_4_torch_empty_without_store():
    """Berkeley RDI: stale GPU memory often still holds the reference answer."""
    code = textwrap.dedent(
        """
        import torch

        def rms_norm_forward(hidden_states, weight, eps=1e-6):
            out = torch.empty_like(hidden_states)
            return out
        """
    )
    violations = check_static(code)
    assert RULE_STALE_EMPTY in rule_ids(code)
    _rule, _line, desc = next(v for v in violations if v[0] == RULE_STALE_EMPTY)
    assert "out" in desc


def test_rule_4_silent_when_buffer_is_written_by_the_kernel():
    assert RULE_STALE_EMPTY not in rule_ids(CLEAN_KERNEL)


def test_rule_5_hardcoded_tensor():
    """CUDA Agent data filter: the output must differ across inputs."""
    code = textwrap.dedent(
        """
        import torch

        def rms_norm_forward(hidden_states, weight, eps=1e-6):
            return torch.tensor([1.0, 2.0])
        """
    )
    assert RULE_HARDCODED_CONSTANT in rule_ids(code)


def test_rule_5_unwritten_constant_allocator():
    """torch.zeros returned unwritten is a constant output by another name."""
    code = textwrap.dedent(
        """
        import torch

        def rms_norm_forward(hidden_states, weight, eps=1e-6):
            out = torch.zeros_like(hidden_states)
            return out
        """
    )
    assert RULE_HARDCODED_CONSTANT in rule_ids(code)


def test_rule_6_try_except():
    """Kevin: try/except means a silent fallback to the torch reference."""
    code = textwrap.dedent(
        """
        import torch

        def rms_norm_forward(hidden_states, weight, eps=1e-6):
            try:
                return _triton_path(hidden_states, weight, eps)
            except Exception:
                return hidden_states * weight
        """
    )
    assert RULE_TRY_EXCEPT in rule_ids(code)


@pytest.mark.parametrize(
    "import_line",
    [
        "import socket",
        "import threading",
        "import multiprocessing",
        "import urllib.request",
        "import requests",
        "from http import client",
        "from threading import Thread",
    ],
)
def test_rule_7_forbidden_imports(import_line):
    code = f"import torch\n{import_line}\n\ndef fwd(x):\n    return x * 2\n"
    assert RULE_UNSAFE_RUNTIME in rule_ids(code)


def test_rule_7_extra_cuda_stream():
    """CUDA-L1: work parked on a side stream is invisible to the timer."""
    code = textwrap.dedent(
        """
        import torch

        def rms_norm_forward(hidden_states, weight, eps=1e-6):
            s = torch.cuda.Stream()
            out = hidden_states * weight
            return out
        """
    )
    assert RULE_UNSAFE_RUNTIME in rule_ids(code)


def test_syntax_error_is_rejected_not_raised():
    """A candidate that does not parse is a -1, never an exception in the verifier."""
    violations = check_static("def fwd(x:\n    return x")
    assert violations
    assert violations[0][0] == RULE_SYNTAX_ERROR


def test_violations_carry_line_numbers_and_descriptions():
    code = textwrap.dedent(
        """
        import torch

        def rms_norm_forward(hidden_states, weight, eps=1e-6):
            return hidden_states
        """
    )
    for rule, line, desc in check_static(code):
        assert isinstance(rule, int)
        assert line > 0
        assert isinstance(desc, str) and desc
