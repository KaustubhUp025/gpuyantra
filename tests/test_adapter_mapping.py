"""The generated deployment contract: validation and the generic adapter.

This is the novel contribution under test (`.claude/rules/implementation-deviations.md`).
Every published system — HF `kernels`, FlashInfer-Bench, Kernel Contracts — has a HUMAN
write the bridge between a benchmark-shaped kernel (`entry(x, weight, eps)`) and a live
module's `forward(self, hidden_states)`. Here the Coder writes it and the verifier
checks it, so the two failure modes worth testing are exactly:

1. a contract that claims an attribute the module does not have, or claims a method as
   data — caught deterministically, BEFORE the sandbox, because `self.eps` instead of
   `self.variance_epsilon` is an AttributeError inside a hot forward on a live server;
2. a contract that validates but binds the wrong thing — caught by checking that the
   generic adapter passes the module's OWN parameters, by keyword, unmodified.

`validate_adapter_mapping` probes the real `Qwen2RMSNorm` / `Qwen2MLP` on the meta
device, so these tests are about the transformers classes actually being patched, not
about a stand-in that could drift from them.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from kernelsmith.inference_server.patchable_ops import build_forward, build_forward_from_mapping
from kernelsmith.verifier.adapter_mapping import (
    is_mappable_op,
    normalize_op_name,
    validate_adapter_mapping,
)

RMSNORM_MAPPING = {"weight": "weight", "eps": "variance_epsilon"}

#: Minimal kernel that the AST checker accepts: a @triton.jit kernel that is actually
#: launched, computing and storing through tl.store. Whether it is CORRECT is the
#: sandbox's problem, and the sandbox must never be reached in these tests.
VALID_KERNEL = """
import torch
import triton
import triton.language as tl


@triton.jit
def _scale(X, W, Y, N, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)
    tl.store(Y + cols, x * w, mask=mask)


def k(x, weight, eps):
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    y = torch.empty_like(x2d)
    N = x2d.shape[-1]
    _scale[(x2d.shape[0],)](x2d, weight, y, N, BLOCK=triton.next_power_of_2(N))
    return y.reshape(x.shape)
"""


# --------------------------------------------------------------- validation


def test_valid_rmsnorm_mapping_has_no_errors():
    assert validate_adapter_mapping("rmsnorm", RMSNORM_MAPPING) == []


def test_op_name_spelling_is_not_a_contract_violation():
    """Spellings like "rms_norm" and "RMSNorm" all name the same module class."""
    for spelling in ("rms_norm", "RMSNorm", "RMS-NORM", "rmsnorm"):
        assert normalize_op_name(spelling) == "rmsnorm"
        assert validate_adapter_mapping(spelling, RMSNORM_MAPPING) == []


def test_swiglu_is_an_alias_for_the_mlp_module():
    """PATCHABLE_OPS calls it "swiglu"; OP_REGISTRY calls it "mlp". Both must resolve."""
    mapping = {"gate_w": "gate_proj.weight", "up_w": "up_proj.weight", "down_w": "down_proj.weight"}
    assert validate_adapter_mapping("swiglu", mapping) == []
    assert validate_adapter_mapping("mlp", mapping) == []


def test_unknown_attribute_is_rejected():
    """The exact bug the layer exists for: `eps` is not what Qwen2RMSNorm calls it."""
    errors = validate_adapter_mapping("rmsnorm", {"weight": "weight", "eps": "eps"})

    assert len(errors) == 1
    assert "Qwen2RMSNorm has no attribute 'eps'" in errors[0]
    # The error goes back into the Coder's next prompt, so it must name the real ones.
    assert "variance_epsilon" in errors[0]


def test_method_name_is_rejected():
    """An adapter may only pass data; binding `forward` would recurse into itself."""
    errors = validate_adapter_mapping("rmsnorm", {"weight": "weight", "fn": "forward"})

    assert len(errors) == 1
    assert "is a method" in errors[0]


def test_submodules_and_activations_are_data_not_methods():
    """`act_fn` is a callable nn.Module — legitimate to hand to a kernel, unlike a method."""
    assert validate_adapter_mapping("mlp", {"act": "act_fn", "gate": "gate_proj"}) == []


def test_the_implicit_input_tensor_may_not_be_mapped():
    """The forward's input is positional; mapping it would pass it twice."""
    errors = validate_adapter_mapping("rmsnorm", {"hidden_states": "weight"})

    assert len(errors) == 1
    assert "implicit" in errors[0]


def test_empty_mapping_is_valid_and_means_use_the_fallback():
    """Seed kernels predate the contract; they must keep working through the adapters."""
    assert validate_adapter_mapping("rmsnorm", {}) == []
    assert validate_adapter_mapping("rmsnorm", None) == []


def test_a_mapping_for_an_unpatchable_op_is_rejected():
    """RoPE is a module-level function: no module, so no attribute to map (deviations doc)."""
    errors = validate_adapter_mapping("rope", RMSNORM_MAPPING)

    assert len(errors) == 1
    assert "no patchable nn.Module" in errors[0]
    assert not is_mappable_op("rope")


def test_a_non_string_attribute_is_rejected():
    errors = validate_adapter_mapping("rmsnorm", {"eps": 1e-6})

    assert len(errors) == 1
    assert "must name a module attribute" in errors[0]


def test_all_bad_entries_are_reported_together():
    """One iteration of the loop is expensive; report every fix at once."""
    errors = validate_adapter_mapping("rmsnorm", {"w": "wieght", "eps": "epsilon"})

    assert len(errors) == 2


# ------------------------------------------------------------ generic adapter


class FakeNorm(nn.Module):
    """A module with the same attribute layout as Qwen2RMSNorm, three orders smaller."""

    def __init__(self, hidden: int = 8, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.weight


def test_generic_adapter_calls_the_kernel_with_mapped_keyword_arguments():
    seen: dict[str, object] = {}

    def kernel(x, weight, eps):  # noqa: ANN001 — mirrors a real wrapper's signature
        seen.update(x=x, weight=weight, eps=eps)
        return x * 2

    module = FakeNorm()
    forward = build_forward_from_mapping(kernel, RMSNORM_MAPPING)
    x = torch.randn(2, 3, 8)

    out = forward(module, x)

    assert torch.equal(out, x * 2)
    assert seen["x"] is x  # the input tensor is positional and implicit
    # The module's OWN parameter, not a copy: a kernel that re-initializes weights is
    # a different model, not a faster one.
    assert seen["weight"] is module.weight
    assert seen["eps"] == module.variance_epsilon


def test_generic_adapter_walks_dotted_attribute_paths():
    class FakeMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(4, 4, bias=False)

    seen: dict[str, object] = {}

    def kernel(x, gate_w):  # noqa: ANN001
        seen.update(gate_w=gate_w)
        return x

    module = FakeMLP()
    build_forward_from_mapping(kernel, {"gate_w": "gate_proj.weight"})(module, torch.randn(1, 4))

    assert seen["gate_w"] is module.gate_proj.weight


def test_generic_adapter_is_bindable_as_a_method():
    """`swap_op` does `types.MethodType(fn, module)`, so `self` must be the first param."""
    import types

    module = FakeNorm()
    forward = build_forward_from_mapping(lambda x, weight, eps: x + weight, RMSNORM_MAPPING)
    module.forward = types.MethodType(forward, module)

    x = torch.randn(1, 2, 8)
    assert torch.allclose(module.forward(x), x + module.weight)


def test_build_forward_prefers_the_mapping_over_the_hard_coded_adapter():
    calls: list[str] = []

    def kernel(x, weight, eps):  # noqa: ANN001
        calls.append("mapped")
        return x

    build_forward("rmsnorm", kernel, RMSNORM_MAPPING)(FakeNorm(), torch.randn(1, 8))

    assert calls == ["mapped"]


def test_build_forward_falls_back_to_the_hard_coded_adapter_without_a_mapping():
    """Backward compatibility: the seed kernel declares no contract and must still bind."""
    seen: dict[str, object] = {}

    def kernel(x, weight, eps):  # noqa: ANN001 — positional, as the seed kernel is called
        seen.update(weight=weight, eps=eps)
        return x

    module = FakeNorm()
    for mapping in (None, {}):
        seen.clear()
        build_forward("rmsnorm", kernel, mapping)(module, torch.randn(1, 8))
        assert seen["weight"] is module.weight
        assert seen["eps"] == module.variance_epsilon


def test_build_forward_still_refuses_an_op_with_neither_mapping_nor_adapter():
    """RoPE has no module to bind to; a silent no-op swap would fake a speedup."""
    with pytest.raises(ValueError, match="no forward adapter"):
        build_forward("rope", lambda x, cos, sin: x)


def test_an_entrypoint_written_as_a_method_is_used_as_is():
    """A Coder that writes `forward(self, x)` directly needs no adapter at all."""

    def forward(self, hidden_states):  # noqa: ANN001
        return hidden_states * self.weight

    assert build_forward("rmsnorm", forward, RMSNORM_MAPPING) is forward


# ------------------------------------------------------- verifier integration


def test_verifier_rejects_a_bad_mapping_without_running_the_sandbox(monkeypatch):
    """Layer 2 must be cheaper than layer 3: an invalid contract never reaches a subprocess."""
    from kernelsmith.tools import verifier_tool

    def explode(*args, **kwargs):
        raise AssertionError("the sandbox must not run on an invalid adapter_mapping")

    monkeypatch.setattr(verifier_tool, "run_in_sandbox", explode)

    verdict = verifier_tool.verify_kernel(
        # Passes the static checker (a real launched kernel), so the ONLY thing that
        # can reject it is the contract.
        kernel_code=VALID_KERNEL,
        entrypoint="k",
        task_spec={"op_name": "rmsnorm", "hidden_size": 1536},
        adapter_mapping={"w": "weight", "eps": "eps"},
    )

    assert verdict["reward"] == -1
    assert verdict["correctness_pass"] is False
    assert verdict["adapter_mapping_errors"]
    assert "adapter_mapping" in verdict["next_action"]
