"""LayerNorm as a first-class patchable op (Task 10, Part C).

`torch.nn.LayerNorm` was on the reject list next to `apply_rotary_pos_emb`, on the
strength of the name alone. It never belonged there: it is an ordinary `nn.Module`,
`named_modules()` finds it, and `types.MethodType` rebinds its forward exactly as it does
Qwen2RMSNorm's. That mistake is what kept GPT-2's 25 normalization layers — its single
biggest bandwidth sink — outside the system.

Three things are asserted here, and the first is the one that would have caught the
original error:

1. a declared contract naming LayerNorm's real attributes VALIDATES, and one naming an
   attribute it does not have does not;
2. the hard-coded fallback adapter reproduces the original forward bit for bit, so a
   swap is a change of implementation and not a change of model;
3. LayerNorm's `bias` and its `eps` (not `variance_epsilon`) are handled — the two
   places a single shared norm adapter would have silently done the wrong thing.

No GPU: everything here is fp32 on the CPU, and the meta-device probe allocates nothing.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from kernelsmith.inference_server.patchable_ops import (
    PATCHABLE_OPS,
    build_forward,
    build_forward_from_mapping,
    find_modules,
    resolve_class_name,
    rollback_op,
    swap_op,
)
from kernelsmith.verifier.adapter_mapping import (
    is_mappable_op,
    normalize_op_name,
    validate_adapter_mapping,
)

HIDDEN = 64
LAYERNORM_MAPPING = {"weight": "weight", "bias": "bias", "eps": "eps"}


def reference_layernorm(x, weight, bias, eps):
    """The signature the verifier benches a layernorm kernel against.

    Matches `_build_layernorm(...).bind` in tools/profiler_tool.py: `entry(x, weight,
    bias, eps)`. The reduction runs in fp32 the way the real reference does.
    """
    f = x.to(torch.float32)
    normed = (f - f.mean(-1, keepdim=True)) * torch.rsqrt(
        f.var(-1, keepdim=True, unbiased=False) + eps
    )
    return (normed.to(x.dtype) * weight) + bias


@pytest.fixture
def layer() -> nn.LayerNorm:
    torch.manual_seed(0)
    module = nn.LayerNorm(HIDDEN)
    with torch.no_grad():
        module.weight.copy_(torch.randn(HIDDEN) * 0.02 + 1.0)
        module.bias.copy_(torch.randn(HIDDEN) * 0.02)
    return module


@pytest.fixture
def probe() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 8, HIDDEN)


# --------------------------------------------------------------------------- #
# Registry wiring
# --------------------------------------------------------------------------- #


def test_layernorm_is_a_patchable_op():
    assert "layernorm" in PATCHABLE_OPS
    assert resolve_class_name("layernorm") == "LayerNorm"


def test_layernorm_is_a_priority_zero_target_like_rmsnorm():
    """Both are the same bottleneck; whichever one the architecture uses is P0."""
    assert PATCHABLE_OPS["layernorm"]["priority"] == PATCHABLE_OPS["rmsnorm"]["priority"] == 0


def test_layernorm_is_reachable_through_named_modules():
    """The claim that made it unpatchable was that it is not a module. It is."""
    model = nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, HIDDEN), nn.LayerNorm(HIDDEN))
    assert len(find_modules(model, "LayerNorm")) == 2


def test_layernorm_is_in_the_verifier_op_registry():
    from kernelsmith.tools.profiler_tool import build_op

    binding = build_op("layernorm", HIDDEN, device="cpu", dtype=torch.float32)
    assert binding.family == "norm"


# --------------------------------------------------------------------------- #
# validate_adapter_mapping
# --------------------------------------------------------------------------- #


def test_layernorm_is_no_longer_on_the_reject_list():
    assert is_mappable_op("layernorm")
    assert is_mappable_op("layer_norm")
    assert is_mappable_op("LayerNorm")
    assert normalize_op_name("Layer_Norm") == "layernorm"


def test_the_real_layernorm_contract_validates():
    assert validate_adapter_mapping("layernorm", LAYERNORM_MAPPING) == []


def test_normalized_shape_is_a_legitimate_binding():
    """A kernel that needs the shape may ask for it; it is data, not a method."""
    assert validate_adapter_mapping("layernorm", {"shape": "normalized_shape"}) == []


def test_a_nonexistent_attribute_is_rejected_with_the_available_ones_named():
    errors = validate_adapter_mapping("layernorm", {"weight": "nonexistent_attr"})

    assert len(errors) == 1
    assert "nonexistent_attr" in errors[0]
    assert "LayerNorm" in errors[0]
    # The error goes into the next iteration's prompt, so it has to be actionable.
    assert "weight" in errors[0] and "bias" in errors[0]


def test_rmsnorms_epsilon_name_is_rejected_for_a_layernorm():
    """`variance_epsilon` is Qwen2RMSNorm's name for it. On a LayerNorm it is an
    AttributeError inside a hot forward — exactly what this gate exists to catch."""
    errors = validate_adapter_mapping("layernorm", {"eps": "variance_epsilon"})
    assert len(errors) == 1
    assert "variance_epsilon" in errors[0]


def test_a_method_cannot_be_bound_as_data():
    errors = validate_adapter_mapping("layernorm", {"weight": "forward"})
    assert len(errors) == 1
    assert "is a method" in errors[0]


def test_the_implicit_input_argument_must_not_be_mapped():
    errors = validate_adapter_mapping("layernorm", {"input": "weight"})
    assert len(errors) == 1
    assert "implicit" in errors[0]


def test_an_empty_contract_is_still_valid_and_falls_back():
    assert validate_adapter_mapping("layernorm", {}) == []
    assert validate_adapter_mapping("layernorm", None) == []


def test_the_probe_allocates_nothing():
    """It is built under torch.device("meta"): the check must cost no memory."""
    from kernelsmith.verifier.adapter_mapping import _OP_MODULES, _probe_instance

    probe = _probe_instance(_OP_MODULES["layernorm"])
    assert isinstance(probe, nn.LayerNorm)
    assert probe.weight.is_meta
    assert probe.bias.is_meta


# --------------------------------------------------------------------------- #
# The hard-coded fallback adapter
# --------------------------------------------------------------------------- #


def test_the_fallback_adapter_reproduces_the_original_forward(layer, probe):
    """A swap must change the implementation, not the model."""
    expected = layer(probe)

    forward = build_forward("layernorm", reference_layernorm)
    got = forward(layer, probe)

    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_the_fallback_adapter_passes_the_modules_own_weight_bias_and_eps(layer, probe):
    """Not a copy and not a re-initialization: the module's own tensors, by identity."""
    seen: dict[str, object] = {}

    def spy(x, weight, bias, eps):
        seen.update(x=x, weight=weight, bias=bias, eps=eps)
        return x

    build_forward("layernorm", spy)(layer, probe)

    assert seen["x"] is probe
    assert seen["weight"] is layer.weight
    assert seen["bias"] is layer.bias
    assert seen["eps"] == layer.eps


def test_the_fallback_adapter_does_not_pass_normalized_shape(layer, probe):
    """It must match the signature the verifier benched — `entry(x, weight, bias, eps)`.

    A kernel that wants the shape declares it and gets it through the generic adapter;
    the fallback cannot add an argument the verified wrapper never accepted.
    """
    calls: list[tuple] = []

    def spy(*args):
        calls.append(args)
        return args[0]

    build_forward("layernorm", spy)(layer, probe)
    assert len(calls[0]) == 4


def test_rmsnorms_adapter_and_layernorms_adapter_are_not_the_same_bridge():
    """LayerNorm has a bias and calls its epsilon `eps`. One adapter cannot cover both."""
    rms_forward = build_forward("rmsnorm", lambda *a: a[0])
    ln_forward = build_forward("layernorm", lambda *a: a[0])
    assert rms_forward is not ln_forward

    # The rmsnorm bridge cannot serve a LayerNorm: it has no `variance_epsilon`.
    with pytest.raises(AttributeError):
        rms_forward(nn.LayerNorm(HIDDEN), torch.randn(2, HIDDEN))


# --------------------------------------------------------------------------- #
# The generic (declared-contract) adapter
# --------------------------------------------------------------------------- #


def test_the_generic_adapter_binds_layernorm_from_the_declared_contract(layer, probe):
    """No hard-coded knowledge of LayerNorm at all — only the Coder's mapping."""
    expected = layer(probe)

    forward = build_forward_from_mapping(reference_layernorm, LAYERNORM_MAPPING)
    got = forward(layer, probe)

    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_a_declared_contract_takes_precedence_over_the_hard_coded_adapter(layer, probe):
    forward = build_forward("layernorm", reference_layernorm, LAYERNORM_MAPPING)
    assert forward.__qualname__.startswith("build_forward_from_mapping")
    torch.testing.assert_close(forward(layer, probe), layer(probe), atol=1e-5, rtol=1e-5)


def test_the_generic_adapter_passes_the_contract_by_keyword(layer, probe):
    seen: dict[str, object] = {}

    def spy(x, **kwargs):
        seen.update(kwargs)
        return x

    build_forward_from_mapping(spy, LAYERNORM_MAPPING)(layer, probe)

    assert seen["weight"] is layer.weight
    assert seen["bias"] is layer.bias
    assert seen["eps"] == layer.eps


# --------------------------------------------------------------------------- #
# End to end: swap and roll back a LayerNorm-based model
# --------------------------------------------------------------------------- #


def test_swapping_every_layernorm_in_a_gpt2_shaped_stack_and_rolling_back():
    """25 LayerNorms is GPT-2's count; the swap has to catch all of them and undo cleanly."""
    torch.manual_seed(2)
    model = nn.Sequential(*[nn.LayerNorm(HIDDEN) for _ in range(25)])
    probe = torch.randn(2, 8, HIDDEN)
    before = model(probe)

    originals = swap_op(
        model, "LayerNorm", build_forward_from_mapping(reference_layernorm, LAYERNORM_MAPPING)
    )
    assert len(originals) == 25

    after_swap = model(probe)
    torch.testing.assert_close(after_swap, before, atol=1e-4, rtol=1e-4)

    assert rollback_op(model, originals) == 25
    torch.testing.assert_close(model(probe), before, rtol=0, atol=0)  # bitwise
