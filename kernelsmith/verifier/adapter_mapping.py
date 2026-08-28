"""Deterministic validation of the Coder's declared deployment contract.

This is layer 2 of the three-layer safety model in
`.claude/rules/implementation-deviations.md`, and the half of the novel contribution
that is NOT an LLM: the Coder *generates* the deployment contract, and this file
*checks* it — with `hasattr`, not with a second model's opinion.

Why the contract exists at all. The verifier calls a kernel as `entry(x, weight, eps)`,
with explicit tensors, because the sandbox has no model to read weights from. The live
model does: `types.MethodType(fn, module)` calls `fn(self, hidden_states)`. Something
has to say which of the module's attributes fill the wrapper's remaining parameters.
Every published system — HF `kernels`, FlashInfer-Bench, Kernel Contracts — has a HUMAN
write that bridge. Here the Coder declares it as `adapter_mapping`
(`{"weight": "weight", "eps": "variance_epsilon"}`) and this module decides, before any
code runs, whether the declaration is true of the class it claims to describe.

A wrong mapping is caught three ways and each is cheaper than the last one it precedes:
this check (no execution at all), the sandbox's 5x3 numerical gate, and the server's
parity gate with auto-rollback. This one exists because `self.eps` instead of
`self.variance_epsilon` is an `AttributeError` inside a hot forward on a live server —
a failure that deserves to happen at validation time, for free.

The probe. `hasattr(Qwen2RMSNorm, "weight")` is False: `weight` is assigned in
`__init__`, so it lives on instances, not on the class. The check therefore builds a
real instance under `torch.device("meta")` — no memory is allocated, no GPU is touched,
and what gets inspected is the actual module the kernel will be bound to. If the probe
cannot be built at all (transformers missing, a signature change), validation falls back
to a declared allowlist rather than blocking a verified kernel on an infrastructure
failure; layers 3 and 4 are still in front of the live server.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _build_qwen2_mlp_probe(cls: Any) -> Any:
    """Qwen2MLP takes a config, not a hidden size. Tiny values: nothing is allocated."""
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

    return cls(Qwen2Config(hidden_size=64, intermediate_size=128))


#: Normalized op name -> the module class the kernel will be bound to, plus the
#: attributes we know it carries and how to build a meta-device instance of it.
#:
#: Ops absent from this table (softmax, silu, rope) have no `nn.Module` to patch —
#: "rope" is a module-level function (see `patchable_ops`) — so a mapping for them
#: cannot be validated OR deployed, and is rejected as such.
#:
#: "layernorm" used to be in that reject list and should not have been:
#: `torch.nn.LayerNorm` is an ordinary `nn.Module`, discoverable through
#: `named_modules()` and swappable by exactly the mechanism RMSNorm uses. It was lumped
#: in with `apply_rotary_pos_emb` on the strength of the name alone. Fixing that is what
#: lets the system optimize GPT-2's normalization layers (Task 10).
_OP_MODULES: dict[str, dict[str, Any]] = {
    "rmsnorm": {
        "module": "transformers.models.qwen2.modeling_qwen2",
        "class_name": "Qwen2RMSNorm",
        "build": lambda cls: cls(64),
        # Fallback only, used when the meta-device probe cannot be built.
        "known_attrs": frozenset({"weight", "variance_epsilon"}),
    },
    "layernorm": {
        "module": "torch.nn",
        "class_name": "LayerNorm",
        # normalized_shape is the one required argument; affine params come with it.
        "build": lambda cls: cls(64),
        "known_attrs": frozenset({"weight", "bias", "eps", "normalized_shape"}),
    },
    "mlp": {
        "module": "transformers.models.qwen2.modeling_qwen2",
        "class_name": "Qwen2MLP",
        "build": _build_qwen2_mlp_probe,
        "known_attrs": frozenset(
            {"gate_proj", "up_proj", "down_proj", "act_fn", "hidden_size", "intermediate_size"}
        ),
    },
}

#: `swiglu` is what `PATCHABLE_OPS` calls the op the verifier calls `mlp`. The Coder
#: sees both names, so both resolve here.
_OP_ALIASES = {"swiglu": "mlp", "qwen2mlp": "mlp", "qwen2rmsnorm": "rmsnorm"}

#: The forward's input tensor is positional and implicit — it is never mapped.
_IMPLICIT_PARAMS = frozenset({"self", "x", "hidden_states", "input", "inputs", "hidden_state"})


def normalize_op_name(op_name: str) -> str:
    """`"RMS_Norm"` -> `"rmsnorm"`. Underscores and case are not a contract violation."""
    key = str(op_name or "").strip().lower().replace("_", "").replace("-", "")
    return _OP_ALIASES.get(key, key)


def is_mappable_op(op_name: str) -> bool:
    """True if this op has a module class an adapter_mapping could describe."""
    return normalize_op_name(op_name) in _OP_MODULES


def validate_adapter_mapping(op_name: str, mapping: Mapping[str, str] | None) -> list[str]:
    """Check a declared adapter mapping against the real target module. Errors, or [].

    Args:
        op_name: The op being optimized ("rmsnorm", "mlp"/"swiglu", ...).
        mapping: kernel parameter name -> module attribute name, e.g.
            `{"weight": "weight", "eps": "variance_epsilon"}`. Dotted attribute paths
            ("gate_proj.weight") are walked. Empty or None means "no contract declared"
            and is valid — the per-op hard-coded adapter is used instead.

    Returns:
        A list of human-readable errors, empty when the mapping is deployable. Every
        error is phrased as something the Coder can act on, because it goes straight
        back into the next iteration's prompt.
    """
    if not mapping:
        return []  # No declared contract: the per-op fallback adapter handles it.

    if not isinstance(mapping, Mapping):
        return [
            f"adapter_mapping must be an object of kernel_param -> module_attr, got {type(mapping).__name__}"
        ]

    errors: list[str] = []
    for param, attr in mapping.items():
        if not isinstance(param, str) or not param.strip():
            errors.append(f"adapter_mapping key {param!r} is not a parameter name")
        elif param.strip().lower() in _IMPLICIT_PARAMS:
            errors.append(
                f"adapter_mapping must not map {param!r}: the forward's input tensor is "
                "passed positionally and is implicit"
            )
        if not isinstance(attr, str) or not attr.strip():
            errors.append(f"adapter_mapping[{param!r}] must name a module attribute, got {attr!r}")
    if errors:
        return errors

    spec = _OP_MODULES.get(normalize_op_name(op_name))
    if spec is None:
        return [
            f"op {op_name!r} has no patchable nn.Module in the served model, so an "
            "adapter_mapping cannot be validated or deployed for it — leave it empty"
        ]

    probe = _probe_instance(spec)
    for param, attr in mapping.items():
        errors.extend(_check_attr(spec, probe, param, attr.strip()))
    return errors


def _check_attr(spec: dict[str, Any], probe: Any, param: str, attr: str) -> list[str]:
    """Validate one kernel_param -> module_attr pair against the probe (or the allowlist)."""
    cls_name = spec["class_name"]

    if probe is None:
        # Degraded path: names only, no type check. Stated, not silent.
        head = attr.split(".", 1)[0]
        if head not in spec["known_attrs"]:
            return [
                f"adapter_mapping[{param!r}] = {attr!r}: {cls_name} has no attribute "
                f"{head!r} (known: {', '.join(sorted(spec['known_attrs']))})"
            ]
        return []

    found, value = _resolve(probe, attr)
    if not found:
        return [
            f"adapter_mapping[{param!r}] = {attr!r}: {cls_name} has no attribute {attr!r} "
            f"(available: {', '.join(_public_attrs(probe))})"
        ]
    if _is_method(value):
        return [
            f"adapter_mapping[{param!r}] = {attr!r}: {cls_name}.{attr} is a method, not a "
            "parameter, buffer or submodule — an adapter may only pass data"
        ]
    return []


def _resolve(obj: Any, dotted: str) -> tuple[bool, Any]:
    """Walk a dotted attribute path. Returns (found, value)."""
    current = obj
    for part in dotted.split("."):
        if not part or not hasattr(current, part):
            return False, None
        current = getattr(current, part)
    return True, current


def _is_method(value: Any) -> bool:
    """True for bound/unbound methods and plain functions — never for an `nn.Module`.

    `act_fn` on Qwen2MLP is a `SiLUActivation` module: callable, but data the adapter
    is allowed to hand to a kernel. A bound `forward` is not.
    """
    import torch

    if isinstance(value, torch.nn.Module | torch.Tensor):
        return False
    import inspect

    return inspect.isroutine(value)


def _public_attrs(probe: Any) -> list[str]:
    """Attribute names an adapter could legitimately reference, for the error message."""
    import torch

    names = {name for name, _ in probe.named_parameters(recurse=False)}
    names |= {name for name, _ in probe.named_buffers(recurse=False)}
    names |= {name for name, _ in probe.named_children()}
    names |= {
        key
        for key, value in vars(probe).items()
        if not key.startswith("_") and isinstance(value, int | float | bool | str | torch.Tensor)
    }
    return sorted(names)


def _probe_instance(spec: dict[str, Any]) -> Any:
    """A meta-device instance of the target class, or None if it cannot be built.

    Meta device means the constructor allocates no storage: building a Qwen2MLP here
    costs microseconds and zero bytes, GPU or otherwise.
    """
    try:
        import importlib

        import torch

        cls = getattr(importlib.import_module(spec["module"]), spec["class_name"])
        with torch.device("meta"):
            return spec["build"](cls)
    except Exception:  # noqa: BLE001 — a missing probe degrades the check, never blocks
        return None
