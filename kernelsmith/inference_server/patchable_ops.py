"""Which ops may be hot-swapped, and the mechanics of swapping them (spec 8.3).

The swap is a `types.MethodType` rebind of `forward` on every matching module. Nothing
is copied and nothing is re-initialized: `self.weight` and `self.variance_epsilon` stay
the original `nn.Parameter` / float, already on the right device in the right dtype.
A new forward that allocates its own weights is not a faster kernel, it is a different
model.

Matching is by class-name SUBSTRING, so one call catches every instance across all 28
decoder layers (`model.layers.0.input_layernorm`, `...post_attention_layernorm`, `norm`).
The same mechanism reaches `torch.nn.LayerNorm` in GPT-2 — nothing here is Qwen-specific
except the class names in `PATCHABLE_OPS`.

The caller must check the returned dict: an empty one means nothing matched and the
model is untouched — a silent no-op that would otherwise be reported as a successful
swap.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable, Mapping
from typing import Any

#: Swappable ops, in the order the demo attempts them. `class_name` is matched as a
#: substring against `type(module).__name__`.
#:
#: Note on "rope": `apply_rotary_pos_emb` is a MODULE-LEVEL function in
#: transformers.models.qwen2.modeling_qwen2, not a method on any module, so
#: `swap_op` cannot reach it — it matches no `named_modules()` entry and returns {}.
#: Patching it needs a module-attribute rebind instead; that is P2 stretch work and is
#: deliberately not implemented here rather than half-implemented.
#:
#: Note on "layernorm": `torch.nn.LayerNorm` IS an `nn.Module` and IS reachable from
#: `named_modules()`, so it swaps by exactly the same mechanism as Qwen2RMSNorm. It was
#: previously grouped with "rope" as unpatchable, which was simply wrong — that is what
#: makes GPT-2's normalization layers optimizable (Task 10). The class name is matched as
#: a substring like every other entry, so a subclass (`FusedLayerNorm`) matches too.
PATCHABLE_OPS: dict[str, dict[str, Any]] = {
    "rmsnorm": {"class_name": "Qwen2RMSNorm", "priority": 0},
    "layernorm": {"class_name": "LayerNorm", "priority": 0},
    "swiglu": {"class_name": "Qwen2MLP", "priority": 1},
    "rope": {"class_name": "apply_rotary_pos_emb", "priority": 2},
}


def resolve_class_name(op_name: str) -> str:
    """Map a registry op name to the class name to patch. Raises KeyError if unknown."""
    key = op_name.strip().lower()
    if key not in PATCHABLE_OPS:
        raise KeyError(f"unknown op {op_name!r}; patchable ops: {sorted(PATCHABLE_OPS)}")
    return str(PATCHABLE_OPS[key]["class_name"])


def swap_op(model: Any, op_class_name: str, new_forward_fn: Callable) -> dict[str, Callable]:
    """Rebind `forward` on every module whose class name contains `op_class_name`.

    Args:
        model: The live `nn.Module` tree.
        op_class_name: Class-name substring, e.g. "Qwen2RMSNorm".
        new_forward_fn: A plain function `(self, x, ...) -> Tensor`. It is bound as a
            method, so it receives the module — and therefore the original weights.

    Returns:
        {module_name: original bound forward}, the handle for `rollback_op`. EMPTY if
        nothing matched: the caller must treat that as a failed swap, not a success.
    """
    originals: dict[str, Callable] = {}
    for name, module in model.named_modules():
        if op_class_name in type(module).__name__:
            originals[name] = module.forward  # bound method; callable as-is for parity
            module.forward = types.MethodType(new_forward_fn, module)
    return originals


def rollback_op(model: Any, originals: dict[str, Callable]) -> int:
    """Restore the forwards saved by `swap_op`. Returns how many modules were restored.

    Where the saved handle is just the class's own method, the instance attribute is
    deleted rather than reassigned: that leaves the module byte-for-byte as it was
    before the swap, with no lingering `module.__dict__["forward"]` bound method (and
    no self-reference cycle) to confuse a later swap or rollback.
    """
    restored = 0
    for name, module in model.named_modules():
        original = originals.get(name)
        if original is None:
            continue
        class_forward = getattr(type(module), "forward", None)
        if class_forward is not None and getattr(original, "__func__", None) is class_forward:
            module.__dict__.pop("forward", None)
        else:
            module.forward = original
        restored += 1
    return restored


def find_modules(model: Any, op_class_name: str) -> list[tuple[str, Any]]:
    """Every module whose class name contains `op_class_name`, in traversal order."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if op_class_name in type(module).__name__
    ]


# --------------------------------------------------------------------------- #
# Verified kernel -> bindable forward
# --------------------------------------------------------------------------- #
#
# Two ways to close the gap below. The hard-coded adapters here are the fallback, kept
# for seed kernels; the generic path is `build_forward_from_mapping`, which uses the
# contract the CODER declared and the verifier validated (see
# `verifier/adapter_mapping.py`). Any new hard-coded op needs its own adapter; an op
# with a declared mapping needs nothing.
#
# The verifier calls a kernel wrapper with explicit weights — `entry(x, weight, eps)`
# for rmsnorm (see `OpBinding.bind` in tools/profiler_tool.py) — because the sandbox
# has no model to read them from. The live model does. These adapters close that gap:
# they pass the module's OWN parameters into the same wrapper, so the swapped-in kernel
# runs against the trained weights, at their device and dtype, with nothing copied and
# nothing re-initialized.


def _rmsnorm_adapter(entry: Callable) -> Callable:
    """Qwen2RMSNorm.forward(self, hidden_states) -> entry(x, self.weight, eps)."""

    def forward(self, hidden_states, *args, **kwargs):  # noqa: ANN001 — bound as a method
        del args, kwargs
        return entry(hidden_states, self.weight, self.variance_epsilon)

    return forward


def _swiglu_adapter(entry: Callable) -> Callable:
    """Qwen2MLP.forward(self, x) -> entry(x, gate_w, up_w, down_w), weights as [out, in]."""

    def forward(self, x, *args, **kwargs):  # noqa: ANN001 — bound as a method
        del args, kwargs
        return entry(x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight)

    return forward


def _layernorm_adapter(entry: Callable) -> Callable:
    """torch.nn.LayerNorm.forward(self, input) -> entry(x, self.weight, self.bias, self.eps).

    LayerNorm carries a `bias`, which RMSNorm does not, and calls its epsilon `eps`
    rather than `variance_epsilon` — the two reasons a single hard-coded norm adapter
    cannot cover both, and a small illustration of why the generated contract exists.

    `self.normalized_shape` is deliberately NOT passed: this fallback must match the
    signature the verifier benched the kernel against, which is
    `entry(x, weight, bias, eps)` (see `_build_layernorm` in tools/profiler_tool.py). A
    kernel that genuinely needs the shape declares it in its `adapter_mapping` and gets
    it through the generic adapter below.
    """

    def forward(self, input, *args, **kwargs):  # noqa: A002, ANN001 — bound as a method
        del args, kwargs
        return entry(input, self.weight, self.bias, self.eps)

    return forward


#: op name -> adapter. "rope" is absent on purpose: it has no module to bind to.
_ADAPTERS: dict[str, Callable[[Callable], Callable]] = {
    "rmsnorm": _rmsnorm_adapter,
    "layernorm": _layernorm_adapter,
    "swiglu": _swiglu_adapter,
}


def build_forward_from_mapping(
    kernel_entry_fn: Callable,
    adapter_mapping: Mapping[str, str],
) -> Callable:
    """The GENERIC adapter: bind a kernel using the contract the Coder declared.

    `adapter_mapping` maps kernel parameter names to module attribute names —
    `{"weight": "weight", "eps": "variance_epsilon"}` — and the forward's input tensor
    is implicit, passed positionally, never mapped. So the returned forward calls
    `kernel_entry_fn(hidden_states, weight=self.weight, eps=self.variance_epsilon)`.

    This is what makes an op the system has never seen deployable without a human
    writing an adapter for it. It is only ever called on a mapping that
    `validate_adapter_mapping` has already accepted (an attribute that exists, and is
    data rather than a method), so the lookups here are resolved eagerly per call
    against the live module — the weights stay the module's own, nothing is copied.

    Args:
        kernel_entry_fn: The verified wrapper, taking the input tensor first.
        adapter_mapping: kernel parameter -> module attribute. Dotted paths are walked.

    Returns:
        `forward(self, hidden_states)`, ready for `types.MethodType` via `swap_op`.
    """
    mapping = {str(k): str(v) for k, v in dict(adapter_mapping).items()}

    def forward(self, hidden_states, *args, **kwargs):  # noqa: ANN001 — bound as a method
        del args, kwargs
        return kernel_entry_fn(
            hidden_states, **{param: _resolve_attr(self, attr) for param, attr in mapping.items()}
        )

    return forward


def _resolve_attr(module: Any, dotted: str) -> Any:
    """Walk a dotted attribute path on the live module. Raises AttributeError if absent."""
    value = module
    for part in dotted.split("."):
        value = getattr(value, part)
    return value


def build_forward(
    op_name: str,
    entrypoint_fn: Callable,
    adapter_mapping: Mapping[str, str] | None = None,
) -> Callable:
    """Turn a verified kernel wrapper into a `forward(self, x)` ready for `swap_op`.

    Three paths, in order of precedence:

    1. A wrapper whose first parameter is `self` is already a forward and is used
       as-is — the path for a Coder that writes the method directly.
    2. A declared, validated `adapter_mapping` goes through the generic adapter above.
       This is the path that works for ops nobody hard-coded an adapter for.
    3. Otherwise the op's hard-coded adapter, kept as the fallback for seed kernels
       written before `adapter_mapping` existed.

    Raises:
        KeyError: unknown op name.
        ValueError: no mapping was declared and the op has no hard-coded adapter
            (e.g. "rope", a module-level function with no module to bind to).
    """
    resolve_class_name(op_name)  # raises KeyError for an unknown op
    key = op_name.strip().lower()

    if _takes_self(entrypoint_fn):
        return entrypoint_fn

    if adapter_mapping:
        return build_forward_from_mapping(entrypoint_fn, adapter_mapping)

    adapter = _ADAPTERS.get(key)
    if adapter is None:
        raise ValueError(
            f"op {key!r} has no forward adapter and no adapter_mapping was declared: it "
            "is not bound to a module (patch it at module scope, declare an "
            "adapter_mapping, or give the entrypoint a `self` first argument)"
        )
    return adapter(entrypoint_fn)


def _takes_self(fn: Callable) -> bool:
    """True if `fn` is already written as a method — first parameter named `self`."""
    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return False
    return bool(params) and params[0] == "self"
