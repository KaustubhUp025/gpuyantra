"""Which ops may be hot-swapped, and the mechanics of swapping them (spec 8.3).

The swap is a `types.MethodType` rebind of `forward` on every matching module. Nothing
is copied and nothing is re-initialized: `self.weight` and `self.variance_epsilon` stay
the original `nn.Parameter` / float, already on the right device in the right dtype.
A new forward that allocates its own weights is not a faster kernel, it is a different
model.

Matching is by class-name SUBSTRING, so one call catches every instance across all 28
decoder layers (`model.layers.0.input_layernorm`, `...post_attention_layernorm`, `norm`).

The caller must check the returned dict: an empty one means nothing matched and the
model is untouched — a silent no-op that would otherwise be reported as a successful
swap.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable
from typing import Any

#: Swappable ops, in the order the demo attempts them. `class_name` is matched as a
#: substring against `type(module).__name__`.
#:
#: Note on "rope": `apply_rotary_pos_emb` is a MODULE-LEVEL function in
#: transformers.models.qwen2.modeling_qwen2, not a method on any module, so
#: `swap_op` cannot reach it — it matches no `named_modules()` entry and returns {}.
#: Patching it needs a module-attribute rebind instead; that is P2 stretch work and is
#: deliberately not implemented here rather than half-implemented.
PATCHABLE_OPS: dict[str, dict[str, Any]] = {
    "rmsnorm": {"class_name": "Qwen2RMSNorm", "priority": 0},
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


#: op name -> adapter. "rope" is absent on purpose: it has no module to bind to.
_ADAPTERS: dict[str, Callable[[Callable], Callable]] = {
    "rmsnorm": _rmsnorm_adapter,
    "swiglu": _swiglu_adapter,
}


def build_forward(op_name: str, entrypoint_fn: Callable) -> Callable:
    """Turn a verified kernel wrapper into a `forward(self, x)` ready for `swap_op`.

    A wrapper whose first parameter is `self` is already a forward and is used as-is —
    that is the path for a Coder that writes the method directly. Anything else goes
    through the op's adapter, which supplies the module's own weights.

    Raises:
        KeyError: unknown op name.
        ValueError: the op has no adapter (e.g. "rope", a module-level function).
    """
    resolve_class_name(op_name)  # raises KeyError for an unknown op
    key = op_name.strip().lower()

    if _takes_self(entrypoint_fn):
        return entrypoint_fn

    adapter = _ADAPTERS.get(key)
    if adapter is None:
        raise ValueError(
            f"op {key!r} has no forward adapter: it is not bound to a module "
            "(patch it at module scope, or give the entrypoint a `self` first argument)"
        )
    return adapter(entrypoint_fn)


def _takes_self(fn: Callable) -> bool:
    """True if `fn` is already written as a method — first parameter named `self`."""
    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return False
    return bool(params) and params[0] == "self"
