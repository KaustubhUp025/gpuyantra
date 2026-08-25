"""Hot-swap mechanics: rebind, parity, rollback (spec 8.3 / 13.1 / 13.4).

Qwen is never loaded here — a 1.5B download is not a unit test. The stand-in is a
tiny `nn.Module` tree whose norm class is named exactly like the transformers one, so
the registry's class-name substring match lands on it the same way it lands on the real
model. What is actually under test is the part that has to be right before anything is
served to a user:

- the swap reuses the ORIGINAL `nn.Parameter` (a kernel that re-initializes weights is
  not an optimization, it is a different model),
- rollback restores the stock forward exactly, including after a second swap,
- a kernel that disagrees numerically is refused and auto-rolled-back,
- `/generate` and `/swap` are serialized by one `asyncio.Lock`.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn

from kernelsmith.config import CORRECTNESS_SEEDS, INFERENCE_HOST, INFERENCE_PORT
from kernelsmith.inference_server import server
from kernelsmith.inference_server.models import TokenMeter
from kernelsmith.inference_server.patchable_ops import (
    PATCHABLE_OPS,
    build_forward,
    find_modules,
    resolve_class_name,
    rollback_op,
    swap_op,
)
from kernelsmith.inference_server.server import apply_swap, check_parity
from kernelsmith.tools import hotswap_tool as hotswap_module
from kernelsmith.tools.hotswap_tool import hotswap_kernel, hotswap_tool

HIDDEN = 32


# --------------------------------------------------------------------------- fakes


class Qwen2RMSNorm(nn.Module):
    """Same class name and same attributes as the real one, three orders smaller.

    The name is the point: `PATCHABLE_OPS["rmsnorm"]["class_name"]` is matched as a
    substring, so a differently named stub would test nothing about the registry.
    """

    def __init__(self, hidden_size: int = HIDDEN):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(hidden_size) * 0.1 + 1.0)
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        var = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normed = hidden_states.to(torch.float32) * torch.rsqrt(var + self.variance_epsilon)
        return normed.to(hidden_states.dtype) * self.weight


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = Qwen2RMSNorm()
        self.post_attention_layernorm = Qwen2RMSNorm()
        self.mlp = nn.Linear(HIDDEN, HIDDEN)  # must NOT be touched by a norm swap


class DummyModel(nn.Module):
    """Three norms across two layers plus a final norm, exactly like the real tree."""

    def __init__(self, layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(DummyLayer() for _ in range(layers))
        self.norm = Qwen2RMSNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


#: A correct RMSNorm in the verifier's wrapper signature: `entry(x, weight, eps)`.
GOOD_KERNEL = '''
import torch

def rmsnorm_forward(x, weight, eps):
    """Stands in for a Triton kernel; the swap mechanics do not care which it is."""
    var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (x.to(torch.float32) * torch.rsqrt(var + eps)).to(x.dtype) * weight
'''

#: Same shape and dtype, wrong numbers — the failure mode parity exists to catch.
WRONG_KERNEL = """
import torch

def rmsnorm_forward(x, weight, eps):
    var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (x.to(torch.float32) * torch.rsqrt(var + eps)).to(x.dtype) * weight * 1.5
"""

#: Already written as a method: `build_forward` must bind it without an adapter.
METHOD_KERNEL = """
import torch

def rmsnorm_forward(self, hidden_states):
    var = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    normed = hidden_states.to(torch.float32) * torch.rsqrt(var + self.variance_epsilon)
    return normed.to(hidden_states.dtype) * self.weight
"""

#: Rule 1 of the static checker: hand the work back to torch and claim the speedup.
HACK_KERNEL = """
import torch

def rmsnorm_forward(x, weight, eps):
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)
"""


@pytest.fixture
def model() -> DummyModel:
    torch.manual_seed(0)
    return DummyModel()


@pytest.fixture
def probe() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 8, HIDDEN)


# ------------------------------------------------------------------- swap_op


def test_swap_op_rebinds_every_matching_module(model, probe):
    norms = dict(find_modules(model, "RMSNorm"))
    assert set(norms) == {
        "layers.0.input_layernorm",
        "layers.0.post_attention_layernorm",
        "layers.1.input_layernorm",
        "layers.1.post_attention_layernorm",
        "norm",
    }

    before = model.norm(probe)
    originals = swap_op(model, "RMSNorm", build_forward("rmsnorm", _load(WRONG_KERNEL)))

    assert set(originals) == set(norms)
    after = model.norm(probe)
    assert not torch.allclose(before, after), "forward did not actually change"
    torch.testing.assert_close(after, before * 1.5)


def test_swap_op_reuses_the_original_weight_tensor(model):
    """Gotcha 3: the learned gain is reused in place, never re-initialized."""
    weights = {name: module.weight for name, module in find_modules(model, "RMSNorm")}
    eps = model.norm.variance_epsilon

    swap_op(model, "RMSNorm", build_forward("rmsnorm", _load(GOOD_KERNEL)))

    for name, module in find_modules(model, "RMSNorm"):
        assert module.weight is weights[name], f"{name}: weight tensor was replaced"
        assert isinstance(module.weight, nn.Parameter)
    assert model.norm.variance_epsilon == eps


def test_swap_op_leaves_non_matching_modules_alone(model, probe):
    mlp = model.layers[0].mlp
    expected = mlp(probe)

    swap_op(model, "RMSNorm", build_forward("rmsnorm", _load(WRONG_KERNEL)))

    assert "forward" not in mlp.__dict__
    torch.testing.assert_close(mlp(probe), expected)


def test_swap_op_on_a_class_nothing_matches_is_a_no_op(model):
    """An empty handle is how the caller learns the swap patched nothing."""
    assert swap_op(model, "Qwen2MLP", lambda self, x: x) == {}


# ---------------------------------------------------------------- rollback_op


def test_rollback_op_restores_the_original_forward(model, probe):
    before = model.norm(probe)
    originals = swap_op(model, "RMSNorm", build_forward("rmsnorm", _load(WRONG_KERNEL)))

    assert rollback_op(model, originals) == len(originals)

    torch.testing.assert_close(model.norm(probe), before)
    for _, module in find_modules(model, "RMSNorm"):
        # Not merely re-pointed at the right function: the instance attribute the swap
        # created is gone, so the module is byte-for-byte what it was.
        assert "forward" not in module.__dict__


def test_rollback_op_ignores_modules_it_did_not_swap(model, probe):
    originals = swap_op(model, "RMSNorm", build_forward("rmsnorm", _load(WRONG_KERNEL)))
    partial = {"norm": originals["norm"]}

    assert rollback_op(model, partial) == 1
    torch.testing.assert_close(model.norm(probe), model.norm.__class__.forward(model.norm, probe))
    assert "forward" in model.layers[0].input_layernorm.__dict__


# --------------------------------------------------------------- build_forward


def test_build_forward_binds_a_method_style_kernel_directly(model, probe):
    forward = build_forward("rmsnorm", _load(METHOD_KERNEL))
    swap_op(model, "RMSNorm", forward)
    torch.testing.assert_close(model.norm(probe), Qwen2RMSNorm.forward(model.norm, probe))


def test_build_forward_rejects_an_op_with_no_module_to_bind_to():
    with pytest.raises(ValueError, match="no forward adapter"):
        build_forward("rope", _load(GOOD_KERNEL))


def test_build_forward_rejects_an_unknown_op():
    with pytest.raises(KeyError):
        build_forward("attention", _load(GOOD_KERNEL))


def test_registry_lists_the_three_patch_targets():
    assert resolve_class_name("rmsnorm") == "Qwen2RMSNorm"
    assert resolve_class_name("swiglu") == "Qwen2MLP"
    assert [
        name for name, _ in sorted(PATCHABLE_OPS.items(), key=lambda kv: kv[1]["priority"])
    ] == [
        "rmsnorm",
        "swiglu",
        "rope",
    ]


# --------------------------------------------------------------- parity gate


def test_check_parity_passes_on_a_matching_kernel(model):
    original = model.norm.forward
    swap_op(model, "RMSNorm", build_forward("rmsnorm", _load(GOOD_KERNEL)))

    parity = check_parity(model.norm, original)

    assert parity["parity_pass"] is True
    assert parity["failures"] == []
    assert parity["seeds"] == CORRECTNESS_SEEDS


def test_check_parity_fails_on_a_mismatching_kernel(model):
    original = model.norm.forward
    swap_op(model, "RMSNorm", build_forward("rmsnorm", _load(WRONG_KERNEL)))

    parity = check_parity(model.norm, original)

    assert parity["parity_pass"] is False
    assert len(parity["failures"]) == CORRECTNESS_SEEDS
    assert "allclose failed" in parity["failures"][0]


def test_check_parity_catches_nan(model):
    original = model.norm.forward
    swap_op(model, "RMSNorm", lambda self, x: x * float("nan"))

    assert "NaN/Inf" in check_parity(model.norm, original)["failures"][0]


# ----------------------------------------------------------------- apply_swap


def test_apply_swap_keeps_a_kernel_that_matches(model):
    result = apply_swap(model, "rmsnorm", GOOD_KERNEL, "rmsnorm_forward")

    assert result["success"] is True
    assert result["modules_patched"] == 5
    assert result["parity"]["parity_pass"] is True
    assert "forward" in model.norm.__dict__, "the kernel should still be live"


def test_apply_swap_auto_rolls_back_a_kernel_that_does_not(model, probe):
    before = model.norm(probe)

    result = apply_swap(model, "rmsnorm", WRONG_KERNEL, "rmsnorm_forward")

    assert result["success"] is False
    assert result["rolled_back"] is True
    assert "numeric parity failed" in result["error"]
    torch.testing.assert_close(model.norm(probe), before)
    for _, module in find_modules(model, "RMSNorm"):
        assert "forward" not in module.__dict__


def test_apply_swap_measures_parity_against_the_true_original(model, probe):
    """A second swap must be judged against stock behaviour, not against kernel #1."""
    first = apply_swap(model, "rmsnorm", GOOD_KERNEL, "rmsnorm_forward")
    baseline = first["originals"]

    result = apply_swap(model, "rmsnorm", WRONG_KERNEL, "rmsnorm_forward", baseline)

    assert result["success"] is False
    assert result["rolled_back"] is True
    # Rolled back to kernel #1, which is what was live when this swap arrived.
    torch.testing.assert_close(model.norm(probe), model.norm.weight * _rms(probe))
    assert rollback_op(model, baseline) == len(baseline)
    torch.testing.assert_close(model.norm(probe), Qwen2RMSNorm.forward(model.norm, probe))


def test_apply_swap_refuses_a_reward_hacked_kernel(model):
    """The static checker runs again at the door: /swap is not a way around it."""
    result = apply_swap(model, "rmsnorm", HACK_KERNEL, "rmsnorm_forward")

    assert result["success"] is False
    assert "static checker" in result["error"]
    assert "forward" not in model.norm.__dict__


def test_apply_swap_refuses_an_unknown_op(model):
    result = apply_swap(model, "attention", GOOD_KERNEL, "rmsnorm_forward")
    assert result["success"] is False
    assert "unknown op" in result["error"]


def test_apply_swap_refuses_a_missing_entrypoint(model):
    result = apply_swap(model, "rmsnorm", GOOD_KERNEL, "not_defined")
    assert result["success"] is False
    assert "could not load entrypoint" in result["error"]
    assert "forward" not in model.norm.__dict__


def test_apply_swap_refuses_when_nothing_matches(model):
    """Silently patching zero modules would be reported as a successful speedup."""
    result = apply_swap(model, "swiglu", GOOD_KERNEL, "rmsnorm_forward")
    assert result["success"] is False
    assert "no module whose class name contains 'Qwen2MLP'" in result["error"]


def test_apply_swap_rolls_back_a_kernel_that_raises(model, probe):
    before = model.norm(probe)
    exploding = "def rmsnorm_forward(x, weight, eps):\n    raise RuntimeError('boom')\n"

    result = apply_swap(model, "rmsnorm", exploding, "rmsnorm_forward")

    assert result["success"] is False
    assert result["rolled_back"] is True
    torch.testing.assert_close(model.norm(probe), before)


# ------------------------------------------------------------------ endpoints


@pytest.fixture
def client(model):
    """A TestClient over a server whose model is the stub.

    Constructed without `with`, so the lifespan handler — and its Qwen download — never
    runs; STATE is populated by hand instead.
    """
    server.STATE.model = model
    server.STATE.tokenizer = None
    server.STATE.meter = TokenMeter(model, None)
    server.STATE.originals.clear()
    try:
        yield TestClient(server.app)
    finally:
        server.STATE.model = None
        server.STATE.meter = None
        server.STATE.originals.clear()


def test_swap_endpoint_holds_the_asyncio_lock():
    """Gotcha 2: one lock, and generation takes it too, or a patch lands mid-decode."""
    assert isinstance(server._SWAP_LOCK, asyncio.Lock)
    for endpoint in (server.swap, server.rollback, server.generate):
        assert "async with _SWAP_LOCK" in inspect.getsource(endpoint), endpoint.__name__


def test_swap_endpoint_patches_and_reports_stats(client):
    response = client.post(
        "/swap",
        json={"op_name": "rmsnorm", "kernel_source": GOOD_KERNEL, "entrypoint": "rmsnorm_forward"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["modules_patched"] == 5
    assert body["stats"]["active_kernel"] == "rmsnorm"
    assert body["stats"]["last_swap_ts"] is not None
    assert "originals" not in body, "rollback handles must not leave the server"
    assert client.get("/stats").json()["active_kernel"] == "rmsnorm"


def test_swap_endpoint_reports_a_parity_failure_without_leaving_it_live(client, model, probe):
    before = model.norm(probe)

    body = client.post(
        "/swap",
        json={"op_name": "rmsnorm", "kernel_source": WRONG_KERNEL, "entrypoint": "rmsnorm_forward"},
    ).json()

    assert body["success"] is False
    assert body["rolled_back"] is True
    assert body["stats"]["active_kernel"] == "none"
    torch.testing.assert_close(model.norm(probe), before)


def test_rollback_endpoint_restores_the_stock_forward(client, model, probe):
    before = model.norm(probe)
    client.post(
        "/swap",
        json={"op_name": "rmsnorm", "kernel_source": GOOD_KERNEL, "entrypoint": "rmsnorm_forward"},
    )

    body = client.post("/rollback", json={"op_name": "rmsnorm"}).json()

    assert body["success"] is True
    assert body["modules_restored"] == 5
    assert body["stats"]["active_kernel"] == "none"
    torch.testing.assert_close(model.norm(probe), before)


def test_rollback_endpoint_after_two_swaps_reaches_the_stock_forward(client, model, probe):
    before = model.norm(probe)
    for _ in range(2):
        assert (
            client.post(
                "/swap",
                json={
                    "op_name": "rmsnorm",
                    "kernel_source": GOOD_KERNEL,
                    "entrypoint": "rmsnorm_forward",
                },
            ).json()["success"]
            is True
        )

    client.post("/rollback", json={"op_name": "rmsnorm"})

    torch.testing.assert_close(model.norm(probe), before)
    for _, module in find_modules(model, "RMSNorm"):
        assert "forward" not in module.__dict__


def test_rollback_endpoint_without_an_active_swap_says_so(client):
    body = client.post("/rollback", json={"op_name": "rmsnorm"}).json()
    assert body["success"] is False
    assert "no active swap" in body["error"]


def test_endpoints_report_503_before_the_model_is_loaded():
    server.STATE.model = None
    server.STATE.meter = None
    client = TestClient(server.app)
    assert client.get("/stats").status_code == 503
    assert (
        client.post(
            "/swap",
            json={"op_name": "rmsnorm", "kernel_source": GOOD_KERNEL, "entrypoint": "f"},
        ).status_code
        == 503
    )


def test_health_reports_the_served_model(client):
    body = client.get("/health").json()
    assert body["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert body["model_loaded"] is True
    assert body["status"] in {"ok", "degraded"}
    assert "available" in body["gpu"]


# ----------------------------------------------------------------- TokenMeter


def test_token_meter_starts_empty():
    meter = TokenMeter(None, None)
    assert meter.stats() == {
        "tokens_per_s": 0.0,
        "tokens_total": 0,
        "active_kernel": "none",
        "last_swap_ts": None,
    }


def test_token_meter_averages_over_its_window():
    meter = TokenMeter(None, None)
    meter.record(50, 1.0)
    meter.record(150, 1.0)

    assert meter.total_tokens_generated == 200
    assert meter.rolling_tokens_per_sec == pytest.approx(100.0)


def test_token_meter_window_is_bounded():
    meter = TokenMeter(None, None, window=2)
    meter.record(10, 1.0)
    meter.record(100, 1.0)
    meter.record(100, 1.0)

    assert meter.total_tokens_generated == 210
    assert meter.rolling_tokens_per_sec == pytest.approx(100.0), "the slow sample should age out"


def test_token_meter_restarts_the_average_on_a_swap():
    """After a swap, tokens/s must describe the NEW kernel, not an average across it."""
    meter = TokenMeter(None, None)
    meter.record(10, 1.0)

    meter.record_swap("rmsnorm")

    assert meter.active_kernel == "rmsnorm"
    assert meter.last_swap_ts is not None
    assert meter.rolling_tokens_per_sec == 0.0
    assert meter.total_tokens_generated == 10, "the total is cumulative across swaps"

    meter.record(90, 1.0)
    assert meter.rolling_tokens_per_sec == pytest.approx(90.0)

    meter.record_rollback()
    assert meter.active_kernel == "none"


# --------------------------------------------------------------- hotswap_tool


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_hotswap_tool_posts_to_the_configured_server(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(url=url, json=json, timeout=timeout)
        return FakeResponse(200, {"success": True, "modules_patched": 5})

    monkeypatch.setattr(hotswap_module.httpx, "post", fake_post)

    result = hotswap_kernel(GOOD_KERNEL, "rmsnorm_forward", "rmsnorm")

    assert result["success"] is True
    assert captured["url"] == f"http://{INFERENCE_HOST}:{INFERENCE_PORT}/swap"
    assert captured["json"] == {
        "op_name": "rmsnorm",
        "kernel_source": GOOD_KERNEL,
        "entrypoint": "rmsnorm_forward",
    }


def test_hotswap_tool_reports_a_refused_swap_as_a_failure(monkeypatch):
    monkeypatch.setattr(
        hotswap_module.httpx,
        "post",
        lambda url, json, timeout: FakeResponse(
            200, {"success": False, "error": "numeric parity failed", "rolled_back": True}
        ),
    )

    result = hotswap_kernel(WRONG_KERNEL, "rmsnorm_forward", "rmsnorm")

    assert result["success"] is False
    assert result["rolled_back"] is True


def test_hotswap_tool_never_reports_an_unreachable_server_as_success(monkeypatch):
    def boom(url, json, timeout):
        raise hotswap_module.httpx.ConnectError("connection refused")

    monkeypatch.setattr(hotswap_module.httpx, "post", boom)

    result = hotswap_kernel(GOOD_KERNEL, "rmsnorm_forward", "rmsnorm")

    assert result["success"] is False
    assert "unreachable" in result["error"]


def test_hotswap_tool_handles_a_broken_server(monkeypatch):
    monkeypatch.setattr(
        hotswap_module.httpx, "post", lambda url, json, timeout: FakeResponse(500, None, "boom")
    )
    assert hotswap_kernel(GOOD_KERNEL, "rmsnorm_forward", "rmsnorm")["success"] is False

    monkeypatch.setattr(
        hotswap_module.httpx, "post", lambda url, json, timeout: FakeResponse(200, None, "<html>")
    )
    assert hotswap_kernel(GOOD_KERNEL, "rmsnorm_forward", "rmsnorm")["success"] is False


def test_hotswap_tool_is_wired_into_the_supervisor():
    from kernelsmith.agents.supervisor import build_supervisor

    tools = [t.name for t in build_supervisor().tools]
    assert tools == ["retrieve_skills_for_agent", "upsert_skill", "hotswap_kernel"]
    assert hotswap_tool.name == "hotswap_kernel"


# ------------------------------------------------------------------- helpers


def _load(source: str):
    namespace: dict = {}
    exec(compile(source, "<test-kernel>", "exec"), namespace)
    return namespace["rmsnorm_forward"]


def _rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (x.to(torch.float32) * torch.rsqrt(var + eps)).to(x.dtype)
