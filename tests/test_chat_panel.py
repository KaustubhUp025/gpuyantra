"""The live inference panel and the target selectors (Task 14).

The panel is the reason the Tokens/s card means anything during a demo: the server's meter
only moves when somebody asks the model for something, and until Task 14 nobody did.

Everything here runs against a **stub inference server** on an ephemeral port — a real one
needs an L4 and 3 GB of VRAM, and the thing under test is the dashboard's half of the
conversation: the request it sends, the numbers it reads back, which forward it attributes
them to, and what it does when the server is not there at all. `kernelsmith/inference_server/`
is off-limits to Task 14, so the stub answers with that server's exact schema
(`/generate` -> `{text, tokens, time_ms}`, `/stats` -> `{tokens_per_s, active_kernel, …}`)
and the tests fail if the dashboard starts expecting something else.

One trap worth recording, because it cost a debugging round and is invisible until a
server is reachable: the prompt box was originally an `st.form`. A form created in this
container leaves a form context behind that Streamlit finds when the SIDEBAR's "Start Run"
button is created on the next run, and refuses it — "st.button() can't be used in an
st.form()". The whole live page died with the panel's first render. It is a plain input
plus a button now, and `test_the_live_page_renders_with_a_server_reachable` is what would
catch a regression.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from kernelsmith.ui.demo_dashboard import (
    CHAT_HISTORY_KEPT,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_OP,
    OP_LABELS,
    PRESET_PROMPTS,
    available_ops,
    chat_throughput_pair,
    hidden_size_label,
    hidden_size_options,
    inference_server_is_up,
    op_deploys,
    op_option_label,
    send_prompt,
    should_autorefresh,
)

APP = Path(__file__).resolve().parent.parent / "kernelsmith" / "ui" / "demo_dashboard.py"
TIMEOUT_S = 180


# --------------------------------------------------------------------------- #
# A stub that speaks the real server's schema
# --------------------------------------------------------------------------- #


class StubServer:
    """`/health`, `/stats` and `/generate`, with the shapes `server.py` actually returns."""

    def __init__(self) -> None:
        self.active_kernel = "none"
        self.tokens_total = 0
        self.calls: list[dict[str, Any]] = []
        self.generate_status = 200
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # noqa: A002 - quiet in tests
                pass

            def _send(self, payload: Any, code: int = 200) -> None:
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - http.server's naming
                if self.path == "/health":
                    self._send({"status": "ok", "model": "stub", "model_loaded": True})
                elif self.path == "/stats":
                    self._send(
                        {
                            "tokens_per_s": 22.9,
                            "tokens_total": stub.tokens_total,
                            "active_kernel": stub.active_kernel,
                            "last_swap_ts": None,
                        }
                    )
                else:
                    self._send({"error": "not found"}, 404)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                request = json.loads(self.rfile.read(length) or b"{}")
                if self.path != "/generate":
                    self._send({"error": "not found"}, 404)
                    return
                stub.calls.append(request)
                if stub.generate_status != 200:
                    self._send({"error": "boom"}, stub.generate_status)
                    return
                tokens = min(int(request.get("max_tokens") or 8), 12)
                stub.tokens_total += tokens
                self._send(
                    {"text": "Paris is the capital of France.", "tokens": tokens, "time_ms": 480.0}
                )

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> StubServer:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def clear_health_cache() -> None:
    """Drop the cached `/health` probe, in this module AND in the app under test.

    `AppTest.from_file` execs the dashboard as a fresh module, so the script's
    `inference_server_is_up` is a different cache-decorated object from the one imported
    here — clearing only ours leaves the script holding a stale "no server", which is
    exactly the failure this helper exists to prevent.
    """
    import streamlit as st

    inference_server_is_up.clear()
    st.cache_data.clear()


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch):
    """A stub server, with the dashboard pointed at it. Health cache cleared both ways."""
    import kernelsmith.config as config

    with StubServer() as server:
        monkeypatch.setattr(config, "INFERENCE_HOST", "127.0.0.1")
        monkeypatch.setattr(config, "INFERENCE_PORT", server.port)
        clear_health_cache()
        yield server
    clear_health_cache()


@pytest.fixture
def no_server(monkeypatch: pytest.MonkeyPatch):
    """A port nothing is listening on — the `--no-server` case, which must not crash."""
    import kernelsmith.config as config

    monkeypatch.setattr(config, "INFERENCE_HOST", "127.0.0.1")
    monkeypatch.setattr(config, "INFERENCE_PORT", 1)  # privileged and unbound
    clear_health_cache()
    yield
    clear_health_cache()


# --------------------------------------------------------------------------- #
# send_prompt
# --------------------------------------------------------------------------- #


def test_a_prompt_reaches_the_server_in_the_shape_it_expects(stub: StubServer):
    record = send_prompt("What is the capital of France?", max_tokens=12)
    assert record["ok"] is True
    assert stub.calls[-1] == {
        "prompt": "What is the capital of France?",
        "max_tokens": 12,
        "temperature": 0.0,
    }


def test_the_throughput_is_the_servers_own_numbers_not_a_stopwatch(stub: StubServer):
    """12 tokens in 480 ms = 25 tok/s, and the dashboard's HTTP overhead is not in it."""
    record = send_prompt("hi", max_tokens=12)
    assert record["tokens"] == 12
    assert record["time_ms"] == pytest.approx(480.0)
    assert record["tokens_per_s"] == pytest.approx(25.0)


def test_the_answer_is_returned_for_the_viewer_to_read(stub: StubServer):
    """Coherent output across a swap is the half of the claim a speedup cannot prove."""
    assert "Paris" in send_prompt("hi")["text"]


def test_the_request_records_which_forward_served_it(stub: StubServer):
    assert send_prompt("hi")["kernel"] == "none"
    stub.active_kernel = "rmsnorm"
    assert send_prompt("hi")["kernel"] == "rmsnorm"


def test_an_unreachable_server_is_an_answer_not_a_crash(no_server):
    record = send_prompt("hi")
    assert record["ok"] is False
    assert record["error"]
    assert record["tokens_per_s"] is None


def test_an_http_error_is_reported_rather_than_swallowed(stub: StubServer):
    stub.generate_status = 500
    record = send_prompt("hi")
    assert record["ok"] is False
    assert "500" in record["error"] or "Error" in record["error"]


def test_zero_tokens_produces_no_rate_rather_than_a_division(stub: StubServer, monkeypatch):
    """A server that returns 0 tokens must not put a fabricated number on the card."""
    import httpx

    class Response:
        status_code = 200

        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, Any]:
            return {"text": "", "tokens": 0, "time_ms": 0.0}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: Response())
    assert send_prompt("hi")["tokens_per_s"] is None


# --------------------------------------------------------------------------- #
# history and the before/after pair
# --------------------------------------------------------------------------- #


def exchange(rate: float | None, kernel: str, ok: bool = True) -> dict[str, Any]:
    return {"ok": ok, "tokens_per_s": rate, "kernel": kernel, "prompt": "hi", "text": "x"}


def test_the_pair_needs_both_sides_of_the_swap():
    assert chat_throughput_pair([exchange(5.7, "none")]) is None
    assert chat_throughput_pair([exchange(22.9, "rmsnorm")]) is None


def test_the_pair_is_the_last_request_on_each_side():
    history = [
        exchange(4.0, "none"),
        exchange(5.7, "none"),
        exchange(20.0, "rmsnorm"),
        exchange(22.9, "rmsnorm"),
    ]
    assert chat_throughput_pair(history) == (5.7, 22.9)


def test_the_pair_ignores_failed_requests_and_unknown_kernels():
    history = [
        exchange(5.7, "none"),
        exchange(99.0, "none", ok=False),
        exchange(99.0, "unknown"),
        exchange(22.9, "rmsnorm"),
    ]
    assert chat_throughput_pair(history) == (5.7, 22.9)


def test_history_is_capped_so_a_long_demo_cannot_grow_forever():
    import streamlit as st

    from kernelsmith.ui.demo_dashboard import record_chat

    st.session_state["chat_history"] = []
    for index in range(CHAT_HISTORY_KEPT + 4):
        record_chat(exchange(float(index), "none"))
    history = st.session_state["chat_history"]
    assert len(history) == CHAT_HISTORY_KEPT
    assert history[-1]["tokens_per_s"] == float(CHAT_HISTORY_KEPT + 3)


# --------------------------------------------------------------------------- #
# The selectors (problem 2)
# --------------------------------------------------------------------------- #


def test_the_op_selector_only_offers_ops_the_profiler_can_build():
    """A dropdown that can produce a KeyError deep in a tool is not a dropdown."""
    from kernelsmith.tools.profiler_tool import OP_REGISTRY

    ops = available_ops()
    assert ops, "the selector must not be empty"
    assert set(ops) <= set(OP_REGISTRY)
    assert set(ops) <= set(OP_LABELS)


def test_the_default_target_is_the_one_the_headline_number_comes_from():
    assert available_ops()[0] == DEFAULT_OP == "rmsnorm"
    assert DEFAULT_HIDDEN_SIZE == 1536


def test_deployable_ops_come_first_in_the_list():
    ops = available_ops()
    deployable = [index for index, op in enumerate(ops) if op_deploys(op)]
    others = [index for index, op in enumerate(ops) if not op_deploys(op)]
    assert not others or not deployable or max(deployable) < min(others)


def test_rope_is_not_offered_as_deployable():
    """`PATCHABLE_OPS` lists it, but `swap_op` cannot reach a module-level function."""
    assert op_deploys("rope") is False
    assert "no live deployment" in op_option_label("rope")


def test_rmsnorm_and_layernorm_are_the_ops_that_reach_a_live_server():
    assert op_deploys("rmsnorm") and op_deploys("layernorm")
    assert "deploys to the live server" in op_option_label("rmsnorm")


def test_every_offered_op_has_a_plain_english_description():
    for op in available_ops():
        label = op_option_label(op)
        assert "—" in label, op
        assert len(label.split()) >= 6, op


def test_the_hidden_sizes_come_from_the_model_registry():
    from kernelsmith.config import MODEL_REGISTRY

    options = hidden_size_options()
    assert {size for size, _ in options} == {
        int(spec["hidden_size"]) for spec in MODEL_REGISTRY.values()
    }


def test_the_served_models_width_is_offered_first_and_says_so():
    size, note = hidden_size_options()[0]
    assert size == DEFAULT_HIDDEN_SIZE
    assert "served model" in note
    assert hidden_size_label(size, note).startswith("1536 — ")


def test_every_width_names_the_model_it_belongs_to():
    for size, note in hidden_size_options():
        assert note.strip(), size
        assert str(size) not in note or "—" in note


# --------------------------------------------------------------------------- #
# The autorefresh gate
# --------------------------------------------------------------------------- #


class FakeConsumer:
    def __init__(self, running: bool) -> None:
        self.is_running = running


def test_the_page_ticks_while_the_agents_are_working():
    assert should_autorefresh(FakeConsumer(True), {}) is True


def test_the_page_ticks_while_turn_two_is_still_owed():
    """The follow-up message is fired from a tick; without one, the run never deploys."""
    assert should_autorefresh(FakeConsumer(False), {"awaiting_followup": True}) is True


def test_the_page_stops_ticking_when_nothing_is_running():
    """A 1 Hz rerun landing mid-generation kills the script that is waiting for it."""
    assert should_autorefresh(FakeConsumer(False), {"awaiting_followup": False}) is False


def test_a_consumer_that_will_not_say_is_treated_as_busy():
    class Broken:
        @property
        def is_running(self) -> bool:
            raise RuntimeError("no")

    assert should_autorefresh(Broken(), {}) is True


# --------------------------------------------------------------------------- #
# The panel, through the real app
# --------------------------------------------------------------------------- #


@pytest.fixture
def app():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    return at


@pytest.fixture
def live_app(stub: StubServer):
    """The app in Live mode, with the stub already reachable on its first run.

    Built after the stub on purpose: an AppTest that runs while nothing is listening
    caches a "no server" probe inside its own module, and the panel then stays hidden for
    the rest of the test.
    """
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    at.sidebar.radio[0].set_value("🔴 Live").run()
    return at


def exceptions(at) -> list[str]:
    return [str(element.value) for element in at.exception]


def test_the_live_sidebar_offers_dropdowns_not_free_text(app):
    app.sidebar.radio[0].set_value("🔴 Live").run()
    assert exceptions(app) == []
    labels = [box.label for box in app.sidebar.selectbox]
    assert "Layer type" in labels
    assert "Model width" in labels
    assert app.sidebar.text_input == []


def test_with_no_server_the_panel_says_so_instead_of_crashing(app, no_server):
    app.sidebar.radio[0].set_value("🔴 Live").run()
    assert exceptions(app) == []
    assert any("No inference server is running" in note.value for note in app.info)


def test_the_live_page_renders_with_a_server_reachable(live_app, stub: StubServer):
    """The regression guard for the `st.form` trap described in this module's docstring."""
    app = live_app
    assert exceptions(app) == []
    labels = [button.label for button in app.button]
    for preset_label, _prompt in PRESET_PROMPTS:
        assert preset_label in labels
    assert "Send" in labels


def test_a_preset_click_sends_the_prompt_and_shows_what_came_back(live_app, stub: StubServer):
    app = live_app
    preset_label, preset_prompt = PRESET_PROMPTS[0]
    next(button for button in app.button if button.label == preset_label).click().run()

    assert exceptions(app) == []
    assert stub.calls and stub.calls[-1]["prompt"] == preset_prompt
    assert any("Paris" in (block.value or "") for block in app.markdown)
    assert any("tokens/s" in (caption.value or "") for caption in app.caption)


def test_the_header_metric_follows_the_server_once_a_request_has_been_sent(
    live_app, stub: StubServer
):
    app = live_app
    next(button for button in app.button if button.label == PRESET_PROMPTS[0][0]).click().run()
    values = {metric.label: metric.value for metric in app.metric}
    assert values["Tokens/s"] == "22.9"


def test_the_before_and_after_pair_appears_once_both_sides_exist(live_app, stub: StubServer):
    app = live_app
    label = PRESET_PROMPTS[0][0]
    next(button for button in app.button if button.label == label).click().run()
    stub.active_kernel = "rmsnorm"
    next(button for button in app.button if button.label == label).click().run()

    assert exceptions(app) == []
    assert any("Before the swap" in (banner.value or "") for banner in app.success)


def test_replay_mode_has_no_inference_panel(live_app, stub: StubServer):
    """Replaying a recording cannot generate text; a panel that looked live would lie."""
    app = live_app
    app.sidebar.radio[0].set_value("📼 Replay").run()
    assert exceptions(app) == []
    labels = [button.label for button in app.button]
    for preset_label, _prompt in PRESET_PROMPTS:
        assert preset_label not in labels
    assert "Send" not in labels
