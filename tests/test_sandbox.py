"""Sandbox subprocess handling with subprocess.run mocked (spec 5.4 / 13.1).

No GPU and no real subprocess here: these tests pin the contract between the sandbox
and the verification script — clean JSON, timeout, crash — so a hostile candidate can
never turn into an exception in the ADK process.
"""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kernelsmith.verifier import sandbox
from kernelsmith.verifier.sandbox import parse_result, run_in_sandbox

GOOD_PAYLOAD = {
    "reward": 3,
    "correctness_pass": True,
    "speedup_vs_eager": 1.42,
    "speedup_vs_torch_compile": 1.18,
    "latency_ms_by_shape": {"1x128": 0.021},
}


@pytest.fixture(autouse=True)
def healthy_gpu(monkeypatch):
    """Default: the GPU is fine. Tests that care about wedging override this."""
    monkeypatch.setattr(sandbox, "gpu_health_probe", lambda *a, **k: True)


def fake_completed(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_normal_completion_parses_json_stdout(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: fake_completed(stdout=json.dumps(GOOD_PAYLOAD))
    )
    result = run_in_sandbox("/tmp/kernelsmith_sandbox/verify_candidate.py")
    assert result["reward"] == 3
    assert result["correctness_pass"] is True
    assert result["speedup_vs_eager"] == pytest.approx(1.42)


def test_json_is_read_from_the_last_stdout_line(monkeypatch):
    """Triton and torch both print to stdout; only the final JSON line is the result."""
    noisy = f"loading model...\nautotuning BLOCK=1024\n{json.dumps(GOOD_PAYLOAD)}\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake_completed(stdout=noisy))
    assert run_in_sandbox("script.py")["reward"] == 3


def test_timeout_returns_sigkill_error(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="python script.py", timeout=kwargs.get("timeout", 60))

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    result = run_in_sandbox("script.py", timeout=5)
    assert result == {
        "reward": -1,
        "correctness_pass": False,
        "error": "timeout_sigkill",
        "stderr_tail": "",
    }


def test_nonzero_returncode_is_handled_gracefully(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: fake_completed(
            stdout="", stderr="CUDA error: an illegal memory access", returncode=1
        ),
    )
    result = run_in_sandbox("script.py")
    assert result["reward"] == -1
    assert result["correctness_pass"] is False
    assert result["error"] == "nonzero_exit:1"
    assert "illegal memory access" in result["stderr_tail"]


def test_nonzero_returncode_ignores_a_claimed_reward(monkeypatch):
    """A crashing candidate does not get to print itself a +3 on the way out."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: fake_completed(stdout=json.dumps(GOOD_PAYLOAD), returncode=139),
    )
    assert run_in_sandbox("script.py")["reward"] == -1


def test_unparseable_stdout_is_minus_one(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: fake_completed(stdout="Segmentation fault (core dumped)")
    )
    result = run_in_sandbox("script.py")
    assert result["reward"] == -1
    assert result["error"] == "unparseable_output"


def test_spawn_failure_is_minus_one(monkeypatch):
    def raise_oserror(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(subprocess, "run", raise_oserror)
    result = run_in_sandbox("script.py")
    assert result["reward"] == -1
    assert result["error"] == "spawn_failed"


def test_env_is_scrubbed_and_cwd_is_the_sandbox(monkeypatch):
    """A candidate must not see ADC paths, API keys, or the repo working tree."""
    captured = {}

    def capture(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return fake_completed(stdout=json.dumps(GOOD_PAYLOAD))

    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/home/me/adc.json")
    monkeypatch.setattr(subprocess, "run", capture)
    run_in_sandbox("script.py")

    assert set(captured["env"]) == {
        "PATH",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "CUBLAS_WORKSPACE_CONFIG",
    }
    assert captured["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert captured["cwd"] == str(sandbox.SANDBOX_DIR)
    assert captured["start_new_session"] is True
    assert captured["cmd"][0] == sys.executable


def test_health_probe_runs_after_every_candidate(monkeypatch):
    """Pass or fail, the GPU is probed — a wedge poisons every later candidate."""
    calls = []
    monkeypatch.setattr(sandbox, "gpu_health_probe", lambda *a, **k: calls.append(1) or True)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: fake_completed(stdout=json.dumps(GOOD_PAYLOAD))
    )
    run_in_sandbox("script.py")
    assert len(calls) == 1


def test_wedged_gpu_triggers_reset(monkeypatch):
    monkeypatch.setattr(sandbox, "gpu_health_probe", lambda *a, **k: False)
    reset_calls = []
    monkeypatch.setattr(sandbox, "run_gpu_reset", lambda: reset_calls.append(1) or True)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: fake_completed(stdout=json.dumps(GOOD_PAYLOAD))
    )
    result = run_in_sandbox("script.py")
    assert result["gpu_wedged"] is True
    assert result["gpu_reset_attempted"] is True
    assert len(reset_calls) == 1


def test_gpu_health_probe_true_on_clean_exit(monkeypatch):
    monkeypatch.undo()  # drop the autouse stub; exercise the real probe body
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake_completed(returncode=0))
    assert sandbox.gpu_health_probe() is True


def test_gpu_health_probe_false_on_timeout(monkeypatch):
    monkeypatch.undo()

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="python -c ...", timeout=10)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    assert sandbox.gpu_health_probe() is False


def test_gpu_health_probe_false_on_failed_probe(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake_completed(returncode=1))
    assert sandbox.gpu_health_probe() is False


def test_run_gpu_reset_invokes_the_script(monkeypatch):
    captured = {}

    def capture(cmd, **kwargs):
        captured["cmd"] = cmd
        return fake_completed(returncode=0)

    monkeypatch.setattr(subprocess, "run", capture)
    assert sandbox.run_gpu_reset() is True
    assert str(sandbox.GPU_RESET_SCRIPT) in captured["cmd"]


def test_gpu_reset_script_exists():
    """run_gpu_reset() silently no-ops if the script is missing — so pin its path."""
    assert sandbox.GPU_RESET_SCRIPT.exists(), f"missing {sandbox.GPU_RESET_SCRIPT}"


def test_parse_result_fills_missing_keys():
    """A script that prints partial JSON must still yield a well-formed reward dict."""
    result = parse_result('{"speedup_vs_eager": 1.2}', "warn: something", returncode=0)
    assert result["reward"] == -1
    assert result["correctness_pass"] is False
    assert result["stderr_tail"] == "warn: something"


def test_parse_result_truncates_long_stderr():
    result = parse_result("", "x" * 100_000, returncode=1)
    assert len(result["stderr_tail"]) == sandbox.STDERR_TAIL_CHARS
