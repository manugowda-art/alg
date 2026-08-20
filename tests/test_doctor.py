"""Preflight checks, against a stub Ollama that can be made to misbehave.

Each test here corresponds to a way a real local-model setup fails. The point
of the doctor is that each one produces a specific message and a fix, rather
than a confusing symptom an hour into a run.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from alg.doctor import (
    FAIL,
    OK,
    WARN,
    check_baseline,
    check_node,
    check_python,
    check_task,
    run_doctor,
    suggest,
)


class _Fake(BaseHTTPRequestHandler):
    models: list[str] = ["qwen3:30b"]
    capabilities: list[str] | None = ["completion", "tools"]
    context: int = 40_960
    chat: dict = {}

    def _send(self, payload: dict, code: int = 200) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/tags":
            self._send({"models": [{"name": m} for m in type(self).models]})
        else:
            self._send({}, 404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path == "/api/show":
            body: dict = {"model_info": {"qwen3.context_length": type(self).context}}
            if type(self).capabilities is not None:
                body["capabilities"] = type(self).capabilities
            self._send(body)
        elif self.path == "/api/chat":
            self._send(type(self).chat)
        else:
            self._send({}, 404)

    def log_message(self, *args) -> None:
        pass


TOOL_CALL = {
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "report_status", "arguments": {"ok": True}}}],
    },
    "done_reason": "stop",
}
JUST_TEXT = {
    "message": {"role": "assistant", "content": "Sure! I would call report_status(ok=true)."},
    "done_reason": "stop",
}


@pytest.fixture
def fake_ollama():
    servers: list[ThreadingHTTPServer] = []

    def start(**overrides):
        handler = type("H", (_Fake,), {"chat": TOOL_CALL, **overrides})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_port}"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def doctor(host: str | None, tmp_path: Path, task="tasks/calc_bug", model="qwen3:30b", **kw):
    kwargs = {"host": host} if host else {}
    kwargs.update(kw)
    return run_doctor(
        task_dir=task, provider="ollama", model=model, work_dir=tmp_path / "w", **kwargs
    )


def status_of(checks, name):
    return next(c for c in checks if c.name == name)


# --- environment ----------------------------------------------------------


def test_python_and_node_are_reported(task_dir):
    assert check_python().status == OK
    from alg.tasks import TaskSpec

    assert check_node(TaskSpec.load(task_dir)).status in (OK, WARN)


def test_a_missing_manifest_is_a_failed_check(tmp_path):
    check, task = check_task(tmp_path)
    assert check.status == FAIL and task is None


def test_the_baseline_check_proves_the_verifier_actually_fails(task_dir, tmp_path):
    from alg.tasks import TaskSpec

    check = check_baseline(task_dir, TaskSpec.load(task_dir), tmp_path / "w")
    assert check.status == OK
    assert "5 failed" in check.detail and "verifier works" in check.detail


def test_an_already_green_task_is_flagged_rather_than_passed(task_dir, tmp_path):
    """A task that cannot fail cannot be solved — that is worth saying out loud."""
    import shutil

    from alg.tasks import TaskSpec
    from conftest import MEAN_BUG, MEAN_FIX, MEDIAN_BUG, MEDIAN_FIX, SOURCE

    fixed = tmp_path / "fixed_task"
    shutil.copytree(task_dir, fixed)
    src = fixed / SOURCE
    src.write_text(src.read_text().replace(MEAN_BUG, MEAN_FIX).replace(MEDIAN_BUG, MEDIAN_FIX))

    check = check_baseline(fixed, TaskSpec.load(fixed), tmp_path / "w")
    assert check.status == WARN and "already green" in check.detail


# --- the model ------------------------------------------------------------


def test_a_healthy_setup_passes_every_check(fake_ollama, tmp_path):
    checks, command = doctor(fake_ollama(), tmp_path)
    assert [c.status for c in checks] == [OK] * len(checks), [
        (c.name, c.detail) for c in checks if c.status != OK
    ]
    assert "report_status" in status_of(checks, "tool call").detail
    assert "--wall-clock" in command


def test_an_unreachable_server_names_the_fix(tmp_path):
    checks, _ = doctor("http://127.0.0.1:1", tmp_path)
    check = status_of(checks, "ollama")
    assert check.status == FAIL
    assert "cannot reach" in check.detail and "ollama serve" in check.fix


def test_a_model_that_is_not_pulled_names_the_pull_command(fake_ollama, tmp_path):
    checks, _ = doctor(fake_ollama(models=["llama3:8b", "qwen3:8b"]), tmp_path)
    check = status_of(checks, "model")
    assert check.status == FAIL
    assert "ollama pull qwen3:30b" in check.fix
    assert "qwen3:8b" in check.fix  # suggests the sibling you do have


def test_a_model_without_tool_support_is_rejected_before_the_run(fake_ollama, tmp_path):
    checks, _ = doctor(fake_ollama(capabilities=["completion"]), tmp_path)
    check = status_of(checks, "model")
    assert check.status == FAIL
    assert "does not advertise tool support" in check.detail
    assert not any(c.name == "tool call" for c in checks)  # stopped before wasting a call


def test_an_older_server_without_capabilities_is_given_the_benefit_of_the_doubt(
    fake_ollama, tmp_path
):
    checks, _ = doctor(fake_ollama(capabilities=None), tmp_path)
    assert status_of(checks, "model").status == OK
    assert status_of(checks, "tool call").status == OK


def test_a_context_window_smaller_than_requested_warns(fake_ollama, tmp_path):
    checks, _ = doctor(fake_ollama(context=8192), tmp_path, num_ctx=32_768)
    check = status_of(checks, "model")
    assert check.status == WARN and "exceeds the model's" in check.fix


def test_a_model_that_answers_in_prose_fails_the_tool_check(fake_ollama, tmp_path):
    checks, _ = doctor(fake_ollama(chat=JUST_TEXT), tmp_path)
    check = status_of(checks, "tool call")
    assert check.status == FAIL
    assert "replied with text" in check.detail
    assert "I would call report_status" in check.fix  # shows what it said instead


# --- the suggestion -------------------------------------------------------


def test_the_suggested_budget_scales_with_measured_latency():
    fast = suggest("tasks/calc_bug", "ollama", "qwen3:30b", elapsed=2.0, max_iterations=10)
    slow = suggest("tasks/calc_bug", "ollama", "qwen3:30b", elapsed=60.0, max_iterations=10)
    assert "--wall-clock 600" in fast
    assert "--wall-clock 1560" in slow  # 60s x 10 x 2.5 rounded up to the minute


def test_no_budget_is_suggested_when_the_model_never_answered():
    assert "--wall-clock" not in suggest("tasks/calc_bug", "ollama", "q", elapsed=None)
