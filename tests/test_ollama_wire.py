"""The Ollama adapter against a real HTTP server.

The scripted-model tests elsewhere bypass the adapter entirely. These do not:
a stub server speaking Ollama's wire format sits on a real socket, so the
request shape we send and the response shape we parse are both exercised. What
remains unverified after this is only how a *particular model* behaves — never
whether the plumbing is right.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from alg.agent import AgentConfig, FixerAgent
from alg.llm.ollama_adapter import OllamaLLM
from conftest import MEAN_BUG, MEAN_FIX, MEDIAN_BUG, MEDIAN_FIX, SOURCE


class _Handler(BaseHTTPRequestHandler):
    responses: list[dict] = []
    requests: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append({"path": self.path, "body": body})
        if type(self).responses:
            payload = type(self).responses.pop(0)
        else:
            payload = {"message": {"role": "assistant", "content": "done"}, "done_reason": "stop"}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # keep pytest output clean
        pass


@pytest.fixture
def ollama():
    """Start a stub Ollama on a free port; yield (host, handler)."""

    def start(responses: list[dict] | None = None):
        handler = type("H", (_Handler,), {"responses": list(responses or []), "requests": []})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started.append(server)
        return f"http://127.0.0.1:{server.server_port}", handler

    started: list[ThreadingHTTPServer] = []
    yield start
    for server in started:
        server.shutdown()
        server.server_close()


def text(content: str) -> dict:
    return {
        "message": {"role": "assistant", "content": content},
        "done_reason": "stop",
        "prompt_eval_count": 100,
        "eval_count": 20,
    }


def tool(name: str, **args) -> dict:
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}],
        },
        "done_reason": "stop",
        "prompt_eval_count": 200,
        "eval_count": 40,
    }


# --- request shape --------------------------------------------------------


def test_the_request_carries_system_prompt_tools_and_context_size(ollama):
    host, handler = ollama([text("hi")])
    llm = OllamaLLM(model="qwen3:30b", host=host, num_ctx=16_384)

    from alg.llm.base import Message, ToolSpec

    llm.complete(
        "you are a fixer",
        [Message.user("go")],
        [ToolSpec(name="run_tests", description="Run tests.", input_schema={"type": "object"})],
    )

    body = handler.requests[0]["body"]
    assert handler.requests[0]["path"] == "/api/chat"
    assert body["model"] == "qwen3:30b"
    assert body["stream"] is False
    assert body["options"]["num_ctx"] == 16_384
    assert body["messages"][0] == {"role": "system", "content": "you are a fixer"}
    assert body["messages"][1] == {"role": "user", "content": "go"}
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["function"]["name"] == "run_tests"
    assert body["tools"][0]["function"]["parameters"] == {"type": "object"}


def test_a_tool_call_and_its_result_round_trip_over_the_wire(ollama, workspace, task, trace):
    host, handler = ollama([tool("run_tests"), text("the suite is red")])
    llm = OllamaLLM(model="qwen3:30b", host=host)
    FixerAgent(
        workspace=workspace, task=task, llm=llm, trace=trace,
        config=AgentConfig(max_attempts=1),
    ).run()

    second = handler.requests[1]["body"]["messages"]
    assistant = next(m for m in second if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["function"]["name"] == "run_tests"

    result = next(m for m in second if m["role"] == "tool")
    assert result["tool_name"] == "run_tests"
    assert "5 failed" in result["content"]  # the real verdict reached the model


def test_usage_counts_come_back_from_the_server(ollama, workspace, task, trace):
    host, _ = ollama([tool("run_tests"), text("red")])
    llm = OllamaLLM(model="qwen3:30b", host=host)
    state = FixerAgent(
        workspace=workspace, task=task, llm=llm, trace=trace,
        config=AgentConfig(max_attempts=1),
    ).run()

    assert state["tokens"]["input"] == 300  # 200 + 100
    assert state["tokens"]["output"] == 60  # 40 + 20


# --- the whole agent, over HTTP -------------------------------------------


def test_the_agent_fixes_the_typescript_task_through_the_ollama_adapter(
    ollama, workspace, task, trace
):
    host, _ = ollama(
        [
            tool("run_tests"),
            tool("read_file", path=SOURCE),
            tool("edit_file", path=SOURCE, old_text=MEAN_BUG, new_text=MEAN_FIX),
            tool("edit_file", path=SOURCE, old_text=MEDIAN_BUG, new_text=MEDIAN_FIX),
            tool("run_tests"),
            text("mean divided by length + 1; median ignored even-length input. Both fixed."),
        ]
    )
    llm = OllamaLLM(model="qwen3:30b", host=host)
    state = FixerAgent(
        workspace=workspace, task=task, llm=llm, trace=trace, config=AgentConfig()
    ).run()

    assert state["green"] is True
    assert state["attempt"] == 1
    assert MEAN_FIX in (workspace.root / SOURCE).read_text()


def test_arguments_arriving_as_a_json_string_are_still_understood(ollama, workspace, task, trace):
    """Some builds serialize tool arguments as a string rather than an object."""
    host, _ = ollama(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "read_file", "arguments": json.dumps({"path": SOURCE})}}
                    ],
                },
                "done_reason": "stop",
            },
            text("read it"),
        ]
    )
    llm = OllamaLLM(model="qwen3:30b", host=host)
    FixerAgent(
        workspace=workspace, task=task, llm=llm, trace=trace,
        config=AgentConfig(max_attempts=1),
    ).run()

    call = trace.of_type("tool.call")[0].payload
    assert call["tool"] == "read_file" and call["ok"] is True


# --- failure modes you will actually hit ----------------------------------


def test_an_unreachable_server_says_so_clearly():
    llm = OllamaLLM(model="qwen3:30b", host="http://127.0.0.1:1")
    from alg.llm.base import Message

    with pytest.raises(RuntimeError, match="cannot reach ollama"):
        llm.complete("s", [Message.user("go")], [])


def test_an_http_error_surfaces_the_server_s_message(ollama):
    class Failing(_Handler):
        def do_POST(self) -> None:  # noqa: N802
            body = b'{"error":"registry.ollama.ai/library/qwen3:30b does not support tools"}'
            self.send_response(400)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Failing)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        llm = OllamaLLM(model="qwen3:30b", host=f"http://127.0.0.1:{server.server_port}")
        from alg.llm.base import Message

        with pytest.raises(RuntimeError, match="does not support tools"):
            llm.complete("s", [Message.user("go")], [])
    finally:
        server.shutdown()
        server.server_close()
