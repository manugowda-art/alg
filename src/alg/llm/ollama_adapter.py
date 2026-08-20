"""Ollama adapter — stdlib HTTP against a local `/api/chat` endpoint.

Kept dependency-free on purpose: the whole point of the neutral interface is
that a local 27B model and a frontier API model are interchangeable to the
harness. Ollama has no tool-call IDs, so we synthesize stable ones and map the
tool-result turn onto its `role: "tool"` messages.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .base import (
    Completion,
    Message,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
    parse_args,
)


class OllamaLLM:
    def __init__(
        self,
        model: str = "gemma3:27b",
        host: str = "http://localhost:11434",
        num_ctx: int | None = 32_768,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, *_encode(messages)],
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
        if self.num_ctx:
            payload["options"] = {"num_ctx": self.num_ctx}

        raw = self._post("/api/chat", payload)
        return _decode(raw)

    # --- introspection, used by `alg doctor` ------------------------------

    def list_models(self) -> list[str]:
        """Model names the server currently has pulled."""
        raw = self._get("/api/tags")
        return sorted(m.get("name", "") for m in raw.get("models", []))

    def describe(self) -> dict[str, Any]:
        """`/api/show` for this model: capabilities, context length, family."""
        return self._post("/api/show", {"model": self.model})

    def context_length(self) -> int | None:
        info = self.describe().get("model_info") or {}
        for key, value in info.items():
            if key.endswith(".context_length"):
                return int(value)
        return None

    def supports_tools(self) -> bool:
        """Ollama reports per-model capabilities; older servers omit the field."""
        caps = self.describe().get("capabilities")
        return True if caps is None else "tools" in caps

    def _get(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(self.host + path, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"ollama HTTP {exc.code}: {exc.read().decode()[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cannot reach ollama at {self.host}: {exc.reason}") from exc

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:  # pragma: no cover - needs a server
            raise RuntimeError(f"ollama HTTP {exc.code}: {exc.read().decode()[:500]}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - needs a server
            raise RuntimeError(f"cannot reach ollama at {self.host}: {exc.reason}") from exc


def _encode(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        tool_results = [b for b in message.blocks if isinstance(b, ToolResult)]
        if tool_results:
            # Ollama wants one `tool` message per result, not a bundled turn.
            for result in tool_results:
                out.append({"role": "tool", "tool_name": result.name, "content": result.content})
            continue
        entry: dict[str, Any] = {"role": message.role, "content": message.text()}
        calls = [b for b in message.blocks if isinstance(b, ToolCall)]
        if calls:
            entry["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.args}} for c in calls
            ]
        out.append(entry)
    return out


def _decode(raw: dict[str, Any]) -> Completion:
    message = raw.get("message") or {}
    blocks: list[Any] = []
    text = message.get("content") or ""
    if text.strip():
        blocks.append(TextBlock(text))
    calls = message.get("tool_calls") or []
    for index, call in enumerate(calls):
        function = call.get("function") or {}
        blocks.append(
            ToolCall(
                id=call.get("id") or f"call_{index}",
                name=function.get("name", ""),
                args=parse_args(function.get("arguments")),
            )
        )
    return Completion(
        blocks=blocks,
        stop_reason="tool_use" if calls else (raw.get("done_reason") or "end_turn"),
        usage=Usage(
            input_tokens=raw.get("prompt_eval_count", 0) or 0,
            output_tokens=raw.get("eval_count", 0) or 0,
        ),
        raw=raw,
    )
