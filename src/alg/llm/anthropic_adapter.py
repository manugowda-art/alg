"""Anthropic adapter — uses the official `anthropic` SDK.

Notes that matter on current models (Claude Opus 5 family):
  * `thinking={"type": "adaptive"}` replaces the removed `budget_tokens`.
  * `temperature` / `top_p` / `top_k` are rejected — steer with the prompt.
  * `max_tokens` caps thinking + text together, so keep it generous.
  * A refusal arrives as HTTP 200 with `stop_reason == "refusal"`, so always
    check `stop_reason` before reading content.
"""

from __future__ import annotations

import os
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

DEFAULT_MODEL = "claude-opus-5"


class AnthropicLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 16_000,
        effort: str | None = "high",
        thinking: bool = True,
        api_key: str | None = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "the anthropic SDK is not installed; `pip install 'alg[anthropic]'`"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking = thinking
        self.name = f"anthropic:{model}"

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [_encode_message(m) for m in messages],
        }
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}

        response = self._client.messages.create(**kwargs)
        return _decode_response(response)


def _encode_message(message: Message) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for block in message.blocks:
        if isinstance(block, TextBlock):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolCall):
            content.append(
                {"type": "tool_use", "id": block.id, "name": block.name, "input": block.args}
            )
        elif isinstance(block, ToolResult):
            content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.call_id,
                    "content": block.content,
                    "is_error": block.is_error,
                }
            )
    return {"role": message.role, "content": content}


def _decode_response(response: Any) -> Completion:
    blocks: list[Any] = []
    for block in response.content:
        kind = getattr(block, "type", None)
        if kind == "text":
            blocks.append(TextBlock(block.text))
        elif kind == "tool_use":
            blocks.append(ToolCall(id=block.id, name=block.name, args=parse_args(block.input)))
        # thinking blocks carry no text under the default display setting and
        # are not replayed by this harness, so they are dropped here.
    usage = getattr(response, "usage", None)
    return Completion(
        blocks=blocks,
        stop_reason=getattr(response, "stop_reason", "end_turn") or "end_turn",
        usage=Usage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        ),
    )
