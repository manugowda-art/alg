"""Provider-neutral model interface.

The harness only ever sees these types. Adapters translate to and from a
provider's wire format, so swapping Claude for a local Ollama model is a
one-line change at the edge and invisible to the loop and the graph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False


Block = TextBlock | ToolCall | ToolResult


@dataclass(frozen=True)
class Message:
    role: str  # "user" | "assistant"
    blocks: list[Block]

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role="user", blocks=[TextBlock(text)])

    @classmethod
    def assistant(cls, blocks: list[Block]) -> "Message":
        return cls(role="assistant", blocks=list(blocks))

    @classmethod
    def tool_results(cls, results: list[ToolResult]) -> "Message":
        # Tool results are a user-role turn in the Anthropic shape; adapters
        # that need a dedicated "tool" role remap it.
        return cls(role="user", blocks=list(results))

    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if isinstance(b, TextBlock))


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class Completion:
    blocks: list[Block]
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens" | provider-specific
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] | None = None

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [b for b in self.blocks if isinstance(b, ToolCall)]

    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if isinstance(b, TextBlock))


@runtime_checkable
class LLM(Protocol):
    """The entire contract the harness depends on."""

    name: str

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> Completion: ...


def parse_args(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a dict or a JSON string depending on provider."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"_unparsed": raw}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    return {}
