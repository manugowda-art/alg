"""Model adapters behind one provider-neutral interface."""

from .base import (
    LLM,
    Block,
    Completion,
    Message,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

__all__ = [
    "LLM",
    "Block",
    "Completion",
    "Message",
    "TextBlock",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "build",
]


def build(provider: str, model: str | None = None, **kwargs) -> LLM:
    """Late-bound factory so importing the harness never imports a provider SDK."""
    if provider == "anthropic":
        from .anthropic_adapter import DEFAULT_MODEL, AnthropicLLM

        return AnthropicLLM(model=model or DEFAULT_MODEL, **kwargs)
    if provider == "ollama":
        from .ollama_adapter import OllamaLLM

        return OllamaLLM(model=model or "gemma3:27b", **kwargs)
    raise ValueError(f"unknown provider: {provider!r} (expected 'anthropic' or 'ollama')")
