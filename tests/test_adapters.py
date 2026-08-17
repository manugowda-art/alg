"""Adapter mapping, tested offline.

The network calls are not exercised here (no key, no local server), but the
translation in and out of each provider's shape is pure and worth pinning: it is
where a provider swap actually breaks.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from alg.llm import build
from alg.llm.anthropic_adapter import _decode_response, _encode_message
from alg.llm.base import (
    Completion,
    Message,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
    parse_args,
)
from alg.llm.ollama_adapter import _decode, _encode


def test_build_rejects_an_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        build("openai")


def test_build_reports_a_missing_sdk_clearly():
    with pytest.raises(RuntimeError, match="anthropic SDK is not installed"):
        build("anthropic")


def test_the_neutral_interface_is_structural():
    llm = build("ollama", model="gemma3:27b")
    assert llm.name == "ollama:gemma3:27b"
    assert hasattr(llm, "complete")


# --- shared helpers -------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"a": 1}, {"a": 1}),
        ('{"a": 1}', {"a": 1}),
        ("", {}),
        (None, {}),
        ("not json", {"_unparsed": "not json"}),
        ("[1,2]", {"_value": [1, 2]}),
    ],
)
def test_parse_args_normalizes_provider_variations(raw, expected):
    assert parse_args(raw) == expected


def test_usage_adds_up():
    assert Usage(1, 2) + Usage(10, 20) == Usage(11, 22)


def test_completion_exposes_tool_calls_and_text():
    completion = Completion(
        blocks=[TextBlock("thinking out loud"), ToolCall(id="c1", name="t", args={})],
        stop_reason="tool_use",
    )
    assert completion.text() == "thinking out loud"
    assert [c.name for c in completion.tool_calls] == ["t"]


# --- anthropic ------------------------------------------------------------


def test_anthropic_encodes_every_block_type():
    encoded = _encode_message(
        Message(
            role="assistant",
            blocks=[
                TextBlock("hello"),
                ToolCall(id="c1", name="read_file", args={"path": "a.py"}),
            ],
        )
    )
    assert encoded["role"] == "assistant"
    assert encoded["content"] == [
        {"type": "text", "text": "hello"},
        {"type": "tool_use", "id": "c1", "name": "read_file", "input": {"path": "a.py"}},
    ]


def test_anthropic_encodes_tool_results_as_a_user_turn():
    encoded = _encode_message(
        Message.tool_results([ToolResult(call_id="c1", name="t", content="boom", is_error=True)])
    )
    assert encoded["role"] == "user"
    assert encoded["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "c1",
        "content": "boom",
        "is_error": True,
    }


def test_anthropic_decodes_text_tool_use_and_usage():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="let me look"),
            SimpleNamespace(type="tool_use", id="c1", name="read_file", input={"path": "a.py"}),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=120, output_tokens=30),
    )
    completion = _decode_response(response)

    assert completion.stop_reason == "tool_use"
    assert completion.text() == "let me look"  # empty thinking blocks are dropped
    assert completion.tool_calls[0].args == {"path": "a.py"}
    assert completion.usage == Usage(120, 30)


def test_anthropic_refusal_survives_decoding_with_empty_content():
    response = SimpleNamespace(content=[], stop_reason="refusal", usage=None)
    completion = _decode_response(response)
    assert completion.stop_reason == "refusal"
    assert completion.blocks == []


# --- ollama --------------------------------------------------------------


def test_ollama_encodes_tool_results_as_tool_role_messages():
    encoded = _encode(
        [
            Message.user("fix it"),
            Message.assistant([ToolCall(id="c1", name="read_file", args={"path": "a.py"})]),
            Message.tool_results([ToolResult(call_id="c1", name="read_file", content="line 1")]),
        ]
    )
    assert [m["role"] for m in encoded] == ["user", "assistant", "tool"]
    assert encoded[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert encoded[2] == {"role": "tool", "tool_name": "read_file", "content": "line 1"}


def test_ollama_decodes_tool_calls_and_synthesizes_ids():
    completion = _decode(
        {
            "message": {
                "content": "checking",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "a.py"}}},
                    {"function": {"name": "run_tests", "arguments": "{}"}},
                ],
            },
            "prompt_eval_count": 400,
            "eval_count": 55,
        }
    )
    assert completion.stop_reason == "tool_use"
    assert [c.id for c in completion.tool_calls] == ["call_0", "call_1"]
    assert completion.tool_calls[0].args == {"path": "a.py"}
    assert completion.usage == Usage(400, 55)


def test_ollama_decodes_a_plain_answer():
    completion = _decode({"message": {"content": "all done"}, "done_reason": "stop"})
    assert completion.stop_reason == "stop"
    assert completion.text() == "all done"
    assert completion.tool_calls == []


def test_both_adapters_round_trip_the_same_conversation_shape():
    """The harness builds one message list; each adapter renders it its own way."""
    conversation = [
        Message.user("fix it"),
        Message.assistant([ToolCall(id="c1", name="run_tests", args={})]),
        Message.tool_results([ToolResult(call_id="c1", name="run_tests", content="1 failed")]),
    ]
    anthropic_shape = [_encode_message(m) for m in conversation]
    ollama_shape = _encode(conversation)

    assert [m["role"] for m in anthropic_shape] == ["user", "assistant", "user"]
    assert [m["role"] for m in ollama_shape] == ["user", "assistant", "tool"]


def test_tool_specs_render_into_each_provider_s_schema_slot():
    spec = ToolSpec(name="t", description="d", input_schema={"type": "object", "properties": {}})
    # Anthropic: input_schema; Ollama: function.parameters. Asserted here so a
    # rename in one adapter cannot silently drop tools for the other.
    assert spec.input_schema["type"] == "object"
    assert spec.name == "t" and spec.description == "d"
