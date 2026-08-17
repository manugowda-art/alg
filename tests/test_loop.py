"""Loop engineering: the cycle runs, feeds evidence back, and every exit is named."""

from __future__ import annotations

import pytest
from conftest import ScriptedLLM, call, say

from alg.llm.base import Completion, TextBlock, ToolCall, ToolResult, Usage
from alg.loop import (
    AgentLoop,
    LoopState,
    max_iterations,
    stalled_evidence,
    token_budget,
    tool_error_streak,
    wall_clock,
)
from alg.tools import Tool, ToolOutcome, ToolRegistry


def counter_tool(log: list[dict]) -> Tool:
    def run(value: str) -> ToolOutcome:
        log.append({"value": value})
        return ToolOutcome(ok=True, content=f"recorded {value}")

    return Tool(
        name="record",
        description="Record a value.",
        input_schema={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        fn=run,
    )


def failing_tool() -> Tool:
    return Tool(
        name="always_fails",
        description="Always fails.",
        input_schema={"type": "object", "properties": {}, "required": []},
        fn=lambda: ToolOutcome.error("nope"),
    )


def build_loop(llm, tools, trace, **kwargs) -> AgentLoop:
    return AgentLoop(
        llm=llm,
        registry=ToolRegistry(tools, trace=trace),
        system="system",
        trace=trace,
        **kwargs,
    )


def test_loop_ends_when_the_model_stops_calling_tools(trace):
    log: list[dict] = []
    llm = ScriptedLLM([call("record", value="a"), say("all done")])
    result = build_loop(llm, [counter_tool(log)], trace).run("go")

    assert result.stop_reason == "model_finished"
    assert result.final_text == "all done"
    assert log == [{"value": "a"}]
    assert result.iterations == 1


def test_tool_results_are_fed_back_as_the_next_turn(trace):
    llm = ScriptedLLM([call("record", value="a"), say("done")])
    build_loop(llm, [counter_tool([])], trace).run("go")

    # Second model call sees: goal, assistant tool_use, tool_result.
    _, messages, _ = llm.calls[1]
    assert [m.role for m in messages] == ["user", "assistant", "user"]
    results = [b for b in messages[-1].blocks if isinstance(b, ToolResult)]
    assert results and results[0].content == "recorded a"
    assert results[0].is_error is False


def test_tool_specs_are_passed_to_the_model(trace):
    llm = ScriptedLLM([say("done")])
    build_loop(llm, [counter_tool([])], trace).run("go")
    _, _, tools = llm.calls[0]
    assert [t.name for t in tools] == ["record"]
    assert tools[0].input_schema["required"] == ["value"]


def test_max_iterations_stops_a_model_that_never_finishes(trace):
    llm = ScriptedLLM(default=call("record", value="x"))
    result = build_loop(llm, [counter_tool([])], trace, stop_rules=[max_iterations(3)]).run("go")

    assert result.stop_reason == "max_iterations(3)"
    assert result.iterations == 3


def test_token_budget_stops_the_loop(trace):
    llm = ScriptedLLM(default=call("record", value="x"))
    result = build_loop(
        llm, [counter_tool([])], trace, stop_rules=[max_iterations(50), token_budget(60)]
    ).run("go")

    assert result.stop_reason.startswith("token_budget(60)")
    assert result.usage.input_tokens + result.usage.output_tokens >= 60


def test_wall_clock_stops_the_loop(trace):
    ticks = iter([0.0, 0.0, 99.0, 99.0])
    llm = ScriptedLLM(default=call("record", value="x"))
    loop = AgentLoop(
        llm=llm,
        registry=ToolRegistry([counter_tool([])], trace=trace),
        system="s",
        trace=trace,
        stop_rules=[max_iterations(50), wall_clock(10.0)],
        clock=lambda: next(ticks),
    )
    assert loop.run("go").stop_reason == "wall_clock(10.0s)"


def test_tool_error_streak_stops_a_model_stuck_on_a_bad_call(trace):
    llm = ScriptedLLM(default=call("always_fails"))
    result = build_loop(
        llm, [failing_tool()], trace, stop_rules=[max_iterations(50), tool_error_streak(2)]
    ).run("go")

    assert result.stop_reason == "tool_error_streak(2)"
    assert result.state.tool_errors == 2


def test_a_successful_call_resets_the_error_streak(trace):
    llm = ScriptedLLM(
        [call("always_fails"), call("record", value="a"), call("always_fails"), say("done")]
    )
    result = build_loop(
        llm,
        [failing_tool(), counter_tool([])],
        trace,
        stop_rules=[max_iterations(50), tool_error_streak(2)],
    ).run("go")

    assert result.stop_reason == "model_finished"
    assert result.state.tool_errors == 2


def test_stalled_evidence_stops_a_loop_that_is_not_converging(trace):
    llm = ScriptedLLM(default=call("record", value="x"))
    result = build_loop(
        llm,
        [counter_tool([])],
        trace,
        stop_rules=[max_iterations(50), stalled_evidence(3)],
        evidence_fn=lambda state: "2 failed",
    ).run("go")

    assert result.stop_reason.startswith("stalled_evidence(3)")


def test_changing_evidence_does_not_trip_the_stall_rule(trace):
    values = iter(["3 failed", "2 failed", "1 failed", "0 failed"])
    llm = ScriptedLLM([*[call("record", value="x") for _ in range(3)], say("done")])
    result = build_loop(
        llm,
        [counter_tool([])],
        trace,
        stop_rules=[max_iterations(50), stalled_evidence(3)],
        evidence_fn=lambda state: next(values),
    ).run("go")

    assert result.stop_reason == "model_finished"


def test_refusal_is_a_named_stop_reason(trace):
    llm = ScriptedLLM([Completion(blocks=[], stop_reason="refusal")])
    assert build_loop(llm, [], trace).run("go").stop_reason == "model_refusal"


def test_parallel_tool_calls_all_run_and_return_in_one_turn(trace):
    log: list[dict] = []
    llm = ScriptedLLM(
        [
            Completion(
                blocks=[
                    ToolCall(id="c1", name="record", args={"value": "a"}),
                    ToolCall(id="c2", name="record", args={"value": "b"}),
                ],
                stop_reason="tool_use",
                usage=Usage(1, 1),
            ),
            say("done"),
        ]
    )
    build_loop(llm, [counter_tool(log)], trace).run("go")

    assert log == [{"value": "a"}, {"value": "b"}]
    _, messages, _ = llm.calls[1]
    results = [b for b in messages[-1].blocks if isinstance(b, ToolResult)]
    assert [r.call_id for r in results] == ["c1", "c2"]


def test_invalid_arguments_come_back_as_an_error_result_not_an_exception(trace):
    llm = ScriptedLLM([call("record", value="a", surprise=1), say("done")])
    result = build_loop(llm, [counter_tool([])], trace).run("go")

    assert result.stop_reason == "model_finished"
    _, messages, _ = llm.calls[1]
    results = [b for b in messages[-1].blocks if isinstance(b, ToolResult)]
    assert results[0].is_error and "unknown argument" in results[0].content


def test_the_trace_records_the_whole_cycle(trace):
    llm = ScriptedLLM([call("record", value="a"), say("done")])
    build_loop(llm, [counter_tool([])], trace).run("go")

    types = trace.types()
    assert types[0] == "loop.start"
    assert types[-1] == "loop.stop"
    assert "model.completion" in types and "tool.call" in types
    stop = trace.of_type("loop.stop")[0].payload
    assert stop["reason"] == "model_finished"
    assert stop["tool_calls"] == 1


def test_stop_rules_are_checked_before_the_first_model_call(trace):
    llm = ScriptedLLM([say("should not be reached")])
    result = build_loop(llm, [], trace, stop_rules=[max_iterations(0)]).run("go")

    assert result.stop_reason == "max_iterations(0)"
    assert llm.calls == []
