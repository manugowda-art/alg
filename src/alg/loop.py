"""Loop engineering: one bounded, instrumented model/tool cycle.

The loop owns exactly five things:

  trigger   a goal message that starts the cycle
  action    a model call that may request tools
  evidence  the tool outcomes, plus anything the caller records as evidence
  feedback  those outcomes fed back into the next model call
  stop rule an explicit, named reason the cycle ended

Every termination path has a name. A loop that can only end by "the model
stopped asking for tools" is the loop that burns a budget on weak work; a loop
with named stop rules tells you *why* it gave up, which is the thing you
actually tune.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .events import Trace
from .llm.base import LLM, Completion, Message, ToolResult, Usage
from .tools import ToolRegistry

MAX_TOOL_RESULT_CHARS = 12_000


@dataclass
class LoopState:
    goal: str
    messages: list[Message] = field(default_factory=list)
    iteration: int = 0
    usage: Usage = field(default_factory=Usage)
    started: float = 0.0
    elapsed: float = 0.0
    last_completion: Completion | None = None
    tool_calls: int = 0
    tool_errors: int = 0
    consecutive_tool_errors: int = 0
    evidence: list[Any] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def latest_evidence(self) -> Any | None:
        return self.evidence[-1] if self.evidence else None


# A stop rule returns a reason string to halt, or None to continue.
StopRule = Callable[[LoopState], str | None]


def max_iterations(limit: int) -> StopRule:
    def rule(state: LoopState) -> str | None:
        return f"max_iterations({limit})" if state.iteration >= limit else None

    return rule


def token_budget(limit: int) -> StopRule:
    def rule(state: LoopState) -> str | None:
        spent = state.usage.input_tokens + state.usage.output_tokens
        return f"token_budget({limit}) spent={spent}" if spent >= limit else None

    return rule


def wall_clock(seconds: float) -> StopRule:
    def rule(state: LoopState) -> str | None:
        return f"wall_clock({seconds}s)" if state.elapsed >= seconds else None

    return rule


def tool_error_streak(limit: int) -> StopRule:
    """Guards the classic failure mode: the model retries a malformed call
    forever because each rejection looks recoverable."""

    def rule(state: LoopState) -> str | None:
        if state.consecutive_tool_errors >= limit:
            return f"tool_error_streak({limit})"
        return None

    return rule


def stalled_evidence(limit: int, key: Callable[[Any], Any] = lambda e: e) -> StopRule:
    """Stop when the last `limit` pieces of evidence are identical — the loop is
    spinning, not converging. This is the anti-reward-hacking rule: progress has
    to show up in the evidence, not in the model's confidence."""

    def rule(state: LoopState) -> str | None:
        if len(state.evidence) < limit:
            return None
        recent = [key(e) for e in state.evidence[-limit:]]
        if all(item == recent[0] for item in recent[1:]):
            return f"stalled_evidence({limit}) at {recent[0]!r}"
        return None

    return rule


@dataclass
class LoopResult:
    state: LoopState
    stop_reason: str
    final_text: str

    @property
    def iterations(self) -> int:
        return self.state.iteration

    @property
    def usage(self) -> Usage:
        return self.state.usage


class AgentLoop:
    def __init__(
        self,
        llm: LLM,
        registry: ToolRegistry,
        system: str,
        trace: Trace,
        stop_rules: list[StopRule] | None = None,
        evidence_fn: Callable[[LoopState], Any] | None = None,
        label: str = "loop",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.system = system
        self.trace = trace
        self.stop_rules = list(stop_rules or [max_iterations(8)])
        self.evidence_fn = evidence_fn
        self.label = label
        self._clock = clock

    def run(self, goal: str, history: list[Message] | None = None) -> LoopResult:
        state = LoopState(goal=goal, messages=list(history or []), started=self._clock())
        state.messages.append(Message.user(goal))
        specs = self.registry.specs()
        self.trace.emit(
            "loop.start",
            label=self.label,
            model=self.llm.name,
            tools=self.registry.names,
            stop_rules=len(self.stop_rules),
        )

        stop_reason = "unset"
        while True:
            state.elapsed = self._clock() - state.started
            triggered = self._check_stop_rules(state)
            if triggered:
                stop_reason = triggered
                break

            self.trace.emit("loop.iteration", label=self.label, n=state.iteration)
            completion = self.llm.complete(self.system, state.messages, specs)
            state.last_completion = completion
            state.usage = state.usage + completion.usage
            self.trace.emit(
                "model.completion",
                label=self.label,
                stop_reason=completion.stop_reason,
                text=completion.text()[:MAX_TOOL_RESULT_CHARS],
                tool_calls=[c.name for c in completion.tool_calls],
                input_tokens=completion.usage.input_tokens,
                output_tokens=completion.usage.output_tokens,
            )

            if completion.stop_reason == "refusal":
                stop_reason = "model_refusal"
                break

            state.messages.append(Message.assistant(completion.blocks))

            calls = completion.tool_calls
            if not calls:
                stop_reason = "model_finished"
                break

            results: list[ToolResult] = []
            iteration_errors = 0
            for call in calls:
                outcome = self.registry.call(call.name, call.args)
                state.tool_calls += 1
                if not outcome.ok:
                    state.tool_errors += 1
                    iteration_errors += 1
                results.append(
                    ToolResult(
                        call_id=call.id,
                        name=call.name,
                        content=outcome.content[:MAX_TOOL_RESULT_CHARS] or "(empty result)",
                        is_error=not outcome.ok,
                    )
                )
            state.consecutive_tool_errors = (
                state.consecutive_tool_errors + 1 if iteration_errors == len(calls) else 0
            )
            state.messages.append(Message.tool_results(results))

            if self.evidence_fn is not None:
                evidence = self.evidence_fn(state)
                if evidence is not None:
                    state.evidence.append(evidence)
                    self.trace.emit("loop.evidence", label=self.label, value=str(evidence))

            state.iteration += 1

        state.elapsed = self._clock() - state.started
        final_text = state.last_completion.text() if state.last_completion else ""
        self.trace.emit(
            "loop.stop",
            label=self.label,
            reason=stop_reason,
            iterations=state.iteration,
            tool_calls=state.tool_calls,
            tool_errors=state.tool_errors,
            input_tokens=state.usage.input_tokens,
            output_tokens=state.usage.output_tokens,
            elapsed=round(state.elapsed, 3),
        )
        return LoopResult(state=state, stop_reason=stop_reason, final_text=final_text)

    def _check_stop_rules(self, state: LoopState) -> str | None:
        for rule in self.stop_rules:
            reason = rule(state)
            if reason:
                return reason
        return None
