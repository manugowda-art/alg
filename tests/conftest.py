"""Test doubles and fixtures for the harness's own tests.

`ScriptedLLM` is a test device, not a provider: it replays a fixed list of
completions so that loop and graph behaviour can be asserted deterministically
and offline. Real runs use the Anthropic or Ollama adapter.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from alg.events import Trace
from alg.llm.base import Completion, Message, TextBlock, ToolCall, ToolSpec, Usage
from alg.tasks import TaskSpec
from alg.workspace import Workspace

TASKS = Path(__file__).resolve().parents[1] / "tasks"


class ScriptedLLM:
    """Replays `script` one completion per call, then falls back to `default`."""

    def __init__(
        self,
        script: list[Completion] | None = None,
        default: Completion | None = None,
        name: str = "scripted",
    ) -> None:
        self.script = list(script or [])
        self.default = default or Completion(
            blocks=[TextBlock("I could not make progress.")], stop_reason="end_turn"
        )
        self.name = name
        self.calls: list[tuple[str, list[Message], list[ToolSpec]]] = []
        self._counter = itertools.count()

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> Completion:
        self.calls.append((system, list(messages), list(tools or [])))
        index = next(self._counter)
        if index < len(self.script):
            return self.script[index]
        return self.default


def say(text: str) -> Completion:
    return Completion(blocks=[TextBlock(text)], stop_reason="end_turn", usage=Usage(10, 5))


def call(name: str, call_id: str = "c1", **args) -> Completion:
    return Completion(
        blocks=[ToolCall(id=call_id, name=name, args=args)],
        stop_reason="tool_use",
        usage=Usage(20, 8),
    )


@pytest.fixture
def trace() -> Trace:
    ticks = itertools.count()
    return Trace(clock=lambda: float(next(ticks)))


@pytest.fixture
def task_dir() -> Path:
    return TASKS / "calc_bug"


@pytest.fixture
def workspace(tmp_path: Path, task_dir: Path) -> Workspace:
    return Workspace.materialize(task_dir, tmp_path / "work")


@pytest.fixture
def task(task_dir: Path) -> TaskSpec:
    return TaskSpec.load(task_dir)


# The seeded defects in tasks/calc_bug/src/stats.ts, and their fixes. Kept here
# so a change to the task shows up as one diff rather than scattered literals.
SOURCE = "src/stats.ts"

MEAN_BUG = "return total / (values.length + 1);"
MEAN_FIX = "return total / values.length;"
MEDIAN_BUG = "    return ordered[middle]!;\n  }\n  return ordered[middle]!;"
MEDIAN_FIX = (
    "    return ordered[middle]!;\n  }\n"
    "  return (ordered[middle - 1]! + ordered[middle]!) / 2;"
)
