"""Tool registry: the harness's permission and validation boundary.

Three rules hold for every tool:
  1. A tool never raises into the loop. Failures come back as an outcome with
     `ok=False`, so the model can read the error and try something else.
  2. Arguments are validated against the declared schema before the tool body
     runs (required keys, unknown keys, coarse types).
  3. Every call is traced with its arguments and a truncated result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..events import Trace
from ..llm.base import ToolSpec
from ..workspace import WorkspaceError

MAX_TRACED_RESULT = 2_000


@dataclass(frozen=True)
class ToolOutcome:
    ok: bool
    content: str
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def error(cls, message: str, **meta: Any) -> "ToolOutcome":
        return cls(ok=False, content=message, meta=meta)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., ToolOutcome]
    mutating: bool = False

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, input_schema=self.input_schema)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None, trace: Trace | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._trace = trace
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, only: list[str] | None = None) -> list[ToolSpec]:
        names = only if only is not None else self.names
        return [self._tools[n].spec() for n in names if n in self._tools]

    def subset(self, names: list[str]) -> "ToolRegistry":
        """A narrower view of the same tools — how read-only phases are enforced."""
        missing = [n for n in names if n not in self._tools]
        if missing:
            raise ValueError(f"unknown tools: {missing}")
        return ToolRegistry([self._tools[n] for n in names], trace=self._trace)

    def call(self, name: str, args: dict[str, Any]) -> ToolOutcome:
        tool = self._tools.get(name)
        if tool is None:
            outcome = ToolOutcome.error(
                f"unknown tool {name!r}; available tools: {', '.join(self.names)}"
            )
            self._emit(name, args, outcome)
            return outcome

        problem = validate_args(tool.input_schema, args)
        if problem:
            outcome = ToolOutcome.error(f"invalid arguments for {name}: {problem}")
            self._emit(name, args, outcome)
            return outcome

        try:
            outcome = tool.fn(**args)
        except WorkspaceError as exc:
            outcome = ToolOutcome.error(f"refused: {exc}")
        except Exception as exc:  # a broken tool must not kill the run
            outcome = ToolOutcome.error(f"tool {name} raised {type(exc).__name__}: {exc}")
        self._emit(name, args, outcome)
        return outcome

    def _emit(self, name: str, args: dict[str, Any], outcome: ToolOutcome) -> None:
        if self._trace is None:
            return
        self._trace.emit(
            "tool.call",
            tool=name,
            args=args,
            ok=outcome.ok,
            result=outcome.content[:MAX_TRACED_RESULT],
            meta=outcome.meta,
        )


_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Return a human-readable problem, or None if the arguments are acceptable."""
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    missing = [key for key in required if key not in args]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"

    unknown = [key for key in args if key not in properties]
    if unknown:
        return (
            f"unknown argument(s): {', '.join(sorted(unknown))}; "
            f"expected: {', '.join(sorted(properties))}"
        )

    for key, value in args.items():
        declared = properties[key].get("type")
        expected = _TYPES.get(declared) if declared else None
        if expected is None:
            continue
        if declared == "number" and isinstance(value, bool):
            return f"argument {key!r} must be a number"
        if not isinstance(value, expected):
            return f"argument {key!r} must be a {declared}, got {type(value).__name__}"
    return None
