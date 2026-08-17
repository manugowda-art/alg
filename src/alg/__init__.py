"""ALG — Agent harness, Loop engineering, Graph engineering, built from scratch.

    from alg.agent import AgentConfig, build_agent
    from alg.events import Trace
    from alg.llm import build

    trace = Trace("runs/demo/trace.jsonl")
    agent = build_agent("tasks/calc_bug", "runs/demo/work", build("anthropic"), trace)
    state = agent.run(checkpoint_path="runs/demo/checkpoints.jsonl")
"""

from .events import Event, Trace
from .graph import END, START, Graph, JsonlCheckpointer
from .loop import AgentLoop, LoopState
from .tools import Tool, ToolOutcome, ToolRegistry
from .workspace import Workspace

__all__ = [
    "AgentLoop",
    "END",
    "Event",
    "Graph",
    "JsonlCheckpointer",
    "LoopState",
    "START",
    "Tool",
    "ToolOutcome",
    "ToolRegistry",
    "Trace",
    "Workspace",
]
