"""The use case: a self-healing test fixer, assembled from the three layers.

    harness  workspace + jailed tools + traced tool calls   (workspace.py, tools/)
    loop     bounded model/tool cycle with named stop rules (loop.py)
    graph    baseline -> repair -> verify -> {done, retry, give up}  (graph.py)

Read `graph()` below first: the topology is the specification. Everything else
is supporting machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .events import Trace
from .graph import END, Graph, JsonlCheckpointer, State
from .llm.base import LLM
from .loop import (
    AgentLoop,
    LoopState,
    max_iterations,
    stalled_evidence,
    token_budget,
    tool_error_streak,
    wall_clock,
)
from .tools import ToolRegistry
from .tools.fs import (
    diff_tool,
    edit_file_tool,
    read_only_tools,
    render_diff,
    restore,
    snapshot,
    write_file_tool,
)
from .tools.patch import apply_patch_tool
from .tools.tests_tool import TestReport, run_tests, run_tests_tool
from .workspace import Workspace

SYSTEM_PROMPT = """\
You are fixing a small Python package so that its existing test suite passes.

The tests encode the intended behaviour. Treat them as the specification: fix the
source, and do not edit, skip, delete, or weaken any test to make it pass.

How to work:
- Read before you write. `run_tests` shows you what is failing; `read_file` and
  `search` show you why.
- Change the smallest amount of code that fixes the actual defect. No refactors,
  no new abstractions, no error handling for cases that cannot happen.
- Use `edit_file` for small changes and `apply_patch` for multi-line ones. Both
  require the surrounding text to match the file exactly.
- After editing, run `run_tests` again to check your fix against real output
  rather than assuming it worked.

When every test passes, stop and reply with one short paragraph: the defect you
found, and the change you made. If you cannot get to green, say plainly what is
still failing and what you tried.
"""

REPAIR_GOAL = """\
The test suite is failing. Diagnose the defect and fix the source.

Current state of the suite:
{report}
"""

RETRY_GOAL = """\
That attempt did not get the suite to green.

Attempt {attempt} of {max_attempts}. Current state of the suite:
{report}

{assessment}

Re-read the relevant source before changing it again, and consider whether your
diagnosis was wrong rather than your edit.
"""


@dataclass
class AgentConfig:
    max_attempts: int = 3
    loop_max_iterations: int = 10
    loop_token_budget: int = 300_000
    loop_wall_clock: float = 600.0
    test_timeout: float = 120.0
    allow_write_file: bool = False
    revert_regressions: bool = True


@dataclass
class FixerAgent:
    """Holds the un-serializable machinery (workspace, model, tools) so graph
    state stays plain JSON and checkpoints stay resumable."""

    workspace: Workspace
    llm: LLM
    trace: Trace
    config: AgentConfig = field(default_factory=AgentConfig)

    def __post_init__(self) -> None:
        self._origin = snapshot(self.workspace)  # pristine copy, for the final diff
        self._best_snapshot = dict(self._origin)
        self._best_score: tuple[int, int] | None = None
        self._observed: list[TestReport] = []

    # --- graph nodes ------------------------------------------------------

    def baseline(self, state: State) -> State:
        report = run_tests(self.workspace, timeout=self.config.test_timeout)
        self.trace.emit("tests.run", phase="baseline", **report.as_dict())
        self._best_score = _score(report)
        return {
            "attempt": 0,
            "reports": [report.as_dict()],
            "green": report.green,
            "last_summary": report.summary(),
        }

    def repair(self, state: State) -> State:
        attempt = int(state.get("attempt", 0)) + 1
        if attempt == 1:
            goal = REPAIR_GOAL.format(report=state["last_summary"])
        else:
            goal = RETRY_GOAL.format(
                attempt=attempt,
                max_attempts=self.config.max_attempts,
                report=state["last_summary"],
                assessment=state.get("assessment", ""),
            )

        registry = self._registry()
        loop = AgentLoop(
            llm=self.llm,
            registry=registry,
            system=SYSTEM_PROMPT,
            trace=self.trace,
            label=f"repair#{attempt}",
            stop_rules=[
                max_iterations(self.config.loop_max_iterations),
                token_budget(self.config.loop_token_budget),
                wall_clock(self.config.loop_wall_clock),
                tool_error_streak(3),
                stalled_evidence(4),
            ],
            evidence_fn=self._evidence,
        )
        result = loop.run(goal)

        return {
            "attempt": attempt,
            "loop_stop_reasons": [*state.get("loop_stop_reasons", []), result.stop_reason],
            "model_note": result.final_text[-1_500:],
            "tokens": {
                "input": state.get("tokens", {}).get("input", 0) + result.usage.input_tokens,
                "output": state.get("tokens", {}).get("output", 0) + result.usage.output_tokens,
            },
        }

    def verify(self, state: State) -> State:
        report = run_tests(self.workspace, timeout=self.config.test_timeout)
        self.trace.emit("tests.run", phase="verify", attempt=state.get("attempt"), **report.as_dict())

        score = _score(report)
        previous = state["reports"][-1]
        previous_score = (previous["failed"] + previous["errors"], -previous["passed"])
        assessment, reverted = self._assess(state, report, score)

        return {
            "reports": [*state["reports"], report.as_dict()],
            "green": report.green,
            "last_summary": report.summary(),
            "assessment": assessment,
            "reverted": reverted,
            "progress": score < previous_score,
        }

    def report(self, state: State) -> State:
        diff = render_diff(self.workspace, self._origin)
        summary = (
            f"green after {state.get('attempt', 0)} attempt(s)"
            if state.get("green")
            else f"not fixed after {state.get('attempt', 0)} attempt(s): {state.get('last_summary')}"
        )
        self.trace.emit("agent.result", green=bool(state.get("green")), summary=summary)
        return {"summary": summary, "diff": diff}

    # --- wiring -----------------------------------------------------------

    def graph(self, max_steps: int = 40) -> Graph:
        graph = Graph(trace=self.trace, max_steps=max_steps)
        graph.add_node("baseline", self.baseline)
        graph.add_node("repair", self.repair)
        graph.add_node("verify", self.verify)
        graph.add_node("report", self.report)

        graph.set_entry("baseline")
        graph.add_conditional_edge(
            "baseline",
            lambda s: "green" if s["green"] else "red",
            {"green": "report", "red": "repair"},
        )
        graph.add_edge("repair", "verify")
        graph.add_conditional_edge("verify", self._verify_router, {"green": "report", "retry": "repair", "exhausted": "report"})
        graph.add_edge("report", END)
        return graph

    def _verify_router(self, state: State) -> str:
        if state["green"]:
            return "green"
        if int(state.get("attempt", 0)) >= self.config.max_attempts:
            return "exhausted"
        return "retry"

    def run(self, checkpoint_path: str | Path | None = None) -> State:
        graph = self.graph()
        checkpointer = JsonlCheckpointer(checkpoint_path) if checkpoint_path else None
        result = graph.invoke(
            {"task": str(self.workspace.root), "max_attempts": self.config.max_attempts},
            checkpointer=checkpointer,
        )
        return result.state

    # --- internals --------------------------------------------------------

    def _registry(self) -> ToolRegistry:
        tools = [
            *read_only_tools(self.workspace),
            edit_file_tool(self.workspace),
            apply_patch_tool(self.workspace),
            diff_tool(self.workspace, self._origin),
            run_tests_tool(
                self.workspace,
                timeout=self.config.test_timeout,
                on_report=self._observed.append,
            ),
        ]
        if self.config.allow_write_file:
            tools.append(write_file_tool(self.workspace))
        return ToolRegistry(tools, trace=self.trace)

    def _evidence(self, state: LoopState) -> str | None:
        """Evidence for the loop's stall detector: the failing-test signature of
        the most recent run the model itself triggered."""
        if not self._observed:
            return None
        report = self._observed[-1]
        return f"{report.passed}p/{report.failed}f:{','.join(report.signature)}"

    def _assess(
        self, state: State, report: TestReport, score: tuple[int, int]
    ) -> tuple[str, bool]:
        best = self._best_score
        if report.green:
            self._best_snapshot = snapshot(self.workspace)
            self._best_score = score
            return "All tests pass.", False

        if best is not None and score > best:
            # Strictly worse than the best state we have seen. Roll back rather
            # than letting the next attempt build on a regression.
            if self.config.revert_regressions:
                reverted = restore(self.workspace, self._best_snapshot)
                self.trace.emit("agent.revert", files=reverted, reason="regression")
                return (
                    "Your last change made the suite worse, so it was reverted. "
                    "The workspace is back to the best state seen so far.",
                    True,
                )
            return "Your last change made the suite worse.", False

        if best is None or score < best:
            self._best_snapshot = snapshot(self.workspace)
            self._best_score = score
            return "That improved the suite but did not finish the job.", False

        return "No measurable change in the suite from that attempt.", False


def _score(report: TestReport) -> tuple[int, int]:
    """Lower is better: fewer failures first, then more passes."""
    return (report.failed + report.errors, -report.passed)


def build_agent(
    task_dir: str | Path,
    work_dir: str | Path,
    llm: LLM,
    trace: Trace,
    config: AgentConfig | None = None,
) -> FixerAgent:
    workspace = Workspace.materialize(task_dir, work_dir)
    return FixerAgent(
        workspace=workspace, llm=llm, trace=trace, config=config or AgentConfig()
    )
