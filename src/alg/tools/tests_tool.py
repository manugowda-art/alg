"""The feedback source: run pytest in the workspace and turn its output into a
structured verdict.

This is the single most important object in the whole harness. Loop engineering
needs evidence that is *comparable across iterations* — "3 failed" vs "1 failed"
is progress, "1 failed" vs "1 failed on a different test" is not. So the runner
returns a report the loop can compare, not a wall of text.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

from ..workspace import Workspace
from . import Tool, ToolOutcome

COUNT = re.compile(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed)")
FAILED_LINE = re.compile(r"^(?:FAILED|ERROR) (\S+)")


@dataclass(frozen=True)
class TestReport:
    exit_code: int
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    failing: tuple[str, ...] = ()
    output: str = ""
    timed_out: bool = False

    @property
    def green(self) -> bool:
        return (
            not self.timed_out
            and self.exit_code == 0
            and self.failed == 0
            and self.errors == 0
        )

    @property
    def signature(self) -> tuple[str, ...]:
        """What "the same failure as last time" means. Used by stop rules to
        detect a loop that is spinning without making progress."""
        return tuple(sorted(self.failing))

    def summary(self) -> str:
        if self.timed_out:
            return "test run TIMED OUT"
        parts = [f"{self.passed} passed", f"{self.failed} failed"]
        if self.errors:
            parts.append(f"{self.errors} errors")
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        head = ", ".join(parts)
        if self.failing:
            head += "\nfailing: " + ", ".join(self.failing)
        return head

    def as_dict(self) -> dict[str, object]:
        return {
            "exit_code": self.exit_code,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "failing": list(self.failing),
            "green": self.green,
            "timed_out": self.timed_out,
        }


def run_tests(workspace: Workspace, target: str | None = None, timeout: float = 120.0) -> TestReport:
    argv = [sys.executable, "-m", "pytest", "-q", "-rf", "--tb=short", "-p", "no:cacheprovider"]
    if target:
        argv.append(target)
    result = workspace.run(argv, timeout=timeout)
    return parse_pytest_output(
        result.stdout + ("\n" + result.stderr if result.stderr else ""),
        exit_code=result.exit_code,
        timed_out=result.timed_out,
    )


def parse_pytest_output(output: str, exit_code: int, timed_out: bool = False) -> TestReport:
    counts: dict[str, int] = {}
    failing: list[str] = []

    for line in output.splitlines():
        match = FAILED_LINE.match(line.strip())
        if match:
            node = match.group(1).rstrip(":")
            if node not in failing:
                failing.append(node)

    # Only the trailing summary line carries authoritative counts; traceback
    # bodies can contain the same words.
    for line in reversed(output.splitlines()):
        if not line.strip():
            continue
        found = COUNT.findall(line)
        if found and ("passed" in line or "failed" in line or "error" in line or "no tests ran" in line):
            for value, label in found:
                key = "errors" if label.startswith("error") else label
                counts[key] = counts.get(key, 0) + int(value)
            break

    return TestReport(
        exit_code=exit_code,
        passed=counts.get("passed", 0),
        failed=counts.get("failed", 0),
        errors=counts.get("errors", 0),
        skipped=counts.get("skipped", 0),
        failing=tuple(failing),
        output=output,
        timed_out=timed_out,
    )


def run_tests_tool(workspace: Workspace, timeout: float = 120.0, on_report=None) -> Tool:
    """Expose the test runner to the model.

    `on_report` lets the surrounding node observe every run the model triggers,
    so the loop's evidence and the model's evidence never diverge.
    """

    def run(target: str | None = None) -> ToolOutcome:
        report = run_tests(workspace, target=target, timeout=timeout)
        if on_report is not None:
            on_report(report)
        body = report.summary()
        if not report.green:
            body += "\n\n" + _tail(report.output, 4_000)
        return ToolOutcome(ok=True, content=body, meta=report.as_dict())

    return Tool(
        name="run_tests",
        description=(
            "Run the test suite with pytest and return the pass/fail counts, the failing test ids, "
            "and the tail of the output. Optionally pass a target such as `tests/test_stats.py::test_mean`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Optional pytest target (file, or file::test).",
                }
            },
            "required": [],
        },
        fn=run,
    )


def _tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return "... [earlier output omitted] ...\n" + text[-limit:]
