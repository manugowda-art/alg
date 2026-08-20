"""The feedback source: run the task's test suite and turn its output into a
structured verdict.

This is the most important object in the whole harness. Loop engineering needs
evidence that is *comparable across iterations* — "3 failed" vs "1 failed" is
progress, "1 failed" vs "1 failed on a different test" is not. So the runner
returns a report the loop can compare, not a wall of text.

The runner is chosen by the task manifest, so adding a language means adding a
parser here, not touching the loop or the graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..tasks import TaskSpec
from ..workspace import Workspace
from . import Tool, ToolOutcome

# pytest
PYTEST_COUNT = re.compile(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed)")
PYTEST_FAILED = re.compile(r"^(?:FAILED|ERROR) (\S+)")

# node --test --test-reporter=tap
TAP_NOT_OK = re.compile(r"^not ok \d+ - (.+?)(?:\s+# .*)?$")
TAP_COUNT = re.compile(r"^# (tests|pass|fail|skipped|todo|cancelled) (\d+)$")
TAP_LOCATION = re.compile(r"^\s+location: '(.+?):\d+:\d+'$")


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
    def collected(self) -> int:
        return self.passed + self.failed + self.errors + self.skipped

    @property
    def green(self) -> bool:
        """Green requires tests to have actually passed.

        A command that collects nothing and exits 0 is a broken verifier, not a
        solved task — and reporting it as success is the worst thing this class
        can do, because every layer above trusts it.
        """
        return (
            not self.timed_out
            and self.exit_code == 0
            and self.failed == 0
            and self.errors == 0
            and self.passed > 0
        )

    @property
    def signature(self) -> tuple[str, ...]:
        """What "the same failure as last time" means. Used by stop rules to
        detect a loop that is spinning without making progress."""
        return tuple(sorted(self.failing))

    def summary(self) -> str:
        if self.timed_out:
            return "test run TIMED OUT"
        if self.collected == 0:
            return (
                "NO TESTS RAN — the test command completed but collected nothing. "
                "The suite did not execute, so its result means nothing."
            )
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


Parser = Callable[..., TestReport]


def parse_node_tap(
    output: str, exit_code: int, timed_out: bool = False, root: Path | None = None
) -> TestReport:
    """Parse TAP 13 as emitted by `node --test --test-reporter=tap`.

    Failing tests are identified as `file::test name` where the file can be
    recovered from the YAML block's `location:` field, so the signature stays
    stable when two files happen to use the same test name.
    """
    counts: dict[str, int] = {}
    failing: list[str] = []
    pending: str | None = None

    for line in output.splitlines():
        match = TAP_NOT_OK.match(line)
        if match:
            if pending is not None:
                failing.append(pending)
            pending = match.group(1).strip()
            continue
        if pending is not None:
            location = TAP_LOCATION.match(line)
            if location:
                path = location.group(1)
                if root is not None:
                    try:
                        path = str(Path(path).relative_to(root))
                    except ValueError:
                        pass
                failing.append(f"{path}::{pending}")
                pending = None
                continue
            if line.startswith("not ok") or line.startswith("ok "):
                failing.append(pending)
                pending = None
        count = TAP_COUNT.match(line)
        if count:
            counts[count.group(1)] = int(count.group(2))
    if pending is not None:
        failing.append(pending)

    deduped = list(dict.fromkeys(failing))
    return TestReport(
        exit_code=exit_code,
        passed=counts.get("pass", 0),
        failed=counts.get("fail", 0),
        errors=counts.get("cancelled", 0),
        skipped=counts.get("skipped", 0),
        failing=tuple(deduped),
        output=output,
        timed_out=timed_out,
    )


def parse_pytest_output(
    output: str, exit_code: int, timed_out: bool = False, root: Path | None = None
) -> TestReport:
    counts: dict[str, int] = {}
    failing: list[str] = []

    for line in output.splitlines():
        match = PYTEST_FAILED.match(line.strip())
        if match:
            node = match.group(1).rstrip(":")
            if node not in failing:
                failing.append(node)

    # Only the trailing summary line carries authoritative counts; traceback
    # bodies can contain the same words.
    for line in reversed(output.splitlines()):
        if not line.strip():
            continue
        found = PYTEST_COUNT.findall(line)
        if found and any(w in line for w in ("passed", "failed", "error", "no tests ran")):
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


PARSERS: dict[str, Parser] = {
    "node-test": parse_node_tap,
    "pytest": parse_pytest_output,
}


def run_tests(
    workspace: Workspace,
    task: TaskSpec,
    target: str | None = None,
    timeout: float | None = None,
) -> TestReport:
    parser = PARSERS.get(task.runner)
    if parser is None:
        raise ValueError(
            f"unknown runner {task.runner!r}; known runners: {', '.join(sorted(PARSERS))}"
        )
    result = workspace.run(task.argv(target), timeout=timeout or task.timeout)
    return parser(
        result.stdout + ("\n" + result.stderr if result.stderr else ""),
        exit_code=result.exit_code,
        timed_out=result.timed_out,
        root=workspace.root,
    )


def run_tests_tool(
    workspace: Workspace,
    task: TaskSpec,
    timeout: float | None = None,
    on_report=None,
) -> Tool:
    """Expose the test runner to the model.

    `on_report` lets the surrounding node observe every run the model triggers,
    so the loop's evidence and the model's evidence never diverge.
    """

    def run(target: str | None = None) -> ToolOutcome:
        report = run_tests(workspace, task, target=target, timeout=timeout)
        if on_report is not None:
            on_report(report)
        body = report.summary()
        if not report.green:
            body += "\n\n" + _tail(report.output, 4_000)
        return ToolOutcome(ok=True, content=body, meta=report.as_dict())

    return Tool(
        name="run_tests",
        description=(
            f"Run the {task.language} test suite (`{' '.join(task.test_command)}`) and return the "
            f"pass/fail counts, the failing test ids, and the tail of the output. Optionally pass "
            f"a target to narrow the run to {task.focus_hint}."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": f"Optional: {task.focus_hint}."}
            },
            "required": [],
        },
        fn=run,
    )


def _tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return "... [earlier output omitted] ...\n" + text[-limit:]
