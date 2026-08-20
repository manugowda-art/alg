"""Preflight: check every link in the chain before spending an hour on a run.

A local-model run has more ways to fail than an API run — the server may be
down, the model may not be pulled, its template may not support tools, its
context may be too small, or Node may be too old to run the task. Each of those
produces a different confusing symptom mid-run. This checks them up front and
names the fix.

The last check also *measures* one model call, which is what makes the
suggested wall-clock budget a number rather than a guess.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .llm.base import Message, ToolSpec
from .tasks import TaskError, TaskSpec
from .tools.tests_tool import run_tests
from .workspace import Workspace

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""

    @property
    def mark(self) -> str:
        return {OK: "✓", WARN: "!", FAIL: "✗"}[self.status]


def check_python() -> Check:
    version = ".".join(str(n) for n in sys.version_info[:3])
    if sys.version_info < (3, 11):
        return Check("python", FAIL, version, "alg needs Python 3.11+")
    return Check("python", OK, version)


def check_node(task: TaskSpec | None) -> Check:
    """Probe the runtime by running a real .ts file.

    Version comparison is a guess; this is a measurement. It also reports the
    exact flag that works, which is what goes into the manifest.
    """
    binary = (task.test_command[0] if task else "node") or "node"
    if shutil.which(binary) is None:
        return Check("node", FAIL, f"{binary} not found on PATH", "install Node 22.18+")
    try:
        version = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return Check("node", FAIL, str(exc))
    if binary != "node":
        return Check("node", OK, f"{binary} {version}")

    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.ts"
        probe.write_text("const answer: number = 42;\nconsole.log(answer);\n")

        def runs(argv: list[str]) -> bool:
            try:
                done = subprocess.run(argv, capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.SubprocessError):
                return False
            return done.returncode == 0 and "42" in done.stdout

        if runs([binary, str(probe)]):
            return Check("node", OK, f"{version}, runs .ts natively")
        if runs([binary, "--experimental-strip-types", str(probe)]):
            return Check(
                "node", WARN,
                f"{version} cannot run .ts without a flag",
                'add "--experimental-strip-types" to test_command in '
                "tasks/calc_bug/alg.task.json, or upgrade to Node 22.18+",
            )
    return Check(
        "node", FAIL, f"{version} cannot run TypeScript at all", "upgrade to Node 22.18+"
    )


def check_task(task_dir: str | Path) -> tuple[Check, TaskSpec | None]:
    try:
        task = TaskSpec.load(task_dir)
    except TaskError as exc:
        return Check("task manifest", FAIL, str(exc)), None
    return (
        Check("task manifest", OK, f"{task.name} ({task.language}, runner={task.runner})"),
        task,
    )


def check_baseline(task_dir: str | Path, task: TaskSpec, work_dir: Path) -> Check:
    """Run the task's own suite. A task that cannot fail cannot be solved."""
    try:
        workspace = Workspace.materialize(task_dir, work_dir)
        report = run_tests(workspace, task)
    except Exception as exc:
        return Check("task baseline", FAIL, f"{type(exc).__name__}: {exc}")
    detail = f"{report.passed} passed, {report.failed} failed"
    if report.collected == 0:
        # Checked before "green": a command that exits 0 having collected
        # nothing looks like success and is the most dangerous state here.
        return Check(
            "task baseline", FAIL, "no tests ran — the verifier is not working",
            f"`{' '.join(task.test_command)}` collected nothing. "
            f"If the suite is TypeScript, this is usually Node not stripping types "
            f"(see the node check above).",
        )
    if report.green:
        return Check(
            "task baseline", WARN, detail + " — already green",
            "the agent will stop at `report` with nothing to do",
        )
    return Check("task baseline", OK, detail + " — the verifier works")


def check_provider(provider: str, model: str | None, **kwargs) -> tuple[list[Check], object | None]:
    from .llm import build

    try:
        llm = build(provider, model, **kwargs)
    except Exception as exc:
        return [Check("provider", FAIL, str(exc))], None

    checks: list[Check] = []
    if provider != "ollama":
        checks.append(Check("provider", OK, llm.name))
        return checks, llm

    try:
        models = llm.list_models()
    except RuntimeError as exc:
        return [
            Check("ollama", FAIL, str(exc), "start it with `ollama serve`, or pass --host")
        ], None
    plural = "model" if len(models) == 1 else "models"
    checks.append(Check("ollama", OK, f"{llm.host} ({len(models)} {plural} pulled)"))

    if llm.model not in models:
        near = [m for m in models if m.split(":")[0] == llm.model.split(":")[0]]
        hint = f"pulled: {', '.join(near or models) or '(none)'}"
        checks.append(
            Check("model", FAIL, f"{llm.model} not pulled", f"`ollama pull {llm.model}` — {hint}")
        )
        return checks, None

    try:
        tools_ok = llm.supports_tools()
        context = llm.context_length()
    except RuntimeError as exc:
        checks.append(Check("model", WARN, f"could not inspect: {exc}"))
        return checks, llm

    if not tools_ok:
        checks.append(
            Check(
                "model", FAIL, f"{llm.model} does not advertise tool support",
                "this harness is tool-driven; use a tool-capable model (qwen3, qwen2.5-coder)",
            )
        )
        return checks, None

    detail = f"{llm.model}, tools supported"
    if context:
        detail += f", context {context:,}"
    status = OK
    fix = ""
    if context and llm.num_ctx and llm.num_ctx > context:
        status, fix = WARN, f"--num-ctx {llm.num_ctx} exceeds the model's {context:,}; it will clip"
    checks.append(Check("model", status, detail, fix))
    return checks, llm


def check_tool_call(llm) -> tuple[Check, float | None]:
    """The one that matters: does this model actually emit a tool call?"""
    spec = ToolSpec(
        name="report_status",
        description="Report whether the check succeeded. Call this immediately.",
        input_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean", "description": "true if you can read this"}},
            "required": ["ok"],
        },
    )
    started = time.monotonic()
    try:
        completion = llm.complete(
            "You are verifying a tool-calling setup. Use the tool you are given.",
            [Message.user("Call report_status with ok set to true. Do not reply with text.")],
            [spec],
        )
    except Exception as exc:
        return Check("tool call", FAIL, f"{type(exc).__name__}: {exc}"), None
    elapsed = time.monotonic() - started

    calls = completion.tool_calls
    if not calls:
        return (
            Check(
                "tool call", FAIL,
                f"model replied with text instead of a tool call after {elapsed:.1f}s",
                "text: " + (completion.text()[:120].replace("\n", " ") or "(empty)"),
            ),
            elapsed,
        )
    if calls[0].name != "report_status":
        return (
            Check("tool call", WARN, f"called {calls[0].name!r} instead of report_status"),
            elapsed,
        )
    return Check("tool call", OK, f"{calls[0].name}({calls[0].args}) in {elapsed:.1f}s"), elapsed


def suggest(task_dir: str, provider: str, model: str | None, elapsed: float | None,
            max_iterations: int = 10) -> str:
    """Size the wall-clock budget from a measured call rather than a guess."""
    parts = [f"alg run {task_dir} --provider {provider}"]
    if model:
        parts.append(f"--model {model}")
    if elapsed:
        # Each iteration is one model call plus tool time; leave generous headroom.
        budget = max(600, int(elapsed * max_iterations * 2.5 / 60 + 1) * 60)
        parts.append(f"--wall-clock {budget}")
    parts.append("--show-diff")
    return " \\\n    ".join(parts)


def run_doctor(
    task_dir: str,
    provider: str,
    model: str | None,
    work_dir: Path,
    max_iterations: int = 10,
    **provider_kwargs,
) -> tuple[list[Check], str]:
    checks: list[Check] = [check_python()]
    task_check, task = check_task(task_dir)
    checks.append(check_node(task))
    checks.append(task_check)
    if task is not None:
        checks.append(check_baseline(task_dir, task, work_dir))

    provider_checks, llm = check_provider(provider, model, **provider_kwargs)
    checks.extend(provider_checks)

    elapsed = None
    if llm is not None:
        check, elapsed = check_tool_call(llm)
        checks.append(check)

    command = suggest(task_dir, provider, model, elapsed, max_iterations)
    return checks, command
