"""Task specification: what the harness needs to know about a repair task.

The engine (harness, loop, graph) is language-agnostic. Everything that is
specific to *this* task — how to run its tests, how to focus one test, which
files count as source — lives in the task's `alg.task.json`, not in Python.

That is what lets the same graph drive a TypeScript task, a Python task, or
anything else with a command that exits non-zero when the work is not done.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

MANIFEST = "alg.task.json"


class TaskError(Exception):
    pass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    language: str
    runner: str
    test_command: tuple[str, ...]
    description: str = ""
    focus_template: tuple[str, ...] = ()
    focus_hint: str = "a test target"
    source_glob: str = "**/*"
    test_glob: str = "**/*"
    search_glob: str = "**/*"
    timeout: float = 120.0

    @classmethod
    def load(cls, task_dir: str | Path) -> "TaskSpec":
        directory = Path(task_dir)
        if not directory.is_dir():
            raise TaskError(
                f"no such task directory: {task_dir} "
                f"(resolved to {directory.resolve()}) — run from the repo root, "
                f"or pass an absolute path"
            )
        path = directory / MANIFEST
        if not path.is_file():
            raise TaskError(
                f"{directory} exists but has no {MANIFEST}; "
                f"every task needs one (see README, 'Adding a task')"
            )
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise TaskError(f"{path} is not valid JSON: {exc}") from exc

        missing = [key for key in ("name", "runner", "test_command") if key not in raw]
        if missing:
            raise TaskError(f"{path} is missing required key(s): {', '.join(missing)}")
        if not isinstance(raw["test_command"], list) or not raw["test_command"]:
            raise TaskError(f"{path}: test_command must be a non-empty array")

        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise TaskError(f"{path}: unknown key(s): {', '.join(unknown)}")

        return cls(
            name=raw["name"],
            language=raw.get("language", "unknown"),
            runner=raw["runner"],
            test_command=tuple(raw["test_command"]),
            description=raw.get("description", ""),
            focus_template=tuple(raw.get("focus_template", ())),
            focus_hint=raw.get("focus_hint", "a test target"),
            source_glob=raw.get("source_glob", "**/*"),
            test_glob=raw.get("test_glob", "**/*"),
            search_glob=raw.get("search_glob", "**/*"),
            timeout=float(raw.get("timeout", 120.0)),
        )

    def argv(self, target: str | None = None) -> list[str]:
        """The command to run, optionally narrowed to one test."""
        argv = list(self.test_command)
        if target and self.focus_template:
            argv.extend(part.replace("{target}", target) for part in self.focus_template)
        elif target:
            argv.append(target)
        return argv
