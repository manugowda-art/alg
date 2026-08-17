"""The sandbox: a writable copy of a task, plus a jailed path resolver and a
bounded command runner.

Harness principle: the model never touches the source of truth. It gets a
disposable copy, every path it names is validated against the workspace root,
and every command it triggers runs with a timeout and captured output.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class WorkspaceError(Exception):
    """Raised when a requested path escapes the workspace."""


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class Workspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace root does not exist: {self.root}")

    @classmethod
    def materialize(cls, source: str | Path, dest: str | Path) -> "Workspace":
        """Copy a task directory into a fresh workspace."""
        source, dest = Path(source).resolve(), Path(dest)
        if not source.is_dir():
            raise WorkspaceError(f"task source does not exist: {source}")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(
            source,
            dest,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
        return cls(dest)

    # --- path jail -------------------------------------------------------

    def resolve(self, relpath: str) -> Path:
        """Resolve a model-supplied path, refusing anything outside the root.

        Absolute paths, `..` traversal, and symlinks that point outward are all
        rejected here rather than in each tool.
        """
        candidate = Path(relpath)
        if candidate.is_absolute():
            raise WorkspaceError(f"absolute paths are not allowed: {relpath}")
        resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise WorkspaceError(f"path escapes workspace: {relpath}")
        return resolved

    def relative(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    def list_files(self, include_hidden: bool = False) -> list[str]:
        out: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            rel = self.relative(path)
            parts = Path(rel).parts
            if any(p in {"__pycache__", ".pytest_cache"} for p in parts):
                continue
            if not include_hidden and any(p.startswith(".") for p in parts):
                continue
            out.append(rel)
        return out

    # --- bounded execution ----------------------------------------------

    def run(
        self,
        argv: list[str],
        timeout: float = 60.0,
        max_output: int = 20_000,
    ) -> CommandResult:
        try:
            proc = subprocess.run(
                argv,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                argv=argv,
                exit_code=-1,
                stdout=_truncate(exc.stdout or "", max_output),
                stderr=_truncate(exc.stderr or "", max_output),
                timed_out=True,
            )
        except FileNotFoundError as exc:
            return CommandResult(argv=argv, exit_code=-1, stdout="", stderr=str(exc), timed_out=False)
        return CommandResult(
            argv=argv,
            exit_code=proc.returncode,
            stdout=_truncate(proc.stdout, max_output),
            stderr=_truncate(proc.stderr, max_output),
            timed_out=False,
        )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n... [{len(text) - limit} chars truncated] ...\n{tail}"
