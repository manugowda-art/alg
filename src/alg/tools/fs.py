"""Read and search tools. Constrained on purpose: every path goes through the
workspace jail, and every result is size-capped so one `read_file` cannot eat
the context window.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..workspace import Workspace, WorkspaceError
from . import Tool, ToolOutcome

MAX_READ_BYTES = 40_000
MAX_MATCHES = 100


def read_only_tools(workspace: Workspace) -> list[Tool]:
    return [
        _list_files(workspace),
        _read_file(workspace),
        _search(workspace),
    ]


def _list_files(workspace: Workspace) -> Tool:
    def run() -> ToolOutcome:
        files = workspace.list_files()
        return ToolOutcome(
            ok=True,
            content="\n".join(files) or "(workspace is empty)",
            meta={"count": len(files)},
        )

    return Tool(
        name="list_files",
        description="List every file in the workspace, as paths relative to the workspace root.",
        input_schema={"type": "object", "properties": {}, "required": []},
        fn=run,
    )


def _read_file(workspace: Workspace) -> Tool:
    def run(path: str) -> ToolOutcome:
        target = workspace.resolve(path)
        if not target.is_file():
            return ToolOutcome.error(f"no such file: {path}")
        data = target.read_text(errors="replace")
        truncated = len(data.encode()) > MAX_READ_BYTES
        if truncated:
            data = data[:MAX_READ_BYTES]
        numbered = "\n".join(
            f"{i:>4} | {line}" for i, line in enumerate(data.splitlines(), start=1)
        )
        suffix = "\n... [truncated]" if truncated else ""
        return ToolOutcome(ok=True, content=numbered + suffix, meta={"truncated": truncated})

    return Tool(
        name="read_file",
        description=(
            "Read one file from the workspace. Returns the contents with 1-based line numbers "
            "prefixed as `NNNN | `; those prefixes are display only and are not part of the file."
        ),
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path relative to the workspace root."}},
            "required": ["path"],
        },
        fn=run,
    )


def _search(workspace: Workspace) -> Tool:
    def run(pattern: str, glob: str = "**/*.py") -> ToolOutcome:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolOutcome.error(f"invalid regular expression: {exc}")

        matches: list[str] = []
        for path in sorted(workspace.root.glob(glob)):
            if not path.is_file():
                continue
            try:
                rel = workspace.relative(path)
            except ValueError:  # pragma: no cover - glob cannot escape root
                continue
            for lineno, line in enumerate(
                path.read_text(errors="replace").splitlines(), start=1
            ):
                if regex.search(line):
                    matches.append(f"{rel}:{lineno}: {line.strip()}")
                    if len(matches) >= MAX_MATCHES:
                        break
            if len(matches) >= MAX_MATCHES:
                break

        return ToolOutcome(
            ok=True,
            content="\n".join(matches) or f"no matches for {pattern!r} in {glob}",
            meta={"count": len(matches)},
        )

    return Tool(
        name="search",
        description="Search workspace files for a Python regular expression. Returns `path:line: text`.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression."},
                "glob": {"type": "string", "description": "Glob to limit the search. Default `**/*.py`."},
            },
            "required": ["pattern"],
        },
        fn=run,
    )


def write_file_tool(workspace: Workspace) -> Tool:
    """Full-file write. Available but discouraged — `edit_file` is safer because
    it fails loudly when the model's picture of the file is stale."""

    def run(path: str, content: str) -> ToolOutcome:
        target = workspace.resolve(path)
        if target.is_dir():
            raise WorkspaceError(f"{path} is a directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.is_file()
        target.write_text(content)
        return ToolOutcome(
            ok=True,
            content=f"{'overwrote' if existed else 'created'} {path} ({len(content)} chars)",
            meta={"path": path, "created": not existed},
        )

    return Tool(
        name="write_file",
        description=(
            "Write a whole file, creating or overwriting it. Prefer `edit_file` for changes to an "
            "existing file — it verifies the text you expect is actually there."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string", "description": "Complete new file contents."},
            },
            "required": ["path", "content"],
        },
        fn=run,
        mutating=True,
    )


def edit_file_tool(workspace: Workspace) -> Tool:
    """Exact-string replacement with a uniqueness requirement.

    This is the staleness check that makes edits safe: if `old_text` is absent
    or ambiguous, nothing is written and the model is told why.
    """

    def run(path: str, old_text: str, new_text: str) -> ToolOutcome:
        target = workspace.resolve(path)
        if not target.is_file():
            return ToolOutcome.error(f"no such file: {path}")
        if not old_text:
            return ToolOutcome.error("old_text must not be empty")
        data = target.read_text()
        occurrences = data.count(old_text)
        if occurrences == 0:
            return ToolOutcome.error(
                f"old_text not found in {path}; re-read the file (line-number prefixes from "
                f"read_file are not part of the content)"
            )
        if occurrences > 1:
            return ToolOutcome.error(
                f"old_text appears {occurrences} times in {path}; include more surrounding "
                f"context so it matches exactly once"
            )
        target.write_text(data.replace(old_text, new_text, 1))
        return ToolOutcome(ok=True, content=f"edited {path}", meta={"path": path})

    return Tool(
        name="edit_file",
        description=(
            "Replace an exact snippet in a file. `old_text` must appear exactly once, and must be "
            "the raw file text without read_file's line-number prefixes."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string", "description": "Exact text to replace; must be unique."},
                "new_text": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "old_text", "new_text"],
        },
        fn=run,
        mutating=True,
    )


def diff_tool(workspace: Workspace, baseline: dict[str, str]) -> Tool:
    """Unified diff of the workspace against its starting state — the harness's
    answer to "what has this agent actually changed?"."""

    def run() -> ToolOutcome:
        text = render_diff(workspace, baseline)
        return ToolOutcome(ok=True, content=text or "(no changes yet)")

    return Tool(
        name="diff",
        description="Show a unified diff of every change made to the workspace so far.",
        input_schema={"type": "object", "properties": {}, "required": []},
        fn=run,
    )


def snapshot(workspace: Workspace) -> dict[str, str]:
    """Text contents of every workspace file, for later diffing."""
    out: dict[str, str] = {}
    for rel in workspace.list_files():
        path = workspace.root / rel
        try:
            out[rel] = path.read_text()
        except UnicodeDecodeError:
            continue
    return out


def render_diff(workspace: Workspace, baseline: dict[str, str]) -> str:
    import difflib

    current = snapshot(workspace)
    chunks: list[str] = []
    for rel in sorted(set(baseline) | set(current)):
        before = baseline.get(rel, "")
        after = current.get(rel, "")
        if before == after:
            continue
        chunks.extend(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
        )
    return "".join(chunks)


def changed_files(workspace: Workspace, baseline: dict[str, str]) -> list[str]:
    current = snapshot(workspace)
    return sorted(
        rel for rel in set(baseline) | set(current) if baseline.get(rel) != current.get(rel)
    )


def restore(workspace: Workspace, baseline: dict[str, str]) -> list[str]:
    """Roll the workspace back to a snapshot. Used when a repair attempt makes
    things worse — a loop needs an undo, not just a retry."""
    reverted = changed_files(workspace, baseline)
    for rel in reverted:
        path = workspace.root / rel
        if rel in baseline:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(baseline[rel])
        elif path.is_file():
            path.unlink()
    return reverted


def ensure_within(workspace: Workspace, path: Path) -> None:  # pragma: no cover - helper
    workspace.relative(path)
