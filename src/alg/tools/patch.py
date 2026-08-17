"""Unified-diff application, implemented from scratch.

A patch tool is where a harness earns its keep or loses the run. This one is
strict about the things that matter and forgiving about the one thing that
doesn't:

  * forgiving: hunk line numbers may be off. Real models miscount. We search
    outward from the stated position for the context block.
  * strict: the context and removed lines must match the file exactly. If they
    do not, nothing is written and the model gets told which hunk failed.
  * atomic: either every hunk in the patch applies, or the workspace is
    untouched. A half-applied patch is the worst possible state to hand back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..workspace import Workspace
from . import Tool, ToolOutcome

HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
SEARCH_WINDOW = 200


class PatchError(Exception):
    pass


@dataclass
class Hunk:
    old_start: int
    lines: list[str] = field(default_factory=list)

    @property
    def before(self) -> list[str]:
        return [line[1:] for line in self.lines if line[:1] in (" ", "-", "")]

    @property
    def after(self) -> list[str]:
        return [line[1:] for line in self.lines if line[:1] in (" ", "+", "")]


@dataclass
class FilePatch:
    path: str
    hunks: list[Hunk] = field(default_factory=list)
    is_new: bool = False
    is_delete: bool = False


def parse_patch(text: str) -> list[FilePatch]:
    """Parse a unified diff. Ignores `diff --git`/`index` noise."""
    files: list[FilePatch] = []
    current: FilePatch | None = None
    hunk: Hunk | None = None
    old_path: str | None = None

    for raw in text.splitlines():
        if raw.startswith("--- "):
            old_path = _strip_prefix(raw[4:].strip())
            hunk = None
            continue
        if raw.startswith("+++ "):
            new_path = _strip_prefix(raw[4:].strip())
            is_new = old_path == "/dev/null"
            is_delete = new_path == "/dev/null"
            path = old_path if is_delete else new_path
            if not path or path == "/dev/null":
                raise PatchError(f"cannot determine target path from header: {raw!r}")
            current = FilePatch(path=path, is_new=is_new, is_delete=is_delete)
            files.append(current)
            hunk = None
            continue
        if raw.startswith("@@"):
            match = HUNK_HEADER.match(raw)
            if match is None:
                raise PatchError(f"malformed hunk header: {raw!r}")
            if current is None:
                raise PatchError("hunk found before any file header (--- / +++)")
            hunk = Hunk(old_start=int(match.group(1)))
            current.hunks.append(hunk)
            continue
        if hunk is not None:
            if raw.startswith(("diff --git", "index ", "new file mode", "deleted file mode")):
                continue
            if raw.startswith("\\"):  # "\ No newline at end of file"
                continue
            if raw[:1] in (" ", "+", "-", ""):
                hunk.lines.append(raw if raw else " ")
                continue
            raise PatchError(f"unexpected line inside hunk: {raw!r}")

    if not files:
        raise PatchError("no file headers found; expected unified diff with --- / +++ lines")
    for spec in files:
        if not spec.hunks and not spec.is_delete:
            raise PatchError(f"patch for {spec.path} contains no hunks")
    return files


def apply_hunks(original: list[str], hunks: list[Hunk], path: str) -> list[str]:
    """Apply hunks to a list of lines (no trailing newlines), left to right."""
    lines = list(original)
    cursor = 0
    for index, hunk in enumerate(hunks, start=1):
        before, after = hunk.before, hunk.after
        at = _locate(lines, before, max(hunk.old_start - 1, cursor))
        if at is None:
            raise PatchError(
                f"hunk {index} of {path} does not match the file. Expected to find:\n"
                + "\n".join(before[:8])
                + ("\n..." if len(before) > 8 else "")
            )
        lines[at : at + len(before)] = after
        cursor = at + len(after)
    return lines


def _locate(lines: list[str], before: list[str], hint: int) -> int | None:
    if not before:
        return min(hint, len(lines))
    hint = max(0, min(hint, len(lines)))
    for offset in range(0, SEARCH_WINDOW + 1):
        for candidate in {hint + offset, hint - offset}:
            if candidate < 0 or candidate + len(before) > len(lines):
                continue
            if lines[candidate : candidate + len(before)] == before:
                return candidate
    return None


def _strip_prefix(path: str) -> str:
    path = path.split("\t")[0].strip()
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def apply_patch_tool(workspace: Workspace) -> Tool:
    def run(patch: str) -> ToolOutcome:
        try:
            specs = parse_patch(patch)
        except PatchError as exc:
            return ToolOutcome.error(f"could not parse patch: {exc}")

        staged: dict[str, str | None] = {}  # path -> new content, or None to delete
        for spec in specs:
            target = workspace.resolve(spec.path)
            if spec.is_delete:
                if not target.is_file():
                    return ToolOutcome.error(f"cannot delete {spec.path}: no such file")
                staged[spec.path] = None
                continue
            if spec.is_new:
                if target.is_file():
                    return ToolOutcome.error(f"patch creates {spec.path} but it already exists")
                original: list[str] = []
            else:
                if not target.is_file():
                    return ToolOutcome.error(f"cannot patch {spec.path}: no such file")
                original = target.read_text().splitlines()
            try:
                updated = apply_hunks(original, spec.hunks, spec.path)
            except PatchError as exc:
                return ToolOutcome.error(str(exc))
            staged[spec.path] = "\n".join(updated) + ("\n" if updated else "")

        # Nothing is written until every hunk in the patch has applied cleanly.
        for path, content in staged.items():
            target = workspace.resolve(path)
            if content is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)

        return ToolOutcome(
            ok=True,
            content="applied patch to: " + ", ".join(sorted(staged)),
            meta={"files": sorted(staged)},
        )

    return Tool(
        name="apply_patch",
        description=(
            "Apply a unified diff to the workspace. Use `--- a/path` / `+++ b/path` headers and "
            "`@@` hunks; line numbers may be approximate but context lines must match the file "
            "exactly. Either the whole patch applies or nothing is written."
        ),
        input_schema={
            "type": "object",
            "properties": {"patch": {"type": "string", "description": "The unified diff to apply."}},
            "required": ["patch"],
        },
        fn=run,
        mutating=True,
    )
