"""Harness layer: workspace jail, bounded execution, tool registry, trace."""

from __future__ import annotations

import sys

import pytest

from alg.events import Trace, read_trace
from alg.tools import Tool, ToolOutcome, ToolRegistry, validate_args
from alg.tools.fs import (
    changed_files,
    edit_file_tool,
    read_only_tools,
    render_diff,
    restore,
    snapshot,
    write_file_tool,
)
from alg.workspace import Workspace, WorkspaceError


# --- workspace -----------------------------------------------------------


def test_materialize_copies_task_and_leaves_source_alone(workspace, task_dir):
    assert (workspace.root / "src" / "stats.ts").is_file()
    (workspace.root / "src" / "stats.ts").write_text("// clobbered\n")
    assert "export function mean" in (task_dir / "src" / "stats.ts").read_text()


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../escape.ts", "src/../../escape.ts", "src/../../../tmp/x"],
)
def test_resolve_refuses_paths_outside_the_workspace(workspace, path):
    with pytest.raises(WorkspaceError):
        workspace.resolve(path)


def test_resolve_allows_paths_inside_the_workspace(workspace):
    assert workspace.resolve("src/stats.ts").is_file()
    assert workspace.resolve("does/not/exist.ts").parent.name == "not"


def test_symlink_pointing_outside_is_refused(workspace, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    (workspace.root / "link.txt").symlink_to(outside)
    with pytest.raises(WorkspaceError):
        workspace.resolve("link.txt")


def test_list_files_skips_caches_and_hidden_files(workspace):
    (workspace.root / "__pycache__").mkdir()
    (workspace.root / "__pycache__" / "x.pyc").write_text("")
    (workspace.root / ".hidden").write_text("")
    files = workspace.list_files()
    assert "src/stats.ts" in files
    assert not any("__pycache__" in f or f.startswith(".") for f in files)


def test_run_captures_exit_code_and_output(workspace):
    result = workspace.run([sys.executable, "-c", "import sys; print('hi'); sys.exit(3)"])
    assert result.exit_code == 3
    assert "hi" in result.stdout
    assert not result.ok


def test_run_enforces_a_timeout(workspace):
    result = workspace.run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.3)
    assert result.timed_out
    assert not result.ok


def test_run_reports_a_missing_binary_instead_of_raising(workspace):
    result = workspace.run(["definitely-not-a-real-binary-xyz"])
    assert not result.ok


# --- trace ---------------------------------------------------------------


def test_trace_is_ordered_and_round_trips_through_jsonl(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = Trace(path, clock=lambda: 1.0)
    trace.emit("a", x=1)
    trace.emit("b", y="two")
    assert [e.seq for e in trace] == [0, 1]
    assert trace.types() == ["a", "b"]
    reloaded = read_trace(path)
    assert [e.type for e in reloaded] == ["a", "b"]
    assert reloaded[1].payload == {"y": "two"}


# --- registry ------------------------------------------------------------


SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["path"],
}


@pytest.mark.parametrize(
    "args,expected_fragment",
    [
        ({}, "missing required"),
        ({"path": "a", "bogus": 1}, "unknown argument"),
        ({"path": 7}, "must be a string"),
        ({"path": "a", "count": "3"}, "must be a integer"),
    ],
)
def test_validate_args_rejects_bad_calls(args, expected_fragment):
    problem = validate_args(SCHEMA, args)
    assert problem and expected_fragment in problem


def test_validate_args_accepts_good_calls():
    assert validate_args(SCHEMA, {"path": "a", "count": 3}) is None


def test_registry_returns_errors_instead_of_raising(trace):
    def boom(path: str) -> ToolOutcome:
        raise RuntimeError("kaboom")

    registry = ToolRegistry(
        [Tool(name="boom", description="", input_schema=SCHEMA, fn=boom)], trace=trace
    )
    outcome = registry.call("boom", {"path": "x"})
    assert not outcome.ok
    assert "kaboom" in outcome.content
    assert trace.of_type("tool.call")[0].payload["ok"] is False


def test_registry_rejects_unknown_tool_and_names_the_alternatives(trace):
    registry = ToolRegistry([], trace=trace)
    outcome = registry.call("nope", {})
    assert not outcome.ok and "unknown tool" in outcome.content


def test_registry_validates_before_running_the_tool(trace):
    seen: list[dict] = []

    def record(path: str) -> ToolOutcome:
        seen.append({"path": path})
        return ToolOutcome(ok=True, content="ok")

    registry = ToolRegistry(
        [Tool(name="record", description="", input_schema=SCHEMA, fn=record)], trace=trace
    )
    assert not registry.call("record", {}).ok
    assert seen == []


def test_subset_narrows_the_visible_tools(workspace, trace):
    registry = ToolRegistry(read_only_tools(workspace), trace=trace)
    narrowed = registry.subset(["read_file"])
    assert narrowed.names == ["read_file"]
    assert [s.name for s in narrowed.specs()] == ["read_file"]


# --- fs tools ------------------------------------------------------------


def test_read_file_numbers_lines_and_reports_missing_files(workspace, trace):
    registry = ToolRegistry(read_only_tools(workspace), trace=trace)
    ok = registry.call("read_file", {"path": "src/stats.ts"})
    assert ok.ok and "   1 | " in ok.content
    missing = registry.call("read_file", {"path": "src/nope.ts"})
    assert not missing.ok and "no such file" in missing.content


def test_search_finds_matches_and_reports_bad_regex(workspace, trace):
    registry = ToolRegistry(read_only_tools(workspace), trace=trace)
    hits = registry.call("search", {"pattern": r"function mean"})
    assert hits.ok and "src/stats.ts" in hits.content
    bad = registry.call("search", {"pattern": "("})
    assert not bad.ok and "invalid regular expression" in bad.content


def test_edit_file_requires_a_unique_match(workspace, trace):
    registry = ToolRegistry([edit_file_tool(workspace)], trace=trace)
    duplicated = registry.call(
        "edit_file",
        {"path": "src/stats.ts", "old_text": "return ordered[middle]!;", "new_text": "x"},
    )
    assert not duplicated.ok and "appears 2 times" in duplicated.content
    assert "return ordered[middle]!;" in (workspace.root / "src/stats.ts").read_text()


def test_edit_file_rejects_stale_text(workspace, trace):
    registry = ToolRegistry([edit_file_tool(workspace)], trace=trace)
    outcome = registry.call(
        "edit_file",
        {"path": "src/stats.ts", "old_text": "this text is not in the file", "new_text": "x"},
    )
    assert not outcome.ok and "not found" in outcome.content


def test_edit_file_applies_a_unique_change(workspace, trace):
    registry = ToolRegistry([edit_file_tool(workspace)], trace=trace)
    outcome = registry.call(
        "edit_file",
        {
            "path": "src/stats.ts",
            "old_text": "return total / (values.length + 1);",
            "new_text": "return total / values.length;",
        },
    )
    assert outcome.ok
    assert "return total / values.length;" in (workspace.root / "src/stats.ts").read_text()


def test_write_file_is_jailed(workspace, trace):
    registry = ToolRegistry([write_file_tool(workspace)], trace=trace)
    outcome = registry.call("write_file", {"path": "../evil.ts", "content": "boom"})
    assert not outcome.ok and "refused" in outcome.content


def test_snapshot_diff_and_restore_round_trip(workspace):
    baseline = snapshot(workspace)
    target = workspace.root / "src" / "stats.ts"
    target.write_text(target.read_text().replace("(values.length + 1)", "values.length"))
    (workspace.root / "new_file.ts").write_text("// added\n")

    assert changed_files(workspace, baseline) == ["new_file.ts", "src/stats.ts"]
    diff = render_diff(workspace, baseline)
    assert "--- a/src/stats.ts" in diff and "+++ b/new_file.ts" in diff

    reverted = restore(workspace, baseline)
    assert reverted == ["new_file.ts", "src/stats.ts"]
    assert changed_files(workspace, baseline) == []
    assert not (workspace.root / "new_file.ts").exists()
