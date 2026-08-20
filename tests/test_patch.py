"""Patch application: forgiving about line numbers, strict about context, atomic."""

from __future__ import annotations

import pytest

from alg.tools import ToolRegistry
from alg.tools.patch import PatchError, apply_hunks, apply_patch_tool, parse_patch

SIMPLE = """\
--- a/src/stats.ts
+++ b/src/stats.ts
@@ -12,7 +12,7 @@
   let total = 0;
   for (const value of values) {
     total += value;
   }
-  return total / (values.length + 1);
+  return total / values.length;
 }
"""


@pytest.fixture
def registry(workspace, trace) -> ToolRegistry:
    return ToolRegistry([apply_patch_tool(workspace)], trace=trace)


def test_parse_extracts_paths_and_hunks():
    specs = parse_patch(SIMPLE)
    assert len(specs) == 1
    assert specs[0].path == "src/stats.ts"
    assert len(specs[0].hunks) == 1
    assert specs[0].hunks[0].before[-2] == "  return total / (values.length + 1);"
    assert specs[0].hunks[0].after[-2] == "  return total / values.length;"


def test_parse_rejects_garbage():
    with pytest.raises(PatchError):
        parse_patch("not a patch at all")


def test_parse_rejects_a_hunk_without_a_file_header():
    with pytest.raises(PatchError):
        parse_patch("@@ -1,1 +1,1 @@\n-a\n+b\n")


def test_apply_patch_edits_the_file(registry, workspace):
    outcome = registry.call("apply_patch", {"patch": SIMPLE})
    assert outcome.ok, outcome.content
    assert "return total / values.length;" in (workspace.root / "src/stats.ts").read_text()


def test_apply_patch_tolerates_wrong_line_numbers(registry, workspace):
    shifted = SIMPLE.replace("@@ -12,7 +12,7 @@", "@@ -140,7 +140,7 @@")
    outcome = registry.call("apply_patch", {"patch": shifted})
    assert outcome.ok, outcome.content
    assert "return total / values.length;" in (workspace.root / "src/stats.ts").read_text()


def test_apply_patch_refuses_when_context_does_not_match(registry, workspace):
    before = (workspace.root / "src/stats.ts").read_text()
    bad = SIMPLE.replace("   let total = 0;", "   let total = 99;")
    outcome = registry.call("apply_patch", {"patch": bad})
    assert not outcome.ok
    assert "does not match" in outcome.content
    assert (workspace.root / "src/stats.ts").read_text() == before


def test_apply_patch_is_atomic_across_files(registry, workspace):
    before = (workspace.root / "src/stats.ts").read_text()
    two_files = SIMPLE + """\
--- a/src/missing.ts
+++ b/src/missing.ts
@@ -1,1 +1,1 @@
-old
+new
"""
    outcome = registry.call("apply_patch", {"patch": two_files})
    assert not outcome.ok and "no such file" in outcome.content
    assert (workspace.root / "src/stats.ts").read_text() == before


def test_apply_patch_creates_a_new_file(registry, workspace):
    patch = """\
--- /dev/null
+++ b/src/extra.ts
@@ -0,0 +1,3 @@
+export function extra(): number {
+  return 1;
+}
"""
    outcome = registry.call("apply_patch", {"patch": patch})
    assert outcome.ok, outcome.content
    assert (workspace.root / "src/extra.ts").read_text() == (
        "export function extra(): number {\n  return 1;\n}\n"
    )


def test_apply_patch_refuses_to_overwrite_via_dev_null(registry):
    patch = """\
--- /dev/null
+++ b/src/stats.ts
@@ -0,0 +1,1 @@
+const x = 1;
"""
    outcome = registry.call("apply_patch", {"patch": patch})
    assert not outcome.ok and "already exists" in outcome.content


def test_apply_patch_deletes_a_file(registry, workspace):
    (workspace.root / "src" / "scratch.ts").write_text("const doomed = 1;\n")
    patch = """\
--- a/src/scratch.ts
+++ /dev/null
@@ -1,1 +0,0 @@
-const doomed = 1;
"""
    outcome = registry.call("apply_patch", {"patch": patch})
    assert outcome.ok, outcome.content
    assert not (workspace.root / "src/scratch.ts").exists()


def test_apply_patch_is_jailed(registry):
    patch = SIMPLE.replace("a/src/stats.ts", "a/../evil.ts").replace(
        "b/src/stats.ts", "b/../evil.ts"
    )
    outcome = registry.call("apply_patch", {"patch": patch})
    assert not outcome.ok and "refused" in outcome.content


def test_multiple_hunks_apply_left_to_right():
    original = [f"line {i}" for i in range(1, 11)]
    hunks = parse_patch(
        """\
--- a/f.txt
+++ b/f.txt
@@ -1,2 +1,2 @@
-line 1
+LINE ONE
 line 2
@@ -8,2 +8,2 @@
 line 8
-line 9
+LINE NINE
"""
    )[0].hunks
    result = apply_hunks(original, hunks, "f.txt")
    assert result[0] == "LINE ONE"
    assert result[8] == "LINE NINE"
    assert len(result) == len(original)


def test_hunk_added_and_removed_line_counts_can_differ():
    original = ["a", "b", "c"]
    hunks = parse_patch(
        """\
--- a/f.txt
+++ b/f.txt
@@ -1,3 +1,4 @@
 a
-b
+b1
+b2
 c
"""
    )[0].hunks
    assert apply_hunks(original, hunks, "f.txt") == ["a", "b1", "b2", "c"]
