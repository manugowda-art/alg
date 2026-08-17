"""Patch application: forgiving about line numbers, strict about context, atomic."""

from __future__ import annotations

import pytest

from alg.tools import ToolRegistry
from alg.tools.patch import PatchError, apply_hunks, apply_patch_tool, parse_patch

SIMPLE = """\
--- a/calc/stats.py
+++ b/calc/stats.py
@@ -18,7 +18,7 @@
     total = 0.0
     for value in values:
         total += value
-    return total / (len(values) + 1)
+    return total / len(values)
"""


@pytest.fixture
def registry(workspace, trace) -> ToolRegistry:
    return ToolRegistry([apply_patch_tool(workspace)], trace=trace)


def test_parse_extracts_paths_and_hunks():
    specs = parse_patch(SIMPLE)
    assert len(specs) == 1
    assert specs[0].path == "calc/stats.py"
    assert len(specs[0].hunks) == 1
    assert specs[0].hunks[0].before[-1] == "    return total / (len(values) + 1)"
    assert specs[0].hunks[0].after[-1] == "    return total / len(values)"


def test_parse_rejects_garbage():
    with pytest.raises(PatchError):
        parse_patch("not a patch at all")


def test_parse_rejects_a_hunk_without_a_file_header():
    with pytest.raises(PatchError):
        parse_patch("@@ -1,1 +1,1 @@\n-a\n+b\n")


def test_apply_patch_edits_the_file(registry, workspace):
    outcome = registry.call("apply_patch", {"patch": SIMPLE})
    assert outcome.ok, outcome.content
    assert "return total / len(values)" in (workspace.root / "calc/stats.py").read_text()


def test_apply_patch_tolerates_wrong_line_numbers(registry, workspace):
    shifted = SIMPLE.replace("@@ -18,7 +18,7 @@", "@@ -140,7 +140,7 @@")
    outcome = registry.call("apply_patch", {"patch": shifted})
    assert outcome.ok, outcome.content
    assert "return total / len(values)" in (workspace.root / "calc/stats.py").read_text()


def test_apply_patch_refuses_when_context_does_not_match(registry, workspace):
    before = (workspace.root / "calc/stats.py").read_text()
    bad = SIMPLE.replace("     total = 0.0", "     total = 1.0")
    outcome = registry.call("apply_patch", {"patch": bad})
    assert not outcome.ok
    assert "does not match" in outcome.content
    assert (workspace.root / "calc/stats.py").read_text() == before


def test_apply_patch_is_atomic_across_files(registry, workspace):
    before = (workspace.root / "calc/stats.py").read_text()
    two_files = SIMPLE + """\
--- a/calc/missing.py
+++ b/calc/missing.py
@@ -1,1 +1,1 @@
-old
+new
"""
    outcome = registry.call("apply_patch", {"patch": two_files})
    assert not outcome.ok and "no such file" in outcome.content
    assert (workspace.root / "calc/stats.py").read_text() == before


def test_apply_patch_creates_a_new_file(registry, workspace):
    patch = """\
--- /dev/null
+++ b/calc/extra.py
@@ -0,0 +1,2 @@
+def extra():
+    return 1
"""
    outcome = registry.call("apply_patch", {"patch": patch})
    assert outcome.ok, outcome.content
    assert (workspace.root / "calc/extra.py").read_text() == "def extra():\n    return 1\n"


def test_apply_patch_refuses_to_overwrite_via_dev_null(registry):
    patch = """\
--- /dev/null
+++ b/calc/stats.py
@@ -0,0 +1,1 @@
+x = 1
"""
    outcome = registry.call("apply_patch", {"patch": patch})
    assert not outcome.ok and "already exists" in outcome.content


def test_apply_patch_deletes_a_file(registry, workspace):
    patch = """\
--- a/conftest.py
+++ /dev/null
@@ -1,2 +0,0 @@
-# Present so pytest puts this directory on sys.path and `import calc` works
-# without installing the package.
"""
    outcome = registry.call("apply_patch", {"patch": patch})
    assert outcome.ok, outcome.content
    assert not (workspace.root / "conftest.py").exists()


def test_apply_patch_is_jailed(registry):
    patch = SIMPLE.replace("a/calc/stats.py", "a/../evil.py").replace(
        "b/calc/stats.py", "b/../evil.py"
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
