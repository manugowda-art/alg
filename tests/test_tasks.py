"""The task manifest: what keeps the engine language-agnostic."""

from __future__ import annotations

import json

import pytest

from alg.tasks import TaskSpec, TaskError
from alg.tools.tests_tool import run_tests


def write_manifest(tmp_path, **overrides):
    manifest = {
        "name": "demo",
        "language": "TypeScript",
        "runner": "node-test",
        "test_command": ["node", "--test"],
    }
    manifest.update(overrides)
    (tmp_path / "alg.task.json").write_text(json.dumps(manifest))
    return tmp_path


def test_the_bundled_task_declares_a_typescript_node_runner(task):
    assert task.name == "calc_bug"
    assert task.language == "TypeScript"
    assert task.runner == "node-test"
    assert task.test_command == ("node", "--test", "--test-reporter=tap")
    assert task.search_glob == "**/*.ts"


def test_argv_appends_the_focus_template(tmp_path):
    spec = TaskSpec.load(
        write_manifest(tmp_path, focus_template=["--test-name-pattern", "{target}"])
    )
    assert spec.argv() == ["node", "--test"]
    assert spec.argv("median") == ["node", "--test", "--test-name-pattern", "median"]


def test_argv_appends_a_positional_target_when_there_is_no_template(tmp_path):
    spec = TaskSpec.load(write_manifest(tmp_path, runner="pytest", test_command=["pytest", "-q"]))
    assert spec.argv("tests/test_x.py::test_y") == ["pytest", "-q", "tests/test_x.py::test_y"]


@pytest.mark.parametrize(
    "broken,fragment",
    [
        ({"runner": None}, "missing required"),
        ({"test_command": []}, "non-empty array"),
        ({"surprise": 1}, "unknown key"),
    ],
)
def test_a_malformed_manifest_is_rejected_with_a_reason(tmp_path, broken, fragment):
    manifest = {
        "name": "demo",
        "runner": "node-test",
        "test_command": ["node", "--test"],
    }
    manifest.update(broken)
    manifest = {k: v for k, v in manifest.items() if v is not None}
    (tmp_path / "alg.task.json").write_text(json.dumps(manifest))
    with pytest.raises(TaskError, match=fragment):
        TaskSpec.load(tmp_path)


def test_a_missing_manifest_is_rejected(tmp_path):
    with pytest.raises(TaskError, match="no alg.task.json"):
        TaskSpec.load(tmp_path)


def test_invalid_json_is_rejected(tmp_path):
    (tmp_path / "alg.task.json").write_text("{not json")
    with pytest.raises(TaskError, match="not valid JSON"):
        TaskSpec.load(tmp_path)


def test_an_unknown_runner_fails_loudly_rather_than_reporting_a_false_green(workspace, tmp_path):
    spec = TaskSpec.load(write_manifest(tmp_path, runner="mocha"))
    with pytest.raises(ValueError, match="unknown runner"):
        run_tests(workspace, spec)
