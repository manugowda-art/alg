"""The task manifest: what keeps the engine language-agnostic."""

from __future__ import annotations

import json

import pytest

from alg.tasks import TaskError, TaskSpec
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


# --- a verifier that does not verify --------------------------------------


def broken_task(tmp_path, command):
    """A task whose test command exits 0 without running anything."""
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "thing.ts").write_text("export const x = 1;\n")
    (tmp_path / "alg.task.json").write_text(
        json.dumps(
            {
                "name": "broken",
                "language": "TypeScript",
                "runner": "node-test",
                "test_command": command,
            }
        )
    )
    return tmp_path


def test_a_command_that_collects_nothing_is_not_green(tmp_path):
    """The failure that Node 22.17 produces: no .ts files matched, exit 0."""
    from alg.workspace import Workspace

    task_dir = broken_task(tmp_path / "task", ["node", "-e", ""])
    workspace = Workspace.materialize(task_dir, tmp_path / "work")
    report = run_tests(workspace, TaskSpec.load(task_dir))

    assert report.exit_code == 0
    assert report.collected == 0
    assert report.green is False, "a suite that ran nothing must never report green"
    assert "NO TESTS RAN" in report.summary()


def test_the_agent_refuses_to_repair_against_a_broken_verifier(tmp_path, trace):
    from alg.agent import AgentConfig, FixerAgent
    from alg.workspace import Workspace
    from conftest import ScriptedLLM, say

    task_dir = broken_task(tmp_path / "task", ["node", "-e", ""])
    workspace = Workspace.materialize(task_dir, tmp_path / "work")
    llm = ScriptedLLM([say("should never be asked")])

    state = FixerAgent(
        workspace=workspace, task=TaskSpec.load(task_dir), llm=llm, trace=trace,
        config=AgentConfig(),
    ).run()

    assert llm.calls == [], "no model budget should be spent on an unfixable task"
    assert state["green"] is False
    assert "verifier is broken" in state["summary"]
    labels = [e.payload["label"] for e in trace.of_type("graph.route")]
    assert labels == ["broken"]


def test_the_doctor_calls_a_zero_collect_baseline_a_failure(tmp_path):
    from alg.doctor import FAIL, check_baseline

    task_dir = broken_task(tmp_path / "task", ["node", "-e", ""])
    check = check_baseline(task_dir, TaskSpec.load(task_dir), tmp_path / "work")

    assert check.status == FAIL
    assert "no tests ran" in check.detail
    assert "stripping types" in check.fix


def test_a_missing_directory_and_a_missing_manifest_read_differently(tmp_path):
    with pytest.raises(TaskError, match="no such task directory"):
        TaskSpec.load(tmp_path / "nope")

    (tmp_path / "empty").mkdir()
    with pytest.raises(TaskError, match="has no alg.task.json"):
        TaskSpec.load(tmp_path / "empty")
