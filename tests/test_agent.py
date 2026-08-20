"""End-to-end: the three layers together, driven by a scripted model.

These are the tests that would catch a harness regression on a real run — the
graph reaches the right terminal node, the loop's evidence drives the branch, and
a regression gets reverted instead of compounding.
"""

from __future__ import annotations

from conftest import (
    MEAN_BUG,
    MEAN_FIX,
    MEDIAN_BUG,
    MEDIAN_FIX,
    SOURCE,
    ScriptedLLM,
    call,
    say,
)

from alg.agent import AgentConfig, FixerAgent
from alg.tools.tests_tool import parse_node_tap, parse_pytest_output, run_tests

FIRST_FAILURE = "test/stats.test.ts::mean of integers"


def fix_mean() -> object:
    return call("edit_file", path=SOURCE, old_text=MEAN_BUG, new_text=MEAN_FIX)


def fix_median() -> object:
    return call("edit_file", path=SOURCE, old_text=MEDIAN_BUG, new_text=MEDIAN_FIX)


def agent(workspace, task, llm, trace, **config) -> FixerAgent:
    return FixerAgent(
        workspace=workspace, task=task, llm=llm, trace=trace, config=AgentConfig(**config)
    )


def apply_fixes(workspace) -> None:
    path = workspace.root / SOURCE
    path.write_text(path.read_text().replace(MEAN_BUG, MEAN_FIX).replace(MEDIAN_BUG, MEDIAN_FIX))


# --- test runner ---------------------------------------------------------


def test_run_tests_reports_the_seeded_failures(workspace, task):
    report = run_tests(workspace, task)
    assert not report.green
    assert report.failed == 5
    assert report.passed == 6
    assert FIRST_FAILURE in report.failing


def test_run_tests_reports_green_once_both_defects_are_fixed(workspace, task):
    apply_fixes(workspace)
    report = run_tests(workspace, task)
    assert report.green
    assert report.failed == 0 and report.failing == ()


def test_run_tests_can_focus_a_single_test(workspace, task):
    report = run_tests(workspace, task, target="median of an even-length list")
    assert report.failed == 1
    assert report.failing == ("test/stats.test.ts::median of an even-length list",)


def test_tap_parser_identifies_failures_as_file_and_name():
    report = parse_node_tap(
        "TAP version 13\n"
        "# Subtest: adds\n"
        "not ok 1 - adds\n"
        "  ---\n"
        "  location: '/w/test/a.test.ts:4:1'\n"
        "  ...\n"
        "1..1\n# tests 1\n# pass 0\n# fail 1\n# skipped 0\n",
        exit_code=1,
    )
    assert report.failed == 1
    assert report.failing == ("/w/test/a.test.ts::adds",)
    assert not report.green


def test_tap_parser_falls_back_to_the_bare_name_without_a_location():
    report = parse_node_tap(
        "not ok 1 - first\nnot ok 2 - second\n# tests 2\n# pass 0\n# fail 2\n", exit_code=1
    )
    assert report.failing == ("first", "second")


def test_tap_parser_reads_a_green_run():
    report = parse_node_tap(
        "ok 1 - adds\n1..1\n# tests 1\n# pass 1\n# fail 0\n# skipped 0\n", exit_code=0
    )
    assert report.green and report.passed == 1 and report.failing == ()


def test_the_pytest_parser_is_still_available_for_python_tasks():
    report = parse_pytest_output(
        "ERROR tests/test_x.py - ImportError: boom\n1 error in 0.01s\n", exit_code=2
    )
    assert report.errors == 1
    assert not report.green
    assert report.failing == ("tests/test_x.py",)


def test_report_signature_is_stable_regardless_of_failure_order():
    a = parse_node_tap("not ok 1 - x\nnot ok 2 - y\n# fail 2\n", exit_code=1)
    b = parse_node_tap("not ok 1 - y\nnot ok 2 - x\n# fail 2\n", exit_code=1)
    assert a.signature == b.signature


# --- happy path ----------------------------------------------------------


def test_agent_fixes_the_task_and_reaches_the_report_node(workspace, task, trace):
    llm = ScriptedLLM(
        [
            call("run_tests"),
            call("read_file", path=SOURCE),
            fix_mean(),
            fix_median(),
            call("run_tests"),
            say("mean divided by length+1 and median ignored even-length inputs; fixed both."),
        ]
    )
    state = agent(workspace, task, llm, trace).run()

    assert state["green"] is True
    assert state["attempt"] == 1
    assert "green after 1 attempt" in state["summary"]
    assert run_tests(workspace, task).green


def test_a_task_that_is_already_green_skips_repair_entirely(workspace, task, trace):
    apply_fixes(workspace)
    llm = ScriptedLLM([say("should never be called")])

    state = agent(workspace, task, llm, trace).run()

    assert state["green"] is True
    assert state["attempt"] == 0
    assert llm.calls == []  # the router went straight to report


def test_the_graph_path_is_visible_in_the_trace(workspace, task, trace):
    llm = ScriptedLLM([fix_mean(), fix_median(), say("fixed")])
    agent(workspace, task, llm, trace).run()

    nodes = [e.payload["node"] for e in trace.of_type("graph.node.enter")]
    assert nodes == ["baseline", "repair", "verify", "report"]
    labels = [e.payload["label"] for e in trace.of_type("graph.route")]
    assert labels == ["red", "green"]


def test_the_model_only_sees_the_tools_the_harness_offers(workspace, task, trace):
    llm = ScriptedLLM([say("nothing to do")])
    agent(workspace, task, llm, trace).run()

    _, _, tools = llm.calls[0]
    assert sorted(t.name for t in tools) == [
        "apply_patch",
        "diff",
        "edit_file",
        "list_files",
        "read_file",
        "run_tests",
        "search",
    ]


def test_the_system_prompt_and_tools_describe_the_task_s_language(workspace, task, trace):
    llm = ScriptedLLM([say("done")])
    agent(workspace, task, llm, trace).run()

    system, _, tools = llm.calls[0]
    assert "TypeScript package" in system
    assert "node --test" in system
    run_tests_spec = next(t for t in tools if t.name == "run_tests")
    assert "TypeScript" in run_tests_spec.description
    assert "test-name pattern" in run_tests_spec.description


def test_search_defaults_to_the_task_s_file_type(workspace, task, trace):
    llm = ScriptedLLM([say("done")])
    agent(workspace, task, llm, trace).run()

    _, _, tools = llm.calls[0]
    search = next(t for t in tools if t.name == "search")
    assert "**/*.ts" in search.input_schema["properties"]["glob"]["description"]


def test_write_file_is_off_by_default_and_opt_in(workspace, task, trace):
    llm = ScriptedLLM([say("done")])
    agent(workspace, task, llm, trace, allow_write_file=True).run()
    _, _, tools = llm.calls[0]
    assert "write_file" in [t.name for t in tools]


# --- partial progress, retries, and giving up ----------------------------


def test_partial_progress_triggers_a_retry_and_then_succeeds(workspace, task, trace):
    llm = ScriptedLLM(
        [
            fix_mean(),  # attempt 1: 4 of 5 failures resolved
            say("fixed the mean bug"),
            fix_median(),  # attempt 2: the rest
            say("fixed the median bug too"),
        ]
    )
    state = agent(workspace, task, llm, trace, max_attempts=3).run()

    assert state["green"] is True
    assert state["attempt"] == 2
    assert [r["failed"] for r in state["reports"]] == [5, 1, 0]
    labels = [e.payload["label"] for e in trace.of_type("graph.route")]
    assert labels == ["red", "retry", "green"]


def test_the_retry_prompt_carries_the_current_failure_state(workspace, task, trace):
    llm = ScriptedLLM([fix_mean(), say("partial"), say("out of ideas")])
    agent(workspace, task, llm, trace, max_attempts=2).run()

    retry_goal = llm.calls[-1][1][0].text()
    assert "Attempt 2 of 2" in retry_goal
    assert "1 failed" in retry_goal
    assert "improved the suite" in retry_goal


def test_the_agent_gives_up_after_max_attempts(workspace, task, trace):
    llm = ScriptedLLM(default=say("I have no idea."))
    state = agent(workspace, task, llm, trace, max_attempts=2).run()

    assert state["green"] is False
    assert state["attempt"] == 2
    assert "not fixed after 2 attempt" in state["summary"]
    labels = [e.payload["label"] for e in trace.of_type("graph.route")]
    assert labels == ["red", "retry", "exhausted"]


def test_a_regression_is_reverted_rather_than_compounded(workspace, task, trace):
    break_clamp = call(
        "edit_file",
        path=SOURCE,
        old_text="  return Math.max(low, Math.min(high, value));",
        new_text="  return NaN;",
    )
    llm = ScriptedLLM([break_clamp, say("tried something")])
    state = agent(workspace, task, llm, trace, max_attempts=1).run()

    assert state["reverted"] is True
    assert "Math.max(low, Math.min(high, value))" in (workspace.root / SOURCE).read_text()
    revert = trace.of_type("agent.revert")
    assert revert and revert[0].payload["files"] == [SOURCE]
    assert "reverted" in state["assessment"]


def test_the_final_diff_shows_only_the_agent_s_changes(workspace, task, trace):
    llm = ScriptedLLM([fix_mean(), fix_median(), say("fixed")])
    state = agent(workspace, task, llm, trace).run()

    diff = state["diff"]
    assert f"--- a/{SOURCE}" in diff
    assert MEAN_FIX in diff
    assert "test/stats.test.ts" not in diff  # the tests were left alone


def test_a_run_is_checkpointed_at_every_node(workspace, task, trace, tmp_path):
    llm = ScriptedLLM([fix_mean(), fix_median(), say("fixed")])
    path = tmp_path / "checkpoints.jsonl"
    agent(workspace, task, llm, trace).run(checkpoint_path=path)

    # Read the file directly: constructing a JsonlCheckpointer truncates it.
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) == 4
    assert '"node": "baseline"' in lines[0]
    assert '"node": "report"' in lines[-1]


def test_loop_stop_reasons_are_recorded_per_attempt(workspace, task, trace):
    llm = ScriptedLLM(default=call("read_file", path=SOURCE))
    state = agent(workspace, task, llm, trace, max_attempts=2, loop_max_iterations=3).run()

    assert state["loop_stop_reasons"] == ["max_iterations(3)", "max_iterations(3)"]
    assert state["green"] is False


def test_token_usage_accumulates_across_attempts(workspace, task, trace):
    llm = ScriptedLLM(default=say("no idea"))
    state = agent(workspace, task, llm, trace, max_attempts=2).run()
    assert state["tokens"]["input"] == 20  # two attempts x one 10-token call
    assert state["tokens"]["output"] == 10


# --- what a real local-model run exposed ----------------------------------


def test_editing_without_re_running_tests_is_not_a_stall(workspace, task, trace):
    """Observed on a live qwen run: the model ran the tests once, then spent
    four iterations reading and editing. Evidence repeated the stale reading,
    `stalled_evidence(4)` fired, and a productive attempt was cut off at
    10 passed / 1 failed. Progress is measured by test runs, not by iterations.
    """
    llm = ScriptedLLM(
        [
            call("run_tests"),                       # one real reading: 6p/5f
            call("read_file", path=SOURCE),          # then four iterations
            call("search", pattern="function mean"),  # of work with no re-run
            fix_mean(),
            fix_median(),
            say("fixed both defects"),
        ]
    )
    state = agent(workspace, task, llm, trace, max_attempts=1).run()

    assert state["loop_stop_reasons"] == ["model_finished"]
    assert state["green"] is True, "the attempt should not have been cut short"


def test_repeated_identical_test_runs_do_still_stall(workspace, task, trace):
    """The rule must keep its teeth: four real runs with the same failures is
    exactly the spinning it exists to catch."""
    llm = ScriptedLLM(default=call("run_tests"))
    state = agent(workspace, task, llm, trace, max_attempts=1, loop_max_iterations=20).run()

    assert len(state["loop_stop_reasons"]) == 1
    reason = state["loop_stop_reasons"][0]
    assert reason.startswith("stalled_evidence(4)")
    assert "6p/5f" in reason  # the reason names the reading it stalled on
    assert state["green"] is False


def test_evidence_is_recorded_once_per_test_run(workspace, task, trace):
    llm = ScriptedLLM(
        [call("run_tests"), call("read_file", path=SOURCE), call("run_tests"), say("done")]
    )
    agent(workspace, task, llm, trace, max_attempts=1).run()

    # Two run_tests calls -> two pieces of evidence, not one per iteration.
    assert len(trace.of_type("loop.evidence")) == 2
