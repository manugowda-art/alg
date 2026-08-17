"""End-to-end: the three layers together, driven by a scripted model.

These are the tests that would catch a harness regression on a real run — the
graph reaches the right terminal node, the loop's evidence drives the branch, and
a regression gets reverted instead of compounding.
"""

from __future__ import annotations

from conftest import MEAN_BUG, MEAN_FIX, MEDIAN_BUG, MEDIAN_FIX, ScriptedLLM, call, say

from alg.agent import AgentConfig, FixerAgent
from alg.graph import JsonlCheckpointer
from alg.tools.tests_tool import parse_pytest_output, run_tests


def fix_mean() -> object:
    return call("edit_file", path="calc/stats.py", old_text=MEAN_BUG, new_text=MEAN_FIX)


def fix_median() -> object:
    return call("edit_file", path="calc/stats.py", old_text=MEDIAN_BUG, new_text=MEDIAN_FIX)


def agent(workspace, llm, trace, **config) -> FixerAgent:
    return FixerAgent(
        workspace=workspace, llm=llm, trace=trace, config=AgentConfig(**config)
    )


# --- test runner ---------------------------------------------------------


def test_run_tests_reports_the_seeded_failures(workspace):
    report = run_tests(workspace)
    assert not report.green
    assert report.failed == 5
    assert report.passed == 6
    assert "tests/test_stats.py::test_mean_of_integers" in report.failing


def test_run_tests_reports_green_once_both_defects_are_fixed(workspace):
    path = workspace.root / "calc" / "stats.py"
    path.write_text(path.read_text().replace(MEAN_BUG, MEAN_FIX).replace(MEDIAN_BUG, MEDIAN_FIX))
    report = run_tests(workspace)
    assert report.green
    assert report.failed == 0 and report.failing == ()


def test_parse_pytest_output_handles_a_collection_error():
    report = parse_pytest_output(
        "ERROR tests/test_x.py - ImportError: boom\n1 error in 0.01s\n", exit_code=2
    )
    assert report.errors == 1
    assert not report.green
    assert report.failing == ("tests/test_x.py",)


def test_report_signature_is_stable_regardless_of_failure_order():
    a = parse_pytest_output("FAILED t.py::x\nFAILED t.py::y\n1 failed in 0s\n", exit_code=1)
    b = parse_pytest_output("FAILED t.py::y\nFAILED t.py::x\n1 failed in 0s\n", exit_code=1)
    assert a.signature == b.signature


# --- happy path ----------------------------------------------------------


def test_agent_fixes_the_task_and_reaches_the_report_node(workspace, trace):
    llm = ScriptedLLM(
        [
            call("run_tests"),
            call("read_file", path="calc/stats.py"),
            fix_mean(),
            fix_median(),
            call("run_tests"),
            say("mean divided by len+1 and median ignored even-length inputs; fixed both."),
        ]
    )
    state = agent(workspace, llm, trace).run()

    assert state["green"] is True
    assert state["attempt"] == 1
    assert "green after 1 attempt" in state["summary"]
    assert run_tests(workspace).green


def test_a_task_that_is_already_green_skips_repair_entirely(workspace, trace):
    path = workspace.root / "calc" / "stats.py"
    path.write_text(path.read_text().replace(MEAN_BUG, MEAN_FIX).replace(MEDIAN_BUG, MEDIAN_FIX))
    llm = ScriptedLLM([say("should never be called")])

    state = agent(workspace, llm, trace).run()

    assert state["green"] is True
    assert state["attempt"] == 0
    assert llm.calls == []  # the router went straight to report


def test_the_graph_path_is_visible_in_the_trace(workspace, trace):
    llm = ScriptedLLM([fix_mean(), fix_median(), say("fixed")])
    agent(workspace, llm, trace).run()

    nodes = [e.payload["node"] for e in trace.of_type("graph.node.enter")]
    assert nodes == ["baseline", "repair", "verify", "report"]
    labels = [e.payload["label"] for e in trace.of_type("graph.route")]
    assert labels == ["red", "green"]


def test_the_model_only_sees_the_tools_the_harness_offers(workspace, trace):
    llm = ScriptedLLM([say("nothing to do")])
    agent(workspace, llm, trace).run()

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


def test_write_file_is_off_by_default_and_opt_in(workspace, trace):
    llm = ScriptedLLM([say("done")])
    agent(workspace, llm, trace, allow_write_file=True).run()
    _, _, tools = llm.calls[0]
    assert "write_file" in [t.name for t in tools]


# --- partial progress, retries, and giving up ----------------------------


def test_partial_progress_triggers_a_retry_and_then_succeeds(workspace, trace):
    llm = ScriptedLLM(
        [
            fix_mean(),  # attempt 1: 3 of 5 failures resolved
            say("fixed the mean bug"),
            fix_median(),  # attempt 2: the rest
            say("fixed the median bug too"),
        ]
    )
    state = agent(workspace, llm, trace, max_attempts=3).run()

    assert state["green"] is True
    assert state["attempt"] == 2
    reports = state["reports"]
    assert [r["failed"] for r in reports] == [5, 1, 0]
    labels = [e.payload["label"] for e in trace.of_type("graph.route")]
    assert labels == ["red", "retry", "green"]


def test_the_retry_prompt_carries_the_current_failure_state(workspace, trace):
    llm = ScriptedLLM([fix_mean(), say("partial"), say("out of ideas")])
    agent(workspace, llm, trace, max_attempts=2).run()

    retry_goal = llm.calls[-1][1][0].text()
    assert "Attempt 2 of 2" in retry_goal
    assert "1 failed" in retry_goal
    assert "improved the suite" in retry_goal


def test_the_agent_gives_up_after_max_attempts(workspace, trace):
    llm = ScriptedLLM(default=say("I have no idea."))
    state = agent(workspace, llm, trace, max_attempts=2).run()

    assert state["green"] is False
    assert state["attempt"] == 2
    assert "not fixed after 2 attempt" in state["summary"]
    labels = [e.payload["label"] for e in trace.of_type("graph.route")]
    assert labels == ["red", "retry", "exhausted"]


def test_a_regression_is_reverted_rather_than_compounded(workspace, trace):
    break_clamp = call(
        "edit_file",
        path="calc/stats.py",
        old_text="    return max(low, min(high, value))",
        new_text="    return None",
    )
    llm = ScriptedLLM([break_clamp, say("tried something")])
    state = agent(workspace, llm, trace, max_attempts=1).run()

    assert state["reverted"] is True
    assert "return max(low, min(high, value))" in (workspace.root / "calc/stats.py").read_text()
    revert = trace.of_type("agent.revert")
    assert revert and revert[0].payload["files"] == ["calc/stats.py"]
    assert "reverted" in state["assessment"]


def test_the_final_diff_shows_only_the_agent_s_changes(workspace, trace):
    llm = ScriptedLLM([fix_mean(), fix_median(), say("fixed")])
    state = agent(workspace, llm, trace).run()

    diff = state["diff"]
    assert "--- a/calc/stats.py" in diff
    assert MEAN_FIX in diff
    assert "tests/test_stats.py" not in diff  # the tests were left alone


def test_a_run_is_checkpointed_at_every_node(workspace, trace, tmp_path):
    llm = ScriptedLLM([fix_mean(), fix_median(), say("fixed")])
    path = tmp_path / "checkpoints.jsonl"
    agent(workspace, llm, trace).run(checkpoint_path=path)

    # Read the file directly: constructing a JsonlCheckpointer truncates it.
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) == 4
    assert '"node": "baseline"' in lines[0]
    assert '"node": "report"' in lines[-1]


def test_loop_stop_reasons_are_recorded_per_attempt(workspace, trace):
    llm = ScriptedLLM(default=call("read_file", path="calc/stats.py"))
    state = agent(workspace, llm, trace, max_attempts=2, loop_max_iterations=3).run()

    assert state["loop_stop_reasons"] == ["max_iterations(3)", "max_iterations(3)"]
    assert state["green"] is False


def test_token_usage_accumulates_across_attempts(workspace, trace):
    llm = ScriptedLLM(default=say("no idea"))
    state = agent(workspace, llm, trace, max_attempts=2).run()
    assert state["tokens"]["input"] == 20  # two attempts x one 10-token call
    assert state["tokens"]["output"] == 10
