"""Graph engineering: explicit topology, guarded cycles, resumable checkpoints."""

from __future__ import annotations

import pytest

from alg.graph import END, Graph, GraphError, JsonlCheckpointer


def build_linear(trace) -> Graph:
    graph = Graph(trace=trace)
    graph.add_node("a", lambda s: {"seen": [*s.get("seen", []), "a"]})
    graph.add_node("b", lambda s: {"seen": [*s.get("seen", []), "b"]})
    graph.set_entry("a").add_edge("a", "b").add_edge("b", END)
    return graph


def test_a_linear_graph_runs_in_order(trace):
    result = build_linear(trace).invoke({})
    assert result.path == ["a", "b"]
    assert result.state["seen"] == ["a", "b"]
    assert result.halted == "end"


def test_nodes_may_return_none_to_leave_state_untouched(trace):
    graph = Graph(trace=trace)
    graph.add_node("noop", lambda s: None)
    graph.set_entry("noop").add_edge("noop", END)
    assert graph.invoke({"x": 1}).state == {"x": 1}


def test_invoke_does_not_mutate_the_caller_s_state(trace):
    initial = {"x": 1}
    graph = Graph(trace=trace)
    graph.add_node("n", lambda s: {"x": 2})
    graph.set_entry("n").add_edge("n", END)
    graph.invoke(initial)
    assert initial == {"x": 1}


def test_conditional_edges_route_on_state(trace):
    graph = Graph(trace=trace)
    graph.add_node("check", lambda s: None)
    graph.add_node("green_path", lambda s: {"took": "green"})
    graph.add_node("red_path", lambda s: {"took": "red"})
    graph.set_entry("check")
    graph.add_conditional_edge(
        "check",
        lambda s: "green" if s["ok"] else "red",
        {"green": "green_path", "red": "red_path"},
    )
    graph.add_edge("green_path", END).add_edge("red_path", END)

    assert graph.invoke({"ok": True}).state["took"] == "green"
    assert graph.invoke({"ok": False}).state["took"] == "red"
    routes = [e.payload for e in trace.of_type("graph.route")]
    assert routes[0]["label"] == "green" and routes[1]["label"] == "red"


def test_a_controlled_cycle_terminates_on_its_own_condition(trace):
    graph = Graph(trace=trace)
    graph.add_node("work", lambda s: {"n": s.get("n", 0) + 1})
    graph.set_entry("work")
    graph.add_conditional_edge(
        "work", lambda s: "done" if s["n"] >= 3 else "again", {"again": "work", "done": END}
    )
    result = graph.invoke({})
    assert result.state["n"] == 3
    assert result.path == ["work", "work", "work"]


def test_max_steps_stops_a_runaway_cycle(trace):
    graph = Graph(trace=trace, max_steps=5)
    graph.add_node("spin", lambda s: {"n": s.get("n", 0) + 1})
    graph.set_entry("spin").add_conditional_edge("spin", lambda s: "again", {"again": "spin"})

    result = graph.invoke({})
    assert result.halted == "max_steps"
    assert result.steps == 5
    assert trace.of_type("graph.halt")[0].payload["reason"] == "max_steps"


def test_validate_catches_structural_mistakes(trace):
    graph = Graph(trace=trace)
    graph.add_node("a", lambda s: None)
    with pytest.raises(GraphError, match="no entry node"):
        graph.validate()

    graph.set_entry("a")
    with pytest.raises(GraphError, match="no outgoing edge"):
        graph.validate()

    graph.add_edge("a", "ghost")
    with pytest.raises(GraphError, match="unknown destination"):
        graph.validate()


def test_a_node_cannot_have_both_a_static_and_a_conditional_edge(trace):
    graph = Graph(trace=trace)
    graph.add_node("a", lambda s: None).add_edge("a", END)
    with pytest.raises(GraphError, match="already has a static edge"):
        graph.add_conditional_edge("a", lambda s: "x", {"x": END})


def test_duplicate_node_names_are_rejected(trace):
    graph = Graph(trace=trace)
    graph.add_node("a", lambda s: None)
    with pytest.raises(GraphError, match="duplicate node"):
        graph.add_node("a", lambda s: None)


def test_a_router_returning_an_unmapped_label_fails_loudly(trace):
    graph = Graph(trace=trace)
    graph.add_node("a", lambda s: None)
    graph.set_entry("a").add_conditional_edge("a", lambda s: "surprise", {"x": END})
    with pytest.raises(GraphError, match="expected one of"):
        graph.invoke({})


def test_checkpoints_record_every_step(tmp_path, trace):
    checkpointer = JsonlCheckpointer(tmp_path / "cp.jsonl")
    build_linear(trace).invoke({}, checkpointer=checkpointer)

    checkpoints = checkpointer.load_all()
    assert [c.node for c in checkpoints] == ["a", "b"]
    assert checkpoints[0].next_node == "b"
    assert checkpoints[-1].next_node == END
    assert checkpoints[-1].state["seen"] == ["a", "b"]


def test_resume_continues_from_the_last_checkpoint(tmp_path, trace):
    checkpointer = JsonlCheckpointer(tmp_path / "cp.jsonl")
    calls: list[str] = []
    crash = {"armed": True}

    def node_b(state):
        if crash["armed"]:
            raise RuntimeError("simulated crash inside node b")
        calls.append("b")
        return {"seen": [*state["seen"], "b"]}

    graph = Graph(trace=trace)
    graph.add_node("a", lambda s: calls.append("a") or {"seen": ["a"]})
    graph.add_node("b", node_b)
    graph.add_node("c", lambda s: calls.append("c") or {"seen": [*s["seen"], "c"]})
    graph.set_entry("a").add_edge("a", "b").add_edge("b", "c").add_edge("c", END)

    with pytest.raises(RuntimeError):
        graph.invoke({}, checkpointer=checkpointer)
    assert calls == ["a"]
    assert checkpointer.latest().next_node == "b"

    # A new process picks the run up where it stopped: node "a" is not re-run.
    crash["armed"] = False
    calls.clear()
    resumed = graph.resume(checkpointer)
    assert calls == ["b", "c"]
    assert resumed.state["seen"] == ["a", "b", "c"]


def test_resume_without_a_checkpoint_is_an_error(tmp_path, trace):
    checkpointer = JsonlCheckpointer(tmp_path / "cp.jsonl")
    with pytest.raises(GraphError, match="no checkpoint"):
        build_linear(trace).resume(checkpointer)


def test_checkpoints_stay_json_clean(tmp_path, trace):
    checkpointer = JsonlCheckpointer(tmp_path / "cp.jsonl")
    graph = Graph(trace=trace)
    graph.add_node("a", lambda s: {"obj": object(), "n": 1})
    graph.set_entry("a").add_edge("a", END)
    graph.invoke({}, checkpointer=checkpointer)

    state = checkpointer.latest().state
    assert state["n"] == 1
    assert state["obj"] == "<object>"


def test_mermaid_renders_nodes_and_labelled_branches(trace):
    graph = Graph(trace=trace)
    graph.add_node("check", lambda s: None)
    graph.add_node("fix", lambda s: None)
    graph.set_entry("check")
    graph.add_conditional_edge("check", lambda s: "red", {"red": "fix", "green": END})
    graph.add_edge("fix", "check")

    diagram = graph.to_mermaid()
    assert "flowchart TD" in diagram
    assert "check -->|red| fix" in diagram
    assert "check -->|green| __end__([end])" in diagram
