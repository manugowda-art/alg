"""Graph engineering: make the topology explicit.

A node is a pure-ish function `state -> update dict`. An edge is either static
(always go to X) or a router (a function that reads state and names the next
node). The executor walks that structure, checkpoints the state after every
node, and refuses to spin forever.

Why bother, when the loop above can already call tools in a cycle? Because
"diagnose, then verify, then decide whether to repair or stop" is a *decision
structure*, and burying it inside a prompt makes it unobservable and untestable.
Pulled out here, every branch is a named edge you can assert on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .events import Trace

START = "__start__"
END = "__end__"

State = dict[str, Any]
Node = Callable[[State], State | None]
Router = Callable[[State], str]


class GraphError(Exception):
    pass


@dataclass
class Checkpoint:
    step: int
    node: str
    next_node: str
    state: State

    def to_json(self) -> str:
        return json.dumps(
            {"step": self.step, "node": self.node, "next": self.next_node, "state": _plain(self.state)},
            sort_keys=True,
            default=str,
        )


class JsonlCheckpointer:
    """Durable, append-only checkpoints.

    A checkpoint after every node is what turns a crashed run into a resumable
    one — and, just as usefully, lets you replay a run to the exact state where
    a decision went wrong.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def save(self, checkpoint: Checkpoint) -> None:
        with self.path.open("a") as fh:
            fh.write(checkpoint.to_json() + "\n")

    def load_all(self) -> list[Checkpoint]:
        out: list[Checkpoint] = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            out.append(
                Checkpoint(
                    step=raw["step"], node=raw["node"], next_node=raw["next"], state=raw["state"]
                )
            )
        return out

    def latest(self) -> Checkpoint | None:
        checkpoints = self.load_all()
        return checkpoints[-1] if checkpoints else None


@dataclass
class GraphResult:
    state: State
    path: list[str]
    steps: int
    halted: str  # "end" | "max_steps"


class Graph:
    def __init__(self, trace: Trace | None = None, max_steps: int = 40) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, str] = {}
        self._routers: dict[str, tuple[Router, dict[str, str]]] = {}
        self._entry: str | None = None
        self.trace = trace
        self.max_steps = max_steps

    # --- construction ----------------------------------------------------

    def add_node(self, name: str, fn: Node) -> "Graph":
        if name in {START, END}:
            raise GraphError(f"{name} is reserved")
        if name in self._nodes:
            raise GraphError(f"duplicate node: {name}")
        self._nodes[name] = fn
        return self

    def set_entry(self, name: str) -> "Graph":
        self._entry = name
        return self

    def add_edge(self, src: str, dst: str) -> "Graph":
        if src in self._routers:
            raise GraphError(f"{src} already has a conditional edge")
        self._edges[src] = dst
        return self

    def add_conditional_edge(self, src: str, router: Router, mapping: dict[str, str]) -> "Graph":
        """`router` returns a label; `mapping` turns labels into node names.

        Keeping the label distinct from the destination is what makes branches
        readable in the trace: you see `route: red -> diagnose`, not just a
        node name appearing out of nowhere.
        """
        if src in self._edges:
            raise GraphError(f"{src} already has a static edge")
        self._routers[src] = (router, dict(mapping))
        return self

    def validate(self) -> None:
        if self._entry is None:
            raise GraphError("no entry node set")
        known = set(self._nodes) | {END}
        if self._entry not in self._nodes:
            raise GraphError(f"entry node {self._entry!r} is not defined")
        for src, dst in self._edges.items():
            if src not in self._nodes:
                raise GraphError(f"edge from unknown node: {src}")
            if dst not in known:
                raise GraphError(f"edge {src} -> {dst}: unknown destination")
        for src, (_, mapping) in self._routers.items():
            if src not in self._nodes:
                raise GraphError(f"conditional edge from unknown node: {src}")
            for label, dst in mapping.items():
                if dst not in known:
                    raise GraphError(f"route {src} -[{label}]-> {dst}: unknown destination")
        dangling = [n for n in self._nodes if n not in self._edges and n not in self._routers]
        if dangling:
            raise GraphError(f"node(s) with no outgoing edge: {sorted(dangling)}")

    def to_mermaid(self) -> str:
        lines = ["flowchart TD", f"    {START}([start]) --> {self._entry}"]
        for src, dst in sorted(self._edges.items()):
            lines.append(f"    {src} --> {_label(dst)}")
        for src, (_, mapping) in sorted(self._routers.items()):
            for label, dst in sorted(mapping.items()):
                lines.append(f"    {src} -->|{label}| {_label(dst)}")
        return "\n".join(lines)

    # --- execution -------------------------------------------------------

    def invoke(
        self,
        state: State,
        checkpointer: JsonlCheckpointer | None = None,
        start_at: str | None = None,
    ) -> GraphResult:
        self.validate()
        current = start_at or self._entry
        assert current is not None
        state = dict(state)
        path: list[str] = []
        step = 0
        halted = "end"

        while current != END:
            if step >= self.max_steps:
                halted = "max_steps"
                self._emit("graph.halt", reason="max_steps", limit=self.max_steps, node=current)
                break
            node = self._nodes.get(current)
            if node is None:
                raise GraphError(f"no such node: {current}")

            self._emit("graph.node.enter", node=current, step=step)
            update = node(state)
            if update:
                state.update(update)
            path.append(current)
            self._emit("graph.node.exit", node=current, step=step, keys=sorted((update or {}).keys()))

            next_node = self._next(current, state)
            if checkpointer is not None:
                checkpointer.save(Checkpoint(step=step, node=current, next_node=next_node, state=state))
            current = next_node
            step += 1

        if halted == "end":
            self._emit("graph.done", steps=step, path=path)
        return GraphResult(state=state, path=path, steps=step, halted=halted)

    def resume(self, checkpointer: JsonlCheckpointer) -> GraphResult:
        """Pick a run back up from its last checkpoint."""
        checkpoint = checkpointer.latest()
        if checkpoint is None:
            raise GraphError("no checkpoint to resume from")
        self._emit("graph.resume", from_node=checkpoint.node, next=checkpoint.next_node)
        if checkpoint.next_node == END:
            return GraphResult(state=checkpoint.state, path=[], steps=0, halted="end")
        return self.invoke(checkpoint.state, checkpointer=None, start_at=checkpoint.next_node)

    def _next(self, current: str, state: State) -> str:
        if current in self._edges:
            return self._edges[current]
        router, mapping = self._routers[current]
        label = router(state)
        if label not in mapping:
            raise GraphError(
                f"router for {current} returned {label!r}; expected one of {sorted(mapping)}"
            )
        destination = mapping[label]
        self._emit("graph.route", node=current, label=label, to=destination)
        return destination

    def _emit(self, type: str, **payload: Any) -> None:
        if self.trace is not None:
            self.trace.emit(type, **payload)


def _label(dst: str) -> str:
    return "__end__([end])" if dst == END else dst


def _plain(state: State) -> dict[str, Any]:
    """Keep checkpoints JSON-clean: drop values that are not plainly serializable."""
    out: dict[str, Any] = {}
    for key, value in state.items():
        try:
            json.dumps(value)
        except TypeError:
            out[key] = f"<{type(value).__name__}>"
        else:
            out[key] = value
    return out
