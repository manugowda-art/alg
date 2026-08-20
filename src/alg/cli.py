"""Command line entry point.

    alg doctor --provider ollama --model qwen3:30b   # check the chain first
    alg run tasks/calc_bug --provider ollama --model qwen3:30b
    alg graph                     # print the topology as mermaid
    alg trace runs/<id>/trace.jsonl
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .agent import AgentConfig, FixerAgent, build_agent
from .doctor import FAIL, WARN, run_doctor
from .events import Trace, read_trace
from .llm import build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the test-fixer agent against a task")
    run.add_argument("task", help="path to a task directory (e.g. tasks/calc_bug)")
    run.add_argument("--provider", default="anthropic", choices=["anthropic", "ollama"])
    run.add_argument("--model", default=None, help="model id; defaults per provider")
    run.add_argument("--runs-dir", default="runs")
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--max-iterations", type=int, default=10)
    run.add_argument("--allow-write-file", action="store_true", help="expose the whole-file write tool")
    run.add_argument("--show-diff", action="store_true")
    _add_provider_flags(run)
    run.add_argument(
        "--wall-clock",
        type=float,
        default=None,
        help="seconds per repair attempt before the loop stops (default 600, or 3600 for ollama)",
    )
    run.add_argument("--test-timeout", type=float, default=120.0, help="seconds per test run")

    doctor = sub.add_parser("doctor", help="check python, node, the task, and the model")
    doctor.add_argument("task", nargs="?", default="tasks/calc_bug")
    doctor.add_argument("--provider", default="ollama", choices=["anthropic", "ollama"])
    doctor.add_argument("--model", default=None)
    doctor.add_argument("--max-iterations", type=int, default=10)
    _add_provider_flags(doctor)

    sub.add_parser("graph", help="print the agent graph as a mermaid diagram")

    trace_cmd = sub.add_parser("trace", help="render a trace file")
    trace_cmd.add_argument("path")
    trace_cmd.add_argument("--type", default=None, help="only show events of this type")

    args = parser.parse_args(argv)

    if args.command == "graph":
        return _cmd_graph()
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "trace":
        return _cmd_trace(args)
    return _cmd_run(args)


def _add_provider_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=None, help="ollama host (default http://localhost:11434)")
    parser.add_argument("--num-ctx", type=int, default=None, help="ollama context window")


def _provider_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict = {}
    if args.provider == "ollama":
        if getattr(args, "host", None):
            kwargs["host"] = args.host
        if getattr(args, "num_ctx", None):
            kwargs["num_ctx"] = args.num_ctx
    return kwargs


def _cmd_doctor(args: argparse.Namespace) -> int:
    import tempfile

    print(f"checking {args.task} with provider={args.provider} model={args.model or '(default)'}\n")
    with tempfile.TemporaryDirectory() as tmp:
        checks, command = run_doctor(
            task_dir=args.task,
            provider=args.provider,
            model=args.model,
            work_dir=Path(tmp) / "work",
            max_iterations=args.max_iterations,
            **_provider_kwargs(args),
        )

    width = max(len(c.name) for c in checks)
    for check in checks:
        print(f"  {check.mark} {check.name:<{width}}  {check.detail}")
        if check.fix:
            print(f"  {' ' * (width + 4)}→ {check.fix}")

    failed = [c for c in checks if c.status == FAIL]
    warned = [c for c in checks if c.status == WARN]
    print()
    if failed:
        print(f"  {len(failed)} check(s) failed — fix those before running.")
        return 1
    if warned:
        print(f"  ready, with {len(warned)} warning(s).")
    else:
        print("  ready.")
    print(f"\n  suggested run:\n    {command}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.runs_dir) / run_id
    trace = Trace(run_dir / "trace.jsonl")

    try:
        llm = build(args.provider, args.model, **_provider_kwargs(args))
    except RuntimeError as exc:
        print(f"cannot build model: {exc}", file=sys.stderr)
        return 2

    agent = build_agent(
        task_dir=args.task,
        work_dir=run_dir / "work",
        llm=llm,
        trace=trace,
        config=AgentConfig(
            max_attempts=args.max_attempts,
            loop_max_iterations=args.max_iterations,
            allow_write_file=args.allow_write_file,
            # A local model is far slower per call than an API one, so the
            # default budget would stop the loop mid-repair.
            loop_wall_clock=(
                args.wall_clock
                if args.wall_clock is not None
                else (3600.0 if args.provider == "ollama" else 600.0)
            ),
            test_timeout=args.test_timeout,
        ),
    )

    print(f"run {run_id}  model={llm.name}  task={args.task}")
    state = agent.run(checkpoint_path=run_dir / "checkpoints.jsonl")

    print()
    for index, report in enumerate(state["reports"]):
        label = "baseline" if index == 0 else f"attempt {index}"
        print(f"  {label:<10} {report['passed']} passed, {report['failed']} failed")
    print()
    print(f"  {state['summary']}")
    print(f"  stop reasons: {', '.join(state.get('loop_stop_reasons', []) or ['-'])}")
    tokens = state.get("tokens", {})
    print(f"  tokens: {tokens.get('input', 0)} in / {tokens.get('output', 0)} out")
    print(f"  trace: {run_dir / 'trace.jsonl'}")

    if args.show_diff and state.get("diff"):
        print()
        print(state["diff"])

    return 0 if state.get("green") else 1


def _cmd_graph() -> int:
    # A topology-only view: no model or workspace needed to print the structure.
    agent = FixerAgent.__new__(FixerAgent)
    agent.trace = None  # type: ignore[assignment]
    agent.config = AgentConfig()
    print(agent.graph().to_mermaid())
    return 0


def _cmd_trace(args: argparse.Namespace) -> int:
    events = read_trace(args.path)
    for event in events:
        if args.type and event.type != args.type:
            continue
        detail = ", ".join(f"{k}={_short(v)}" for k, v in sorted(event.payload.items()))
        print(f"{event.seq:>4} {event.type:<22} {detail}")
    return 0


def _short(value: object, limit: int = 120) -> str:
    text = str(value).replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 3] + "..."


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
