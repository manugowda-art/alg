# ALG

**A**gent Harness -> **L**oop Engineering -> **G**raph Engineering.

environment → feedback → flow.

# What do they do
1. Harness engineering builds the machinery around the model.
2. Loop engineering designs the repeated work-and-feedback cycle.
3. Graph engineering makes the workflow topology explicit: nodes, branches, joins, state transitions and controlled cycles.


# Agent Architecture Comparison

| Question | Agent harness | Loop engineering | Graph engineering |
| :--- | :--- | :--- | :--- |
| **Primary concern** | Operational capability | Iterative progress and feedback | Explicit control flow |
| **Core object** | Model wrapper / runtime | A bounded repeatable cycle | A directed graph of steps |
| **Typical building blocks** | Tools, memory, sandbox, middleware, permissions, traces | Trigger, goal, action, evidence, feedback, stop rule | Nodes, edges, shared state, branches, joins, interrupts, cycles |
| **Failure it fixes** | "The model cannot safely do the work." | "The agent stops too early or repeats weak work." | "The workflow is hard to reason about or control." |
| **Best fit** | General agent platform or task-specific runtime | Open-ended work that improves through verification | Complex multi-step processes with known decision points |
| **Main risk** | A bloated, opaque runtime | Infinite retries, token burn, reward hacking | Over-engineered diagrams and brittle paths |

---

# The use case: a self-healing test fixer

All three layers, built from scratch on the standard library, applied to one task
small enough to hold in your head and complete enough to exercise every layer:

> Given a Python package with a failing test suite, diagnose the defect, patch the
> source, and prove the fix by running the tests. Do not touch the tests.

That task is chosen because **the feedback is free, objective, and unfakeable**.
`pytest` decides whether the work is done, not the model.

```
             ┌─────────────── graph.py ────────────────┐
             │  baseline ─red→ repair → verify ─retry→ ⤾ │
             │      └─green→ report ←─green/exhausted─┘  │
             └───────────────────┬─────────────────────┘
                                 │  each repair node runs
                     ┌───────────▼───────────┐
                     │       loop.py         │  bounded model/tool cycle
                     │  action → evidence →  │  with named stop rules
                     │  feedback → stop rule │
                     └───────────┬───────────┘
                                 │  every tool call goes through
        ┌────────────────────────▼────────────────────────┐
        │  workspace.py + tools/    (the harness)         │
        │  disposable copy · path jail · schema validation │
        │  contained errors · bounded execution · traces   │
        └─────────────────────────────────────────────────┘
```

## Layout

```
src/alg/
  events.py       Event trace (JSONL). The only output channel.
  workspace.py    Disposable task copy, path jail, bounded command runner.
  tools/
    __init__.py   Tool registry: schema validation, error containment, tracing.
    fs.py         list_files, read_file, search, edit_file, write_file, diff
    patch.py      Unified-diff parser and applier (forgiving offsets, strict context, atomic)
    tests_tool.py pytest runner → structured TestReport
  loop.py         Bounded model/tool cycle + five stop rules.
  graph.py        Node/edge executor, conditional routing, JSONL checkpoints, resume.
  agent.py        The wiring: baseline → repair → verify → report.
  llm/            Provider-neutral interface + Anthropic and Ollama adapters.
  cli.py          alg run | graph | trace
tasks/calc_bug/   A stats module with two seeded defects and 11 tests.
tests/            108 tests covering all three layers, no network required.
docs/             One document per layer, plus the roadmap.
```

## Try it

```bash
pip install -e '.[dev]'
pytest                                    # 108 tests, offline
alg graph                                 # print the topology as mermaid
```

Live runs need a model. Either provider works — the harness cannot tell them apart:

```bash
pip install -e '.[anthropic]'
export ANTHROPIC_API_KEY=...
alg run tasks/calc_bug --show-diff

# or a local model, no key, no cloud
alg run tasks/calc_bug --provider ollama --model gemma3:27b
```

Then read what actually happened:

```bash
alg trace runs/<id>/trace.jsonl                     # everything
alg trace runs/<id>/trace.jsonl --type tool.call    # just the tool calls
alg trace runs/<id>/trace.jsonl --type graph.route  # just the branches
cat runs/<id>/checkpoints.jsonl                     # state after every node
```

## Reading order

| Document | Layer |
| :--- | :--- |
| [docs/01-harness.md](docs/01-harness.md) | What the model is allowed to do, see, and have recorded |
| [docs/02-loop.md](docs/02-loop.md) | Why every exit from a cycle needs a name, and why evidence must be external |
| [docs/03-graph.md](docs/03-graph.md) | Why retry, undo, and "what next" belong outside the loop |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Live runs, evals, and the port to LangGraph |

The one line worth carrying out of each layer:

- **Harness** — a tool failure is data, not an exception. Contained errors become
  feedback; raised exceptions end runs.
- **Loop** — every way out has a name, and the evidence comes from the verifier,
  not from the model's own account of its progress.
- **Graph** — retry, revert, and "what next" are control flow. Put them in the
  topology where they can be traced, tested, and bounded twice.
