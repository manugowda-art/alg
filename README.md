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

> Given a **TypeScript** package with a failing test suite, diagnose the defect,
> patch the source, and prove the fix by running the tests. Do not touch the tests.

That task is chosen because **the feedback is free, objective, and unfakeable**.
`node --test` decides whether the work is done, not the model.

The engine is Python; the task under repair is TypeScript. Nothing in the loop or
the graph knows either language — the task's `alg.task.json` supplies the test
command, and a parser keyed by `runner` turns its output into a verdict.

```
          ┌──────────────────── graph.py ─────────────────────┐
          │  baseline ──red──→ repair ──→ verify ──retry──┐   │
          │                      ↑                        │   │
          │                      └────────────────────────┘   │
          │                                                   │
          │  baseline / verify ──green | exhausted──→ report  │
          └─────────────────────────┬─────────────────────────┘
                                    │  each repair node runs one
              ┌───────────────── loop.py ──────────────────┐
              │  action → evidence → feedback → stop rule  │  bounded cycle, five named stop rules
              └──────────────────────┬─────────────────────┘
                                     │  every tool call goes through
      ┌─────── workspace.py + tools/   (the harness) ───────┐
      │  disposable copy · path jail · schema validation    │
      │  contained errors · bounded execution · full trace  │
      └─────────────────────────────────────────────────────┘
```

## Prerequisites

| | Why |
| :--- | :--- |
| Python 3.11+ | The engine: harness, loop, graph, CLI |
| Node 22.18+ | The task: `node --test` runs `.ts` directly, no build step |

Verified on Python 3.11 and Node 22.22. If `node --test` fails to parse the
`.ts` files on your Node build, add `--experimental-strip-types` to
`test_command` in `tasks/calc_bug/alg.task.json` — the harness takes the command
from there, so nothing in the engine changes.

## Try it

```bash
pip install -e '.[dev]'
pytest                                    # 143 tests, offline
(cd tasks/calc_bug && npm test)           # watch the task fail on its own
alg graph                                 # print the topology as mermaid
```

Live runs need a model. Either provider works — the harness cannot tell them
apart. Check the whole chain first; `doctor` makes one real model call, so a
model that cannot emit tool calls is caught in seconds rather than an hour in:

```bash
# a local model — no key, no cloud
alg doctor --provider ollama --model qwen3:30b
alg run tasks/calc_bug --provider ollama --model qwen3:30b --show-diff

# or the API
pip install -e '.[anthropic]'
export ANTHROPIC_API_KEY=...
alg run tasks/calc_bug --show-diff
```

Local models are slower per call, so `--wall-clock` defaults to 3600s per repair
attempt for Ollama against 600s for API providers. See
[docs/04-local-model.md](docs/04-local-model.md) for the full walkthrough,
including what each stop reason tells you to change.

Then read what actually happened:

```bash
alg trace runs/<id>/trace.jsonl                     # everything
alg trace runs/<id>/trace.jsonl --type tool.call    # just the tool calls
alg trace runs/<id>/trace.jsonl --type graph.route  # just the branches
cat runs/<id>/checkpoints.jsonl                     # state after every node
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
    tests_tool.py Test runners → structured TestReport (node --test TAP, pytest)
  loop.py         Bounded model/tool cycle + five stop rules.
  graph.py        Node/edge executor, conditional routing, JSONL checkpoints, resume.
  agent.py        The wiring: baseline → repair → verify → report.
  tasks.py        TaskSpec: reads alg.task.json so the engine stays language-agnostic.
  llm/            Provider-neutral interface + Anthropic and Ollama adapters.
  doctor.py       Preflight: python, node, task, server, model, tool call.
  cli.py          alg doctor | run | graph | trace
tasks/calc_bug/   TypeScript stats module: two seeded defects, 11 tests, zero deps.
  alg.task.json   Test command, focus flag, source/test/search globs.
  src/stats.ts    The code under repair.
  test/           node:test suite — the specification, and off-limits to the agent.
tests/            143 tests covering all three layers, no network required.
  conftest.py     Fixtures + ScriptedLLM, the offline test double for a model.
  test_harness.py Jail, bounded execution, registry, fs tools
  test_patch.py   Diff parsing and application
  test_loop.py    Stop rules, feedback, evidence
  test_graph.py   Routing, cycles, checkpoints, resume
  test_tasks.py   Manifest loading and validation
  test_agent.py   The three layers end to end, plus the test-output parsers
  test_ollama_wire.py  The Ollama adapter against a stub HTTP server
  test_doctor.py  Preflight checks, including the ways a local setup fails
docs/             One document per layer, plus the roadmap.
```

## Adding a task

A task is a directory with an `alg.task.json`. The engine reads it and never
names a language itself. This is the bundled one, verbatim:

```json
{
  "name": "calc_bug",
  "language": "TypeScript",
  "description": "A statistics module with two seeded defects and 11 tests.",
  "runner": "node-test",
  "test_command": ["node", "--test", "--test-reporter=tap"],
  "focus_template": ["--test-name-pattern", "{target}"],
  "focus_hint": "a test-name pattern, e.g. \"median\"",
  "source_glob": "src/**/*.ts",
  "test_glob": "test/**/*.ts",
  "search_glob": "**/*.ts"
}
```

| Key | Required | Meaning |
| :--- | :--- | :--- |
| `name` | yes | Task identifier |
| `runner` | yes | Selects the output parser from `PARSERS` in `tests_tool.py` |
| `test_command` | yes | Argv run inside the workspace; its exit code and output are the verdict |
| `language` | no | Named in the system prompt and the `run_tests` description |
| `description` | no | Free text for humans |
| `focus_template` | no | Argv fragment for narrowing to one test; `{target}` is substituted. Omitted → the target is appended positionally (what pytest wants) |
| `focus_hint` | no | How the target is described to the model |
| `source_glob` / `test_glob` | no | Which files are source and which are specification |
| `search_glob` | no | Default glob for the `search` tool |
| `timeout` | no | Seconds per test run (default 120) |

Unknown keys are rejected rather than ignored, so a typo fails at load time
instead of silently doing nothing. `node-test` and `pytest` parsers ship today:
a Vitest or Jest task costs a manifest if its reporter emits TAP, and a manifest
plus one parser function if it does not.

The bar for a good task is not the language. It is: **a verifier that returns
structure, and a definition of "solved" the agent cannot fake.**

## Reading order

| Document | Layer |
| :--- | :--- |
| [docs/01-harness.md](docs/01-harness.md) | What the model is allowed to do, see, and have recorded |
| [docs/02-loop.md](docs/02-loop.md) | Why every exit from a cycle needs a name, and why evidence must be external |
| [docs/03-graph.md](docs/03-graph.md) | Why retry, undo, and "what next" belong outside the loop |
| [docs/04-local-model.md](docs/04-local-model.md) | Running it against Ollama, and reading the result |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Live runs, evals, and the port to LangGraph |

The one line worth carrying out of each layer:

- **Harness** — a tool failure is data, not an exception. Contained errors become
  feedback; raised exceptions end runs.
- **Loop** — every way out has a name, and the evidence comes from the verifier,
  not from the model's own account of its progress.
- **Graph** — retry, revert, and "what next" are control flow. Put them in the
  topology where they can be traced, tested, and bounded twice.
