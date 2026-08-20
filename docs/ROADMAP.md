# Roadmap

Phase 1 is done: harness, loop, and graph built from scratch in Python, wired into
one TypeScript use case, all exercised offline by 123 tests.

## Phase 2 — run it for real

Nothing here is built yet; the point of phase 1 was to make phase 2 measurable.

1. **A live run against Claude.** `pip install 'alg[anthropic]'`, set
   `ANTHROPIC_API_KEY`, `alg run tasks/calc_bug --show-diff`. Read the trace, not
   just the verdict.
2. **The same run against local Ollama.** `alg run tasks/calc_bug --provider ollama
   --model <your 27B>`. Expect it to fail differently: smaller models produce more
   malformed tool calls and more premature "I fixed it". Those are exactly the
   failures the stop rules exist to catch — check *which* rule fires.
3. **More tasks.** One bug per task is too easy to be informative. Add: a bug
   whose fix requires reading two files; a task where the obvious fix breaks
   another test; a task with a genuinely ambiguous spec. `tasks/` is a directory
   because the interesting work is comparative — and since a task is just a
   directory with an `alg.task.json`, a Vitest task, a Go task, or a Python task
   costs a manifest plus (if the output format is new) one parser.
4. **An eval harness.** N tasks × M runs → solve rate, attempts to green, tokens
   per solve, and the distribution of stop reasons. Without this, every tuning
   decision is vibes.

## Phase 3 — port to LangGraph, then compare

You built the primitives so the framework's choices are legible. The mapping:

| This repo | LangGraph / Agent SDK equivalent |
| :--- | :--- |
| `Graph`, `add_conditional_edge` | `StateGraph`, `add_conditional_edges` |
| `State` dict merged per node | Typed state with reducers (`Annotated[list, add]`) |
| `JsonlCheckpointer` | `checkpointer=` (`MemorySaver`, `SqliteSaver`) + thread ids |
| `AgentLoop` | `create_react_agent`, or the SDK tool runner |
| `ToolRegistry` + `validate_args` | `@tool` decorators with pydantic schemas |
| `Trace` | LangSmith tracing / callbacks |
| Manual `Message`/`Block` types | Provider SDK message types |

Port `agent.py` and keep `tasks/`, then answer for yourself:

- Which of my 200 lines did the framework replace with 20, and which did it
  replace with 200 lines of configuration?
- What can I no longer see? (Traces are the usual answer.)
- What can I no longer control? (`stalled_evidence` and the revert-on-regression
  are the ones to check — both are custom logic, not framework features.)

## Backlog — harness

- Permission policy per tool: allow / ask / deny, with the ask path checkpointing
  and exiting so a human can approve out of band.
- Context management: compaction when the message list grows, and a token
  accounting view per attempt.
- Prompt caching: freeze the system prompt, keep volatile content after the last
  cache breakpoint, and verify with `cache_read_input_tokens`.
- Secrets and network policy: run tests with no network; assert the agent cannot
  reach it.

## Backlog — loop

- Non-improvement stall detection (not just exact repetition).
- Verifier as a separate model call with a fresh context, instead of trusting the
  same conversation to grade itself.
- Per-attempt strategy: force a read-only diagnosis phase before any edit is
  allowed, using `ToolRegistry.subset()`.

## Backlog — graph

- Parallel fan-out per failing test, with a join that detects conflicting edits.
- A `triage` node that picks a prompt by failure class.
- Sub-graphs: make `repair` itself a graph so its internal steps are traceable.

## Toward Wirespect

The layers transfer; the task definition is what changes. What makes this repo a
usable rehearsal for a real product is the shape of `tasks/`: a directory, a
verifier that returns structure, and a definition of "solved" that the agent
cannot fake. Before pointing this at real work, be able to answer:

1. What is the automatic verifier? (Here: `node --test`. If there is no verifier, loop
   engineering has nothing to close over and you are back to one-shot prompting.)
2. What is the blast radius of a wrong action, and what is the undo?
3. What does the eval set look like, and how will you know a change to the prompt
   or the graph made it better rather than different?
