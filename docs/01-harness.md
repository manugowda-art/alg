# 1. Agent harness

> The machinery around the model. Fixes: *"the model cannot safely do the work."*

A harness is not "the code that calls the API". It is everything that decides
**what the model is allowed to do, what it can see, and what is recorded**. In
this repo that is five files:

| File | Responsibility |
| :--- | :--- |
| `workspace.py` | A disposable copy of the task, a path jail, a bounded command runner |
| `tools/__init__.py` | Tool registry: schema validation, error containment, tracing |
| `tools/*.py` | The actual capabilities: read, search, edit, patch, run tests |
| `tasks.py` | The task manifest — what makes the harness language-agnostic |
| `events.py` | The trace — the only output channel |

## The four properties worth building

**1. The model never touches the source of truth.**
`Workspace.materialize()` copies the task into `runs/<id>/work/`. A destructive
mistake costs a directory, not your repository. Everything downstream can then be
generous with what it allows, because the blast radius is already bounded.

**2. Every path is validated in one place.**
`Workspace.resolve()` rejects absolute paths, `..` traversal, and symlinks
pointing outward. Tools call `resolve()` and stop worrying. The alternative —
each tool doing its own checking — is how the one tool that forgot becomes the
hole. Try it:

```python
workspace.resolve("../../etc/passwd")   # WorkspaceError
```

**3. A tool failure is data, not an exception.**
`ToolRegistry.call()` catches everything and returns `ToolOutcome(ok=False, ...)`.
This is the single highest-leverage decision in the harness. A raised exception
kills the run; a returned error becomes a `tool_result` with `is_error=True`, the
model reads it and adapts. Malformed arguments, missing files, ambiguous edits,
even a tool that raises `RuntimeError` — all become feedback.

That is also why `validate_args()` runs *before* the tool body. A model calling
`edit_file` with a typo'd argument name gets told the valid argument names, which
is recoverable; a `TypeError` from Python's argument binding is not.

**4. Everything is traced, nothing is printed.**
No module in `src/alg/` writes to stdout. They emit events. The CLI renders them.
That constraint is what makes runs inspectable after the fact:

```
alg trace runs/<id>/trace.jsonl --type tool.call
```

## Tool design: the interesting decisions

**`edit_file` requires a unique match.** The tool refuses when `old_text` is
absent (the model's picture of the file is stale) or appears more than once (the
edit is ambiguous). Both refusals carry the reason. This one constraint prevents
the most common silent corruption: an edit that lands in the wrong place and
"works" until the test output makes no sense.

**`apply_patch` is forgiving about line numbers and strict about context.** Models
miscount lines constantly, so hunks are located by searching outward from the
stated position for the context block. But if the context does not match, nothing
is written — and if a patch touches three files and the third hunk fails, *none*
of them are written. A half-applied patch is worse than a rejected one, because
the model now reasons about a file state that nobody intended.

**`run_tests` returns structure, not text.** `TestReport` carries counts, the
failing test ids, and a `signature`. Loop engineering needs to compare evidence
across iterations, and you cannot compare two walls of test output.

**The runner is data, not code.** `tasks.py` reads the task's `alg.task.json` —
test command, focus flag, source/test/search globs — and `tests_tool.py` picks a
parser from a registry keyed by `runner`. The bundled task is TypeScript on
`node --test`; a pytest parser ships alongside it. Adding a language means adding
a parser, not touching the loop or the graph. This is the seam that keeps
"what the engine does" separate from "what this task happens to be written in".

**Narrow tool sets are a feature.** `ToolRegistry.subset()` exists so a phase can
be made read-only by construction rather than by asking the model nicely.
`write_file` is off by default (`AgentConfig.allow_write_file`) because a
whole-file overwrite discards the staleness check that makes `edit_file` safe.

## Where the seams are

Read `tests/test_harness.py`. Every property above has a test, and the tests are
the fastest way to see what the harness actually guarantees:

```
test_resolve_refuses_paths_outside_the_workspace
test_registry_returns_errors_instead_of_raising
test_registry_validates_before_running_the_tool
test_edit_file_requires_a_unique_match
test_apply_patch_is_atomic_across_files
test_run_enforces_a_timeout
```

## What is deliberately missing

No permission prompts, no approval gates, no per-tool rate limits, no context
compaction. Each is a real harness concern and each is a good exercise — see
`ROADMAP.md`. The point of stopping here is that the four properties above are
what a harness *must* have; the rest is what a harness *may* have.
