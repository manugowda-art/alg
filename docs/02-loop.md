# 2. Loop engineering

> The repeated work-and-feedback cycle. Fixes: *"the agent stops too early or
> repeats weak work."*

`loop.py` is 200 lines and one idea: **every way out of the loop has a name.**

```
trigger    a goal message
action     a model call that may request tools
evidence   tool outcomes, plus whatever the caller records as evidence
feedback   those outcomes go back in as the next turn
stop rule  a named reason the cycle ended
```

## The five stop rules and the failure each one prevents

| Rule | Prevents |
| :--- | :--- |
| `max_iterations(n)` | An unbounded run. The floor, not the strategy. |
| `token_budget(n)` | Cost blowup on a task the model will not solve. |
| `wall_clock(s)` | A run that hangs behind a slow tool. |
| `tool_error_streak(k)` | The model retrying a malformed call forever, because each rejection looks recoverable. |
| `stalled_evidence(k)` | The loop spinning: `k` iterations with identical evidence. |

`stalled_evidence` is the one that matters most, and the one most loops lack.
Without it, the only signals available are "the model said it was done" and "the
budget ran out" — and the first is exactly the signal you cannot trust, because a
model that has stopped making progress rarely announces it.

Named reasons also change what you can learn from a run. `loop_stop_reasons` in
the final state distinguishes these three outcomes, which look identical if all
you record is "failed":

```
["model_finished", "model_finished"]        the model believed it was done, twice, and was wrong
["max_iterations(10)", "max_iterations(10)"] it ran out of room; raise the ceiling
["stalled_evidence(4)", "tool_error_streak(3)"] it was stuck; the prompt or tools are wrong
```

## Evidence has to be external

`evidence_fn` is called after each iteration and its return value is appended to
`state.evidence`. In this repo it returns the failing-test signature of the most
recent run the model itself triggered:

```python
def _evidence(self, state: LoopState) -> str | None:
    report = self._observed[-1]
    return f"{report.passed}p/{report.failed}f:{','.join(report.signature)}"
```

The important part is that this comes from the **test runner** (`node --test`), not
from the model's own account of how it is doing. That is the anti-reward-hacking property:
progress must show up in something the model does not author. If evidence were
"did the model say it fixed it", the stall detector would never fire.

Note the shape of the signature. `3p/2f` alone is not enough — a run that trades
one failure for a different one shows the same counts while making no progress,
so the failing test *ids* are part of the signature.

## A real bug in this file, found by a real run

The first live run against a local Qwen produced this:

```
  baseline   6 passed, 5 failed
  attempt 1  10 passed, 1 failed
  attempt 2  11 passed, 0 failed
  stop reasons: stalled_evidence(4) at '6p/5f:...', model_finished
```

Green — but look at attempt 1. It *made progress* (5 failures down to 1) and
was still stopped by the stall detector, at the **baseline** reading. Both
things cannot be true: a loop that went from 5 failures to 1 was not stalled.

The bug was in the evidence function, not the rule:

```python
# before — returns the last reading even if it is old
if not self._observed:
    return None
return signature_of(self._observed[-1])
```

`self._observed` only grows when the *model* calls `run_tests`. The model ran
the tests once, then spent four iterations reading and editing — good, normal
work. Every one of those iterations re-reported the same stale reading, four
identical values landed in `state.evidence`, and the rule fired.

So `stalled_evidence` was not measuring "the model is not making progress". It
was measuring **"the model has not re-run the tests lately"** — a different
thing, and one that is often true precisely when the model is mid-edit and
doing fine. The fix is one condition:

```python
# after — silence is not evidence
if len(self._observed) <= self._reported:
    return None
self._reported = len(self._observed)
return signature_of(self._observed[-1])
```

Three lessons worth more than the fix:

1. **An iteration is not an observation.** The loop ticks on model turns; the
   evidence ticks on verifier runs. Coupling them conflates "time passed" with
   "nothing changed".
2. **Absence of evidence must not be encoded as evidence.** Returning a stale
   value where `None` was correct is what created a phantom pattern for the
   rule to match.
3. **The graph hid it.** The retry loop recovered — attempt 2 finished the job,
   the run went green, and a pass/fail view would have shown nothing wrong.
   Only the *stop reasons* exposed it. This is the argument for naming every
   exit: the run succeeded and was still broken.

The gap the fix leaves open is worth knowing: a model that edits and *never*
re-runs the tests now produces no evidence at all, so `stalled_evidence` cannot
fire for it. That is a different failure and wants its own named rule — see the
exercises.

## Where the loop stops and the graph starts

The loop deliberately cannot do three things:

- retry after failure — it has no notion of "attempt 2"
- undo — it cannot roll the workspace back
- decide what happens next — it returns a stop reason and exits

Those are decisions about *control flow*, and burying them inside a single cycle
is what makes agents hard to reason about. They belong in the graph (`03-graph.md`).

The division: **the loop is one bounded conversation; the graph decides how many
conversations happen and what comes between them.**

## Exercises

1. Add a `no_mutation` stop rule: halt if `k` iterations pass with only read-only
   tool calls. (The model is exploring, not working.)
2. Add its mirror, `unverified_edits(k)`: `k` iterations that mutate files with
   no `run_tests` in between. That is the gap the evidence fix leaves open, and
   it is arguably better handled by *nudging* the model than by stopping —
   which makes it a good place to think about what a stop rule is actually for.
3. Make `stalled_evidence` smarter — stop on *non-improvement* rather than exact
   repetition, using the score function from `agent.py`.
4. Add context compaction: when `state.messages` exceeds N turns, summarize the
   middle. Note what this does to prompt caching and to the trace.
5. Measure it. Run the same task at `max_iterations` 3, 6, and 12 and plot
   attempts-to-green against tokens. That curve is the whole subject.
