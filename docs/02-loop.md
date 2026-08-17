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

The important part is that this comes from the **test runner**, not from the
model's own account of how it is doing. That is the anti-reward-hacking property:
progress must show up in something the model does not author. If evidence were
"did the model say it fixed it", the stall detector would never fire.

Note the shape of the signature. `3p/2f` alone is not enough — a run that trades
one failure for a different one shows the same counts while making no progress,
so the failing test *ids* are part of the signature.

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
2. Make `stalled_evidence` smarter — stop on *non-improvement* rather than exact
   repetition, using the score function from `agent.py`.
3. Add context compaction: when `state.messages` exceeds N turns, summarize the
   middle. Note what this does to prompt caching and to the trace.
4. Measure it. Run the same task at `max_iterations` 3, 6, and 12 and plot
   attempts-to-green against tokens. That curve is the whole subject.
