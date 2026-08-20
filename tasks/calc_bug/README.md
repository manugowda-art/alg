# task: calc_bug (TypeScript)

A four-function statistics module with a failing test suite. Two defects are
seeded in `src/stats.ts`; one of them also breaks `variance`, so the failure
count does not map one-to-one onto the number of bugs.

The tests are the specification. A run counts as solved when the suite is green
and no file under `test/` has been modified.

That second condition is currently a *prompt* instruction, not a harness rule —
`edit_file` and `apply_patch` will happily write to `test/` if the model asks.
The manifest already carries `test_glob`, so enforcing it is a small change; it
is left undone on purpose, as one of the exercises.

```bash
npm test          # node --test --test-reporter=tap
```

A correct starting point looks like this — if you see anything else, the
environment is wrong, not the task:

```
# tests 11
# pass 6
# fail 5
```

The five failures come from two defects, because `variance` calls `mean` and
inherits its bug. That is deliberate: an agent that counts failures instead of
reading code will conclude there are more bugs than there are.

No dependencies and no build step: Node 22.18+ runs `.ts` directly (type
stripping is on by default) and `node --test` discovers `test/*.test.ts` on its
own. That keeps the harness hermetic — a workspace copy is ready to run the
instant it is created, with no `npm install` and no `node_modules` to clone.

`alg.task.json` is what the harness reads: the test command, how to focus a
single test, and which globs count as source, tests, and searchable files.
