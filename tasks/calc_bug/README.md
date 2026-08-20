# task: calc_bug (TypeScript)

A four-function statistics module with a failing test suite. Two defects are
seeded in `src/stats.ts`; one of them also breaks `variance`, so the failure
count does not map one-to-one onto the number of bugs.

The tests are the specification. A run counts as solved when the suite is green
and no file under `test/` has been modified.

```bash
npm test          # node --test --test-reporter=tap
```

No dependencies and no build step: Node 22.18+ runs `.ts` directly (type
stripping is on by default) and `node --test` discovers `test/*.test.ts` on its
own. That keeps the harness hermetic — a workspace copy is ready to run the
instant it is created, with no `npm install` and no `node_modules` to clone.

`alg.task.json` is what the harness reads: the test command, how to focus a
single test, and which globs count as source, tests, and searchable files.
