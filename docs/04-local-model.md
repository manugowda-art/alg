# 4. Running it against a local model

Everything below assumes a Mac with Ollama. The harness cannot tell a local
model from an API one — the difference you will feel is *latency* and *tool-call
reliability*, and both are things the run is instrumented to show you.

## 0. Get the repo running with no model at all

Do this first. If it does not pass, a model will not help.

```bash
git clone <your repo> && cd alg
git checkout claude/agentic-dev-learning-bseqtx

python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

pytest                              # 143 tests, no network
(cd tasks/calc_bug && npm test)     # expect: 11 tests, 6 pass, 5 fail
```

That second command failing *is* the correct outcome — it is the task. If you
see anything other than 6 pass / 5 fail, your Node is the problem, not the task.
Node 22.18+ runs `.ts` with no flag and no `npm install`; on older Node, add
`--experimental-strip-types` to `test_command` in `tasks/calc_bug/alg.task.json`.

## 1. Pick a model that can call tools

This harness is entirely tool-driven. A model without tool support will
produce plausible prose and change nothing, so check before you spend an hour:

```bash
ollama serve                        # if it is not already running
ollama pull qwen3:30b               # or qwen2.5-coder:32b, qwen3:14b
ollama list
```

Qwen3 and Qwen2.5-Coder both support tools in Ollama. Bigger is not
automatically better here: the job is following a tool schema and reading test
output, which the coder-tuned models tend to do more reliably than a general
model of the same size.

## 2. Preflight

```bash
alg doctor --provider ollama --model qwen3:30b
```

This checks every link in the chain and — importantly — makes one real model
call, so you find out *now* whether your model emits tool calls:

```
  ✓ python         3.13.1
  ✓ node           v22.22.0
  ✓ task manifest  calc_bug (TypeScript, runner=node-test)
  ✓ task baseline  6 passed, 5 failed — the verifier works
  ✓ ollama         http://localhost:11434 (4 models pulled)
  ✓ model          qwen3:30b, tools supported, context 40,960
  ✓ tool call      report_status({'ok': True}) in 12.4s

  ready.

  suggested run:
    alg run tasks/calc_bug --provider ollama --model qwen3:30b \
        --wall-clock 3120 --show-diff
```

The suggested `--wall-clock` is derived from that measured call, not guessed.
Every failure names its own fix. The one worth understanding:

> ✗ tool call   model replied with text instead of a tool call after 31.2s
>               → text: Sure! I would call report_status with ok=true...

That model will never solve the task. Switch models rather than tuning prompts.

## 3. Run it

```bash
alg run tasks/calc_bug --provider ollama --model qwen3:30b --show-diff
```

Expect it to take a while — a 30B model on a Mac is tens of seconds per call,
and a repair attempt is up to 10 calls. `--wall-clock` defaults to 3600s per
attempt for Ollama (600s for API providers) precisely because the API default
would cut a local run off mid-repair.

Useful knobs:

| Flag | When |
| :--- | :--- |
| `--max-iterations 6` | Shorter leash per attempt; fail faster while experimenting |
| `--max-attempts 1` | One shot, no retry — the cleanest signal about raw capability |
| `--num-ctx 16384` | Lower memory pressure; raise if you see truncated reasoning |
| `--host` | Ollama on another machine |

## 4. Read what happened — this is the actual exercise

The verdict is the least interesting output. The run directory is the point:

```bash
alg trace runs/<id>/trace.jsonl --type graph.route   # which branches were taken
alg trace runs/<id>/trace.jsonl --type tool.call     # every call, args, ok/error
alg trace runs/<id>/trace.jsonl --type loop.stop     # why each attempt ended
cat runs/<id>/checkpoints.jsonl                      # state after every node
```

**The single most informative line is the stop reason.** It tells you which
failure mode you hit, and they call for different fixes:

| Stop reason | What it means | What to change |
| :--- | :--- | :--- |
| `model_finished` while still red | The model believed it was done and was wrong | Prompt, or a stronger model |
| `max_iterations(N)` | Ran out of room mid-repair | Raise `--max-iterations` |
| `stalled_evidence(4)` | Four iterations, identical failing tests | The model is stuck; retrying will not help |
| `tool_error_streak(3)` | Three straight malformed tool calls | Schema too complex for this model, or its template is weak |
| `wall_clock(Ns)` | Ran out of time, not ideas | Raise `--wall-clock` |
| `model_refusal` | Provider declined | Not applicable locally |

A small model failing is not a failed experiment — it is the experiment. The
stop rules exist to name *how* it failed, and each name points at a different
part of the system.

## 5. What to expect from a local model

Honest calibration, so a bad run does not read as a broken repo:

- **Tool-call malformation is the common failure**, not bad reasoning. Watch for
  `tool.call` events with `ok=false` — argument names invented, `old_text` that
  does not match the file, patches with fabricated context lines.
- **`edit_file` is the friendliest tool** for a small model: exact string, unique
  match, clear error. `apply_patch` demands correct context lines and is where
  weaker models fail most.
- **Premature completion** — "I have fixed the bug" with the suite still red — is
  what `stalled_evidence` and the `verify` node exist to catch. The graph
  re-runs the tests regardless of what the model claims.
- Expect the model to need the retry loop. Getting to green on attempt 2 or 3 is
  a good outcome; the interesting comparison is *how many attempts* and *which
  stop reasons*, not pass/fail.

## 6. If it does not work

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `cannot reach ollama` | Server not running | `ollama serve` |
| `does not support tools` | Model template has no tool support | Use qwen3 / qwen2.5-coder |
| Model narrates instead of calling tools | Weak tool template | Different model; confirm with `alg doctor` |
| Every `edit_file` errors "old_text not found" | Model is copying `read_file`'s line-number prefixes | Known small-model failure; the error message already says this, and a capable model recovers |
| Run dies at `wall_clock` | Local latency | Raise `--wall-clock` |
| Tests time out inside the run | Cold Node start under memory pressure | `--test-timeout 300` |
| Nonsense after several turns | Context overflow | Lower `--max-iterations`, or raise `--num-ctx` if the model allows |

## 7. The comparison worth running

Once one run works, the exercise is comparative — that is the whole point of
having stop reasons and traces:

```bash
alg run tasks/calc_bug --provider ollama --model qwen3:30b --max-attempts 1
alg run tasks/calc_bug --provider ollama --model qwen2.5-coder:32b --max-attempts 1
alg run tasks/calc_bug --provider anthropic --max-attempts 1   # needs a key
```

Same task, same harness, same graph. Compare attempts-to-green, tool-error
rate, and the distribution of stop reasons. That table is loop engineering:
you are no longer asking "is the model good", you are asking "which part of my
system failed, and does the evidence say so".
