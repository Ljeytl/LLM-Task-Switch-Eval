# Architecture

## The one idea

Every conversation is generated from a list of `Op` objects, and that **same list goes
to two places**:

```
                        ┌──────────────────────┐
                        │  sample_ops(seed…)   │
                        │  a list[Op]          │
                        └──────────┬───────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
      surface.render(op)                   state.ground_truth(ops)
      "Add milk to the list."              {"shopping": ["milk"], …}
                 │                                   │
                 ▼                                   │
      runner.run_conversation ──► model's answer     │
                 │                                   │
                 └──────────────► scorer.score ◄─────┘
                                       │
                                       ▼
                          Score(joint_correct, failures)
```

Because the transcript and the answer key are produced from one source, grading is a
dict comparison. There is no judge model, no annotation, and no subjectivity in the
primary metric. Everything else in the repo exists to protect that property.

## Data flow, module by module

| Module | Responsibility | Key invariant |
|---|---|---|
| `ops.py` | The operation grammar: `TaskKind`, `OpKind`, `Op` | Illegal (task, kind) pairs raise at construction. `INERT_KINDS` names the ops that must never move state. |
| `state.py` | **The oracle.** `ShoppingState` (a set), `ScheduleState` (an ordered map), `ground_truth()` | `apply()` is *total* — it never raises, whatever it is handed. A crash here would be recorded as a model failure. |
| `surface.py` | Natural-language rendering, ≥4 paraphrases per (task, kind) | Templated, never model-generated: a generated turn could say something the op list does not encode and break the answer key. |
| `generator.py` | Op sampling, ordering, and **pairing** | Renders each op *once*, then reorders the rendered turns. Asserts the two orderings hold the identical multiset of strings. |
| `runner.py` | Ollama client, response cache, runtime token verification | Only the final turn is generated. Assistant turns are prefilled with a fixed `"Got it."`. |
| `scorer.py` | Diff + failure taxonomy | Pure: no model calls, no I/O. The corpus can be re-scored from cache without re-running inference. |
| `stats.py` | Wilson, McNemar, paired bootstrap, clustered SE | See `STATS.md`. Every estimator is chosen over a more obvious alternative for a stated reason. |
| `plots.py` | Dumbbell (primary), taxonomy bars | The dumbbell's *slope* is the switch cost. |
| `run.py` | CLI: demo, calibrate, check-constrained, sweep, analyse, rescore | Checkpoints results after every cell so a long sweep is resumable. |

## The three independent knobs

This is the part of the design that took the most thought, because two of these
quantities want to move together and must not.

```
total context ≈  n_tasks  ×  ops_per_task  ×  tokens_per_turn   +   n_noise × tokens_per_turn
                    │             │                                      │
              live states    state updates                        irrelevant padding
```

- **`n_tasks`** — how many states are live at once.
- **`n_ops`** — how many state updates occur. **Total per conversation, not per task.**
  If it were per-task, four tasks would produce roughly twice the conversation of two,
  and "more tasks is harder" would silently be partly "longer is harder".
- **`n_noise`** — task-irrelevant turns. This is the context-length lever.

Varying length by adding *operations* would have confounded length with state-update
load. Varying it with noise keeps them separate. It is also far cheaper: noise never
enters the final answer, so it costs only fast prefill (~300 tok/s) and never slow
generation (~16 tok/s).

## Why pairing happens where it does

`build_pair()` is the single most important function in the repo.

```python
ops        = sample_ops(seed, tasks, n_ops, n_false, n_noise)
rng        = Random(seed ^ 0x5EED)
text_by_idx = {o.idx: render(o, rng) for o in ops}      # ← rendered ONCE
for ordering in (BLOCKED, INTERLEAVED):
    seq   = order_ops(ops, ordering, tasks)
    turns = [text_by_idx[o.idx] for o in seq]           # ← then reordered
```

Rendering once is not an optimisation. Rendering per-ordering would draw from the RNG
in a different sequence, select different paraphrases, and silently break the token
match the entire comparison depends on.

## RNG discipline

Each `(seed, task, purpose)` gets its own RNG, derived with blake2b:

```python
_subrng(seed, task.value, "mut", n_mut)   # the mutating operation stream
_subrng(seed, task.value, "mix", n_noise, n_false)   # placement of inert turns
```

This is not tidiness. An earlier version passed one shared RNG through every task, so
generating noise for task 1 consumed a variable number of draws and shifted task 2's
entire operation stream — meaning raising `n_noise` silently changed the *operations*
as well as the padding. That is the same class of confound the project exists to
isolate, one level down. `blake2b` rather than `hash()`, which is salted per process
and would make runs irreproducible across invocations.

## Caching

`cache_key` hashes the fully rendered message list plus every generation parameter, not
the seed. A change to any template, or to the ack string, therefore invalidates stale
entries instead of silently reusing them. The cache makes a long sweep resumable and,
because `scorer.score` is pure, lets the taxonomy be revised and re-applied to the
whole corpus with `--rescore` at zero inference cost.
