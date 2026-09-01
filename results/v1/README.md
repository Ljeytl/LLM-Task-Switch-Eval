# v1 results (pre-slot design)

These are the results of the **first** complete sweep, run before task identity was
refactored from `TaskKind` to per-instance slots (`docs/DECISIONS.md` D17).

**They are not reproducible from the current code.** The refactor changed every user
turn — templates now name the specific list or calendar ("add milk to my grocery list"
rather than "add milk to the shopping list") — so prompts, token counts and cache keys
all differ. They are preserved because they are real measurements, honestly obtained,
and because the v1→v2 comparison is itself informative.

## What v1 measured

Two concurrent tasks only, distinguished by kind. 480 conversations, 8 cells, 2 models,
480/480 token match exact.

| model | condition | blocked | interleaved | delta | b/c | p |
|---|---|---:|---:|---:|---|---:|
| qwen2.5-coder:7b | ctrl_1task *(control)* | 0.600 | 0.600 | +0.0 | 0/0 | 1.000 |
| qwen2.5-coder:7b | len_short | 0.900 | 0.700 | −20.0 | 7/1 | 0.070 |
| qwen2.5-coder:7b | **len_medium (primary)** | 0.467 | 0.433 | −3.3 | 6/5 | 1.000 |
| qwen2.5-coder:7b | len_long | 0.667 | 0.333 | −33.3 | 12/2 | 0.013 |
| gemma4:12b | ctrl_1task *(control)* | 0.933 | 0.933 | +0.0 | 0/0 | 1.000 |
| gemma4:12b | len_short | 0.900 | 0.900 | +0.0 | 0/0 | 1.000 |
| gemma4:12b | len_medium | 0.967 | 0.933 | −3.3 | 2/1 | 1.000 |
| gemma4:12b | len_long | 0.833 | 0.800 | −3.3 | 3/2 | 1.000 |

Failure composition: absorbed/false-assertion 38.1%, dropped/vanished 30.9%,
absorbed/hallucination 21.6%, absorbed/stale 6.7%, dropped/wrong-value 2.6%,
**misattributed 0.0%**.

## Why v1 could not answer the headline question

v1 had **no task-count arm**. Task identity was `TaskKind` and only two kinds existed,
so "3 tasks" collapsed to two states. The question the project set out to ask — *how many
live tasks can a model hold?* — was unanswerable in v1 and is answerable in v2.

## The one caveat v1 raised about its own zero

v1 recorded **zero misattribution across 480 conversations** and could not say whether
that meant models do not misfile across dissimilar tasks, or that two tasks was simply
too few for misfiling to arise. v2 can distinguish these, because it runs three and four
concurrent tasks with disjoint vocabularies per instance — a grocery item appearing on
the hardware list is unmistakable.
