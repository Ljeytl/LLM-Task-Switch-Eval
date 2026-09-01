# Experimental design

## The manipulation

One operation sequence, two orderings, with operation content and token count held fixed.

```
ops:  [S1 S2 S3]  [C1 C2 C3]            (S = shopping, C = calendar)

BLOCKED      S1 S2 S3 C1 C2 C3          1 task transition
INTERLEAVED  S1 C1 S2 C2 S3 C3          5 task transitions
```

Both conversations contain the **same rendered strings**. Within-task order is
identical in both. The sequence differs, so the ground truth is provably the same
(`test_ordering_does_not_change_ground_truth`) and the token count is provably the same
(`test_pair_is_token_matched`, plus runtime `prompt_eval_count` verification).

Sequence is not a single-factor manipulation here. Fixed blocked-task order also changes
the serial position and recency of each task's updates. The current run therefore
estimates an **ordering effect**; a causal switch-cost estimate requires counterbalancing
blocked-task order and the interleaved starting slot.

## The grid, as shipped

Seven cells per model. `n_ops = 6` throughout.

| label | n_tasks | n_ops | n_noise | role |
|---|---|---|---|---|
| `ctrl_1task` | 1 | 6 | 40 | **Negative control.** Nothing to interleave, so the delta must be exactly 0. |
| `len_short` | 2 | 6 | 0 | |
| `len_medium` | 2 | 6 | 40 | **Repository-prespecified primary**; shared corner of the L |
| `len_long` | 2 | 6 | 120 | Does the ordering difference change with context length? |
| `same_kind_2` | 2 | 6 | 40 | Targeted follow-up: two shopping-list instances |
| `tasks_3` | 3 | 6 | 40 | |
| `tasks_4` | 4 | 6 | 40 | Does the ordering difference grow with the number of live states? |

**Why a length arm and not a single point.** Holding context length fixed licenses a
claim at one length only. Varying it turns the result from a number into a shape, and
answers the criticism levelled at the closest prior work: rather than holding length
constant *or* confounding it with interleaving, this varies length while the
interleaving contrast is made independently inside each cell.

### What the task-count arm actually asks

`n_ops` is **total per conversation** (D4), so fewer tasks means *more operations per
task*: the 1-task control puts all six ops on one list, the 2-task cell puts three on
each, the 4-task cell puts one or two on each.

That is a deliberate trade and it must be stated precisely, because it changes the claim:

> The arm does **not** answer "does adding another task hurt?"
> It answers "**at fixed total work and fixed token count, does splitting that work
> across more live states cost you?**"

The second is the more useful question for someone sizing an assistant — real budgets are
in tokens, not tasks — but it is a different claim and only it is supported here.

It also explains a result that looks backwards at first glance: the 1-task control scores
*below* the 2-task cell (0.320 vs 0.640 on `qwen2.5-coder:7b`). The control is not an
easier condition; it is a single list six items long instead of two lists three items
long.

The alternative — holding ops *per task* constant — would make a 4-task conversation
twice the length of a 2-task one and re-confound task count with context length, which is
exactly the error D4 fixed. No design avoids both; this one names which it accepts.

### The task-count arm, cut in v1 and restored in v2

v1 could not run it. Task identity was `TaskKind` and only two kinds existed, so "3
tasks" resolved to `[SHOPPING, SCHEDULE, SHOPPING]` — the duplicate merged into the
first list's state and its turns were emitted twice. The token-match assertion caught it
and the arm was cut (`DECISIONS.md` D14).

v2 keys state on per-instance **slots** with disjoint vocabularies (D17), so 1–4
concurrent tasks are genuinely independent and misattribution between two lists of the
same kind is detectable. The arm is in `configs/main.yaml` as `tasks_3` and `tasks_4`.

## Repository prespecification

**Primary comparison:** `len_medium` (2 tasks, 6 ops, 40 noise) on `qwen2.5-coder:7b`,
joint goal accuracy, McNemar exact test, reported uncorrected. Git history records this
choice before the v2 sweep; there was no external registry.

The one-task controls are validation checks rather than inferential tests. The other 11
model-condition comparisons form one **secondary and exploratory** family, with raw and
Bonferroni-adjusted p-values reported. Prespecifying one comparison is cleaner than
selecting a hypothesis after seeing the data; correction does not make an exploratory
hypothesis confirmatory.

## Unit of analysis

**The conversation, not the task.** Tasks inside one conversation share a context
window, a system prompt, and a single model call; if the model loses the thread it
plausibly loses several at once.

- **Primary metric:** joint goal accuracy — every task in the conversation exactly
  right. One binary outcome per conversation, so no clustering problem.
- **Secondary metric:** per-task accuracy, with standard errors clustered on
  conversation. The naive SE is printed alongside so the ratio is visible.

## Calibration and what actually ran

`run.py --calibrate` sweeps `n_ops` at small n to find where blocked joint accuracy sits
in the 0.6–0.8 band. The measured v1 grid selected `n_ops=6` for qwen.

This is not optional bookkeeping. If blocked accuracy is at 1.0 or 0.0 there is no room
for a delta in either direction, and the entire sweep returns a null for reasons that
have nothing to do with interleaving. Small models in particular have effective context
well below their nominal window, so the band has to be found per model rather than
assumed. The shipped v2 run did not repeat that calibration after changing templates and
noise construction, and gemma was not calibrated separately. The primary stayed in the
measurable band, but the calibration does not transfer as a completed v2 validation gate.

## Confounds, and how each is handled

| Confound | Handling |
|---|---|
| Context bloat masquerading as switch cost | Token-matched orderings, identical by construction and verified at run time against `prompt_eval_count` |
| Task count confounded with conversation length | `n_ops` is total per conversation, not per task (D4); the arm was cut in v1 and restored with slot identity in v2 (D17). |
| Context length confounded with state-update load | Length varied with noise turns, which never enter the answer (D5) |
| Noise level silently changing operations or earlier rendering | v2 uses nested noise pools and operation-keyed rendering, covered by invariance tests (D6, D17). |
| **Output-schema ambiguity correlating with condition** | Named keys in the schema; scorer decides positional pairs by content (D7) — this one actually happened, see below |
| Constrained decoding degrading accuracy | Measured directly on a subset, not assumed away (`--check-constrained`) |
| Format failure scored as tracking failure | Separate taxonomy bucket, excluded from state accuracy |
| Model pattern-matching one phrasing | ≥4 paraphrases per (task, kind), seed-randomised; some imperative templates remain syntactically ambiguous. |
| Ordering confounded with serial position and recency | **Unresolved in this run.** Confirmatory design must rotate blocked-task order and interleaved starting slot. |
| Nondeterminism at temperature 0 | Not claimed. Seeds and digests logged; intervals reported, not point estimates |

## The confound that nearly got through

The output schema originally asked for the schedule as an array of two-element string
arrays, without specifying which element was the time. The first live run returned
`[title, time]` in the blocked condition and `[time, title]` in the interleaved one —
same model, same tokens, different format choice **per condition**.

The scorer read that as six tracking failures in a conversation the model had tracked
perfectly. Had it gone unnoticed, it would have produced a large, clean, entirely
fictitious switch cost.

The lesson generalises past this bug: an ambiguous measurement instrument is not merely
noisy. If the ambiguity resolves differently under different conditions, it becomes a
**systematic** effect pointing in whatever direction the ambiguity happens to lean.


---

## What the two arms showed

The point of crossing ordering with both factors was to inspect a shape rather than a
single number. The exploratory qwen point estimates were:

```
task count   2 -> 3 -> 4 tasks     delta  -12.0  -20.0  -28.0 pp    monotone
context      0 -> 40 -> 120 noise  delta  -16.0  -12.0  -12.0 pp    flat
```

Both arms hold `n_ops = 6` and are token-matched within every cell. The task-count arm
reduces operations per task as it adds tasks, but task composition, same-kind-pair count,
serial position and recency still prevent a clean causal task-count interpretation.

**The honest caveats, in order of importance:**

1. The **repository-prespecified primary is inconclusive** (−12.0pp, p=0.508).
2. `tasks_4` reaches raw p=0.039 but Bonferroni-adjusted p=0.430 in the 11-test
   exploratory family. No secondary cell establishes a confirmatory effect.
3. The monotone task-count point estimates are a hypothesis worth testing, not a trend
   test. Composition and operations per task change across those cells.
4. The length arm's nearly flat qwen point estimates are a low-power observation, not
   evidence that context length does not affect the ordering difference.
5. Fixed ordering leaves serial position and recency unresolved, so the current outcome
   should be called an ordering effect rather than an isolated switch cost.
