# Experimental design

## The manipulation

One operation sequence, two orderings, everything else held fixed.

```
ops:  [S1 S2 S3]  [C1 C2 C3]            (S = shopping, C = calendar)

BLOCKED      S1 S2 S3 C1 C2 C3          1 task transition
INTERLEAVED  S1 C1 S2 C2 S3 C3          5 task transitions
```

Both conversations contain the **same rendered strings**. Within-task order is
identical in both. Only the sequence differs, so the ground truth is provably the same
(`test_ordering_does_not_change_ground_truth`) and the token count is provably the same
(`test_pair_is_token_matched`, plus runtime `prompt_eval_count` verification).

## The grid, as shipped

Four cells per model. `n_ops = 6` throughout — measured, not guessed (see Calibration).

| label | n_tasks | n_ops | n_noise | role |
|---|---|---|---|---|
| `ctrl_1task` | 1 | 6 | 40 | **Negative control.** Nothing to interleave, so the delta must be exactly 0. |
| `len_short` | 2 | 6 | 0 | |
| `len_medium` | 2 | 6 | 40 | **Primary**, pre-registered |
| `len_long` | 2 | 6 | 120 | Does switch cost grow with context length? |

**Why a length arm and not a single point.** Holding context length fixed licenses a
claim at one length only. Varying it turns the result from a number into a shape, and
answers the criticism levelled at the closest prior work: rather than holding length
constant *or* confounding it with interleaving, this varies length while the
interleaving contrast is made independently inside each cell.

### The task-count arm was cut

The plan called for a second arm at 2/3/4 concurrent tasks. It turned out to be
undeliverable: task identity is `TaskKind` and there are only two kinds, so "3 tasks"
resolved to `[SHOPPING, SCHEDULE, SHOPPING]` — the duplicate merged into the first
list's state and its turns were emitted twice. The token-match assertion caught it.
Full reasoning, and why it was cut rather than fixed, in `DECISIONS.md` D14.

## Pre-registration

**Primary comparison:** `len_medium` (2 tasks, 6 ops, 40 noise) on `qwen2.5-coder:7b`,
joint goal accuracy, McNemar exact test, reported uncorrected.

Everything else is **secondary and exploratory**, with Bonferroni applied to that family
and noted as conservative. Pre-registering one comparison is cleaner and more honest
than running several and correcting across all of them — a correction cannot undo a
hypothesis chosen after seeing the data.

## Unit of analysis

**The conversation, not the task.** Tasks inside one conversation share a context
window, a system prompt, and a single model call; if the model loses the thread it
plausibly loses several at once.

- **Primary metric:** joint goal accuracy — every task in the conversation exactly
  right. One binary outcome per conversation, so no clustering problem.
- **Secondary metric:** per-task accuracy, with standard errors clustered on
  conversation. The naive SE is printed alongside so the ratio is visible.

## Calibration comes before the sweep

`run.py --calibrate` sweeps `n_ops` at small n to find where blocked joint accuracy sits
in the 0.6–0.8 band, **per model**.

This is not optional bookkeeping. If blocked accuracy is at 1.0 or 0.0 there is no room
for a delta in either direction, and the entire sweep returns a null for reasons that
have nothing to do with interleaving. Small models in particular have effective context
well below their nominal window, so the band has to be found per model rather than
assumed.

## Confounds, and how each is handled

| Confound | Handling |
|---|---|
| Context bloat masquerading as switch cost | Token-matched orderings, identical by construction and verified at run time against `prompt_eval_count` |
| Task count confounded with conversation length | `n_ops` is total per conversation, not per task (D4). The task-count arm was ultimately cut (D14). |
| Context length confounded with state-update load | Length varied with noise turns, which never enter the answer (D5) |
| Noise level silently changing the operations | Independent RNG per (seed, task, purpose) (D6) |
| **Output-schema ambiguity correlating with condition** | Named keys in the schema; scorer decides positional pairs by content (D7) — this one actually happened, see below |
| Constrained decoding degrading accuracy | Measured directly on a subset, not assumed away (`--check-constrained`) |
| Format failure scored as tracking failure | Separate taxonomy bucket, excluded from state accuracy |
| Model pattern-matching one phrasing | ≥4 paraphrases per (task, kind), seed-randomised |
| Ordering confounded with recency | Per-task accuracy reported by position of the task's op block |
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
