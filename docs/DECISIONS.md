# Decision log

Every fork, what was chosen, what was rejected, and why. Dated. The entries marked
**changed direction** are the ones where the first answer turned out to be wrong.

---

### D1 — Mechanical ground truth over an LLM judge
*Chosen: a state machine consumes the same op list the generator emitted.*

MultiChallenge abandoned rule-based grading because "there is no single ground-truth
answer for most test conversations." Constraining the domain to structured state buys
that back: a set and an ordered map are independently diffable, so the transcript and
the answer key come out of one function as a pair.

**Cost, stated honestly:** the tasks are far simpler than real assistant work. This is
an existence proof that switch cost is separable and mechanically measurable, not a
general benchmark. Lineage: MultiWOZ joint goal accuracy, tau-bench final-DB-state
comparison.

---

### D2 — Two task types, not three
*Chosen: shopping (set) and schedule (ordered map). Expenses (running sum) cut.*

Chosen so misattribution is **visible**: a grocery item appearing in the schedule is
unambiguous, whereas two shopping lists would not be. An expense tracker was cut
because arithmetic error would confound state-tracking error — a wrong total could mean
a dropped operation or a failed addition, and the taxonomy could not tell them apart.

---

### D3 — Prefilled assistant acknowledgements
*Chosen: every user turn is followed by a fixed `"Got it."` the model did not generate.*
*Rejected: letting the model reply to each turn.*

This is the decision the token-match claim rests on. With prefilled acks, the blocked
and interleaved prompts are the **identical multiset of strings in a different order**,
so they tokenise to the same length by construction rather than by estimate. Confirmed
empirically: 0 of 5 demo pairs drifted, max delta 0 tokens.

The alternative also failed on cost. Generated turns mean 24–48 calls per conversation
instead of one — roughly 10× — which put the sweep at 50+ hours on this hardware.

**Cost, stated honestly:** the model is not doing interactive work mid-conversation.
This measures extraction and aggregation over a transcript, not live turn-by-turn state
maintenance. Named as next work.

---

### D4 — `n_ops` is total per conversation, not per task
*Changed direction.*

The original interface specified `sample_ops(seed, tasks, n_ops, …)` without saying
which. If it were per-task, four tasks would produce roughly twice the conversation of
two, and any "more concurrent tasks is harder" result would silently be partly "longer
conversations are harder" — reproducing the exact confound this project criticises the
closest prior work for, one level down.

Now total, dealt round-robin with the remainder going to earlier tasks
deterministically. Pinned by `test_n_ops_is_total_not_per_task`.

---

### D5 — Context length varied with noise turns, not more operations
*Chosen: hold `n_ops` at 24, vary `n_noise`.*

Two knobs both drive token count: how many state updates occur, and how much
irrelevant text surrounds them. Driving length with *operations* would confound length
with state-update load — the same error as D4.

An unplanned benefit showed up in the benchmark: prefill runs at ~300 tok/s while
generation runs at ~16 tok/s, so cost is dominated by *output* length. Noise never
enters the final answer, so the length arm is nearly free. An op-count length arm would
have inflated the final state and hit the slow path.

**Cost, stated honestly:** length and task-irrelevant content move together here, so
the length arm cannot separate "more context" from "more distraction."

---

### D6 — Independent RNG per (seed, task, purpose)
*Changed direction, after a test failure.*

The first implementation threaded one shared `Random` through every task. Generating
noise for task 1 consumed a variable number of draws and shifted task 2's entire
operation stream — so raising `n_noise` silently changed the *operations* as well as
the padding, which is the D5 confound sneaking back in through the RNG.

Now each `(seed, task, purpose)` slice derives its own generator via blake2b.
`blake2b` rather than `hash()`, which is salted per process and would make runs
irreproducible across invocations. Caught by
`test_noise_lengthens_context_without_changing_the_answer`.

---

### D7 — Named keys in the output schema, not positional pairs
*Changed direction, after the first live run.*

The schedule was originally requested as an array of two-element string arrays. The
schema never said which element was the time. The first demo run returned
`['one-on-one', '14:00']` in the blocked condition and `['14:00', 'one-on-one']` in the
interleaved one — **the same model, the same tokens, a different format choice per
condition** — and the scorer read the difference as six tracking failures in a
conversation the model had tracked perfectly.

An ambiguous schema does not merely add noise. It added noise *correlated with the
condition*, which would have manufactured a large switch cost out of nothing.

Fixed twice over: the schema now uses named `{"time", "title"}` keys, and `_canon`
decides a positional pair by which element matches a clock-time pattern rather than by
position. Regression test:
`test_positional_pair_is_read_by_content_not_position`.

---

### D8 — Retries cover transport failures only
*Chosen: no retry on parse failure.*

The interface sketch called for retrying a parse failure up to twice. Under greedy
decoding (temperature 0, top_k 1) a retry reproduces the same tokens exactly, so the
loop would burn wall-clock and change nothing. Recording `parse_ok=False` immediately
is faster and more honest. Transport errors are retried, with backoff.

---

### D9 — Exact binomial McNemar, not the chi-square approximation
The textbook `χ² = (|b-c|-1)²/(b+c)` is a normal approximation to a binomial that is
unreliable when the discordant count is small — exactly the regime a real but modest
effect produces. `scipy.stats.binomtest` is exact and costs nothing here. See
`STATS.md` §3.

---

### D10 — Case- and format-insensitive comparison
*Chosen: normalise case, whitespace, and time format before diffing.*

The question is whether the model tracked the state, not whether it capitalised it.
Grading `"Milk" != "milk"` as a tracking failure would inflate the headline number with
formatting noise. Everything normalised away is recorded in `scorer.py` so the choice
is auditable, and format failures have their own taxonomy bucket so they are never
reported as tracking failures.

---

### D11 — Bespoke harness, not `inspect_ai`
The framework would hide the state machine, the scorer and the paired design — which
are the parts worth owning and the parts that get interrogated. Borrowed its separation
of concerns and its JSONL conventions.

---

### D12 — Models: whatever was already installed
*Chosen: `qwen2.5-coder:7b` and `gemma4:12b`. Rejected: pulling the Qwen2.5 3B/7B/14B ladder.*

Two of the four locally installed models turned out to be unusable: `qwen3.6` is 36B at
23 GB and `qwen3-coder` is 30.5B MoE at 18 GB, and with a ~18 GB Metal wired limit on
24 GB neither leaves room for a KV cache.

**Consequence, stated for the record:** `qwen2.5-coder:7b` and `gemma4:12b` differ in
family, instruction tuning *and* parameter count simultaneously. There is no scaling
ladder and no isolated variable across them. The **primary result is unaffected** —
blocked vs interleaved is paired within a single model — but cross-model comparison is
descriptive only, and must not be read as a scale trend.

---

### D13 — `think=False` is mandatory for gemma4
`gemma4:12b` reasons by default. With thinking on and a 64-token budget it returned
**empty content**, having spent the entire budget reasoning — a silent 100%
format-failure rate across the whole sweep had this not been caught in a pre-flight
check. Disabling it also cut latency from 16 s to 3.1 s.

Because the token match is on the *prompt*, thinking never threatened the match itself
— only cost and cross-model comparability.

---

### D14 — The task-count arm was cut because it was silently degenerate
*Changed direction, after the token-match assertion failed.*

The plan called for an L-shaped design: a length arm plus a task-count arm at 2, 3 and 4
concurrent tasks. It is not implementable in this codebase, and the reason is
structural rather than a bug that could be patched.

**Task identity is `TaskKind`**, and there are exactly two kinds. So `tasks_for(3)`
returned `[SHOPPING, SCHEDULE, SHOPPING]`, and:

- `build_states` keys its dict on `TaskKind`, so the duplicate collapsed — "3 tasks"
  built **2** state machines.
- `order_ops` builds `{t: [ops for t] for t in tasks}` and then iterates `tasks`, so in
  blocked ordering the shopping turns were emitted **twice** (13 turns vs 8).

The second problem is what surfaced it: `build_pair` raised `TokenMatchError`. Had only
the first problem existed, the 3- and 4-task cells would have quietly measured two tasks
and reported the result as a task-count effect.

**Why cut rather than fix.** A correct task-count arm needs per-instance identity — two
separately named shopping lists — which is a refactor across `ops`, `state`, `surface`,
`generator`, `runner` and `scorer`, and invalidates the whole response cache. It is
also the *scientifically weaker* half of the design by this project's own analysis:
`LIMITATIONS.md` §5 already noted that two same-type tasks make misattribution nearly
undetectable, and misattribution is the diagnostic this design exists to produce.

**What replaced it.** A `ctrl_1task` negative control. With one task there is nothing to
interleave, so both orderings are the identical sequence and the measured delta must be
exactly zero. Any non-zero switch cost there would mean the harness manufactures
differences on its own.

**Guarded so it cannot recur silently.** `build_pair` now rejects duplicate task kinds
with an explicit error, and `tasks_for` refuses `n` beyond the number of distinct kinds.
Tests: `test_duplicate_task_kinds_are_rejected`, `test_single_task_pair_is_a_null_control`.

The general lesson matches D7: the assertions that pay for themselves are the ones that
check an *invariant you believe*, not the output you expect. Both bugs that would have
produced fabricated findings were caught by invariant checks, not by looking at results.


---

### D15 — `stale` split out from `hallucination`
*Changed direction, after the grounding audit disagreed with the scorer.*

`tools/audit_taxonomy.py` checks that every failure the scorer emits corresponds to a
real expected-vs-reported discrepancy. It flagged two cases the scorer had labelled
`hallucination`. Tracing one back through the op list:

```
add    one-on-one at 11:30
remove one-on-one
add    one-on-one at 14:30
```

The model reported the meeting **twice**, at 11:30 and at 14:30. The 11:30 entry is
ungrounded in the final state — but the model did not invent it. It failed to apply the
removal. That is a dropped operation leaving a stale value, and calling it a
hallucination would point anyone debugging it at entirely the wrong problem: "the model
makes things up" and "the model does not retract" have nothing in common as failures.

Now reported as `absorbed … stale`, with `hallucination` reserved for content that was
never mentioned. Tests: `test_stale_duplicate_is_not_called_a_hallucination`,
`test_true_hallucination_still_reads_as_hallucination`.

Two things worth noting about how this was found:

1. It came from an **automated grounding check**, not from reading accuracy numbers.
   The accuracy was correct throughout — only the diagnosis was wrong. Re-scoring
   confirmed it: `--rescore` changed the labels and **0 outcomes**.
2. The audit tool was wrong twice before it was right. Its identity parser split on
   whitespace, turning `"olive oil"` into `"olive"`, and its identity-only view could
   not see a value mismatch. A verifier is code too — when it disagrees with the thing
   it verifies, that is a bug report against *whichever side is wrong*, and you have to
   look rather than trust either one.


---

### D16 — Free-form runs get a second extraction pass
*Changed direction, after the first constrained-vs-free-form check produced a nonsense number.*

The check is meant to answer: does schema-constrained decoding degrade state tracking?
Published estimates put the effect as high as 8.7 points, so it needs measuring rather
than assuming.

The first implementation ran the same conversation with and without `format=schema` and
compared joint accuracy. It reported constrained decoding as **+66.7pp better**, with a
free-form parse rate of **0/12**. That number is meaningless. Looking at an actual
free-form response explains why:

```
Sure, here's the current state:

**Shopping List:**
- Yoghurt
- Olive oil
- Tomatoes

**Meeting Schedule:**
- Onboarding at 12:00
- Retro at 13:30
```

The answer is *correct*. It is prose, because nothing asked for JSON, and a JSON parser
scores it as a total failure. The check was measuring my parser, not the model.

`runner.extract_state` now adds a second call that converts a free-form answer into the
schema, with a prompt that forbids adding, removing or correcting anything — so the
extractor cannot rescue a wrong answer, only reformat a right one. Reasoning happens in
the first call, serialisation in the second, which is what makes the comparison fair.
This is what the original design doc proposed; the first implementation simply skipped it.

Fourth instance of the same pattern in this project (see D7, D14, D15): **the measuring
apparatus was broken, and it failed in a direction that flattered the result.** A check
that produces a suspiciously large effect deserves the same scrutiny as one that
produces none.


---

### D17 — Task identity is a *slot*, not a `TaskKind`
*Reverses D14. The task-count arm is now deliverable.*

D14 recorded the task-count arm as impossible and cut it. That was the right call at the
time — a correct fix needed a refactor across six modules — but it left the project
unable to answer the question it set out to ask: *how many live tasks can a model hold?*

**The change.** An `Op` now carries a `slot: int` alongside its `TaskKind`, and every
state bucket is keyed on `slot_key(task, slot)` — `shopping_0`, `shopping_1` — rather
than on the kind. `assign_slots` turns `[SHOPPING, SCHEDULE, SHOPPING]` into slots
`[0, 0, 1]`, giving three independent states where there were two.

**The part that matters more than the plumbing: disjoint vocabularies.** Each slot draws
from its own entity pool — grocery items, hardware items, pharmacy items; work meetings,
personal appointments. This is not decoration. The original two-task design got its
diagnostic power from using two *different kinds*, so a grocery item appearing in a
schedule was unmistakable. Two generic shopping lists would have destroyed that, and
`LIMITATIONS.md` said so. Naming the instances and separating their vocabularies
preserves the property while lifting the two-task cap: `screws` on the grocery list is
just as unmistakable as `milk` in a calendar.

Every surface template now names its list explicitly ("add milk to my **grocery list**"),
because with two lists of one kind an unnamed turn would be genuinely ambiguous — and an
ambiguous turn makes the answer key *unanswerable*, not merely hard. That is the D7
lesson applied before it could bite rather than after.

**Two bugs the refactor surfaced, both caught immediately:**

1. Nesting broke. Rendering walked one shared RNG in list order, so adding a noise turn
   changed the paraphrase chosen for every op after it. Each op now derives its own
   generator from a key stable under changes elsewhere in the list: its slot, kind,
   payload, and how many identical ops precede it.
2. The sweep crashed on its first row: `parsed` became a plain slot-keyed dict and
   `run.py` still called `.model_dump()` on it. Crashing loudly on row one is the good
   outcome; it is now handled and regression-tested.

**Cost, stated honestly.** v1 results are not reproducible from this code — every prompt
changed, so token counts and cache keys all differ. They are preserved under
`results/v1/` with their provenance rather than silently replaced.

---

### D18 — Cells specify a task *composition*, not just a task count

**Date:** 2026-09-01

**Context.** The v2 task-count arm produced its headline finding — misattribution rising
0 → 8 → 59 across 2, 3 and 4 tasks — and I nearly reported it as a task-count effect.
Checking the compositions first showed it is not cleanly one. `tasks_for(n)` deals from
`[SHOPPING, SCHEDULE]` round-robin, so the count of *same-kind pairs* runs 0, 0, 1, 2 as
n runs 1..4 — perfectly collinear with n across every cell in the sweep. Since kinds have
disjoint vocabularies, a same-kind pair is the only place misattribution can occur at all.
Task count and confusability were varying together and the design could not separate them.

**Decision.** A config cell may name its kinds explicitly (`tasks: [shopping, shopping]`)
instead of, or in place of, `n_tasks`. Added the `same_kind_2` cell to break the
collinearity.

**What this cost.** Three things had to change, none of them the config:

1. `run.py:resolve_tasks` accepts a count or an explicit list; `cell_tasks` reads either
   key off a cell.
2. `Conversation.cell` had to distinguish compositions a bare `t{n}` cannot. A kind
   signature is appended **only** when the composition differs from `default_tasks(n)`,
   so `t2_o6_n40` and friends keep their exact ids and the committed results stay
   addressable, while `[shopping, shopping]` becomes `t2shsh_o6_n40`. Signing every cell
   unconditionally would have been simpler and would have orphaned every result on disk.
3. Result rows now record `tasks` (the kinds), not just `n_tasks`. `--rescore` rebuilt
   the state machines from `tasks_for(row["n_tasks"])`, which silently assumes the
   canonical pool; on any explicit composition it would have graded against the wrong
   states. Old rows fall back to the count.

**Rejected: adding more task kinds instead.** Four distinct kinds would also break the
collinearity, and more directly. It needs a third and fourth state machine, vocabulary and
template set — roughly the same work as the original two — and it would answer a *different*
question (does confusability matter?) less sharply than the one-line same-kind cell answers
this one (is confusability what is actually driving the number I am about to report?).

**What it revealed immediately.** `cell.get("tasks", cell["n_tasks"])` evaluates its
default eagerly, so it raised `KeyError` on precisely the cells that supply `tasks` and no
`n_tasks` — every cell the feature exists for. Caught by `tests/test_run.py`, which was
written in the same change and which exists because of D14's lesson: the entry point had
no test coverage at all, and 543 green tests once coexisted with a CLI that crashed on its
first task-count cell.
