# Limitations

Ordered roughly by how much they should change your reading of the result.

---

## 1. This measures extraction, not live tracking

Assistant turns are **prefilled** with a fixed `"Got it."` that the model never
generated. The model sees the whole conversation exactly once, at the end, and is asked
to report state.

That is what makes the token match provable — both orderings become the identical
multiset of strings — but it changes what is being measured. A deployed assistant
responds to each turn, and its own replies enter the context and may act as a scratchpad
that helps it track state. **This design removes that scratchpad.** The number here is
the cost of interleaving for a model reading a transcript, which is a lower bound on
capability and possibly an upper bound on switch cost.

The generative variant is the single most valuable next experiment. It was not run
because it costs ~10× the compute and would not fit the window.

---

## 2. One synthetic domain, templated turns

Turns come from four-plus paraphrase templates per operation. Real users are messier,
more elliptical, and more ambiguous. A model could in principle pattern-match the
templates rather than track state; seed-randomised paraphrase selection makes that
harder but does not rule it out.

No claim of generalisation to real conversations is made or supported.

---

## 3. Cross-model comparison is descriptive, not causal

`qwen2.5-coder:7b` and `gemma4:12b` differ in **family, instruction tuning, and
parameter count at the same time**. There is no isolated variable across them and no
scaling ladder.

The primary result is unaffected — blocked vs interleaved is paired *within* a single
model — but any read of "the bigger model was better" is confounded three ways and
should not be taken as evidence about scale. `qwen2.5-coder` is additionally a
code-tuned model being asked to do a natural-language tracking task.

---

## 4. Noise is a difficulty driver, not neutral padding — and I calibrated the wrong thing

Context length is varied by adding **noise turns**, so "more context" and "more
task-irrelevant content" move together. A result showing switch cost rising with length
cannot distinguish "the model struggles with distance" from "the model struggles with
distractors."

The run made clear this is not a minor caveat — **for one of the two models**. On
`qwen2.5-coder:7b`, blocked joint accuracy fell from **0.90 at zero noise to 0.47 with 40
noise turns**, a 43-point drop from padding alone, more than double the largest switch
cost measured anywhere in the sweep. On `gemma4:12b`, the identical padding on the
identical seeds moved accuracy from **0.90 to 0.97** — no cost at all.

I wrote the general version of this claim ("noise is a difficulty driver, not neutral
padding") from the qwen data before gemma4 had run. It is not general. Noise sensitivity
is a property of the model, and a single-model result would have shipped it as a property
of the design.

**The v2 refactor invalidated the calibration a second time.** `n_ops = 6` was measured
against v1 templates. v2 names the target list in every turn ("add milk to my grocery
list"), lengthening every turn and the system prompt, and it also nested the noise pools —
so two things changed at once.

Difficulty moved **non-uniformly**, which is the part that matters:

| cell | v1 blocked | v2 blocked |
|---|---:|---:|
| `ctrl_1task` | 0.600 | 0.320 |
| `len_short` | 0.900 | 0.840 |
| `len_medium` | 0.467 | 0.640 |

The control got harder; the primary got *easier* and moved into the 0.6–0.8 band it was
supposed to be in. So "v2 is harder" — which is what the control alone suggested — is
wrong, and any single-cell read of this table would have been wrong too.

Nothing here is attributable, because template wording and noise nesting changed together
and v1 cannot be re-run under v2 code. What it does establish is that the calibration does
not transfer across a change of this size. Re-running `--calibrate` against the current
templates before the next sweep is the fix, and this is the same mistake as the one below
in a different costume: calibrating one thing, then changing another underneath it.

**The original methodological error, kept because the pattern repeated.** `run.py --calibrate` sweeps
`n_ops` at `n_noise = 0`, and I picked `n_ops = 6` from that curve. But difficulty is
driven by *both* knobs, so the primary cell — which adds 40 noise turns on top — landed
at 0.47 blocked rather than in the 0.6–0.8 band I had calibrated for. It is still
measurable, so the sweep is not wasted, but the calibration should have been run over
the (`n_ops`, `n_noise`) *pair* rather than one knob with the other pinned at zero.
Calibrating one dimension of a two-dimensional difficulty surface tells you about a
slice, not the surface.

The alternative — driving length with more operations — would have confounded length
with state-update load instead, which is worse, because that is the confound the whole
project exists to remove. There is no clean third option within this design.

**And it was calibrated against one model.** `--calibrate` was run on
`qwen2.5-coder:7b` only; `n_ops = 6` is the value that put *that* model in the 0.6–0.8
band. `gemma4:12b` then ran the same cells at the same difficulty setting and sat far
higher. A single `n_ops` cannot put two models of different capability in the same band,
so the cross-model rows are not comparing them at matched difficulty — they are comparing
them at matched *task parameters*, which is a different and weaker thing.

This does not invalidate either model's within-model contrast, which is what the paired
design measures and which is unaffected by where the model sits on the difficulty curve.
It does mean a null for the easier model is ambiguous between "this model does not pay a
switch cost" and "this task is not taxing enough for this model to pay one." Per-model
calibration — one `n_ops` per model, chosen to land each in the same accuracy band — is
the fix. The harness already supports it (`--calibrate` takes `--model`); the gap is that
I ran it once, on qwen, and then reused that `n_ops` for both:

```bash
python run.py --calibrate --model gemma4:12b     # never run
```

so this costs a calibration pass, not a redesign. I did not do it, and the cross-model
reading has to carry the caveat.

---

## 4b. The length arm varies noise *placement* as well as quantity

`_subrng(seed, task, "mix", n_noise, n_false)` includes `n_noise` in the RNG key, so the
40-noise and 120-noise conditions for the same seed draw **entirely different noise
placements** rather than nested ones. The conditions are not "the same conversation with
more padding added"; they are different padding draws of different sizes.

The consequence is visible in the data. Across noise levels, 19 of 30 seeds change their
blocked outcome, and five of them go 1 -> 0 -> 1: correct with no padding, wrong at 40,
correct again at 120. Additive difficulty cannot produce that pattern; a fresh random
placement per condition can. The cell base rates (0.90 / 0.47 / 0.67) are also
statistically indistinguishable between the last two — Wilson [0.30, 0.64] and
[0.49, 0.81] overlap heavily.

**What this does and does not affect.** The *within-cell* blocked-vs-interleaved contrast
is unaffected: both orderings inside a cell share one op sample, one rendering pass and
one noise placement, which is the pairing the whole design rests on. What is affected is
*between-cell* comparison — any read of a trend across noise levels carries placement
variance on top of quantity.

**Fixed in v2.** `n_noise` was removed from the mix-RNG key, one pool of `MAX_NOISE_POOL`
positions is drawn per slot, and each condition takes a prefix — so a longer condition is
now a strict superset of a shorter one. Two subtleties had to be fixed alongside it: the
extras list was shuffled (which reassigned positions whenever it grew), and rendering
walked one shared RNG in list order (which changed paraphrases for everything after an
inserted turn). Both are covered by
`test_noise_conditions_are_nested_not_independent_draws` and
`test_mutating_stream_is_invariant_to_noise_level`.

---

## 4c. Task count and operations-per-task move inversely

`n_ops` is total per conversation, so raising the task count *lowers* the operations per
task. A reader who takes the task-count arm as "adding a task costs X" will be wrong; it
measures splitting a fixed amount of work across more live states, at a fixed token
count.

Visible in the data as the 1-task control scoring below the 2-task cell (0.320 vs 0.640):
the control is one six-item list, not a simpler condition.

Holding ops-per-task constant instead would re-confound task count with conversation
length (the D4 error). Both cannot be held at once, and this design says which it gave up.

---

## 5. The task-count arm is now run, but the instances are synthetic

v1 had no task-count result at all; task identity was `TaskKind` and a third task merged
into the first (D14). v2 keys state on per-instance slots with disjoint vocabularies
(D17), so 1-4 concurrent tasks are genuinely independent and misattribution between two
lists of the *same kind* is detectable.

What remains synthetic: the instances are distinguished by an explicit name in every
turn ("add milk to my **grocery list**"). Real users are far more elliptical — they say
"add milk" and expect the assistant to infer which list from context. This design
therefore measures tracking with the *routing problem removed*, which is easier than the
real thing. Making the reference implicit is the natural follow-up and would likely
raise misattribution sharply.

---

## 5b. Misattribution tracks same-kind *pairs*, which this design cannot separate from task count

This is the sharpest confound in the new arm, and it undercuts the obvious reading of the
headline number.

`tasks_for(n)` deals from `TASK_POOL = [SHOPPING, SCHEDULE]` round-robin, so the number of
*same-kind pairs* — the only pairs between which misattribution is even possible, since
the vocabularies of different kinds are disjoint — rises in lockstep with task count:

| tasks | slots | same-kind pairs | misattribution events |
|---:|---|---:|---:|
| 1 | `shopping_0` | 0 | 0 |
| 2 | `shopping_0, schedule_0` | 0 | 0 |
| 3 | `+ shopping_1` | 1 | 8 |
| 4 | `+ schedule_1` | 2 | 59 |

Pair count and task count are perfectly collinear across the cells actually run. So the
claim the data supports is **"misattribution requires a confusable neighbour,"** not
"misattribution grows with the number of concurrent tasks." Those are different claims
and I can only demonstrate the first. A 4-task conversation over four *distinct* kinds
might show none at all.

**The experiment that separates them is one config line, and it is now shipped.** The
`same_kind_2` cell is `tasks: [shopping, shopping]` — two tasks, one same-kind pair —
which breaks the collinearity:

| cell | tasks | same-kind pairs |
|---|---:|---:|
| `len_medium` | 2 | 0 |
| `same_kind_2` | 2 | **1** |
| `tasks_3` | 3 | 1 |

`len_medium` and `same_kind_2` hold task count fixed and vary pair count; `same_kind_2`
and `tasks_3` hold pair count fixed and vary task count. Misattribution in `same_kind_2`
means similarity is the driver and the count is incidental; none means the count matters
in its own right.

Making this expressible needed a real change, not just a config edit — cells were
specified by *count* only, and the cell id `t{n}_o{ops}_n{noise}` could not tell
`[shopping, schedule]` from `[shopping, shopping]` (D18). The library never assumed
distinct kinds; `build_states`, `order_ops` and the scorer all key on slot, so the
constraint was entirely in the CLI.

### Interleaving is not the main cause of misattribution

Splitting the events by ordering is deflationary in a way worth saying out loud:

| cell | blocked | interleaved |
|---|---:|---:|
| 3 tasks | 2 events (1/25 convs) | 6 events (4/25 convs) |
| 4 tasks | 25 events (11/25 convs) | 34 events (12/25 convs) |

Misattribution is substantial in *blocked* ordering, where the two shopping lists are
never interleaved with each other at all. Interleaving adds roughly a third on top. So the
mechanism is mostly "two similar states in one context bleed together," with ordering as a
secondary modulator — **not** "switching between them causes the bleed." The joint-accuracy
delta remains a genuine ordering effect; the misattribution count largely is not, and
reporting it as one would overclaim.

---

## 5c. The negative control is weaker than it looks

`ctrl_1task` reports delta **+0.0pp** with **0/25** discordance on both models, and that
reads as a strong validation. Most of it is true by construction.

With one task there is nothing to interleave, so `order_ops` returns the same sequence for
both orderings and the two conversations render **byte-identical prompts** (verified: the
turn lists hash equal at `n_tasks=1` and differ at `n_tasks=2`). The response cache is
keyed on the rendered prompt, so the second call is a cache hit on the first. One
inference happens and its result is compared against itself.

So the control establishes:

- that a one-task conversation really has no ordering degrees of freedom — a genuine
  property of the generator, and the thing most likely to break if `order_ops` were
  wrong;
- that the pipeline runs end to end and the scorer agrees with itself.

It does **not** establish that the model is stable under repetition, because the model was
asked once. A reviewer reading `+0.0pp, 0/25` as evidence of run-to-run determinism would
be reading more into it than is there, and I presented it that way before checking.

`tools/determinism_check.py` closes the gap: it runs both halves of a one-task pair with
the cache **disabled**, so two independent calls receive the identical prompt, and reports
how often they agree. That is the number that bounds how much of a measured delta could be
noise. It asserts the byte-identity premise before measuring, so it cannot silently
degrade into measuring ordering instead.

---

## 6. Bit-exact reproducibility is not achievable

Batch-invariant kernels do not exist on Metal, so identical inputs can produce slightly
different logits depending on batching. Seeds, model tags and digests are logged, and
results are reported as intervals rather than point estimates. No claim of exact
reproduction is made.

---

## 7. Normalisation hides real formatting differences

Comparison is case-insensitive, whitespace-trimmed, and canonicalises `9:00` to `09:00`.
This is defensible — the question is state tracking, not capitalisation — but it does
mean a model that tracks perfectly and formats erratically scores identically to one
that does both well. Anyone who cares about output discipline should read the FORMAT
bucket rather than the accuracy number.

---

## 8. The constrained-decoding check is underpowered and should be read as such

Schema-constrained decoding on the final turn could itself degrade accuracy; published
estimates run as high as 8.7 points. `run.py --check-constrained` measures it rather
than assuming it away, comparing constrained decoding against free-form decoding
followed by a separate extraction call (`runner.extract_state`), so that reasoning and
serialisation are separated.

Measured on `qwen2.5-coder:7b`, n=12 conversations:

| arm | joint accuracy | parse rate |
|---|---|---|
| constrained | 0.667 (8/12) | 1.000 |
| free-form + extraction | 0.583 (7/12) | 1.000 |

**That is a one-conversation difference.** The headline "+8.3pp" is not a finding; at
n=12 it is indistinguishable from noise, and it happens to be almost exactly the
magnitude of the published concern while pointing the opposite way — which is a good
reason to treat it as noise rather than as counter-evidence.

What the check does establish: constrained decoding is not catastrophically harmful
here, and the free-form arm is measurable at all only because of the extraction pass
(parse rate went from 0.000 to 1.000 once prose answers were converted rather than
discarded). What it does not establish: that constraining is free. Ruling out an
8.7-point effect would need roughly an order of magnitude more conversations, and it was
measured on one condition rather than all of them.

---

## 9. Statistical power — the sweep is badly underpowered, and here is by how much

This is not a hedge; it is simulated from the discordance actually observed
(`tools/power_analysis.py`, output in `results/POWER.md`). At the sweep's n=30 pairs per
cell, exact McNemar at alpha=0.05 with the observed 27% discordance rate:

| true effect | n=30 | n=60 | n=120 | n=240 | n=480 |
|---|---:|---:|---:|---:|---:|
| 8.2 pp | 6% | 18% | 36% | 64% | 94% |
| 13.7 pp | 20% | 44% | 79% | 99% | 100% |
| 19.2 pp | 39% | 79% | 99% | 100% | 100% |

**At n=30 this design has ~6% power against an 8-point effect and ~20% against a
14-point one.** It cannot reliably detect anything smaller than roughly 20 points.

The consequence for reading the results: the pre-registered primary cell came back null,
and that null carries almost no information. Its discordant counts were b=6, c=5 —
interleaving broke six conversations and fixed five. That is not "ordering did not
matter"; it is two effects cancelling in a sample too small to separate them.

Reaching 80% power against a 14-point effect needs roughly **120 pairs per cell**, four
times what was run. That is the single cheapest improvement available and it is the
first thing more compute should buy.

---

## 10. The taxonomy has genuine edge cases

`ABSORBED` covers both planted counterfactuals and free-standing hallucinations; they
share a bucket because both are ungrounded content entering state, and are separated
only by the detail string. `DROPPED` covers both "vanished entirely" and "present with
the wrong value." These are defensible groupings, not obvious ones, and a different
analyst would reasonably split them differently.


---

## 11. `wall_seconds` in the result rows is not a clean benchmark

The constrained-vs-free-form check was launched while the main sweep was still running,
so the two contended for the GPU. Accuracy is unaffected — decoding is greedy and
deterministic, and the response cache is keyed on the prompt and parameters, not on
timing — but the `wall_seconds` column in `results/sweep.jsonl` is inflated for any row
that overlapped, and should not be read as a throughput measurement.

Clean per-model timings are the ones recorded separately during the pre-flight
benchmark: prefill ~274-348 tok/s and generation ~16-21 tok/s on `qwen2.5-coder:7b`,
measured with nothing else on the GPU.
