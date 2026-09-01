# Walkthrough prep

Sixty minutes. Roughly 20–25 of content, the rest cross-examination. Slides are
optional and the project is the focus, so lead with the repo open, not a deck.

---

## Six beats

### 1. The problem, and why I picked it (~90 seconds, no build-up)
People interleave tasks. Every deployed stateful assistant runs in that mode. GoodAI LTM
already shows that interleaving makes long conversations harder. My intended step was a
paired, mechanically graded comparison with the same operations and matched tokens.

Say “matched-token ordering instrument,” not “nobody measures switching” or “this fully
isolates switch cost.” Fixed task order leaves serial position and recency unresolved,
and that is the first limitation to volunteer.

### 2. Walkthrough
Open `docs/WALKTHROUGH.md`. Show one generated transcript beside its ground truth, then
the same operation list in both orderings with identical token counts. The design is
legible in about thirty seconds if you show the pair rather than describe it.

Then `state.py` — this is the oracle, it is 90 lines, and it is why there is no judge
model.

### 3. Approach and deliberate omissions
- Mechanical grading over an LLM judge, and what that cost (D1)
- Bespoke over `inspect_ai`, because the state machine, scorer and paired design are
  the parts worth owning (D11)
- Prefilled acks: what it bought (a provable token match, a 10× cheaper sweep) and what
  it cost (this measures extraction, not live tracking) (D3)
- Cut on purpose: the expenses task, the 32B ceiling, Phase 2 entirely

### 4. The dead end — **required beat, and there are four real ones**

Pick two to tell in full. The schema one is the most striking; the task-count one is
the most honest.

**The schema ambiguity (the good one).** First live run, the schedule schema said
"array of two strings" and never said which was the time. The model returned
`[title, time]` in the blocked condition and `[time, title]` in the interleaved one.
The scorer read that as six tracking failures in a conversation the model had tracked
perfectly.

The lesson is not "validate your schema." It is that **an ambiguous instrument is not
merely noisy — if the ambiguity resolves differently under different conditions, it
becomes a systematic effect** pointing wherever the ambiguity leans. I would have
shipped a large, clean, fictitious switch cost.

**The shared RNG (the subtle one).** A single `Random` threaded through every task meant
that generating noise for task 1 shifted task 2's entire operation stream — so the
length arm was silently varying the operations too. Same class of confound the project
exists to remove, one level down, in my own generator. Caught by a test that asserted
the invariant rather than the output.

**The task-count arm: cut, then rebuilt (the one that shows judgement changing).** Half
the planned design was 2/3/4 concurrent tasks. Task identity was `TaskKind` and only two
kinds existed, so "3 tasks" resolved to `[shopping, schedule, shopping]` — the duplicate
merged into the first list's state and blocked ordering emitted its turns twice. The
token-match assertion caught it. I cut it rather than refactor six modules mid-run and
shipped v1 without it.

Then I rebuilt it. Task identity became a **slot**, and — the part that actually matters
— each slot got its own **disjoint vocabulary**. The original design's diagnostic power
came from using two different *kinds*, so a grocery item in a calendar was unmistakable;
two generic lists would have destroyed that, and my own limitations doc said so. Naming
the instances preserves the property while lifting the cap.

The tail on this one is the better story: I lifted the cap in the library, left a stale
guard in the CLI, and the sweep crashed on the first task-count cell. **543 green tests
and a broken entry point**, because every test called `build_pair` directly and nothing
exercised `tasks_for`. The guard's error message even cited D14, which D17 had already
reversed — a stale assertion is worse than none, because it reads as deliberate.

**The audit that was wrong twice (the one about method).** I wrote a checker that
verifies every emitted failure against the actual diff. It disagreed with the scorer
eight times. Six of those were the *checker's* bugs — it split identities on whitespace,
so "olive oil" became "olive". The other two were real: the scorer was calling stale
values hallucinations. A verifier is code too; a disagreement is a bug report against
whichever side is wrong, and you have to look rather than trust either one.

### 4b. The finding that only exists because I rebuilt the cut arm

Worth its own beat, because it is the payoff for the D14 → D17 reversal.

v1 reported **zero misattribution across 480 conversations**. I wrote that up as the most
interesting result — the failure the whole taxonomy was designed to catch never happened
— and honestly flagged that I could not tell whether models simply do not misfile, or
whether a shopping list and a calendar are too dissimilar to confuse.

v2 probes it. For qwen, `tasks_3` is the first canonical condition with **two lists of
the same kind**:

| | same-kind pair? | conversations | misattribution |
|---|---|---:|---:|
| qwen 2-task different-kind cells | no | 150 | **0** |
| `tasks_3` | yes | 50 | **8** |

Every event is a grocery item landing on the hardware list. None cross between a list and
a calendar.

**The v1 zero did not establish that the models never misfile.** The same-kind probe
exposed a failure mode the dissimilar-task instrument did not observe. An eval that never
produces a failure mode has not shown the failure mode is absent.

The defensible claim is narrower: a same-kind neighbour was sufficient to expose qwen
misattribution in this instrument. The stronger task-count and necessity claims do not
survive audit.

### 4c. …and then I checked whether that finding was what it looked like. It wasn't.

**Lead with this if there is time for only one methodology beat.** It is the strongest
evidence of how I work, because it is me auditing my own headline after I already liked it.

The task-count arm gave a clean monotone story — misattribution **0 → 8 → 59** across 2,
3 and 4 tasks, and joint-accuracy delta **−12 → −20 → −28pp**. I had the sentence written:
switch cost scales with the number of live states.

Then I looked at what the compositions actually were. `tasks_for(n)` deals from
`[SHOPPING, SCHEDULE]` round-robin:

| tasks | slots | same-kind pairs | misattribution |
|---:|---|---:|---:|
| 2 | shopping_0, schedule_0 | 0 | 0 |
| 3 | + shopping_1 | 1 | 8 |
| 4 | + schedule_1 | 2 | 59 |

Kinds have **disjoint vocabularies**, so same-kind neighbours make cross-slot
misattribution easier to observe — and pair count is perfectly collinear with task count
across every cell I ran. The number is real. The *explanation* I was about to attach to it
was not identified.

Two things I did about it:

1. **Reported the weaker claim.** A same-kind neighbour was sufficient to expose
   misattribution here; the data establish neither necessity nor a task-count effect.
2. **Split the entries by ordering.** Interleaved transcripts contain 50 and blocked
   transcripts 29, but those are clustered descriptive counts without an interval or
   test. They motivate a follow-up; they do not establish causal amplification.

**And built the experiment that probes it** (D18): `same_kind_2` is
`[shopping, shopping]` — two tasks, one same-kind pair. Against `len_medium` it holds task
count fixed and varies pair count; against `tasks_3` it holds pair count fixed and varies
task count. **It resolved cleanly:**

| qwen cell | tasks | same-kind pairs | misattribution |
|---|---:|---:|---:|
| `len_medium` | 2 | 0 | 0 |
| `same_kind_2` | 2 | 1 | **12** |
| `tasks_3` | 3 | 1 | 8 |
| `tasks_4` | 4 | 2 | 59 |

Pairs fixed, tasks 2 → 3: **12 → 8**, down. Tasks fixed, pairs 0 → 1: **0 → 12**.
That is enough to reject my original monotone task-count explanation, not enough to prove
that similarity is necessary or task count has no effect. The entries are clustered
within conversations and the compositions differ.

It also produced the sweep's largest blocked-accuracy difference: replacing the calendar
with a second shopping list takes qwen from 0.640 to **0.000**. That changes task mechanics
and semantics together, and the cell is on the floor, so neither a clean similarity effect
nor its ordering difference is identified. `gemma4:12b` is not at the floor
(0.880 → 0.720) and logs zero misattribution across its 350 conversations.

*If asked "why not just add more task types?"* — four distinct kinds would break the
collinearity too, but it needs two more state machines, vocabularies and template sets, and
answers a different question less sharply. The one-line cell answers the question I
actually have, which is whether the number I am about to report means what I think.

*If asked what it cost* — not a config edit. Cells were specified by count only, and the
cell id `t{n}_o{ops}_n{noise}` could not distinguish `[shopping, schedule]` from
`[shopping, shopping]`. The signature is appended only for non-canonical compositions so
committed result ids stay byte-identical; and rows now record their kinds, because
`--rescore` rebuilt states from the count and would have graded an explicit composition
against the wrong answer key.

### 5. Quality, and where it falls short
- Token match verified against Ollama's own `prompt_eval_count`, not estimated
- Calibration before the sweep: the originally planned `n_ops=24` sat at **0.00** blocked
  accuracy — the whole sweep would have returned a null for reasons unrelated to switching
- Unit of analysis is the conversation; the naive SE is printed next to the clustered
  one so the gap is visible
- The taxonomy grounding audit found a real mislabelling that **no accuracy number
  would ever have surfaced** — stale values (a dropped REMOVE) reported as
  hallucinations. Re-scoring changed the labels and 0 outcomes, which is the correct
  behaviour and shows diagnosis and scoring are properly separated.
- The task-count headline is reported with its confound attached, in the results table
  itself rather than in a footnote — see beat 4c
- The primary is inconclusive, and `tasks_4` raw p=0.039 becomes Bonferroni-adjusted
  p=0.430 in the 11-test exploratory family
- Fixed blocked-task order confounds switching with serial position and recency; the
  current run is an ordering-effect prototype, not a final causal estimate
- Honest limitation: one synthetic domain, templated turns, no generalisation claim

### 6. Next
First counterbalance task order and clean the ambiguous imperative templates. Then run
generative turns, a same-family scaling ladder, and a second domain.

---

## Questions to have answers ready for

**Why not an LLM judge?**
Because the domain was chosen so I would not need one. Structured state is
independently diffable, so the transcript and the answer key come out of one function.
A judge would add cost, latency, and a second model's failure modes to a measurement
whose whole point is that grading is unambiguous. The cost is that the tasks are much
simpler than real assistant work, and I say that in the limitations.

**How do you know the token match is real?**
Two ways. By construction: with prefilled acks the two orderings are the identical
multiset of strings in a different sequence, so they cannot tokenise differently.
And empirically: every pair's actual `prompt_eval_count` from Ollama is compared, drift
is reported, and any drifting pair is excluded. It ran at zero drift.

**Why these models and not a frontier model?**
Honestly: these were what was installed and what fits 24 GB. `qwen3.6` (36B) and
`qwen3-coder` (30.5B MoE) do not leave room for a KV cache. And I should be clear that
the two I used differ in family, tuning *and* size at once, so there is no scaling
claim — the primary result is paired within a single model and does not depend on the
comparison.

**Why is `n_ops` only 6?**
Because I measured it. The difficulty curve for qwen2.5-coder:7b runs 0.62 / 0.38 /
0.12 / 0.00 at 6 / 12 / 18 / 24 ops. My original plan used 24, which is the floor. The
calibration step exists precisely to catch that, and it did.

**What happens at 4 tasks?**
v1 could not measure it; v2 can. Task identity is now a slot with a disjoint vocabulary
per instance, so four concurrent tasks are four independent states and misattribution
between two same-kind lists is detectable. Be precise about what the arm asks though:
`n_ops` is *total*, so fewer tasks means more ops per task. It measures splitting a fixed
amount of work across more live states at a fixed token count — not the marginal cost of
adding a task. That is the unavoidable price of not re-confounding task count with
conversation length.

There is a *second* confound in the same arm, and I would raise it before being asked:
same-kind pairs rise in lockstep with task count under the canonical composition
(0, 0, 1, 2 for 1-4 tasks), while same-kind neighbours make cross-slot misattribution
easier to observe. So the arm cannot separate "more tasks" from "a confusable neighbour".
The `same_kind_2` cell breaks the simple collinearity but also changes task mechanics and
exposure — see beat 4c.

**Is the switch-cost story just the misattribution story?**
No, and keeping them apart matters. Joint accuracy falls under interleaving in cells with
*zero* same-kind pairs (`len_short` -16.0pp, `len_medium` -12.0pp, both two dissimilar
tasks), so the ordering effect exists without any possibility of misfiling. Conversely
misattribution is substantial under *blocked* ordering (25 of 59 events at 4 tasks), where
same-kind lists are never interleaved with each other. Two distinct mechanisms that the
task-count cells happen to vary together: interleaving costs accuracy, and similarity
costs attribution.

**Would this transfer to a real product conversation?**
Unknown, and I do not claim it. This is a prototype showing that a paired matched-token
ordering comparison can be mechanically graded. Templated turns in one synthetic domain,
plus unresolved recency, are not a benchmark or a causal switch-cost estimate.

**Derive McNemar.**
See `docs/STATS.md` §3. Lay out the 2×2; `a` and `d` carry no information about
ordering; under the null each discordant pair is a coin flip, so `b ~ Binomial(b+c, ½)`.
I use the exact binomial rather than the chi-square approximation because the
approximation misbehaves when `b+c` is small, which is exactly where a modest real
effect lives.

**Which part did the model write, which part did you?**
The design, the confound analysis, the taxonomy, the decision to pair, the choice of
estimators, and every judgement recorded in `DECISIONS.md`. Claude Code wrote
implementation bodies against those decisions and caught two of my own bugs by writing
tests that asserted my invariants. I read every diff. See the README's Tools section.

**What would you do with a week?**
Counterbalanced order and cleaned templates, generative turns, `n_pairs` at 200+, a
same-family ladder, and a second domain. The first two protect the interpretation; the
larger sample changes the error bars.
