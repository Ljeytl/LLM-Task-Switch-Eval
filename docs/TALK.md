# Walkthrough prep

Sixty minutes. Roughly 20–25 of content, the rest cross-examination. Slides are
optional and the project is the focus, so lead with the repo open, not a deck.

---

## Six beats

### 1. The problem, and why I picked it (~90 seconds, no build-up)
People interleave tasks. Every deployed stateful assistant runs in that mode. The
benchmark landscape measures one task that evolves, or long-context retrieval, or
recall of one thread. GoodAI LTM does interleave — and confounds interleaving with
context length. Nobody isolates the switch.

Say "isolates," not "nobody measures it." The overstated version is easy to puncture
and someone in the room will know GoodAI.

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

**The task-count arm was undeliverable (the honest one).** Half the planned design was
2/3/4 concurrent tasks. Task identity is `TaskKind` and there are only two kinds, so
"3 tasks" resolved to `[shopping, schedule, shopping]` — the duplicate merged into the
first list's state, and blocked ordering emitted its turns twice. The token-match
assertion caught it. I cut it rather than refactor six modules mid-run, **so the
headline question the project set out to ask — how many live tasks can a model hold —
is not answered here.** Say that plainly; it is the biggest gap in the work and trying
to talk around it would be worse than owning it.

**The audit that was wrong twice (the one about method).** I wrote a checker that
verifies every emitted failure against the actual diff. It disagreed with the scorer
eight times. Six of those were the *checker's* bugs — it split identities on whitespace,
so "olive oil" became "olive". The other two were real: the scorer was calling stale
values hallucinations. A verifier is code too; a disagreement is a bug report against
whichever side is wrong, and you have to look rather than trust either one.

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
- Honest limitation: one synthetic domain, templated turns, no generalisation claim

### 6. Next
Generative turns is the experiment that would most change the conclusion. A same-family
scaling ladder to replace a cross-model comparison that is confounded three ways. A
second domain.

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
I do not know, and the honest answer is that I could not measure it. Task identity is
`TaskKind` with two kinds, so a third task repeated one and merged into its state — the
arm was degenerate and I cut it rather than refactor six modules mid-run. Doing it
properly needs per-instance identity, two separately named shopping lists, which is the
first item of next work. It is the biggest gap in the project.

**Would this transfer to a real product conversation?**
Unknown, and I do not claim it. This is an existence proof that switch cost is
separable from context length and mechanically measurable. Templated turns in one
synthetic domain are not a benchmark.

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
Generative turns, a same-family ladder, `n_pairs` at 200+, and a second domain. In that
order, because the first one is the only change that could move the *interpretation*
rather than just the error bars.
