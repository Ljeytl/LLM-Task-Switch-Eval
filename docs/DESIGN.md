# Design: why this is worth measuring

## The problem

People do not use assistants one task at a time. In a single conversation a user will
start a list, jump to a schedule, come back to the list, correct something, and drop an
aside that is not a task at all. Every deployed stateful assistant runs in this mode.

The question someone building such an assistant actually has is: **how many live tasks
can this model hold, and what does switching between them cost me?**

## What the existing benchmarks measure instead

| Benchmark | What it measures | Why it does not answer the question |
|---|---|---|
| MultiChallenge, Multi-IF, MT-Bench-101 | A single task that *evolves* over turns | One task, not several concurrent ones |
| RULER, needle-in-a-haystack | Long-context retrieval | Retrieval of static content, not maintained state |
| LoCoMo, LongMemEval | Long-term recall of one thread | One thread |
| **GoodAI LTM** (NeurIPS 2024) | Interleaved concurrent tasks | Closest prior art — but varies interleaving *and* context length together |

The intended contribution is deliberately narrow. GoodAI LTM already measures
interleaved concurrent tasks and finds that they are harder. This project builds a
mechanically graded paired instrument whose two arms contain the same rendered strings
and matched token counts. The current run is a prototype, not a novelty or causal-isolation
claim: fixed task order still leaves serial position and recency unresolved.

## What this measures

Given N concurrent stateful tasks in one conversation, the accuracy difference between
blocked and interleaved orderings while holding operation content and token count fixed.

### The manipulation

One operation sequence, two orderings:

- **Blocked** — all task-1 operations, then all task-2, then all task-3
- **Interleaved** — round-robin across tasks

Identical operations. Identical token count, asserted by construction and verified at
run time against the model's own `prompt_eval_count`. The delta is an ordering effect,
not context bloat. Counterbalancing task order and update recency is still required
before attributing it solely to switching and re-entry.

### Crossing rather than holding fixed

Holding context length constant licenses a claim at exactly one length. This design
**crosses** interleaving with context length instead — an L-shaped grid with a length
arm and a task-count arm sharing a corner — so the result is not a single number but a
shape: does switch cost grow with the amount of surrounding context, or with the number
of live states?

A flat length pattern would motivate the hypothesis that the ordering effect tracks live
state rather than distance. At this sample size it would not establish that interaction.

### Secondary probe

Counterfactual assertions injected mid-conversation ("I thought about adding rice, but
decided against it") that are *not* operations. Measures whether ungrounded context
contaminates tracked state. The phrasing matters: "I put rice on the list" is a
legitimate instruction in ordinary conversation, so scoring a model wrong for honouring
it would be scoring our own prompt.

## Why the ground truth can be mechanical

Two tasks with independently diffable state:

| Task | State type | Operations |
|---|---|---|
| Shopping list | set | add, remove, query membership |
| Meeting schedule | ordered map | insert at time, move, remove, query |

A plain state machine consumes the same operation sequence the generator emitted and
produces the expected final state. Transcript and answer key come out of one function
as a pair. No annotation, no judge model, no subjectivity — grading is a diff.

MultiChallenge had to abandon rule-based grading because "there is no single
ground-truth answer for most test conversations." Constraining the domain to structured
state buys that back. The two task types were chosen so **misattribution is visible**: a
grocery item appearing in the schedule is unambiguous, whereas two shopping lists would
not be.

## Failure taxonomy

Every discrepancy classifies into one bucket, and which one dominates is more
interesting than the accuracy number:

1. **Dropped** — an operation never landed anywhere
2. **Misattributed** — an operation landed in the wrong task's state
3. **Absorbed** — ungrounded content entered state. Three sub-cases kept separable by
   the detail string: `false_assert` (a planted counterfactual treated as real),
   `stale` (a value that *was* true and was never retracted — almost always a dropped
   REMOVE or UPDATE), and `hallucination` (never mentioned at all)
4. **Format** — the response did not parse or validate

Format failures are logged separately and excluded from state accuracy, so a formatting
problem is never reported as a tracking problem.

"Forgot an item" and "filed it under the wrong task" are different engineering
problems with different fixes. An aggregate accuracy number cannot tell you which one
you have; the taxonomy comes free from the same diff.

## The finding the two-task design could not produce

v1 recorded **zero misattribution in 480 conversations** and could not interpret it. Two
readings were available and it had no way to separate them: either models do not misfile
operations across tasks, or two tasks of very different kinds are simply too dissimilar
to confuse.

v2's task-count arm probes it. For qwen, `tasks_3` is the first canonical condition
containing **two lists of the same kind** — a grocery list and a hardware list, with
disjoint vocabularies so a confusion is unmistakable:

| cell | same-kind pair? | conversations | misattribution events |
|---|---|---:|---:|
| qwen 2-task different-kind cells | no | 150 | **0** |
| `tasks_3` (2 shopping + 1 schedule) | yes | 50 | **8** |

Every event is `shopping_0` ↔ `shopping_1`: grocery items landing on the hardware list.
None cross between a list and a calendar.

So the v1 zero was not evidence that the models never misfile. A same-kind neighbour was
sufficient to expose misattribution here. That does not establish that similarity is
necessary or isolate it from changed task mechanics.

## What this does not show

One synthetic domain with templated turns is not a benchmark. It is a prototype showing
that a matched-token ordering effect is mechanically measurable. Fixed blocked-task order
still confounds that effect with serial position and recency, and generalisation to real
conversations is untested. See `LIMITATIONS.md`, which is deliberately longer than this
section.
