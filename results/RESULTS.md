### How to read this

`len_medium` on `qwen2.5-coder:7b` is the **pre-registered primary comparison**. It was
chosen before the sweep ran and is reported uncorrected. Every other row is
**exploratory**: if one of them shows a larger effect than the primary, that is a
hypothesis to test, not a finding to report. Swapping the headline to whichever cell
came out strongest is exactly the practice pre-registration exists to prevent, and the
temptation is real precisely when the primary comes back null.

`ctrl_1task` is a **negative control**. With one task there is nothing to interleave, so
both orderings are the identical prompt and the delta must be exactly 0.0. A non-zero
value there would invalidate everything below it.


### Joint goal accuracy by condition

| model | condition | n pairs | blocked | interleaved | delta (pp) | 95% CI | McNemar b/c | p |
|---|---|---:|---:|---:|---:|---|---|---:|
| `gemma4:12b` | ctrl_1task *(control)* | 25 | 0.76 <sub>[0.57,0.89]</sub> | 0.76 <sub>[0.57,0.89]</sub> | +0.0 | [+0.0, +0.0] | 0/0 | 1.000 |
| `gemma4:12b` | len_short *(exploratory)* | 25 | 0.80 <sub>[0.61,0.91]</sub> | 0.88 <sub>[0.70,0.96]</sub> | +8.0 | [+0.0, +20.0] | 0/2 | 0.500 |
| `gemma4:12b` | len_medium *(exploratory)* | 25 | 0.88 <sub>[0.70,0.96]</sub> | 0.96 <sub>[0.80,0.99]</sub> | +8.0 | [+0.0, +20.0] | 0/2 | 0.500 |
| `gemma4:12b` | len_long *(exploratory)* | 25 | 0.96 <sub>[0.80,0.99]</sub> | 0.84 <sub>[0.65,0.94]</sub> | -12.0 | [-28.0, +4.0] | 4/1 | 0.375 |
| `gemma4:12b` | tasks_3 *(exploratory)* | 25 | 0.68 <sub>[0.48,0.83]</sub> | 0.64 <sub>[0.45,0.80]</sub> | -4.0 | [-24.0, +16.0] | 4/3 | 1.000 |
| `qwen2.5-coder:7b` | ctrl_1task *(control)* | 25 | 0.32 <sub>[0.17,0.52]</sub> | 0.32 <sub>[0.17,0.52]</sub> | +0.0 | [+0.0, +0.0] | 0/0 | 1.000 |
| `qwen2.5-coder:7b` | len_short *(exploratory)* | 25 | 0.84 <sub>[0.65,0.94]</sub> | 0.68 <sub>[0.48,0.83]</sub> | -16.0 | [-36.0, +0.0] | 5/1 | 0.219 |
| `qwen2.5-coder:7b` | len_medium **(PRIMARY)** | 25 | 0.64 <sub>[0.45,0.80]</sub> | 0.52 <sub>[0.33,0.70]</sub> | -12.0 | [-36.0, +12.0] | 6/3 | 0.508 |
| `qwen2.5-coder:7b` | len_long *(exploratory)* | 25 | 0.56 <sub>[0.37,0.73]</sub> | 0.44 <sub>[0.27,0.63]</sub> | -12.0 | [-36.0, +12.0] | 6/3 | 0.508 |
| `qwen2.5-coder:7b` | tasks_3 *(exploratory)* | 25 | 0.40 <sub>[0.23,0.59]</sub> | 0.20 <sub>[0.09,0.39]</sub> | -20.0 | [-36.0, -4.0] | 5/0 | 0.062 |
| `qwen2.5-coder:7b` | tasks_4 *(exploratory)* | 25 | 0.40 <sub>[0.23,0.59]</sub> | 0.12 <sub>[0.04,0.30]</sub> | -28.0 | [-48.0, -8.0] | 8/1 | 0.039 |

### Failure composition by ordering

| model | ordering | dropped | misattributed | absorbed | format | n failures |
|---|---|---:|---:|---:|---:|---:|
| `gemma4:12b` | blocked | 48% | 0% | 52% | 0% | 44 |
| `gemma4:12b` | interleaved | 44% | 0% | 56% | 0% | 39 |
| `qwen2.5-coder:7b` | blocked | 40% | 22% | 38% | 0% | 125 |
| `qwen2.5-coder:7b` | interleaved | 39% | 21% | 39% | 0% | 188 |

### What actually goes wrong

| failure mode | count | share |
|---|---:|---:|
| dropped / vanished | 154 | 38.9% |
| absorbed / hallucination | 87 | 22.0% |
| absorbed / false assertion | 70 | 17.7% |
| misattributed | 67 | 16.9% |
| absorbed / stale value | 10 | 2.5% |
| dropped / wrong value | 8 | 2.0% |

**Misattribution occurred, and the obvious reading of it is wrong.**

| condition | tasks | same-kind pairs | events | conversations affected |
|---|---:|---:|---:|---|
| ctrl_1task | 1 | 0 | 0 | 0/100 |
| len_short | 2 | 0 | 0 | 0/100 |
| len_medium | 2 | 0 | 0 | 0/100 |
| len_long | 2 | 0 | 0 | 0/100 |
| tasks_3 | 3 | 1 | 8 | 5/100 |
| tasks_4 | 4 | 2 | 59 | 23/50 |

Two things keep this from being a clean task-count effect. First, kinds have disjoint vocabularies, so a same-kind pair is the *only* place a misattribution can occur — and under the canonical composition the pair count rises in lockstep with the task count, so the two cannot be separated by the count cells alone. That is what the `same_kind_2` cell exists to break. Second, the events split 27 blocked / 40 interleaved: substantial misattribution happens in *blocked* ordering, where same-kind lists are never interleaved with each other at all. Similarity drives the bulk of it and ordering modulates it. The joint-accuracy delta above is a genuine ordering effect; this count largely is not.

### Validation counters

- **Token match:** 550/550 conversations exact (0 drifted), verified against Ollama's `prompt_eval_count`.
- **Parse rate:** 550/550 (100.0%) under schema-constrained decoding.
- **Per-task accuracy:** 0.766, clustered SE 0.0183 vs naive SE 0.0122 — ratio **1.50x** (the naive SE understates uncertainty).
- **Rows:** 550 conversations across 11 model-condition cells.
