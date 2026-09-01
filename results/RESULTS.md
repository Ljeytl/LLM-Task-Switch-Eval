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
| `gemma4:12b` | ctrl_1task *(control)* | 30 | 0.93 <sub>[0.79,0.98]</sub> | 0.93 <sub>[0.79,0.98]</sub> | +0.0 | [+0.0, +0.0] | 0/0 | 1.000 |
| `gemma4:12b` | len_long *(exploratory)* | 30 | 0.83 <sub>[0.66,0.93]</sub> | 0.80 <sub>[0.63,0.90]</sub> | -3.3 | [-16.7, +10.0] | 3/2 | 1.000 |
| `gemma4:12b` | len_medium *(exploratory)* | 30 | 0.97 <sub>[0.83,0.99]</sub> | 0.93 <sub>[0.79,0.98]</sub> | -3.3 | [-13.3, +6.7] | 2/1 | 1.000 |
| `gemma4:12b` | len_short *(exploratory)* | 30 | 0.90 <sub>[0.74,0.97]</sub> | 0.90 <sub>[0.74,0.97]</sub> | +0.0 | [+0.0, +0.0] | 0/0 | 1.000 |
| `qwen2.5-coder:7b` | ctrl_1task *(control)* | 30 | 0.60 <sub>[0.42,0.75]</sub> | 0.60 <sub>[0.42,0.75]</sub> | +0.0 | [+0.0, +0.0] | 0/0 | 1.000 |
| `qwen2.5-coder:7b` | len_long *(exploratory)* | 30 | 0.67 <sub>[0.49,0.81]</sub> | 0.33 <sub>[0.19,0.51]</sub> | -33.3 | [-53.3, -10.0] | 12/2 | 0.013 |
| `qwen2.5-coder:7b` | len_medium **(PRIMARY)** | 30 | 0.47 <sub>[0.30,0.64]</sub> | 0.43 <sub>[0.27,0.61]</sub> | -3.3 | [-23.3, +16.7] | 6/5 | 1.000 |
| `qwen2.5-coder:7b` | len_short *(exploratory)* | 30 | 0.90 <sub>[0.74,0.97]</sub> | 0.70 <sub>[0.52,0.83]</sub> | -20.0 | [-36.7, -3.3] | 7/1 | 0.070 |

### Failure composition by ordering

| model | ordering | dropped | misattributed | absorbed | format | n failures |
|---|---|---:|---:|---:|---:|---:|
| `gemma4:12b` | blocked | 41% | 0% | 59% | 0% | 17 |
| `gemma4:12b` | interleaved | 35% | 0% | 65% | 0% | 20 |
| `qwen2.5-coder:7b` | blocked | 30% | 0% | 70% | 0% | 66 |
| `qwen2.5-coder:7b` | interleaved | 34% | 0% | 66% | 0% | 91 |

### What actually goes wrong

| failure mode | count | share |
|---|---:|---:|
| absorbed / false assertion | 74 | 38.1% |
| dropped / vanished | 60 | 30.9% |
| absorbed / hallucination | 42 | 21.6% |
| absorbed / stale value | 13 | 6.7% |
| dropped / wrong value | 5 | 2.6% |
| misattributed | 0 | 0.0% |

**Misattribution never occurred.** The taxonomy was built around it — the two task types were chosen precisely so that a grocery item appearing in a schedule would be unmissable — and it did not happen once. The failure that dominates instead is absorbing content the user explicitly negated. Read with care: with only two tasks of very different types, this may mean models do not misfile across dissimilar tasks, or it may mean two tasks is too few for misfiling to arise. The task-count arm that would separate those was cut (see D14), so this run cannot tell you which.

### Validation counters

- **Token match:** 480/480 conversations exact (0 drifted), verified against Ollama's `prompt_eval_count`.
- **Parse rate:** 480/480 (100.0%) under schema-constrained decoding.
- **Per-task accuracy:** 0.842, clustered SE 0.0212 vs naive SE 0.0126 — ratio **1.68x** (the naive SE understates uncertainty).
- **Rows:** 480 conversations across 8 model-condition cells.
