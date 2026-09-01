### How to read this

`len_medium` on `qwen2.5-coder:7b` is the **repository-prespecified primary comparison**.
Git history records it before the v2 sweep; there was no external registry. It is
reported uncorrected. Every other inferential row is **exploratory**, except the
negative-control rows. Exploratory p-values receive one Bonferroni correction across the
complete model-condition family; both raw and adjusted values are shown. If one shows a
larger effect than the primary, that is a hypothesis to test, not a finding to report.

`ctrl_1task` is a **negative control**. With one task there is nothing to interleave, so
both orderings are the identical prompt and the delta must be exactly 0.0. A non-zero
value there would invalidate everything below it.


Exploratory Bonferroni family size: **m=11**.

### Joint goal accuracy by condition

| model | condition | n pairs | blocked | interleaved | delta (pp) | 95% paired-bootstrap interval | McNemar b/c | raw p | Bonferroni p |
|---|---|---:|---:|---:|---:|---|---|---:|---:|
| `gemma4:12b` | ctrl_1task *(control)* | 25 | 0.76 <sub>[0.57,0.89]</sub> | 0.76 <sub>[0.57,0.89]</sub> | +0.0 | [+0.0, +0.0] | 0/0 | 1.000 | — |
| `gemma4:12b` | len_short *(exploratory)* | 25 | 0.80 <sub>[0.61,0.91]</sub> | 0.88 <sub>[0.70,0.96]</sub> | +8.0 | [+0.0, +20.0] | 0/2 | 0.500 | 1.000 |
| `gemma4:12b` | len_medium *(exploratory)* | 25 | 0.88 <sub>[0.70,0.96]</sub> | 0.96 <sub>[0.80,0.99]</sub> | +8.0 | [+0.0, +20.0] | 0/2 | 0.500 | 1.000 |
| `gemma4:12b` | len_long *(exploratory)* | 25 | 0.96 <sub>[0.80,0.99]</sub> | 0.84 <sub>[0.65,0.94]</sub> | -12.0 | [-28.0, +4.0] | 4/1 | 0.375 | 1.000 |
| `gemma4:12b` | same_kind_2 *(exploratory)* | 25 | 0.72 <sub>[0.52,0.86]</sub> | 0.76 <sub>[0.57,0.89]</sub> | +4.0 | [-16.0, +24.0] | 3/4 | 1.000 | 1.000 |
| `gemma4:12b` | tasks_3 *(exploratory)* | 25 | 0.68 <sub>[0.48,0.83]</sub> | 0.64 <sub>[0.45,0.80]</sub> | -4.0 | [-24.0, +16.0] | 4/3 | 1.000 | 1.000 |
| `gemma4:12b` | tasks_4 *(exploratory)* | 25 | 0.72 <sub>[0.52,0.86]</sub> | 0.64 <sub>[0.45,0.80]</sub> | -8.0 | [-28.0, +12.0] | 4/2 | 0.688 | 1.000 |
| `qwen2.5-coder:7b` | ctrl_1task *(control)* | 25 | 0.32 <sub>[0.17,0.52]</sub> | 0.32 <sub>[0.17,0.52]</sub> | +0.0 | [+0.0, +0.0] | 0/0 | 1.000 | — |
| `qwen2.5-coder:7b` | len_short *(exploratory)* | 25 | 0.84 <sub>[0.65,0.94]</sub> | 0.68 <sub>[0.48,0.83]</sub> | -16.0 | [-36.0, +0.0] | 5/1 | 0.219 | 1.000 |
| `qwen2.5-coder:7b` | len_medium **(PRIMARY)** | 25 | 0.64 <sub>[0.45,0.80]</sub> | 0.52 <sub>[0.33,0.70]</sub> | -12.0 | [-36.0, +12.0] | 6/3 | 0.508 | — |
| `qwen2.5-coder:7b` | len_long *(exploratory)* | 25 | 0.56 <sub>[0.37,0.73]</sub> | 0.44 <sub>[0.27,0.63]</sub> | -12.0 | [-36.0, +12.0] | 6/3 | 0.508 | 1.000 |
| `qwen2.5-coder:7b` | same_kind_2 *(exploratory)* | 25 | 0.00 <sub>[0.00,0.13]</sub> | 0.04 <sub>[0.01,0.20]</sub> | +4.0 | [+0.0, +12.0] | 0/1 | 1.000 | 1.000 |
| `qwen2.5-coder:7b` | tasks_3 *(exploratory)* | 25 | 0.40 <sub>[0.23,0.59]</sub> | 0.20 <sub>[0.09,0.39]</sub> | -20.0 | [-36.0, -4.0] | 5/0 | 0.062 | 0.688 |
| `qwen2.5-coder:7b` | tasks_4 *(exploratory)* | 25 | 0.40 <sub>[0.23,0.59]</sub> | 0.12 <sub>[0.04,0.30]</sub> | -28.0 | [-48.0, -8.0] | 8/1 | 0.039 | 0.430 |

### Failure composition by ordering

| model | ordering | dropped | misattributed | absorbed | format | n failures |
|---|---|---:|---:|---:|---:|---:|
| `gemma4:12b` | blocked | 49% | 0% | 51% | 0% | 73 |
| `gemma4:12b` | interleaved | 49% | 0% | 51% | 0% | 72 |
| `qwen2.5-coder:7b` | blocked | 55% | 15% | 30% | 0% | 188 |
| `qwen2.5-coder:7b` | interleaved | 47% | 19% | 33% | 0% | 257 |

### What actually goes wrong

| failure mode | count | share |
|---|---:|---:|
| dropped / vanished | 287 | 48.6% |
| absorbed / hallucination | 117 | 19.8% |
| absorbed / false assertion | 89 | 15.1% |
| misattributed | 79 | 13.4% |
| absorbed / stale value | 10 | 1.7% |
| dropped / wrong value | 8 | 1.4% |

**Misattribution occurred, and the obvious reading of it is wrong.**

| model | condition | tasks | same-kind pairs | entries | conversations affected |
|---|---|---:|---:|---:|---|
| `gemma4:12b` | ctrl_1task | 1 | 0 | 0 | 0/50 |
| `gemma4:12b` | len_short | 2 | 0 | 0 | 0/50 |
| `gemma4:12b` | len_medium | 2 | 0 | 0 | 0/50 |
| `gemma4:12b` | len_long | 2 | 0 | 0 | 0/50 |
| `gemma4:12b` | same_kind_2 | 2 | 1 | 0 | 0/50 |
| `gemma4:12b` | tasks_3 | 3 | 1 | 0 | 0/50 |
| `gemma4:12b` | tasks_4 | 4 | 2 | 0 | 0/50 |
| `qwen2.5-coder:7b` | ctrl_1task | 1 | 0 | 0 | 0/50 |
| `qwen2.5-coder:7b` | len_short | 2 | 0 | 0 | 0/50 |
| `qwen2.5-coder:7b` | len_medium | 2 | 0 | 0 | 0/50 |
| `qwen2.5-coder:7b` | len_long | 2 | 0 | 0 | 0/50 |
| `qwen2.5-coder:7b` | same_kind_2 | 2 | 1 | 12 | 6/50 |
| `qwen2.5-coder:7b` | tasks_3 | 3 | 1 | 8 | 5/50 |
| `qwen2.5-coder:7b` | tasks_4 | 4 | 2 | 59 | 23/50 |

The canonical task-count arm changes same-kind-pair count with task count. `same_kind_2` shows that adding a same-kind neighbour is sufficient to expose misattribution in this instrument and that the original monotone task-count explanation was not identified. These are conversation-clustered discrepancy entries, not independent events or a causal comparison. The entries split 29 blocked / 50 interleaved; that is a descriptive diagnostic without an interval or test. See docs/LIMITATIONS.md 5b.

### Validation counters

- **Token match:** 700/700 conversations exact (0 drifted), verified against Ollama's `prompt_eval_count`.
- **Parse rate:** 700/700 (100.0%) under schema-constrained decoding.
- **Per-task accuracy:** 0.767, clustered SE 0.0117 vs naive SE 0.0106 — ratio **1.11x** (the naive SE understates uncertainty).
- **Rows:** 700 conversations across 14 model-condition cells.
