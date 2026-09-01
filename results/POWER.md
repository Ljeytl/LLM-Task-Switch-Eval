## Observed discordance

`b` = interleaving broke it, `c` = interleaving fixed it. Concordant pairs carry
no information about ordering, so the discordant share IS the usable sample.

| model | cell | n | b | c | discordant |
|---|---|---:|---:|---:|---:|
| `gemma4:12b` | ctrl_1task | 25 | 0 | 0 | 0% |
| `gemma4:12b` | len_long | 25 | 4 | 1 | 20% |
| `gemma4:12b` | len_medium | 25 | 0 | 2 | 8% |
| `gemma4:12b` | len_short | 25 | 0 | 2 | 8% |
| `gemma4:12b` | same_kind_2 | 25 | 3 | 4 | 28% |
| `gemma4:12b` | tasks_3 | 25 | 4 | 3 | 28% |
| `gemma4:12b` | tasks_4 | 25 | 4 | 2 | 24% |
| `qwen2.5-coder:7b` | ctrl_1task | 25 | 0 | 0 | 0% |
| `qwen2.5-coder:7b` | len_long | 25 | 6 | 3 | 36% |
| `qwen2.5-coder:7b` | len_medium | 25 | 6 | 3 | 36% |
| `qwen2.5-coder:7b` | len_short | 25 | 5 | 1 | 24% |
| `qwen2.5-coder:7b` | same_kind_2 | 25 | 0 | 1 | 4% |
| `qwen2.5-coder:7b` | tasks_3 | 25 | 5 | 0 | 20% |
| `qwen2.5-coder:7b` | tasks_4 | 25 | 8 | 1 | 36% |

Mean discordance across non-control cells: **23%**

## Power

Simulated, exact McNemar, alpha = 0.05, discordance 23%, 400 replicates per cell.

| true effect | n=30 | n=60 | n=120 | n=240 | n=480 |
|---|---:|---:|---:|---:|---:|
| 6.8 pp | 5% | 14% | 25% | 56% | 88% |
| 11.3 pp | 14% | 32% | 70% | 95% | 100% |
| 15.9 pp | 28% | 71% | 96% | 100% | 100% |

**Read this before reading any null in the results.** At n=30 this design has
roughly 15% power against a 12-point effect. A non-significant cell is evidence
of insufficient sample, not evidence of no effect.
