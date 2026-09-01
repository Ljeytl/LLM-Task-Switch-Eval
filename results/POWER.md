## Observed discordance

`b` = interleaving broke it, `c` = interleaving fixed it. Concordant pairs carry
no information about ordering, so the discordant share IS the usable sample.

| model | cell | n | b | c | discordant |
|---|---|---:|---:|---:|---:|
| `gemma4:12b` | ctrl_1task | 30 | 0 | 0 | 0% |
| `gemma4:12b` | len_short | 30 | 0 | 0 | 0% |
| `qwen2.5-coder:7b` | ctrl_1task | 30 | 0 | 0 | 0% |
| `qwen2.5-coder:7b` | len_long | 30 | 12 | 2 | 47% |
| `qwen2.5-coder:7b` | len_medium | 30 | 6 | 5 | 37% |
| `qwen2.5-coder:7b` | len_short | 30 | 7 | 1 | 27% |

Mean discordance across non-control cells: **27%**

## Power

Simulated, exact McNemar, alpha = 0.05, discordance 27%, 400 replicates per cell.

| true effect | n=30 | n=60 | n=120 | n=240 | n=480 |
|---|---:|---:|---:|---:|---:|
| 8.2 pp | 6% | 18% | 36% | 64% | 94% |
| 13.7 pp | 20% | 44% | 79% | 99% | 100% |
| 19.2 pp | 39% | 79% | 99% | 100% | 100% |

**Read this before reading any null in the results.** At n=30 this design has
roughly 15% power against a 12-point effect. A non-significant cell is evidence
of insufficient sample, not evidence of no effect.
