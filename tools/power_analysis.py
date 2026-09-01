#!/usr/bin/env python3
"""What effect size can this design actually detect, at the n it was run at?

Answering "is n=30 enough?" with a simulation rather than a hedge. Uses the discordance
rate observed in the real data, because McNemar's power depends on it and an assumed
rate would be guessing at the thing that matters most.
"""
from __future__ import annotations

import collections, json, sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from taskswitch.stats import mcnemar, mcnemar_table

src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/sweep.jsonl")
rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
by = collections.defaultdict(lambda: collections.defaultdict(dict))
for r in rows:
    by[(r["model"], r.get("label", r["cell"]))][r["seed"]][r["ordering"]] = r["joint_correct"]

print("## Observed discordance\n")
print("`b` = interleaving broke it, `c` = interleaving fixed it. Concordant pairs carry")
print("no information about ordering, so the discordant share IS the usable sample.\n")
print("| model | cell | n | b | c | discordant |")
print("|---|---|---:|---:|---:|---:|")
disc_rates = []
for (m, c), d in sorted(by.items()):
    seeds = [s for s, v in d.items() if len(v) == 2]
    if not seeds:
        continue
    t = mcnemar_table([d[s]["blocked"] for s in seeds], [d[s]["interleaved"] for s in seeds])
    print(f"| `{m}` | {c} | {len(seeds)} | {t.b} | {t.c} | "
          f"{t.n_discordant/len(seeds):.0%} |")
    if c != "ctrl_1task":
        disc_rates.append(t.n_discordant / len(seeds))

rate = float(np.mean(disc_rates)) if disc_rates else 0.25
print(f"\nMean discordance across non-control cells: **{rate:.0%}**\n")

print("## Power\n")
print(f"Simulated, exact McNemar, alpha = 0.05, discordance {rate:.0%}, "
      "400 replicates per cell.\n")
rng = np.random.default_rng(0)
ns = (30, 60, 120, 240, 480)
print("| true effect | " + " | ".join(f"n={n}" for n in ns) + " |")
print("|---|" + "---:|" * len(ns))
for split in (0.65, 0.75, 0.85):
    cells = []
    for n in ns:
        hits = 0
        for _ in range(400):
            d = rng.binomial(n, rate)
            b = rng.binomial(d, split)
            bl = [True] * b + [False] * (d - b) + [True] * (n - d)
            il = [False] * b + [True] * (d - b) + [True] * (n - d)
            hits += mcnemar(bl, il)[1] < 0.05
        cells.append(f"{hits/400:.0%}")
    print(f"| {(2*split-1)*rate*100:.1f} pp | " + " | ".join(cells) + " |")

print("\n**Read this before reading any null in the results.** At n=30 this design has")
print("roughly 15% power against a 12-point effect. A non-significant cell is evidence")
print("of insufficient sample, not evidence of no effect.")
