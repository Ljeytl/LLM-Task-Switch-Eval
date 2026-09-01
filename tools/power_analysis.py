#!/usr/bin/env python3
"""What effect size can this design actually detect, at the n it was run at?

Answering "is this n enough?" with a simulation rather than a hedge. Uses the discordance
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
#: Reading order for conditions; alphabetical puts len_long before len_medium before
#: len_short, backwards along the axis the arm varies.
CELL_ORDER = ["ctrl_1task", "len_short", "len_medium", "len_long",
              "same_kind_2", "tasks_3", "tasks_4"]


def _rank(key):
    model, label = key
    return (model, CELL_ORDER.index(label) if label in CELL_ORDER else len(CELL_ORDER),
            label)


for (m, c), d in sorted(by.items(), key=_rank):
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
# 25 is the n the sweep ACTUALLY ran. The grid opened at 30 -- a value no cell used --
# so every power figure quoted from this table was for a sample larger than the one that
# produced the results it was being used to qualify.
ns = (25, 50, 100, 200, 400)
print("| true effect | " + " | ".join(f"n={n}" for n in ns) + " |")
print("|---|" + "---:|" * len(ns))
sweep_power: dict[float, str] = {}
for split in (0.65, 0.75, 0.85):
    cells = []
    row_effect = (2 * split - 1) * rate * 100
    for n in ns:
        hits = 0
        for _ in range(400):
            d = rng.binomial(n, rate)
            b = rng.binomial(d, split)
            bl = [True] * b + [False] * (d - b) + [True] * (n - d)
            il = [False] * b + [True] * (d - b) + [True] * (n - d)
            hits += mcnemar(bl, il)[1] < 0.05
        cells.append(f"{hits/400:.0%}")
    print(f"| {row_effect:.1f} pp | " + " | ".join(cells) + " |")
    sweep_power[round(row_effect, 1)] = cells[0]

# Derived, not asserted: this sentence previously quoted n=30 and 15% -- a sample size no
# cell ran and a figure from an older table. A hand-written summary of a generated table
# is one edit away from contradicting it.
mid = sorted(sweep_power)[len(sweep_power) // 2]
print(f"\n**Read this before reading any null in the results.** At the n={ns[0]} per cell "
      f"this sweep actually ran, the design has")
print(f"roughly {sweep_power[mid]} power against an {mid:.0f}-point effect. "
      f"A non-significant cell is evidence")
print("of insufficient sample, not evidence of no effect.")
