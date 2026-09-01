#!/usr/bin/env python3
"""Render the Results section of the README from results/sweep.jsonl.

Kept as a script so the numbers in the README cannot drift from the data.
Writes results/RESULTS.md, which README.md includes by reference.
"""
from __future__ import annotations

import collections, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from taskswitch.stats import (clustered_se, mcnemar, mcnemar_table, naive_se,
                              paired_bootstrap, wilson)

rows = [json.loads(l) for l in Path("results/sweep.jsonl").read_text().splitlines() if l.strip()]

def paired(rs):
    by = collections.defaultdict(dict)
    for r in rs:
        by[r["seed"]][r["ordering"]] = r
    seeds = sorted(s for s, d in by.items() if len(d) == 2)
    return ([by[s]["blocked"]["joint_correct"] for s in seeds],
            [by[s]["interleaved"]["joint_correct"] for s in seeds])

groups = collections.defaultdict(list)
for r in rows:
    groups[(r["model"], r.get("label", r["cell"]))].append(r)

PREAMBLE = """### How to read this

`len_medium` on `qwen2.5-coder:7b` is the **pre-registered primary comparison**. It was
chosen before the sweep ran and is reported uncorrected. Every other row is
**exploratory**: if one of them shows a larger effect than the primary, that is a
hypothesis to test, not a finding to report. Swapping the headline to whichever cell
came out strongest is exactly the practice pre-registration exists to prevent, and the
temptation is real precisely when the primary comes back null.

`ctrl_1task` is a **negative control**. With one task there is nothing to interleave, so
both orderings are the identical prompt and the delta must be exactly 0.0. A non-zero
value there would invalidate everything below it.

"""

out = [PREAMBLE, "### Joint goal accuracy by condition\n",
       "| model | condition | n pairs | blocked | interleaved | delta (pp) | 95% CI | McNemar b/c | p |",
       "|---|---|---:|---:|---:|---:|---|---|---:|"]
for (model, label), rs in sorted(groups.items()):
    ok = [r for r in rs if r["parse_ok"]]
    b, i = paired(ok)
    if not b:
        continue
    stat, p = mcnemar(b, i)
    d, lo, hi = paired_bootstrap(b, i, 10000, 0)
    t = mcnemar_table(b, i)
    bl, il = sum(b)/len(b), sum(i)/len(i)
    wb, wi = wilson(sum(b), len(b)), wilson(sum(i), len(i))
    if label == "len_medium" and "qwen" in model:
        star = " **(PRIMARY)**"
    elif label == "ctrl_1task":
        star = " *(control)*"
    else:
        star = " *(exploratory)*"
    out.append(f"| `{model}` | {label}{star} | {len(b)} | {bl:.2f} "
               f"<sub>[{wb[0]:.2f},{wb[1]:.2f}]</sub> | {il:.2f} "
               f"<sub>[{wi[0]:.2f},{wi[1]:.2f}]</sub> | {d*100:+.1f} | "
               f"[{lo*100:+.1f}, {hi*100:+.1f}] | {t.b}/{t.c} | {p:.3f} |")

fails = [f for r in rows for f in r["failures"]]
out += ["\n### Failure composition by ordering\n",
        "| model | ordering | dropped | misattributed | absorbed | format | n failures |",
        "|---|---|---:|---:|---:|---:|---:|"]
fg = collections.defaultdict(list)
for r in rows:
    fg[(r["model"], r["ordering"])].extend(r["failures"])
for (model, ordering), fs in sorted(fg.items()):
    n = len(fs) or 1
    out.append(f"| `{model}` | {ordering} | " + " | ".join(
        f"{fs.count(k)/n:.0%}" for k in ("dropped", "misattributed", "absorbed", "format")
    ) + f" | {len(fs)} |")

# Sub-bucket breakdown. The headline buckets hide the most informative split: what
# KIND of ungrounded content got absorbed, and whether misattribution happened at all.
sub = collections.Counter()
for r in rows:
    for d in r.get("failure_details", []):
        if d.endswith("false_assert"):     sub["absorbed / false assertion"] += 1
        elif d.endswith("stale"):          sub["absorbed / stale value"] += 1
        elif d.endswith("hallucination"):  sub["absorbed / hallucination"] += 1
        elif d.endswith("wrong value"):    sub["dropped / wrong value"] += 1
        elif "other task" in d:            sub["misattributed"] += 1
        else:                              sub["dropped / vanished"] += 1
tot_sub = sum(sub.values()) or 1
out += ["\n### What actually goes wrong\n",
        "| failure mode | count | share |", "|---|---:|---:|"]
for k, v in sub.most_common():
    out.append(f"| {k} | {v} | {v/tot_sub:.1%} |")
if not sub.get("misattributed"):
    out.append(f"| misattributed | 0 | 0.0% |")
    out.append("\n**Misattribution never occurred.** The taxonomy was built around it — the two "
               "task types were chosen precisely so that a grocery item appearing in a schedule "
               "would be unmissable — and it did not happen once. The failure that dominates "
               "instead is absorbing content the user explicitly negated. Read with care: with "
               "only two tasks of very different types, this may mean models do not misfile "
               "across dissimilar tasks, or it may mean two tasks is too few for misfiling to "
               "arise. The task-count arm that would separate those was cut (see D14), so this "
               "run cannot tell you which.")

per_task, clusters = [], []
for r in rows:
    if r["per_task_correct"]:
        for v in r["per_task_correct"].values():
            per_task.append(v); clusters.append(r["seed"])
cse, nse = clustered_se(per_task, clusters), naive_se(per_task)
drift = sum(1 for r in rows if not r.get("token_matched", True))
parse = sum(r["parse_ok"] for r in rows)

out += ["\n### Validation counters\n",
        f"- **Token match:** {len(rows)-drift}/{len(rows)} conversations exact "
        f"({drift} drifted), verified against Ollama's `prompt_eval_count`.",
        f"- **Parse rate:** {parse}/{len(rows)} ({parse/len(rows):.1%}) under "
        f"schema-constrained decoding.",
        f"- **Per-task accuracy:** {sum(per_task)/len(per_task):.3f}, clustered SE "
        f"{cse:.4f} vs naive SE {nse:.4f} — ratio **{cse/max(nse,1e-9):.2f}x**"
        + (" (the naive SE understates uncertainty)." if cse/max(nse,1e-9) > 1.05
           else " (no material clustering effect at this n)."),
        f"- **Rows:** {len(rows)} conversations across {len(groups)} model-condition cells."]

Path("results/RESULTS.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
