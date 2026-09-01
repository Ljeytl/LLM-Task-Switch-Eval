#!/usr/bin/env python3
"""Render the Results section of the README from results/sweep.jsonl.

Kept as a script so the numbers in the README cannot drift from the data.
Writes results/RESULTS.md, which README.md includes by reference.
"""
from __future__ import annotations

import collections, json, sys
import sys
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

#: Reading order for conditions. Alphabetical puts len_long before len_medium before
#: len_short, i.e. exactly backwards along the axis the arm varies. Unknown labels sort
#: last, alphabetically, so a new cell appears rather than vanishing.
CELL_ORDER = ["ctrl_1task", "len_short", "len_medium", "len_long",
              "same_kind_2", "tasks_3", "tasks_4"]


def cell_rank(label: str) -> tuple[int, str]:
    return (CELL_ORDER.index(label) if label in CELL_ORDER else len(CELL_ORDER), label)


out = [PREAMBLE, "### Joint goal accuracy by condition\n",
       "| model | condition | n pairs | blocked | interleaved | delta (pp) | 95% CI | McNemar b/c | p |",
       "|---|---|---:|---:|---:|---:|---|---|---:|"]
for (model, label), rs in sorted(groups.items(),
                                key=lambda kv: (kv[0][0], cell_rank(kv[0][1]))):
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
               "instead is absorbing content the user explicitly negated. Read with care: if "
               "this run included only two tasks of different kinds it may mean models do not "
               "misfile across dissimilar tasks; the task-count cells (3 and 4 tasks) include "
               "two lists of the SAME kind with disjoint vocabularies, where misfiling would "
               "be unmistakable, so check whether those cells are present before concluding "
               "anything general.")
else:
    # Misattribution occurred, which is the v2 result. It is confounded, and the
    # confound has to travel with the number rather than sit in a footnote nobody
    # opens -- see docs/LIMITATIONS.md 5b.
    import collections as _c

    def _same_kind_pairs(row):
        kinds = _c.Counter(t.split("_")[0] for t in (row.get("expected") or {}))
        return sum(c * (c - 1) // 2 for c in kinds.values())

    by_cell = _c.defaultdict(lambda: [0, 0, 0])   # events, convs, pairs
    for r in rows:
        n = r["failures"].count("misattributed")
        rec = by_cell[(r.get("label", r["cell"]), _same_kind_pairs(r), r["n_tasks"])]
        rec[0] += n
        rec[1] += 1 if n else 0
        rec[2] += 1
    out += ["\n**Misattribution occurred, and the obvious reading of it is wrong.**\n",
            "| condition | tasks | same-kind pairs | events | conversations affected |",
            "|---|---:|---:|---:|---|"]
    for (label, pairs, n_tasks), (ev, convs, tot) in sorted(
            by_cell.items(), key=lambda kv: cell_rank(kv[0][0])):
        out.append(f"| {label} | {n_tasks} | {pairs} | {ev} | {convs}/{tot} |")

    by_ord = _c.defaultdict(int)
    for r in rows:
        by_ord[r["ordering"]] += r["failures"].count("misattributed")
    out.append(
        f"\nTwo things keep this from being a clean task-count effect. First, kinds have "
        f"disjoint vocabularies, so a same-kind pair is the *only* place a misattribution "
        f"can occur — and under the canonical composition the pair count rises in lockstep "
        f"with the task count, so the two cannot be separated by the count cells alone. "
        f"That is what the `same_kind_2` cell exists to break. Second, the events split "
        f"{by_ord['blocked']} blocked / {by_ord['interleaved']} interleaved: substantial "
        f"misattribution happens in *blocked* ordering, where same-kind lists are never "
        f"interleaved with each other at all. Similarity drives the bulk of it and "
        f"ordering modulates it. The joint-accuracy delta above is a genuine ordering "
        f"effect; this count largely is not.")

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


# --- Inject the compact table into README.md ------------------------------------------
# The README carried a hand-maintained copy of this table, which drifted: it still showed
# v1 numbers after the v2 sweep replaced them. A summary duplicated by hand is a summary
# that will be wrong, so it is generated between markers and rewritten here.
BEGIN, END = "<!-- BEGIN:results-table -->", "<!-- END:results-table -->"

compact = ["| model | condition | blocked | interleaved | delta (pp) | McNemar b/c | p |",
           "|---|---|---:|---:|---:|---|---:|"]
for (model, label), rs in sorted(groups.items(),
                                 key=lambda kv: (kv[0][0], cell_rank(kv[0][1]))):
    ok = [r for r in rs if r["parse_ok"]]
    b, i = paired(ok)
    if not b:
        continue
    stat, pv = mcnemar(b, i)
    t = mcnemar_table(b, i)
    bl, il = sum(b) / len(b), sum(i) / len(i)
    d = (il - bl) * 100
    name = (f"**{label} (PRIMARY)**" if label == "len_medium" and "qwen" in model
            else f"{label} *(control)*" if label == "ctrl_1task" else label)
    mark = "**" if pv < 0.05 else ""
    compact.append(f"| `{model}` | {name} | {bl:.3f} | {il:.3f} | "
                   f"{mark}{d:+.1f}{mark} | {t.b}/{t.c} | "
                   f"{'**' if pv < 0.05 else ''}{pv:.3f}{'**' if pv < 0.05 else ''} |")

readme = Path("README.md")
text = readme.read_text()
if BEGIN in text and END in text:
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(head + BEGIN + "\n" + "\n".join(compact) + "\n" + END + tail)
    print("updated README.md results table", file=sys.stderr)
else:
    print("WARNING: README.md has no results-table markers; table not updated",
          file=sys.stderr)

print("\n".join(out))
