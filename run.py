#!/usr/bin/env python3
"""CLI for the taskswitch experiment.

    python run.py --demo                      5 pairs, one model, writes sample output
    python run.py --calibrate                 find the op count that avoids floor/ceiling
    python run.py --check-constrained         constrained vs free-form, before the sweep
    python run.py --config configs/main.yaml  the full sweep
    python run.py --analyse                   stats + plots from existing results
    python run.py --rescore                   re-score from cache with no inference
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from taskswitch.generator import Ordering, build_pair
from taskswitch.ops import TaskKind, default_tasks
from taskswitch.plots import dumbbell, taxonomy_bars
from taskswitch.runner import ModelSpec, resolve_model, run_conversation, verify_token_match
from taskswitch.scorer import score
from taskswitch.stats import (bonferroni, clustered_se, mcnemar, mcnemar_table, naive_se,
                              paired_bootstrap, wilson)

RESULTS = Path("results")


def tasks_for(n: int) -> list[TaskKind]:
    """`n` concurrent tasks, cycling the kinds.

    Repeated kinds are fine: each becomes its own named instance with its own disjoint
    vocabulary (D17), so [SHOPPING, SCHEDULE, SHOPPING] is a grocery list, a work
    calendar and a hardware list -- three independent states.

    This function kept a D14-era cap at two tasks long after D17 lifted it in the
    library, and the sweep crashed on the first task-count cell. The tests called
    `build_pair` directly with 3 and 4 tasks and passed, because nothing exercised this
    boundary. `build_pair` enforces the real limit (how many distinct vocabularies
    exist), so no cap belongs here.
    """
    return default_tasks(n)


def resolve_tasks(spec: int | list[TaskKind] | list[str]) -> list[TaskKind]:
    """Accept a task count or an explicit list of kind names.

    An explicit list is what lets a cell hold task count fixed while varying how many
    *same-kind pairs* the conversation contains. Those two move together under
    `tasks_for` (LIMITATIONS.md 5b), so with counts alone the misattribution result
    cannot be attributed to either one.
    """
    if isinstance(spec, int):
        return tasks_for(spec)
    out = []
    for item in spec:
        if isinstance(item, TaskKind):
            out.append(item)
            continue
        try:
            out.append(TaskKind(str(item).strip().lower()))
        except ValueError:
            legal = ", ".join(t.value for t in TaskKind)
            raise SystemExit(f"unknown task kind {item!r}; legal kinds: {legal}") from None
    if not out:
        raise SystemExit("a cell's `tasks` list may not be empty")
    return out


def cell_tasks(cell: dict) -> list[TaskKind]:
    """The composition a config cell asks for: an explicit `tasks` list, else `n_tasks`.

    Written as an explicit branch rather than `cell.get("tasks", cell["n_tasks"])`,
    which evaluates its default eagerly and raises KeyError on exactly the cells that
    supply `tasks` instead of `n_tasks` -- that is, on every cell this feature exists for.
    """
    if "tasks" in cell:
        return resolve_tasks(cell["tasks"])
    if "n_tasks" not in cell:
        raise SystemExit(f"cell {cell.get('label', cell)!r} needs `tasks` or `n_tasks`")
    return resolve_tasks(cell["n_tasks"])


def row_of(res, sc, cell_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    c = res.conversation
    return {
        "seed": c.seed, "model": res.model.name, "digest": res.model.digest,
        "ordering": c.ordering.value, "cell": c.cell,
        "n_tasks": len(c.tasks),
        # The kinds themselves, not just how many. --rescore rebuilds states from this;
        # inferring them from n_tasks silently assumed the canonical alternating pool and
        # would mis-score any cell with an explicit composition.
        "tasks": [t.value for t in c.tasks],
        "n_ops": c.n_ops, "n_noise": c.n_noise,
        "n_false": c.n_false, "n_turns": len(c.turns),
        "joint_correct": sc.joint_correct,
        "per_task_correct": sc.per_task_correct,
        "failures": sc.failure_kinds,
        "failure_details": [d for _, d in sc.failures],
        "parse_ok": res.parse_ok, "constrained": res.constrained,
        "prompt_tokens": res.prompt_tokens, "eval_tokens": res.eval_tokens,
        "wall_seconds": round(res.wall_seconds, 3),
        "expected": c.expected,
        # `parsed` is a plain dict keyed by slot (runner.parse_slots); the extraction
        # path may still hand back a pydantic model, so accept either.
        "reported": (res.parsed.model_dump() if hasattr(res.parsed, "model_dump")
                     else res.parsed),
        **(cell_extra or {}),
    }


def run_cell(model: ModelSpec, tasks_spec: int | list, n_ops: int, n_noise: int,
             n_false: int, n_pairs: int, seed0: int, constrained: bool = True,
             desc: str = "", extract: bool = False) -> tuple[list[dict], dict]:
    """Run one condition and return (rows, token-match diagnostics).

    `tasks_spec` is either a count (canonical composition) or an explicit list of kinds.
    """
    tasks = resolve_tasks(tasks_spec)
    n_tasks = len(tasks)
    rows, drift = [], []
    for k in tqdm(range(n_pairs), desc=desc or f"t{n_tasks} o{n_ops} n{n_noise}",
                  leave=False, file=sys.stderr):
        blocked, inter = build_pair(seed0 + k, tasks, n_ops, n_false, n_noise)
        rb = run_conversation(blocked, model, constrained, extract=extract)
        ri = run_conversation(inter, model, constrained, extract=extract)
        matched, delta = verify_token_match(rb, ri)
        drift.append(delta)
        # A pair whose orderings did not tokenise identically is not a valid comparison.
        # Recorded and excluded rather than quietly averaged in.
        for res in (rb, ri):
            rows.append(row_of(res, score(res), {"token_matched": matched,
                                                 "token_delta": delta}))
    diag = {"pairs": n_pairs, "max_drift": max(drift) if drift else 0,
            "n_drifted": sum(1 for d in drift if d != 0)}
    return rows, diag


def paired_arrays(rows: list[dict]) -> tuple[list[bool], list[bool]]:
    """Line up blocked and interleaved outcomes by seed. Order is the pairing."""
    by = defaultdict(dict)
    for r in rows:
        by[r["seed"]][r["ordering"]] = r
    seeds = sorted(s for s, d in by.items() if len(d) == 2)
    return ([by[s]["blocked"]["joint_correct"] for s in seeds],
            [by[s]["interleaved"]["joint_correct"] for s in seeds])


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# --- commands ------------------------------------------------------------------------

def cmd_demo(args) -> None:
    model = resolve_model(args.model, num_ctx=args.num_ctx)
    print(f"model {model.name} digest={model.digest or '?'}")
    rows, diag = run_cell(model, 2, 12, 4, 2, 5, seed0=1000, desc="demo")
    write_jsonl(rows, RESULTS / "demo.jsonl")
    ok = sum(1 for r in rows if r["joint_correct"])
    print(f"\n{ok}/{len(rows)} conversations exactly correct")
    print(f"token match: {diag['n_drifted']}/{diag['pairs']} pairs drifted "
          f"(max {diag['max_drift']} tokens)")
    for r in rows[:2]:
        print(f"\n  [{r['ordering']}] expected {r['expected']}")
        print(f"  {' ' * len(r['ordering'])}   reported {r['reported']}")
        if r["failure_details"]:
            print(f"  {' ' * len(r['ordering'])}   failures {r['failure_details']}")
    print(f"\nwrote {RESULTS / 'demo.jsonl'}")


def cmd_calibrate(args) -> None:
    """Find the op count where blocked accuracy sits in the measurable band.

    Run BEFORE the sweep. If blocked accuracy is at 1.0 or 0.0 there is no room for a
    delta in either direction, and the whole sweep would return a null for reasons that
    have nothing to do with interleaving.
    """
    model = resolve_model(args.model, num_ctx=args.num_ctx)
    print(f"calibrating {model.name} (target: blocked joint accuracy 0.6-0.8)\n")
    print(f"{'n_ops':>6} {'blocked':>9} {'interleaved':>12} {'prompt_tok':>11}")
    for n_ops in args.op_grid:
        rows, _ = run_cell(model, 2, n_ops, args.n_noise, 2, args.n_pairs,
                           seed0=5000, desc=f"n_ops={n_ops}")
        b, i = paired_arrays(rows)
        pt = int(sum(r["prompt_tokens"] for r in rows) / max(1, len(rows)))
        bl = sum(b) / len(b) if b else 0
        il = sum(i) / len(i) if i else 0
        flag = "  <-- in band" if 0.55 <= bl <= 0.85 else ""
        print(f"{n_ops:6d} {bl:9.2f} {il:12.2f} {pt:11d}{flag}")


def cmd_check_constrained(args) -> None:
    """Measure what schema-constrained decoding costs, before trusting it everywhere.

    Published estimates put the effect as high as 8.7 points. If the delta here is
    near zero, constrained decoding stays and we say so with evidence. If it is large,
    the constraint is a measured confound and the design switches to free-form plus a
    second extraction call.
    """
    model = resolve_model(args.model, num_ctx=args.num_ctx)
    out = {}
    for constrained in (True, False):
        # Free-form runs get a second extraction pass, otherwise this compares JSON
        # emission rather than state tracking -- see runner.extract_state.
        rows, _ = run_cell(model, 2, args.n_ops, args.n_noise, 2, args.n_pairs,
                           seed0=9000, constrained=constrained, extract=not constrained,
                           desc="constrained" if constrained else "free-form+extract")
        parse_rate = sum(r["parse_ok"] for r in rows) / len(rows)
        scored = [r for r in rows if r["parse_ok"]]
        acc = sum(r["joint_correct"] for r in scored) / max(1, len(scored))
        out[constrained] = (acc, parse_rate, len(rows))
        print(f"{'constrained' if constrained else 'free-form  '}: "
              f"joint acc {acc:.3f} on parseable, parse rate {parse_rate:.3f}, n={len(rows)}")
    d = (out[True][0] - out[False][0]) * 100
    print(f"\nconstrained - (free-form + extraction) = {d:+.1f} pp on state accuracy")
    if out[False][1] < 0.5:
        print("WARNING: the free-form arm still has a low parse rate even after "
              "extraction, so this delta is not interpretable as a decoding effect.")
    else:
        print("A large NEGATIVE delta would mean constraining hurts tracking and should "
              "be reported as a measured confound. Near zero means the constraint is "
              "safe to keep.")


def cmd_sweep(args) -> None:
    cfg = yaml.safe_load(Path(args.config).read_text())
    all_rows: list[dict] = []
    for mname in cfg["models"]:
        model = resolve_model(mname, num_ctx=cfg.get("num_ctx", 16384))
        print(f"\n=== {model.name} (digest {model.digest or '?'}) ===")
        for cell in cfg["cells"]:
            rows, diag = run_cell(
                model, cell_tasks(cell),
                cell["n_ops"], cell["n_noise"],
                cfg.get("n_false", 3), cfg["n_pairs"], seed0=cfg.get("seed0", 1),
                desc=f"{mname} {cell['label']}")
            for r in rows:
                r["label"] = cell["label"]
            all_rows.extend(rows)
            b, i = paired_arrays(rows)
            bl = sum(b) / len(b) if b else 0
            il = sum(i) / len(i) if i else 0
            print(f"  {cell['label']:>14}  blocked {bl:.3f}  interleaved {il:.3f}  "
                  f"delta {(il-bl)*100:+.1f}pp  drift {diag['n_drifted']}/{diag['pairs']}")
            write_jsonl(all_rows, RESULTS / "sweep.jsonl")   # checkpoint after each cell
    print(f"\nwrote {len(all_rows)} rows to {RESULTS / 'sweep.jsonl'}")
    analyse(all_rows)


def analyse(rows: list[dict]) -> None:
    """Per-condition estimates, the paired test, and the plots."""
    print("\n" + "=" * 96)
    print(f"{'model':<20}{'cell':<16}{'n':>5}{'blocked':>10}{'inter':>10}"
          f"{'delta pp':>10}{'95% CI':>18}{'p':>9}")
    print("-" * 96)
    # Exclude pairs whose two orderings did not tokenise identically. Such a pair is
    # not a valid comparison -- it compares two different-length conversations -- so it
    # is reported in the drift counter and dropped here rather than quietly averaged in.
    usable = [r for r in rows if r.get("token_matched", True)]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in usable:
        groups[(r["model"], r["cell"])].append(r)

    secondary_p = []
    for (model, cell), rs in sorted(groups.items()):
        b, i = paired_arrays([r for r in rs if r["parse_ok"]])
        if not b:
            continue
        stat, p = mcnemar(b, i)
        d, lo, hi = paired_bootstrap(b, i, n_boot=10000, seed=0)
        t = mcnemar_table(b, i)
        print(f"{model.split(':')[0]:<20}{cell:<16}{len(b):>5}"
              f"{sum(b)/len(b):>10.3f}{sum(i)/len(i):>10.3f}"
              f"{d*100:>+10.1f}{f'[{lo*100:+.1f},{hi*100:+.1f}]':>18}{p:>9.4f}")
        secondary_p.append(p)

    print("-" * 96)
    # Per-task accuracy with SEs clustered on conversation, plus the naive SE for
    # contrast -- the gap is the reason clustering is not optional here.
    per_task, clusters = [], []
    for r in rows:
        if r["per_task_correct"]:
            for v in r["per_task_correct"].values():
                per_task.append(v); clusters.append(r["seed"])
    if per_task:
        cse, nse = clustered_se(per_task, clusters), naive_se(per_task)
        ratio = cse / max(nse, 1e-9)
        # Only call the naive SE "too narrow" when it actually is. With few clusters or
        # negative within-cluster correlation the ratio can fall below 1, and asserting
        # understatement there would be the same kind of overclaiming this project is
        # trying to avoid.
        verdict = (f"naive is {ratio:.2f}x too narrow" if ratio > 1.05
                   else f"clustered/naive ratio {ratio:.2f}, no material clustering effect")
        print(f"per-task accuracy {sum(per_task)/len(per_task):.3f}  "
              f"clustered SE {cse:.4f}  naive SE {nse:.4f}  ({verdict})")

    fails = [f for r in rows for f in r["failures"]]
    if fails:
        comp = {k: fails.count(k) / len(fails) for k in set(fails)}
        print("failure composition: " + "  ".join(
            f"{k} {v:.1%}" for k, v in sorted(comp.items(), key=lambda kv: -kv[1])))
    drift = [r for r in rows if not r.get("token_matched", True)]
    print(f"token match: {len(rows)-len(drift)}/{len(rows)} rows exact "
          f"({len(drift)} drifted)")

    RESULTS.mkdir(exist_ok=True)
    dumbbell(rows, RESULTS / "dumbbell.png")
    taxonomy_bars(rows, RESULTS / "taxonomy.png")
    print(f"wrote {RESULTS/'dumbbell.png'} and {RESULTS/'taxonomy.png'}")


def cmd_analyse(args) -> None:
    analyse(read_jsonl(Path(args.results)))


def cmd_rescore(args) -> None:
    """Recompute every score from the response cache. No inference.

    This is why the scorer is pure: the taxonomy can be revised after looking at real
    failures without paying for the sweep twice.
    """
    src = Path(args.results)
    rows = read_jsonl(src)
    models = {name: resolve_model(name) for name in {r["model"] for r in rows}}
    out = []
    for r in tqdm(rows, desc="rescore", file=sys.stderr):
        # Prefer the recorded composition; fall back for rows written before it existed.
        tasks = resolve_tasks(r.get("tasks") or r["n_tasks"])
        pair = build_pair(r["seed"], tasks, r["n_ops"], r["n_false"], r["n_noise"])
        conv = pair[0] if r["ordering"] == "blocked" else pair[1]
        res = run_conversation(conv, models[r["model"]], r.get("constrained", True))
        new = row_of(res, score(res), {"token_matched": r.get("token_matched", True),
                                       "token_delta": r.get("token_delta", 0)})
        new["label"] = r.get("label", "")
        out.append(new)
    write_jsonl(out, src)
    changed = sum(1 for a, b in zip(rows, out) if a["joint_correct"] != b["joint_correct"])
    print(f"rescored {len(out)} rows ({changed} outcomes changed)")
    analyse(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen2.5-coder:7b")
    p.add_argument("--num-ctx", type=int, default=16384)
    p.add_argument("--n-pairs", type=int, default=20)
    p.add_argument("--n-ops", type=int, default=24)
    p.add_argument("--n-noise", type=int, default=0)
    p.add_argument("--op-grid", type=int, nargs="+", default=[8, 16, 24, 32, 48])
    p.add_argument("--results", default="results/sweep.jsonl")
    p.add_argument("--config")
    g = p.add_mutually_exclusive_group()
    for flag in ("demo", "calibrate", "check-constrained", "sweep", "analyse", "rescore"):
        g.add_argument(f"--{flag}", action="store_true")
    a = p.parse_args()

    # --config implies --sweep, matching the documented usage at the top of this file.
    if a.config and not any([a.demo, a.calibrate, a.check_constrained, a.analyse, a.rescore]):
        a.sweep = True
    if not any([a.demo, a.calibrate, a.check_constrained, a.sweep, a.analyse, a.rescore]):
        p.error("pick one of --demo --calibrate --check-constrained --sweep --analyse --rescore "
                "(or pass --config, which implies --sweep)")
    if a.sweep and not a.config:
        a.config = "configs/main.yaml"

    if a.demo: cmd_demo(a)
    elif a.calibrate: cmd_calibrate(a)
    elif a.check_constrained: cmd_check_constrained(a)
    elif a.sweep: cmd_sweep(a)
    elif a.analyse: cmd_analyse(a)
    elif a.rescore: cmd_rescore(a)


if __name__ == "__main__":
    main()
