#!/usr/bin/env python3
"""CLI for the bugqueue experiment (sequential related defects)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

from bugqueue.generator import build_pair
from bugqueue.runner import resolve_model, run_conversation
from bugqueue.scorer import score
from taskswitch.stats import mcnemar

RESULTS = Path("results")


def run_pair(seed: int, n_bugs: int, n_noise: int, n_false: int, model_name: str,
             label: str) -> list[dict]:
    acc, ticket = build_pair(seed, n_bugs, n_noise, n_false)
    model = resolve_model(model_name)
    rows = []
    for conv in (acc, ticket):
        res = run_conversation(conv, model)
        sc = score(res)
        rows.append({
            "seed": seed, "model": model.name, "digest": model.digest,
            "ordering": conv.ordering.value, "cell": label,
            "n_bugs": n_bugs, "n_noise": n_noise,
            "joint_correct": sc.joint_correct,
            "failures": sc.failure_kinds,
            "parse_ok": res.parse_ok,
            "prompt_tokens": res.prompt_tokens,
            "eval_tokens": res.eval_tokens,
            "wall_seconds": res.wall_seconds,
        })
    return rows


def demo(model: str, n_pairs: int) -> None:
    print(f"bugqueue demo: {n_pairs} pairs, model={model}")
    all_rows: list[dict] = []
    for seed in tqdm(range(1, n_pairs + 1)):
        all_rows.extend(run_pair(seed, n_bugs=4, n_noise=2, n_false=1, model_name=model,
                                 label="bugs_4"))
    out = RESULTS / "bugs_demo.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    acc_n = sum(1 for r in all_rows if r["ordering"] == "accumulate" and r["joint_correct"])
    tick_n = sum(1 for r in all_rows if r["ordering"] == "ticket" and r["joint_correct"])
    print(f"accumulate accuracy: {acc_n}/{n_pairs} = {acc_n/n_pairs:.3f}")
    print(f"ticket accuracy:     {tick_n}/{n_pairs} = {tick_n/n_pairs:.3f}")
    by_seed: dict[int, dict[str, bool]] = {}
    for r in all_rows:
        by_seed.setdefault(r["seed"], {})[r["ordering"]] = r["joint_correct"]
    pairs = [(d["accumulate"], d["ticket"]) for d in by_seed.values()
             if "accumulate" in d and "ticket" in d]
    if pairs:
        blocked = [a for a, t in pairs]
        ticket_ok = [t for a, t in pairs]
        delta, p = mcnemar(blocked, ticket_ok)
        print(f"McNemar (accumulate vs ticket): delta={delta*100:+.1f}pp p={p:.3f}")
    print(f"written {out}")


def sweep(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    models = cfg.get("models", ["qwen2.5-coder:7b"])
    n_pairs = cfg.get("n_pairs", 10)
    seed0 = cfg.get("seed0", 1)
    n_false = cfg.get("n_false", 1)
    out = RESULTS / "bugs_sweep.jsonl"
    rows: list[dict] = []
    if out.exists():
        rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]

    for cell in cfg.get("cells", []):
        label = cell["label"]
        n_bugs = cell.get("n_bugs", 4)
        n_noise = cell.get("n_noise", 0)
        for model_name in models:
            for i in tqdm(range(n_pairs), desc=f"{label}/{model_name}"):
                seed = seed0 + i
                if any(r.get("seed") == seed and r.get("model") == model_name
                       and r.get("cell") == label for r in rows):
                    continue
                rows.extend(run_pair(seed, n_bugs, n_noise, n_false, model_name, label))

    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"written {out} ({len(rows)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(description="bugqueue: sequential defect accumulation")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--config", type=Path)
    p.add_argument("--model", default="qwen2.5-coder:7b")
    p.add_argument("--pairs", type=int, default=3)
    args = p.parse_args()

    if args.demo:
        demo(args.model, args.pairs)
    elif args.config:
        sweep(args.config)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
