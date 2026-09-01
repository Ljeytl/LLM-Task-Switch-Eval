#!/usr/bin/env python3
"""Verify --rescore needs no inference, by checking cache keys rather than counting files.

Counting cache entries against result rows looks equivalent and is not. The `ctrl_1task`
cell has one task, so `order_ops` returns the same sequence for both orderings and the two
conversations render byte-identical prompts; the cache is keyed on the rendered prompt, so
one entry legitimately serves two rows. A count-based check reports a shortfall of exactly
the control's size and calls a complete cache incomplete -- which it did, before this.

This rebuilds each row's conversation, computes its cache key, and checks the file. Exact,
and it reports which rows would actually trigger inference.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import resolve_tasks                                   # noqa: E402
from taskswitch.generator import build_pair                     # noqa: E402
from taskswitch.runner import CACHE_DIR, cache_key, resolve_model  # noqa: E402


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/sweep.jsonl")
    if not src.exists():
        print(f"no {src}", file=sys.stderr)
        return 1
    rows = [json.loads(ln) for ln in src.read_text().splitlines() if ln.strip()]
    models = {name: resolve_model(name) for name in {r["model"] for r in rows}}

    keys, missing = set(), Counter()
    for r in rows:
        tasks = resolve_tasks(r.get("tasks") or r["n_tasks"])
        pair = build_pair(r["seed"], tasks, r["n_ops"], r["n_false"], r["n_noise"])
        conv = pair[0] if r["ordering"] == "blocked" else pair[1]
        key = cache_key(conv, models[r["model"]], r.get("constrained", True))
        keys.add(key)
        if not (CACHE_DIR / f"{key}.json").exists():
            missing[(r["model"], r.get("label") or r["cell"])] += 1

    shared = len(rows) - len(keys)
    print(f"{len(rows)} rows -> {len(keys)} distinct cache keys "
          f"({shared} rows share a key with another; expected for one-task cells, "
          f"whose two orderings render identical prompts)")
    if missing:
        print(f"MISSING {sum(missing.values())} rows would re-run inference:")
        for (m, c), n in missing.most_common():
            print(f"    {m} {c}: {n}")
        return 1
    print("cache covers every row -- --rescore needs no inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
