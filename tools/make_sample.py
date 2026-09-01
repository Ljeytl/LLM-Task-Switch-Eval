#!/usr/bin/env python3
"""Cut a small, *stratified* sample of result rows for committing to the repo.

The point of a committed sample is that a reviewer who runs nothing still sees what the
harness produces. `head -n 40` does not achieve that: rows are written cell by cell, so
the first 40 are 40 rows of one condition from one model — the reviewer sees the negative
control and concludes the tool only does negative controls.

This takes a fixed number of pairs from every (model, cell, ordering) group, so the sample
spans the whole design, and it keeps both halves of each pair together: a blocked row
whose interleaved twin is missing cannot demonstrate the one comparison the project makes.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PAIRS_PER_CELL = 2


def stratified(rows: list[dict], pairs_per_cell: int = PAIRS_PER_CELL) -> list[dict]:
    by_pair: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_pair[(r["model"], r["cell"], r["seed"])][r["ordering"]] = r

    picked: dict[tuple, int] = defaultdict(int)
    out: list[dict] = []
    # Sort so the selection is deterministic regardless of row order on disk.
    for key in sorted(by_pair, key=lambda k: (k[0], k[1], k[2])):
        sides = by_pair[key]
        if len(sides) < 2:
            continue                      # never ship half a pair
        cell = (key[0], key[1])
        if picked[cell] >= pairs_per_cell:
            continue
        picked[cell] += 1
        out.append(sides["blocked"])
        out.append(sides["interleaved"])
    return out


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/sweep.jsonl")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "results/sample.jsonl")
    if not src.exists():
        print(f"no {src}; nothing to sample", file=sys.stderr)
        return 1
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    sample = stratified(rows)
    dst.write_text("".join(json.dumps(r) + "\n" for r in sample))
    cells = sorted({(r["model"], r.get("label") or r["cell"]) for r in sample})
    print(f"wrote {dst} ({len(sample)} rows, {len(cells)} model-condition cells)")
    for model, label in cells:
        print(f"    {model:<20} {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
