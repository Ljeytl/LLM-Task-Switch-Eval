"""Figures for the matched-token ordering deltas and failure taxonomy."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .stats import wilson  # noqa: E402

BLOCKED_C, INTER_C = "#4C78A8", "#E45756"


def _acc(rows: list[dict[str, Any]]) -> tuple[float, float, float, int]:
    ok = [r for r in rows if r["parse_ok"]]
    n = len(ok)
    if n == 0:
        return (0.0, 0.0, 0.0, 0)
    s = sum(1 for r in ok if r["joint_correct"])
    lo, hi = wilson(s, n)
    return (s / n, lo, hi, n)


def _same_kind_pairs(row: dict[str, Any]) -> int:
    """How many pairs of same-kind task instances the conversation held.

    Derived from the ground-truth keys (`shopping_0`, `shopping_1`, ...) rather than
    stored, so it is correct for rows written before compositions were configurable.
    Same-kind pairs make cross-slot misattribution easier to expose because different
    kinds draw from disjoint vocabularies.
    """
    kinds: dict[str, int] = defaultdict(int)
    for slot in (row.get("expected") or {}):
        kinds[slot.rsplit("_", 1)[0]] += 1
    return sum(c * (c - 1) // 2 for c in kinds.values())


def _cell_order(row: dict[str, Any]) -> tuple:
    """Sort conditions the way they vary, not alphabetically.

    Sorting on the cell string puts `n120` before `n40`, which reads as a reversed dose
    on a chart whose whole point is the trend across padding.

    Same-kind pair count is part of the key because without it `len_medium` and
    `same_kind_2` are identical -- same task count, ops and noise -- and tie. They would
    then order by dict insertion, which is to say by whatever the sweep happened to run
    first, and the one dimension that actually distinguishes them would be the one the
    chart ignored.
    """
    return (row.get("n_tasks", 0), _same_kind_pairs(row),
            row.get("n_ops", 0), row.get("n_noise", 0))


def _cell_label(row: dict[str, Any]) -> str:
    return row.get("label") or row["cell"]


def dumbbell(rows: list[dict[str, Any]], out_path: Path) -> None:
    """One row per (model, condition): blocked accuracy and interleaved accuracy joined
    by a line. A long line means ordering mattered; a flat one means it did not."""
    groups: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    order: dict[tuple[str, str], tuple] = {}
    for r in rows:
        key = (r["model"], _cell_label(r))
        groups[key][r["ordering"]].append(r)
        order[key] = _cell_order(r)

    # Reverse so the first condition sits at the top of the chart.
    keys = sorted(groups, key=lambda k: (k[0], order[k]), reverse=True)
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(keys) + 2.2))
    for y, key in enumerate(keys):
        b, lo_b, hi_b, _nb = _acc(groups[key]["blocked"])
        i, lo_i, hi_i, _ni = _acc(groups[key]["interleaved"])
        ax.plot([b, i], [y, y], color="#999", lw=2, zorder=1)
        ax.hlines(y, lo_b, hi_b, color=BLOCKED_C, lw=1, alpha=.5)
        ax.hlines(y, lo_i, hi_i, color=INTER_C, lw=1, alpha=.5)
        ax.scatter([b], [y], color=BLOCKED_C, s=70, zorder=3,
                   label="blocked" if y == 0 else None)
        ax.scatter([i], [y], color=INTER_C, s=70, zorder=3,
                   label="interleaved" if y == 0 else None)
        ax.text(1.02, y, f"{(i-b)*100:+.1f} pp", va="center", fontsize=8,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{m.split(':')[0]}  {c}" for m, c in keys], fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel("joint goal accuracy (Wilson 95% CI)")
    ax.set_title("Switch cost: same operations, same tokens, different ordering")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=.25); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)


def taxonomy_bars(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Failure composition by model and ordering. Which failure dominates is the more
    interesting result -- forgetting and misfiling are different problems."""
    kinds = ["dropped", "misattributed", "absorbed", "format"]
    colors = ["#4C78A8", "#E45756", "#F58518", "#B279A2"]
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["ordering"])].extend(r["failures"])

    keys = sorted(groups)
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(keys) + 2.2))
    for y, key in enumerate(keys):
        total = len(groups[key]) or 1
        left = 0.0
        for kind, col in zip(kinds, colors):
            w = sum(1 for f in groups[key] if f == kind) / total
            if w:
                ax.barh(y, w, left=left, color=col, height=.62,
                        label=kind if y == 0 else None)
            left += w
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{m.split(':')[0]}  {o}" for m, o in keys], fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel("share of failures")
    ax.set_title("Failure composition")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.18),
              frameon=False, fontsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)
