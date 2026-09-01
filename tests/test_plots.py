"""Tests for the figures.

Plots are usually left untested, but two of this project's bugs were presentation bugs
(cells sorted alphabetically so `n120` preceded `n40`; a delta annotation that could
mislead), and a chart is the first thing a reader trusts.
"""

import json
from pathlib import Path

import pytest

from taskswitch.plots import _cell_label, _cell_order, dumbbell, taxonomy_bars


def _row(model, label, ordering, correct, n_tasks=2, n_ops=6, n_noise=0, failures=()):
    return {"model": model, "label": label, "cell": f"t{n_tasks}_o{n_ops}_n{n_noise}",
            "ordering": ordering, "joint_correct": correct, "parse_ok": True,
            "n_tasks": n_tasks, "n_ops": n_ops, "n_noise": n_noise,
            "failures": list(failures), "seed": 0}


ROWS = [
    _row("m:1", "len_short", "blocked", True, n_noise=0),
    _row("m:1", "len_short", "interleaved", False, n_noise=0),
    _row("m:1", "len_long", "blocked", True, n_noise=120, failures=["dropped"]),
    _row("m:1", "len_long", "interleaved", False, n_noise=120, failures=["absorbed"]),
    _row("m:1", "len_medium", "blocked", True, n_noise=40),
    _row("m:1", "len_medium", "interleaved", True, n_noise=40),
]


def test_cells_sort_by_dose_not_alphabetically():
    """Regression: sorting on the cell string put `n120` before `n40`, reading as a
    reversed dose on a chart whose whole point is the trend across padding."""
    ordered = sorted({_cell_label(r): _cell_order(r) for r in ROWS}.items(),
                     key=lambda kv: kv[1])
    assert [k for k, _ in ordered] == ["len_short", "len_medium", "len_long"]


def test_cell_order_ranks_tasks_before_ops_before_noise():
    assert _cell_order(_row("m", "a", "blocked", True, n_tasks=1)) < \
           _cell_order(_row("m", "a", "blocked", True, n_tasks=2))


def test_cell_label_prefers_the_readable_label():
    assert _cell_label(ROWS[0]) == "len_short"
    bare = dict(ROWS[0]); bare.pop("label")
    assert _cell_label(bare) == bare["cell"]


def test_dumbbell_writes_a_png(tmp_path: Path):
    out = tmp_path / "d.png"
    dumbbell(ROWS, out)
    assert out.exists() and out.stat().st_size > 1000


def test_taxonomy_bars_writes_a_png(tmp_path: Path):
    out = tmp_path / "t.png"
    taxonomy_bars(ROWS, out)
    assert out.exists() and out.stat().st_size > 1000


@pytest.mark.parametrize("fn", [dumbbell, taxonomy_bars])
def test_plots_handle_empty_input_without_crashing(fn, tmp_path: Path):
    """An empty or partial sweep must not take the analysis down with it."""
    fn([], tmp_path / "empty.png")


def test_dumbbell_survives_a_cell_with_only_one_ordering(tmp_path: Path):
    """Checkpointed sweeps can be interrupted mid-cell."""
    dumbbell([_row("m:1", "len_short", "blocked", True)], tmp_path / "partial.png")


def test_unparseable_rows_are_excluded_from_accuracy(tmp_path: Path):
    """Format failures must never be averaged into state accuracy."""
    from taskswitch.plots import _acc
    rows = [_row("m", "c", "blocked", True), dict(_row("m", "c", "blocked", False),
                                                  parse_ok=False)]
    acc, _, _, n = _acc(rows)
    assert n == 1 and acc == 1.0
