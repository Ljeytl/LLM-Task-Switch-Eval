"""Tests for the CLI entry point's own helpers.

These exist because of a specific failure. `run.py:tasks_for` kept a two-task cap long
after D17 lifted it in the library, and the sweep crashed on the first task-count cell --
while 543 tests passed, because every one of them called `build_pair` directly and none
ever went through the entry point. Green tests coexisted with a broken binary. Anything
`run.py` computes on its own is now covered here.
"""

from __future__ import annotations

import pytest
import yaml

import run as R
from taskswitch.generator import build_pair
from taskswitch.ops import TaskKind, default_tasks, kind_signature


class TestTasksFor:
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
    def test_length_matches_request(self, n):
        assert len(R.tasks_for(n)) == n

    def test_no_cap_at_two(self):
        """The regression itself: this raised before D17 was carried into the CLI."""
        assert len(R.tasks_for(4)) == 4

    def test_cycles_the_pool(self):
        assert R.tasks_for(3) == [TaskKind.SHOPPING, TaskKind.SCHEDULE, TaskKind.SHOPPING]

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_every_count_the_sweep_uses_actually_builds(self, n):
        """End-to-end through the real generator, which is what `tasks_for` feeds."""
        blocked, inter = build_pair(1, R.tasks_for(n), n_ops=6, n_false=2, n_noise=10)
        assert blocked.token_count == inter.token_count
        assert len(blocked.expected) == n, "one state per task instance"


class TestResolveTasks:
    def test_int_matches_tasks_for(self):
        assert R.resolve_tasks(3) == R.tasks_for(3)

    def test_explicit_names(self):
        assert R.resolve_tasks(["shopping", "shopping"]) == [TaskKind.SHOPPING] * 2

    def test_accepts_kind_objects(self):
        assert R.resolve_tasks([TaskKind.SCHEDULE]) == [TaskKind.SCHEDULE]

    def test_case_and_whitespace_insensitive(self):
        assert R.resolve_tasks([" ShOpPiNg "]) == [TaskKind.SHOPPING]

    def test_unknown_kind_is_rejected_with_the_legal_set(self):
        with pytest.raises(SystemExit, match="legal kinds"):
            R.resolve_tasks(["expenses"])

    def test_empty_is_rejected(self):
        with pytest.raises(SystemExit):
            R.resolve_tasks([])

    def test_same_kind_pair_yields_two_independent_states(self):
        tasks = R.resolve_tasks(["shopping", "shopping"])
        blocked, _ = build_pair(1, tasks, n_ops=6, n_false=2, n_noise=10)
        assert set(blocked.expected) == {"shopping_0", "shopping_1"}


class TestCellId:
    """The id has to stay stable for canonical compositions -- committed results are
    addressed by it -- while distinguishing compositions a bare count cannot."""

    @pytest.mark.parametrize("n,expected", [
        (1, "t1_o6_n40"), (2, "t2_o6_n40"), (3, "t3_o6_n40"), (4, "t4_o6_n40"),
    ])
    def test_canonical_ids_are_unchanged(self, n, expected):
        blocked, _ = build_pair(1, default_tasks(n), 6, 2, 40)
        assert blocked.cell == expected

    def test_non_canonical_composition_gets_a_signature(self):
        blocked, _ = build_pair(1, [TaskKind.SHOPPING, TaskKind.SHOPPING], 6, 2, 40)
        assert blocked.cell == "t2shsh_o6_n40"

    def test_signature_distinguishes_same_count_compositions(self):
        a, _ = build_pair(1, [TaskKind.SHOPPING, TaskKind.SCHEDULE], 6, 2, 40)
        b, _ = build_pair(1, [TaskKind.SHOPPING, TaskKind.SHOPPING], 6, 2, 40)
        assert a.cell != b.cell, "otherwise the two collapse into one row group"

    def test_order_is_reflected(self):
        assert kind_signature([TaskKind.SCHEDULE, TaskKind.SHOPPING]) == "scsh"
        assert kind_signature([TaskKind.SHOPPING, TaskKind.SCHEDULE]) == "shsc"

    def test_both_orderings_of_a_pair_share_a_cell(self):
        blocked, inter = build_pair(1, [TaskKind.SHOPPING, TaskKind.SHOPPING], 6, 2, 40)
        assert blocked.cell == inter.cell, "cell groups the pair; ordering separates it"


class TestShippedConfig:
    """The config is a deliverable: a typo in it breaks the sweep, not a test."""

    @pytest.fixture(scope="class")
    def cfg(self):
        with open("configs/main.yaml") as fh:
            return yaml.safe_load(fh)

    def test_a_cell_needs_one_of_tasks_or_n_tasks(self):
        with pytest.raises(SystemExit, match="needs"):
            R.cell_tasks({"label": "broken", "n_ops": 6, "n_noise": 0})

    def test_explicit_tasks_cell_needs_no_n_tasks(self):
        """`cell.get("tasks", cell["n_tasks"])` evaluates its default eagerly and raised
        KeyError on precisely the cells this feature was added for."""
        assert R.cell_tasks({"tasks": ["shopping", "shopping"]}) == [TaskKind.SHOPPING] * 2

    def test_every_cell_resolves(self, cfg):
        for cell in cfg["cells"]:
            tasks = R.cell_tasks(cell)
            assert tasks, cell["label"]

    def test_cell_ids_are_unique(self, cfg):
        ids = []
        for cell in cfg["cells"]:
            tasks = R.cell_tasks(cell)
            blocked, _ = build_pair(1, tasks, cell["n_ops"], cfg["n_false"],
                                    cell["n_noise"])
            ids.append(blocked.cell)
        assert len(ids) == len(set(ids)), f"two cells share a row group: {ids}"

    def test_labels_are_unique(self, cfg):
        labels = [c["label"] for c in cfg["cells"]]
        assert len(labels) == len(set(labels))

    def test_control_cell_has_one_task(self, cfg):
        ctrl = next(c for c in cfg["cells"] if c["label"] == "ctrl_1task")
        assert len(R.cell_tasks(ctrl)) == 1

    def test_every_cell_pairs_without_token_drift(self, cfg):
        """If a shipped cell cannot produce a token-matched pair, the sweep dies mid-run
        after burning GPU hours on the cells before it."""
        for cell in cfg["cells"]:
            tasks = R.cell_tasks(cell)
            blocked, inter = build_pair(1, tasks, cell["n_ops"], cfg["n_false"],
                                        cell["n_noise"])
            assert blocked.token_count == inter.token_count, cell["label"]
