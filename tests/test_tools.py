"""Tests for the analysis tools.

The taxonomy audit found a real scorer bug (D15) after being wrong twice itself — its
identity parser split on whitespace, turning "olive oil" into "olive". A verifier is code
too, so it gets tests.
"""

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from make_sample import stratified  # noqa: E402
from audit_taxonomy import (BUCKETS, canon_schedule, canon_shopping, entries,  # noqa: E402
                            grounding_pass, identities, split_detail)


# --- detail parsing ------------------------------------------------------------------

@pytest.mark.parametrize("detail,expected", [
    ("shopping_0:milk", ("shopping_0", "milk", "")),
    ("shopping_0:olive oil", ("shopping_0", "olive oil", "")),
    ("schedule_0:one-on-one wrong value", ("schedule_0", "one-on-one", "wrong value")),
    ("shopping_1:wood glue false_assert", ("shopping_1", "wood glue", "false_assert")),
    ("schedule_0:demo hallucination", ("schedule_0", "demo", "hallucination")),
    ("schedule_0:retro stale", ("schedule_0", "retro", "stale")),
    ("shopping_0:milk -> other task", ("shopping_0", "milk", "-> other task")),
    ("shopping_0:screws from other task", ("shopping_0", "screws", "from other task")),
])
def test_split_detail_handles_multiword_identities_and_every_suffix(detail, expected):
    """Regression: splitting on whitespace turned "olive oil" into "olive", so the
    grounding check looked up a name that never existed and reported six false
    ungrounded failures."""
    assert split_detail(detail) == expected


def test_split_detail_does_not_mistake_a_name_ending_in_a_suffix_word():
    assert split_detail("shopping_0:sticky tape")[1] == "sticky tape"


# --- canonicalisation ----------------------------------------------------------------

def test_canon_schedule_reads_named_keys_and_both_positional_orders():
    """A positional pair has no intrinsic order; decide by which element is a time."""
    assert canon_schedule([{"time": "09:00", "title": "Standup"}]) == {"standup"}
    assert canon_schedule([["09:00", "standup"]]) == {"standup"}
    assert canon_schedule([["standup", "09:00"]]) == {"standup"}


def test_canon_shopping_normalises_and_drops_blanks():
    assert canon_shopping([" Milk ", "", "EGGS"]) == {"milk", "eggs"}


def test_entries_keeps_values_so_a_wrong_time_is_visible():
    """An identity-only view cannot see "present but at the wrong hour"."""
    e = entries({"schedule_0": [{"time": "09:00", "title": "standup"}]})
    assert e["schedule_0"] == {("09:00", "standup")}


def test_identities_are_keyed_by_slot():
    got = identities({"shopping_0": ["milk"], "shopping_1": ["screws"]})
    assert got == {"shopping_0": {"milk"}, "shopping_1": {"screws"}}


# --- grounding pass ------------------------------------------------------------------

def _row(expected, reported, details, kinds):
    return {"parse_ok": True, "expected": expected, "reported": reported,
            "failure_details": details, "failures": kinds, "seed": 1,
            "ordering": "blocked", "model": "m"}


def test_grounding_accepts_a_real_dropped_item():
    row = _row({"shopping_0": ["milk", "olive oil"]}, {"shopping_0": ["milk"]},
               ["shopping_0:olive oil"], ["dropped"])
    checked, ungrounded, _ = grounding_pass([row])
    assert (checked, ungrounded) == (1, 0)


def test_grounding_accepts_a_real_stale_duplicate():
    row = _row({"schedule_0": [["14:30", "standup"]]},
               {"schedule_0": [{"time": "11:30", "title": "standup"},
                               {"time": "14:30", "title": "standup"}]},
               ["schedule_0:standup stale"], ["absorbed"])
    checked, ungrounded, _ = grounding_pass([row])
    assert (checked, ungrounded) == (1, 0)


def test_grounding_rejects_a_failure_that_names_nothing_real():
    """The check must actually be capable of failing, or it proves nothing."""
    row = _row({"shopping_0": ["milk"]}, {"shopping_0": ["milk"]},
               ["shopping_0:caviar"], ["dropped"])
    checked, ungrounded, problems = grounding_pass([row])
    assert (checked, ungrounded) == (1, 1) and problems


def test_grounding_skips_unparseable_rows():
    row = _row({"shopping_0": ["milk"]}, None, [], [])
    row["parse_ok"] = False
    assert grounding_pass([row]) == (0, 0, [])


# --- CLI smoke -----------------------------------------------------------------------

@pytest.mark.parametrize("script", ["tools/power_analysis.py", "tools/audit_taxonomy.py"])
def test_tool_runs_end_to_end_on_a_small_file(script, tmp_path: Path):
    data = tmp_path / "rows.jsonl"
    rows = []
    for seed in range(6):
        for ordering, ok in (("blocked", True), ("interleaved", seed % 2 == 0)):
            rows.append({"model": "m:1", "label": "c", "cell": "t2_o6_n0", "seed": seed,
                         "ordering": ordering, "joint_correct": ok, "parse_ok": True,
                         "per_task_correct": {"shopping_0": ok}, "failures": [],
                         "failure_details": [], "expected": {"shopping_0": []},
                         "reported": {"shopping_0": []}, "n_tasks": 2, "n_ops": 6,
                         "n_noise": 0})
    data.write_text("\n".join(json.dumps(r) for r in rows))
    r = subprocess.run([sys.executable, str(ROOT / script), str(data)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-800:]


def test_buckets_cover_every_failure_enum_value():
    from taskswitch.scorer import Failure
    assert set(BUCKETS) == {f.value for f in Failure}


class TestStratifiedSample:
    """The committed sample is the only output a reviewer who runs nothing will see.
    `head -n 40` gave 40 rows of one condition from one model, because rows are written
    cell by cell — the reviewer would have concluded the harness only runs controls."""

    @staticmethod
    def _rows(models=("m1", "m2"), cells=("c1", "c2", "c3"), seeds=range(5)):
        return [{"model": m, "cell": c, "seed": s, "ordering": o, "label": c}
                for m in models for c in cells for s in seeds
                for o in ("blocked", "interleaved")]

    def test_spans_every_model_and_cell(self):
        out = stratified(self._rows())
        assert {(r["model"], r["cell"]) for r in out} == {
            (m, c) for m in ("m1", "m2") for c in ("c1", "c2", "c3")}

    def test_keeps_both_halves_of_every_pair(self):
        out = stratified(self._rows())
        seen = defaultdict(set)
        for r in out:
            seen[(r["model"], r["cell"], r["seed"])].add(r["ordering"])
        assert all(v == {"blocked", "interleaved"} for v in seen.values())

    def test_respects_the_per_cell_budget(self):
        out = stratified(self._rows(), pairs_per_cell=2)
        per_cell = Counter((r["model"], r["cell"]) for r in out)
        assert set(per_cell.values()) == {4}, "2 pairs = 4 rows per cell"

    def test_drops_unpaired_rows(self):
        rows = [{"model": "m", "cell": "c", "seed": 1, "ordering": "blocked"}]
        assert stratified(rows) == [], "half a pair demonstrates no comparison"

    def test_is_deterministic_regardless_of_row_order(self):
        rows = self._rows()
        a = stratified(rows)
        b = stratified(list(reversed(rows)))
        assert [(r["model"], r["cell"], r["seed"], r["ordering"]) for r in a] == \
               [(r["model"], r["cell"], r["seed"], r["ordering"]) for r in b]


class TestReadmeTableMarkers:
    """The README carried a hand-maintained copy of the results table and it drifted --
    it still showed v1 numbers after v2 replaced them. It is generated between markers
    now; if the markers go missing the generator degrades to a warning on stderr and the
    stale table silently survives, so their presence is asserted."""

    @staticmethod
    def _readme():
        return Path("README.md").read_text()

    def test_markers_present_and_ordered(self):
        text = self._readme()
        begin, end = "<!-- BEGIN:results-table -->", "<!-- END:results-table -->"
        assert begin in text and end in text, "generator would silently skip the table"
        assert text.index(begin) < text.index(end)

    def test_exactly_one_marker_pair(self):
        text = self._readme()
        assert text.count("<!-- BEGIN:results-table -->") == 1
        assert text.count("<!-- END:results-table -->") == 1

    def test_region_holds_a_table_not_prose(self):
        text = self._readme()
        body = text.split("<!-- BEGIN:results-table -->")[1].split(
            "<!-- END:results-table -->")[0].strip()
        lines = [ln for ln in body.splitlines() if ln.strip()]
        assert lines, "results table region is empty"
        assert all(ln.startswith("|") for ln in lines), "region must hold only the table"
