"""Tests for bugqueue — no model required."""

from bugqueue.generator import Ordering, build_pair, order_core, sample_core_ops
from bugqueue.ops import OpKind
from bugqueue.state import ground_truth


def test_accumulate_and_ticket_same_final_state():
    acc, ticket = build_pair(seed=42, n_bugs=4, n_noise=2, n_false=1)
    assert acc.expected == ticket.expected
    assert acc.expected["open_bugs"] == []


def test_ticket_interleaves_report_fix():
    core, _ = sample_core_ops(1, 3)
    ordered = order_core(core, Ordering.TICKET)
    for i in range(0, len(ordered) - 1, 2):
        assert ordered[i].kind is OpKind.REPORT
        assert ordered[i + 1].kind is OpKind.FIX


def test_accumulate_piles_reports_first():
    core, _ = sample_core_ops(1, 3)
    ordered = order_core(core, Ordering.ACCUMULATE)
    reports = [i for i, o in enumerate(ordered) if o.kind is OpKind.REPORT]
    fixes = [i for i, o in enumerate(ordered) if o.kind is OpKind.FIX]
    assert max(reports) < min(fixes)


def test_ground_truth_all_fixed_at_end():
    acc, _ = build_pair(seed=99, n_bugs=4)
    assert acc.expected["open_bugs"] == []
    for sym in acc.symbols:
        assert acc.expected["symbols"][sym.name] == sym.expected
