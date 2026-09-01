"""Tests for bugqueue — no model required."""

from bugqueue.generator import Ordering, build_pair, order_core, sample_core_ops
from bugqueue.ops import OpKind
from bugqueue.state import ground_truth


def test_accumulate_and_ticket_same_final_state():
    acc, ticket = build_pair(seed=42, n_bugs=4, n_noise=2, n_false=1)
    assert acc.expected == ticket.expected
    assert acc.expected["open_bugs"] == []
    assert all(v for v in acc.expected["symbols"].values())


def test_ticket_interleaves_report_fix():
    core, _ = sample_core_ops(1, 3)
    ordered = order_core(core, Ordering.TICKET)
    kinds = [o.kind for o in ordered]
    assert kinds.count(OpKind.REPORT) == 3
    assert kinds.count(OpKind.FIX) == 3
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
    exp = acc.expected
    assert exp["open_bugs"] == []
    for sym in acc.symbols:
        assert exp["symbols"][sym.name] == sym.expected


def test_open_bugs_mid_accumulate():
    core, symbols = sample_core_ops(7, 3)
    # Only reports — bugs should stay open
    reports = [o for o in core if o.kind is OpKind.REPORT]
    mid = ground_truth(reports, symbols)
    assert len(mid["open_bugs"]) == 3


def test_ordering_does_not_change_ground_truth_when_fully_fixed():
    core, symbols = sample_core_ops(5, 4)
    from bugqueue.generator import order_core, sample_inert_ops
    inert = sample_inert_ops(5, 0, 0, symbols)
    for ordering in (Ordering.ACCUMULATE, Ordering.TICKET):
        seq = inert + order_core(core, ordering)
        gt = ground_truth(seq, symbols)
        assert gt["open_bugs"] == []
