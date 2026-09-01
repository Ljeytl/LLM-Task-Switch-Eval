"""Tests for the ground-truth oracle.

These run before anything else exists, because every downstream number is only as
trustworthy as `ground_truth`. Two of these tests are not routine coverage -- they are
the assumptions the experiment's validity rests on, written down as assertions.
"""

import random

import pytest

from taskswitch.ops import INERT_KINDS, Op, OpKind, TaskKind
from taskswitch.state import ScheduleState, ShoppingState, ground_truth

TASKS = [TaskKind.SHOPPING, TaskKind.SCHEDULE]


def _blocked(ops: list[Op]) -> list[Op]:
    """Group by task, preserving within-task order."""
    out: list[Op] = []
    for t in TASKS:
        out.extend(o for o in ops if o.task == t)
    return out


def _interleaved(ops: list[Op]) -> list[Op]:
    """Round-robin across tasks, preserving within-task order."""
    per = {t: [o for o in ops if o.task == t] for t in TASKS}
    out: list[Op] = []
    while any(per.values()):
        for t in TASKS:
            if per[t]:
                out.append(per[t].pop(0))
    return out


def _sample(seed: int, n: int = 24) -> list[Op]:
    """A messy but well-formed op list: mutations, queries, noise, false assertions."""
    rng = random.Random(seed)
    items = ["milk", "eggs", "bread", "rice", "apples"]
    titles = ["standup", "review", "1:1", "retro"]
    ops: list[Op] = []
    live_items: set[str] = set()
    live_titles: set[str] = set()
    for i in range(n):
        task = rng.choice(TASKS)
        if task is TaskKind.SHOPPING:
            kind = rng.choice([OpKind.ADD, OpKind.ADD, OpKind.REMOVE, OpKind.QUERY,
                               OpKind.NOISE, OpKind.FALSE_ASSERT])
            item = rng.choice(sorted(live_items)) if (kind is OpKind.REMOVE and live_items) \
                else rng.choice(items)
            if kind is OpKind.ADD:
                live_items.add(item)
            if kind is OpKind.REMOVE:
                live_items.discard(item)
            payload = {"text": "by the way it rained"} if kind is OpKind.NOISE else {"item": item}
        else:
            kind = rng.choice([OpKind.ADD, OpKind.ADD, OpKind.UPDATE, OpKind.REMOVE,
                               OpKind.QUERY, OpKind.NOISE, OpKind.FALSE_ASSERT])
            title = rng.choice(sorted(live_titles)) if (
                kind in (OpKind.UPDATE, OpKind.REMOVE) and live_titles) else rng.choice(titles)
            time = f"{rng.randrange(8, 18):02d}:{rng.choice(['00', '30'])}"
            if kind is OpKind.ADD:
                live_titles.add(title)
            if kind is OpKind.REMOVE:
                live_titles.discard(title)
            payload = {"text": "by the way it rained"} if kind is OpKind.NOISE else (
                {"title": title, "new_time": time} if kind is OpKind.UPDATE
                else {"title": title, "time": time})
        ops.append(Op(task=task, kind=kind, payload=payload, idx=i))
    return ops


# --- the two load-bearing invariants -------------------------------------------------

@pytest.mark.parametrize("seed", range(30))
def test_ordering_does_not_change_ground_truth(seed: int) -> None:
    """Blocked and interleaved orderings of ONE op list must agree.

    This is what makes the headline comparison fair. Ops for different tasks touch
    disjoint state, so reordering across tasks cannot change the result -- but the
    experiment's entire claim depends on it, so it is asserted rather than assumed.
    """
    ops = _sample(seed)
    assert ground_truth(_blocked(ops), TASKS) == ground_truth(_interleaved(ops), TASKS)


@pytest.mark.parametrize("seed", range(30))
def test_inert_ops_mutate_nothing(seed: int) -> None:
    """QUERY, NOISE and FALSE_ASSERT must leave every snapshot byte-identical.

    If a FALSE_ASSERT could move state, the "does ungrounded context contaminate
    tracking?" probe would be measuring nothing. This also licenses using NOISE as the
    context-length lever: padding must not change the answer key.
    """
    ops = _sample(seed)
    mutating_only = [o for o in ops if o.kind not in INERT_KINDS]
    assert ground_truth(ops, TASKS) == ground_truth(mutating_only, TASKS)


# --- totality ------------------------------------------------------------------------

@pytest.mark.parametrize("state_cls", [ShoppingState, ScheduleState])
@pytest.mark.parametrize("payload", [{}, {"item": ""}, {"title": "   "}, {"title": "x"},
                                     {"time": "09:00"}, {"item": None}, {"title": "x", "new_time": ""}])
def test_apply_is_total(state_cls, payload) -> None:
    """apply() must never raise on any payload. A crash here would be recorded as a
    model failure, which would be a lie about the model."""
    st = state_cls()
    for kind in (OpKind.ADD, OpKind.REMOVE, OpKind.UPDATE, OpKind.QUERY):
        task = TaskKind.SHOPPING if state_cls is ShoppingState else TaskKind.SCHEDULE
        if kind is OpKind.UPDATE and state_cls is ShoppingState:
            continue
        st.apply(Op(task=task, kind=kind, payload=payload, idx=0))
    st.snapshot()


# --- documented edge-case policy -----------------------------------------------------

def test_shopping_add_is_idempotent_and_remove_absent_is_noop() -> None:
    st = ShoppingState()
    st.apply(Op(TaskKind.SHOPPING, OpKind.ADD, {"item": "Milk"}, 0))
    st.apply(Op(TaskKind.SHOPPING, OpKind.ADD, {"item": "milk"}, 1))
    assert st.snapshot() == ["milk"]
    st.apply(Op(TaskKind.SHOPPING, OpKind.REMOVE, {"item": "caviar"}, 2))
    assert st.snapshot() == ["milk"]


def test_schedule_add_is_last_write_wins_on_title() -> None:
    st = ScheduleState()
    st.apply(Op(TaskKind.SCHEDULE, OpKind.ADD, {"time": "09:00", "title": "Standup"}, 0))
    st.apply(Op(TaskKind.SCHEDULE, OpKind.ADD, {"time": "11:00", "title": "standup"}, 1))
    assert st.snapshot() == [["11:00", "standup"]]


def test_schedule_permits_double_booking() -> None:
    """Two meetings may hold one slot. Rejecting would encode a conflict rule the user
    never stated into the answer key."""
    st = ScheduleState()
    st.apply(Op(TaskKind.SCHEDULE, OpKind.ADD, {"time": "09:00", "title": "standup"}, 0))
    st.apply(Op(TaskKind.SCHEDULE, OpKind.ADD, {"time": "09:00", "title": "review"}, 1))
    assert st.snapshot() == [["09:00", "review"], ["09:00", "standup"]]


def test_schedule_update_absent_title_is_noop() -> None:
    st = ScheduleState()
    st.apply(Op(TaskKind.SCHEDULE, OpKind.UPDATE, {"title": "ghost", "new_time": "12:00"}, 0))
    assert st.snapshot() == []


def test_snapshot_is_json_round_trippable() -> None:
    """Snapshots are compared against JSON parsed from the model, so they must survive
    a JSON round trip unchanged (lists, not tuples or sets)."""
    import json
    ops = _sample(7)
    gt = ground_truth(ops, TASKS)
    assert json.loads(json.dumps(gt)) == gt


def test_untracked_task_ops_are_ignored() -> None:
    """An op for a task this conversation is not tracking must not create a bucket."""
    ops = [Op(TaskKind.SCHEDULE, OpKind.ADD, {"time": "09:00", "title": "standup"}, 0)]
    assert ground_truth(ops, [TaskKind.SHOPPING]) == {"shopping": []}


def test_illegal_op_construction_is_rejected() -> None:
    with pytest.raises(ValueError):
        Op(TaskKind.SHOPPING, OpKind.UPDATE, {"item": "milk"}, 0)
