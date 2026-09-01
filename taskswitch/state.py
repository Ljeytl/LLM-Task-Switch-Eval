"""State machines: the ground-truth oracle.

This module is the reason the project needs no judge model. A plain state machine
consumes the same `Op` list the generator emitted and produces the expected final
state, so the transcript and its answer key come out of one function as a pair and
grading reduces to a dict comparison.

Two properties are load-bearing and are asserted in `tests/test_state.py`:

1. **Inert ops mutate nothing.** QUERY, NOISE and FALSE_ASSERT must leave every
   snapshot byte-identical. If a FALSE_ASSERT could move state, the "did ungrounded
   context contaminate tracking?" probe would be meaningless.
2. **Order independence across tasks.** Applying one op list in blocked order and in
   interleaved order must yield the SAME ground truth, because ops for different tasks
   touch disjoint state. This is precisely what makes the blocked-vs-interleaved
   comparison fair, so it is asserted rather than assumed.

Every `apply` must be **total**: it never raises, whatever it is handed. A crash here
would be scored as a model failure, which would be a lie.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .ops import Op, OpKind, TaskKind


@runtime_checkable
class TaskState(Protocol):
    def apply(self, op: Op) -> None: ...
    def snapshot(self) -> Any: ...   # JSON-serialisable, comparable with ==


class ShoppingState:
    """A set of item names.

    Edge-case policy (documented because it defines what "correct" means):
      - ADD of an item already present is idempotent. A set has no multiplicity, and a
        user saying "add milk" twice plainly still wants one milk.
      - REMOVE of an absent item is a no-op rather than an error. The user asked for a
        postcondition ("no milk"), and that postcondition already holds.
    Both choices keep `apply` total.
    """

    def __init__(self) -> None:
        self._items: set[str] = set()

    def apply(self, op: Op) -> None:
        if not op.mutating:
            return
        item = str(op.payload.get("item", "")).strip().lower()
        if not item:
            return
        if op.kind is OpKind.ADD:
            self._items.add(item)
        elif op.kind is OpKind.REMOVE:
            self._items.discard(item)

    def snapshot(self) -> list[str]:
        """Sorted so equality is order-insensitive: a set has no intrinsic order, and
        grading the model on recall order would be grading the wrong thing."""
        return sorted(self._items)


class ScheduleState:
    """An ordered map from time to meeting title.

    Backed by a dict keyed on title (titles are the user-facing handle for move and
    remove) with the time as the value. `snapshot` sorts by time, so ordering is
    derived rather than stored.
    """

    def __init__(self) -> None:
        self._by_title: dict[str, str] = {}  # title -> "HH:MM"

    def apply(self, op: Op) -> None:
        """Mutation policy, chosen so the oracle is total and the semantics are
        defensible out loud:

        - ADD is last-write-wins on the title. Titles are the user-facing handle, so
          re-adding a title reads as restating its time, not as creating a second
          meeting with the same name.
        - ADD does **not** reject a time another meeting already holds. Real calendars
          double-book, and rejecting would mean inventing a conflict-resolution rule
          the user never stated -- which would bake a policy into the answer key that
          the model was never told about.
        - REMOVE of an absent title is a silent no-op, matching ShoppingState: the user
          asked for a postcondition and it already holds.
        - UPDATE of an absent title is a no-op for a *different* reason worth keeping
          straight: moving a meeting that does not exist has no achievable
          postcondition at all. Nothing to do, rather than nothing left to do.

        The generator only emits REMOVE/UPDATE against titles it knows are present, so
        these branches are a safety net rather than a routine path. They are still
        tested, because a total oracle is the entire basis for trusting the scores.
        """
        if not op.mutating:
            return

        title = str(op.payload.get("title", "")).strip().lower()
        if not title:
            return

        if op.kind is OpKind.ADD:
            time = str(op.payload.get("time", "")).strip()
            if time:
                self._by_title[title] = time
        elif op.kind is OpKind.REMOVE:
            self._by_title.pop(title, None)
        elif op.kind is OpKind.UPDATE:
            new_time = str(op.payload.get("new_time", "")).strip()
            if new_time and title in self._by_title:
                self._by_title[title] = new_time

    def snapshot(self) -> list[list[str]]:
        """`[[time, title], ...]` sorted by time, then title to break ties
        deterministically. Lists rather than tuples so the snapshot round-trips
        through JSON unchanged and compares equal to a parsed model response."""
        return [[t, n] for n, t in sorted(self._by_title.items(), key=lambda kv: (kv[1], kv[0]))]


_CONSTRUCTORS: dict[TaskKind, type] = {
    TaskKind.SHOPPING: ShoppingState,
    TaskKind.SCHEDULE: ScheduleState,
}


def build_states(tasks: list[TaskKind]) -> dict[TaskKind, TaskState]:
    """Fresh, independent state machines for the tasks a conversation tracks."""
    return {t: _CONSTRUCTORS[t]() for t in tasks}


def ground_truth(ops: list[Op], tasks: list[TaskKind]) -> dict[str, Any]:
    """Apply every op in sequence and return `{task_name: snapshot}`.

    Ops addressed to a task the conversation is not tracking are ignored, so a stray
    op can never silently create an untracked state bucket. NOISE and FALSE_ASSERT
    mutate nothing by construction (see `Op.mutating`).
    """
    states = build_states(tasks)
    for op in ops:
        st = states.get(op.task)
        if st is not None:
            st.apply(op)
    return {t.value: states[t].snapshot() for t in tasks}
