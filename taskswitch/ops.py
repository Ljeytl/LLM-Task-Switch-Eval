"""Operation grammar.

An `Op` is the atomic unit of this experiment. Every conversation is generated from a
list of Ops, and the *same* list is fed to two places: `surface.render` turns each Op
into a natural-language user turn, and `state.ground_truth` applies each Op to a state
machine. Because both sides consume one list, the transcript and its expected answer
are produced as a pair and grading is a diff rather than a judgement.

Two op kinds deliberately do not advance state:

- NOISE carries a `task` anyway, so filler turns are *topically* about a real task.
  A distractor that mentions groceries is a harder distractor than a generic aside,
  and this is also the lever used to vary context length independently of state load.
- FALSE_ASSERT claims a state change that must NOT land. It measures whether ungrounded
  context contaminates tracked state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class TaskKind(str, Enum):
    """The concurrent tasks a conversation tracks.

    Chosen so that misattribution is *visible*: a grocery item appearing in the
    schedule is unambiguous. Two shopping lists would not be. Deliberately two,
    not three -- an expense tracker (running sum) was cut because arithmetic error
    would confound state-tracking error.
    """

    SHOPPING = "shopping"  # backed by a set
    SCHEDULE = "schedule"  # backed by an ordered map (time -> title)


class OpKind(str, Enum):
    ADD = "add"
    REMOVE = "remove"
    UPDATE = "update"
    QUERY = "query"
    NOISE = "noise"                # not an operation; advances no state
    FALSE_ASSERT = "false_assert"  # claims a change that must NOT land


#: Op kinds that must leave every state machine byte-identical. Asserted in tests.
INERT_KINDS: frozenset[OpKind] = frozenset({OpKind.QUERY, OpKind.NOISE, OpKind.FALSE_ASSERT})

#: Which (task, kind) pairs are legal. A set has no UPDATE; an ordered map does.
LEGAL_OPS: Mapping[TaskKind, frozenset[OpKind]] = MappingProxyType({
    TaskKind.SHOPPING: frozenset({OpKind.ADD, OpKind.REMOVE, OpKind.QUERY,
                                  OpKind.NOISE, OpKind.FALSE_ASSERT}),
    TaskKind.SCHEDULE: frozenset({OpKind.ADD, OpKind.REMOVE, OpKind.UPDATE, OpKind.QUERY,
                                  OpKind.NOISE, OpKind.FALSE_ASSERT}),
})


#: Distinct named instances per task kind. Task identity used to be TaskKind itself,
#: which capped the design at two concurrent tasks and silently merged a third into the
#: first (see docs/DECISIONS.md D14). Naming the instances lifts that cap AND restores
#: the taxonomy's diagnostic power: each instance draws from its own vocabulary, so an
#: item in the wrong list is unmistakable rather than merely wrong.
SLOT_NAMES: Mapping[TaskKind, tuple[str, ...]] = MappingProxyType({
    TaskKind.SHOPPING: ("grocery list", "hardware list", "pharmacy list", "garden list"),
    TaskKind.SCHEDULE: ("work calendar", "personal calendar", "team calendar",
                        "family calendar"),
})


def slot_key(task: TaskKind, slot: int) -> str:
    """Stable identifier for one task instance, e.g. `shopping_0`.

    Used as the dict key everywhere state is held, so two shopping lists are two
    independent states rather than one merged one.
    """
    return f"{task.value}_{slot}"


def slot_label(task: TaskKind, slot: int) -> str:
    """Human-facing name for a task instance, e.g. `hardware list`."""
    names = SLOT_NAMES[task]
    return names[slot % len(names)]


def assign_slots(tasks: list[TaskKind]) -> list[int]:
    """Per-kind instance index for each position: [SHOP, SCHED, SHOP] -> [0, 0, 1]."""
    seen: dict[TaskKind, int] = {}
    out: list[int] = []
    for t in tasks:
        out.append(seen.get(t, 0))
        seen[t] = seen.get(t, 0) + 1
    return out


@dataclass(frozen=True)
class Op:
    """One operation in the emitted sequence.

    `payload` shape depends on (task, kind); see `state.py` for the contract:

        SHOPPING ADD/REMOVE/QUERY/FALSE_ASSERT  {"item": str}
        SCHEDULE ADD/FALSE_ASSERT               {"time": "HH:MM", "title": str}
        SCHEDULE REMOVE/QUERY                   {"title": str}
        SCHEDULE UPDATE                         {"title": str, "new_time": "HH:MM"}
        *        NOISE                          {}   -- carries no payload; the filler
                                                     text lives entirely in the surface
                                                     templates, so noise cannot smuggle
                                                     task content into a turn

    `idx` is the position in the *emitted* sequence, before ordering is applied. It is
    what lets a blocked and an interleaved conversation be recognised as carrying the
    identical operations, and it is preserved through reordering.

    `slot` distinguishes two instances of the same kind -- a grocery list and a hardware
    list are two independent states, not one.
    """

    task: TaskKind
    kind: OpKind
    payload: dict[str, Any]
    idx: int
    slot: int = 0     # which instance of `task` this op addresses

    @property
    def key(self) -> str:
        """The state bucket this op belongs to."""
        return slot_key(self.task, self.slot)

    def __post_init__(self) -> None:
        if self.kind not in LEGAL_OPS[self.task]:
            raise ValueError(f"{self.kind.value!r} is not a legal op for {self.task.value!r}")

    @property
    def mutating(self) -> bool:
        """True if this op is expected to change state. The negation of INERT_KINDS."""
        return self.kind not in INERT_KINDS
