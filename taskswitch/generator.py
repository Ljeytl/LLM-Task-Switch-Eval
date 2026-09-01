"""Turns an operation sample into a matched pair of conversations.

This is where the experiment's central guarantee is made. A blocked conversation and
an interleaved conversation are built from ONE op sample and ONE rendering pass, then
the *already-rendered turns* are reordered. Rendering once is not an optimisation --
rendering twice would draw from the RNG in a different order, pick different
paraphrases, and silently break the token match the whole comparison depends on.

Because both orderings are the identical multiset of strings in a different sequence,
their token counts are equal by construction rather than by estimate. `build_pair`
asserts the multiset equality directly (exact, free) and the runner separately verifies
the real tokenisation at run time via Ollama's `prompt_eval_count`.

Three quantities are varied independently, which is the point of the design:
  n_tasks  how many states are live at once
  n_ops    how many state updates occur      (total per conversation, not per task)
  n_noise  how much task-irrelevant context surrounds them  -> the length lever
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import Any

from .ops import (Op, OpKind, TaskKind, assign_slots, default_tasks, kind_signature,
                  slot_key)
from .state import ground_truth
from .surface import render, render_final_request, system_prompt, vocabulary


class Ordering(str, Enum):
    BLOCKED = "blocked"
    INTERLEAVED = "interleaved"


@dataclass
class Conversation:
    seed: int
    tasks: list[TaskKind]
    ordering: Ordering
    ops: list[Op]
    turns: list[str]
    expected: dict[str, Any]
    token_count: int
    n_ops: int = 0
    n_noise: int = 0
    n_false: int = 0
    system: str = ""
    final_request: str = ""

    @property
    def cell(self) -> str:
        """Condition label used to group rows in analysis.

        A bare `t{n}` cannot distinguish `[shopping, schedule]` from
        `[shopping, shopping]` -- both are two tasks, but only the second contains a
        same-kind pair, which is the whole point of running it (LIMITATIONS.md 5b). A kind
        signature is appended when, and only when, the composition departs from
        `default_tasks(n)`, so every id produced before compositions were configurable is
        unchanged and the committed results stay addressable.
        """
        n = len(self.tasks)
        sig = "" if self.tasks == default_tasks(n) else kind_signature(self.tasks)
        return f"t{n}{sig}_o{self.n_ops}_n{self.n_noise}"


def approx_tokens(texts: list[str]) -> int:
    """Deterministic size proxy used for pairing assertions and budgeting.

    Intentionally NOT presented as a true token count. The real tokenisation is
    verified at run time from `prompt_eval_count`; this exists so the generator can
    assert and report a stable number without loading a tokeniser.
    """
    return sum(len(t.split()) for t in texts)


#: Noise pool size. Conditions take a prefix of one pool of this size, which is what
#: makes longer length conditions strict supersets of shorter ones.
MAX_NOISE_POOL = 512


def _deal(total: int, k: int) -> list[int]:
    """Split `total` across `k` tasks, remainder to earlier tasks.

    Deterministic by design: the split must not depend on the RNG, or two runs of the
    same seed could allocate ops differently and break reproducibility.
    """
    return [total // k + (1 if i < total % k else 0) for i in range(k)]


def _subrng(seed: int, *parts: object) -> Random:
    """A deterministic, independent RNG for one (seed, task, purpose) slice.

    Independence matters more than it looks. When every task drew from one shared
    RNG, generating noise for task 1 consumed a variable number of draws and shifted
    task 2's entire operation stream -- so raising `n_noise` silently changed the
    operations as well as the padding. That is the same class of confound this
    project exists to isolate, one level down. Splitting the streams makes the
    mutating op sequence a function of (seed, task, n_mut) alone.

    Uses blake2b rather than hash(), which is salted per process and would make runs
    irreproducible across invocations.
    """
    key = "|".join(str(p) for p in (seed, *parts)).encode()
    return Random(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))


def _sample_task_ops(task: TaskKind, slot: int, n_mut: int, n_noise: int, n_false: int,
                     seed: int) -> list[Op]:
    """Ops for one task, in the order that task's turns will appear.

    Maintains a live model of the task's own state so REMOVE and UPDATE always target
    something that exists. The oracle tolerates dangling references, but *testing* the
    model on them would conflate state tracking with the model's guess at our
    edge-case policy -- a different question than the one being asked.
    """
    # Streams are keyed on the SLOT, not the kind, so two shopping lists in one
    # conversation draw independently instead of sharing a generator.
    key = slot_key(task, slot)
    # Stream A: the mutating operations. Must not depend on n_noise or n_false.
    rng = _subrng(seed, key, "mut", n_mut)
    # Stream B: placement of inert turns. Varying this cannot disturb Stream A.
    # NOT keyed on n_noise: keying on it made every noise level draw a fresh placement,
    # so the length conditions were independent draws rather than nested ones and
    # between-cell comparison carried placement variance (LIMITATIONS.md 4b). One pool
    # is drawn at MAX_NOISE_POOL and each condition takes a prefix, so a longer
    # condition is now a strict superset of a shorter one.
    mix = _subrng(seed, key, "mix", n_false)

    ops: list[Op] = []
    if task is TaskKind.SHOPPING:
        pool = list(vocabulary(task, slot))
        rng.shuffle(pool)
        live: list[str] = []
        for _ in range(n_mut):
            can_remove = len(live) >= 2
            kind = rng.choices(
                [OpKind.ADD, OpKind.REMOVE, OpKind.QUERY],
                weights=[6, 3 if can_remove else 0, 2],
            )[0]
            if kind is OpKind.ADD:
                unused = [i for i in pool if i not in live]
                if not unused:
                    kind, item = OpKind.QUERY, rng.choice(live)
                else:
                    item = unused[0]
                    live.append(item)
            elif kind is OpKind.REMOVE:
                item = rng.choice(live)
                live.remove(item)
            else:
                item = rng.choice(live) if live else pool[0]
            ops.append(Op(task, kind, {"item": item}, 0, slot))
        for _ in range(n_false):
            ops.append(Op(task, OpKind.FALSE_ASSERT,
                          {"item": mix.choice([i for i in pool if i not in live] or pool)},
                          0, slot))
    else:
        pool = list(vocabulary(task, slot))
        rng.shuffle(pool)
        # Distinct times: double-booking is legal in the oracle but ambiguous to a
        # reader, and we are not trying to test the model on our conflict policy.
        times = [f"{h:02d}:{m}" for h in range(8, 19) for m in ("00", "30")]
        rng.shuffle(times)
        live: dict[str, str] = {}
        for _ in range(n_mut):
            can_touch = len(live) >= 2
            kind = rng.choices(
                [OpKind.ADD, OpKind.UPDATE, OpKind.REMOVE, OpKind.QUERY],
                weights=[6, 3 if can_touch else 0, 2 if can_touch else 0, 2],
            )[0]
            if kind is OpKind.ADD:
                unused = [t for t in pool if t not in live]
                if not unused or not times:
                    kind = OpKind.QUERY
                else:
                    title, time = unused[0], times.pop()
                    live[title] = time
                    ops.append(Op(task, kind, {"title": title, "time": time}, 0, slot))
                    continue
            if kind is OpKind.UPDATE:
                title = rng.choice(sorted(live))
                new_time = times.pop() if times else "12:00"
                live[title] = new_time
                ops.append(Op(task, kind, {"title": title, "new_time": new_time}, 0, slot))
            elif kind is OpKind.REMOVE:
                title = rng.choice(sorted(live))
                del live[title]
                ops.append(Op(task, kind, {"title": title}, 0, slot))
            else:
                title = rng.choice(sorted(live)) if live else pool[0]
                ops.append(Op(task, kind, {"title": title}, 0, slot))
        for _ in range(n_false):
            spare = [t for t in pool if t not in live] or pool
            ops.append(Op(task, OpKind.FALSE_ASSERT,
                          {"title": mix.choice(spare),
                           "time": mix.choice(times) if times else "12:00"}, 0, slot))

    for _ in range(n_noise):
        ops.append(Op(task, OpKind.NOISE, {}, 0, slot))

    # Shuffle noise and false assertions into the mutating stream. Mutating ops keep
    # their relative order -- their sequence is what the oracle depends on.
    #
    # Insertion positions come from a fixed-length pool truncated to the number of
    # extras, so raising n_noise ADDS turns without moving the ones already there.
    mutating = [o for o in ops if o.kind not in (OpKind.NOISE, OpKind.FALSE_ASSERT)]
    # Stable order, NOT shuffled: false assertions first (their count is fixed across
    # length conditions), then noise. Shuffling reassigned positions whenever the list
    # grew, which broke nesting.
    extras = ([o for o in ops if o.kind is OpKind.FALSE_ASSERT]
              + [o for o in ops if o.kind is OpKind.NOISE])
    pool = [mix.randrange(len(mutating) + 1) for _ in range(MAX_NOISE_POOL)]
    # Pair each extra with its own draw BEFORE sorting, so an extra keeps its position
    # when later ones are added. Sorting the positions alone would reshuffle them.
    placed = sorted(zip(pool[:len(extras)], range(len(extras))))
    out: list[Op] = []
    e = 0
    for i in range(len(mutating) + 1):
        while e < len(placed) and placed[e][0] == i:
            out.append(extras[placed[e][1]]); e += 1
        if i < len(mutating):
            out.append(mutating[i])
    return out


def sample_ops(seed: int, tasks: list[TaskKind], n_ops: int,
               n_false: int, n_noise: int) -> list[Op]:
    """Sample a full op list. `n_ops`, `n_false` and `n_noise` are TOTALS for the
    conversation, dealt across tasks -- so raising the task count does not lengthen
    the conversation. That decision is what keeps task count from being confounded
    with context length."""
    k = len(tasks)
    slots = assign_slots(tasks)
    muts, falses, noises = _deal(n_ops, k), _deal(n_false, k), _deal(n_noise, k)
    out: list[Op] = []
    for i, t in enumerate(tasks):
        out.extend(_sample_task_ops(t, slots[i], muts[i], noises[i], falses[i], seed))
    return [Op(o.task, o.kind, o.payload, i, o.slot) for i, o in enumerate(out)]


def order_ops(ops: list[Op], ordering: Ordering, tasks: list[TaskKind]) -> list[Op]:
    """BLOCKED groups by task slot; INTERLEAVED round-robins across slots.

    Grouped by SLOT, not kind. Grouping by kind meant two shopping lists were one
    bucket, so blocked ordering emitted their turns twice and interleaving never
    switched between them (docs/DECISIONS.md D14).

    Within-slot order is preserved identically by both orderings, which is what makes
    the two conversations carry the same operations rather than merely the same counts.
    """
    keys = [slot_key(t, s) for t, s in zip(tasks, assign_slots(tasks))]
    per = {k: [o for o in ops if o.key == k] for k in keys}
    if ordering is Ordering.BLOCKED:
        return [o for k in keys for o in per[k]]
    out: list[Op] = []
    idx = {k: 0 for k in keys}
    while any(idx[k] < len(per[k]) for k in keys):
        for k in keys:
            if idx[k] < len(per[k]):
                out.append(per[k][idx[k]]); idx[k] += 1
    return out


class TokenMatchError(AssertionError):
    """Raised when a pair's two orderings do not carry identical turn text.

    Loud by design. A silent drift here would turn the headline measurement into a
    comparison of two different conversations.
    """


def build_pair(seed: int, tasks: list[TaskKind], n_ops: int,
               n_false: int, n_noise: int) -> tuple[Conversation, Conversation]:
    """Build the (blocked, interleaved) pair from one op sample.

    Renders every op exactly once, then reorders the rendered turns. Asserts that the
    two orderings hold the identical multiset of turn strings.
    """
    # Repeated kinds ARE supported now -- each becomes its own named instance with its
    # own disjoint vocabulary (D14 / D17). The cap is the number of named instances.
    from .surface import SLOT_ITEMS, SLOT_TITLES
    caps = {TaskKind.SHOPPING: len(SLOT_ITEMS), TaskKind.SCHEDULE: len(SLOT_TITLES)}
    for kind in set(tasks):
        n = sum(1 for t in tasks if t == kind)
        if n > caps[kind]:
            raise ValueError(f"{n} instances of {kind.value} requested but only "
                             f"{caps[kind]} distinct vocabularies exist")

    ops = sample_ops(seed, tasks, n_ops, n_false, n_noise)

    # One rendering pass. Each op gets its OWN generator, derived from a key that is
    # stable under changes to the rest of the list: its slot, kind, payload, and how
    # many identical ops precede it in the same slot.
    #
    # A single shared generator walked in list order would pick different paraphrases
    # for the same op as soon as anything earlier was added or removed -- which broke
    # nesting across noise levels just as surely as the placement did.
    seen: dict[tuple, int] = {}
    text_by_idx: dict[int, str] = {}
    for o in ops:
        ident = (o.key, o.kind.value, tuple(sorted(o.payload.items())))
        n = seen.get(ident, 0)
        seen[ident] = n + 1
        text_by_idx[o.idx] = render(o, _subrng(seed, "render", *ident, n))

    expected = ground_truth(ops, tasks)
    sys_prompt, final = system_prompt(tasks), render_final_request(tasks)

    convs: list[Conversation] = []
    for ordering in (Ordering.BLOCKED, Ordering.INTERLEAVED):
        seq = order_ops(ops, ordering, tasks)
        turns = [text_by_idx[o.idx] for o in seq]
        convs.append(Conversation(
            seed=seed, tasks=list(tasks), ordering=ordering, ops=seq, turns=turns,
            expected=expected, token_count=approx_tokens([sys_prompt, *turns, final]),
            n_ops=n_ops, n_noise=n_noise, n_false=n_false,
            system=sys_prompt, final_request=final,
        ))

    blocked, interleaved = convs
    if sorted(blocked.turns) != sorted(interleaved.turns):
        raise TokenMatchError(
            f"seed={seed}: orderings carry different turn text "
            f"({len(blocked.turns)} vs {len(interleaved.turns)} turns)")
    if blocked.token_count != interleaved.token_count:
        raise TokenMatchError(
            f"seed={seed}: token proxy differs {blocked.token_count} vs {interleaved.token_count}")
    if blocked.expected != interleaved.expected:
        raise TokenMatchError(f"seed={seed}: ground truth differs between orderings")
    return blocked, interleaved
