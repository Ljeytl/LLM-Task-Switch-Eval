"""Tests for natural-language rendering.

The surface layer is where an ambiguous turn would make the answer key *unanswerable*
rather than merely hard, so these check disambiguation and paraphrase coverage rather
than exact wording.
"""

import re
from random import Random

import pytest

from taskswitch.ops import SLOT_NAMES, Op, OpKind, TaskKind, slot_label
from taskswitch.surface import (SLOT_ITEMS, SLOT_TITLES, _TEMPLATES, render,
                                render_final_request, system_prompt, vocabulary)

ALL_PAIRS = sorted(_TEMPLATES, key=lambda k: (k[0].value, k[1].value))


@pytest.mark.parametrize("key", ALL_PAIRS)
def test_at_least_four_paraphrases_per_op_kind(key):
    """Seed-randomised paraphrase choice is what stops a model passing by
    pattern-matching one fixed phrasing instead of tracking state."""
    assert len(_TEMPLATES[key]) >= 4


@pytest.mark.parametrize("key", ALL_PAIRS)
def test_templates_are_distinct(key):
    assert len(set(_TEMPLATES[key])) == len(_TEMPLATES[key])


@pytest.mark.parametrize("key", [k for k in ALL_PAIRS if k[1] is not OpKind.NOISE])
def test_every_non_noise_template_names_its_list(key):
    """With two lists of one kind, an unnamed turn is genuinely ambiguous. NOISE is
    exempt -- it addresses no list by design."""
    assert all("{slot}" in t for t in _TEMPLATES[key])


def test_noise_templates_carry_no_payload_placeholders():
    """NOISE ops have an empty payload, so a placeholder would raise at render time."""
    for kind in (TaskKind.SHOPPING, TaskKind.SCHEDULE):
        for t in _TEMPLATES[(kind, OpKind.NOISE)]:
            assert not re.search(r"\{(item|title|time|new_time|slot)\}", t)


@pytest.mark.parametrize("slot", range(4))
def test_vocabularies_are_disjoint_across_slots(slot):
    """The property that makes misattribution detectable between same-kind lists."""
    for pools in (SLOT_ITEMS, SLOT_TITLES):
        others = set().union(*(set(p) for i, p in enumerate(pools) if i != slot))
        assert not (set(pools[slot]) & others)


def test_slot_labels_are_unique_within_a_kind():
    for kind in (TaskKind.SHOPPING, TaskKind.SCHEDULE):
        names = SLOT_NAMES[kind]
        assert len(set(names)) == len(names)


@pytest.mark.parametrize("seed", range(20))
def test_render_is_deterministic_for_a_given_rng_seed(seed):
    op = Op(TaskKind.SHOPPING, OpKind.ADD, {"item": "milk"}, 0, 0)
    assert render(op, Random(seed)) == render(op, Random(seed))


@pytest.mark.parametrize("kind,payload", [
    (OpKind.ADD, {"item": "milk"}), (OpKind.REMOVE, {"item": "milk"}),
    (OpKind.QUERY, {"item": "milk"}), (OpKind.FALSE_ASSERT, {"item": "milk"}),
    (OpKind.NOISE, {}),
])
def test_shopping_ops_render_without_error(kind, payload):
    out = render(Op(TaskKind.SHOPPING, kind, payload, 0, 1), Random(0))
    assert out and "{" not in out


@pytest.mark.parametrize("kind,payload", [
    (OpKind.ADD, {"title": "standup", "time": "09:00"}),
    (OpKind.UPDATE, {"title": "standup", "new_time": "10:00"}),
    (OpKind.REMOVE, {"title": "standup"}),
    (OpKind.QUERY, {"title": "standup"}),
    (OpKind.FALSE_ASSERT, {"title": "standup", "time": "09:00"}),
    (OpKind.NOISE, {}),
])
def test_schedule_ops_render_without_error(kind, payload):
    out = render(Op(TaskKind.SCHEDULE, kind, payload, 0, 1), Random(0))
    assert out and "{" not in out


def test_render_uses_the_slot_label_not_the_kind():
    a = render(Op(TaskKind.SHOPPING, OpKind.ADD, {"item": "milk"}, 0, 0), Random(0))
    b = render(Op(TaskKind.SHOPPING, OpKind.ADD, {"item": "milk"}, 0, 1), Random(0))
    assert slot_label(TaskKind.SHOPPING, 0) in a
    assert slot_label(TaskKind.SHOPPING, 1) in b
    assert a != b


@pytest.mark.parametrize("n", range(1, 5))
def test_final_request_names_every_tracked_list(n):
    tasks = [TaskKind.SHOPPING, TaskKind.SCHEDULE,
             TaskKind.SHOPPING, TaskKind.SCHEDULE][:n]
    req = render_final_request(tasks)
    from taskswitch.ops import assign_slots
    for t, s in zip(tasks, assign_slots(tasks)):
        assert slot_label(t, s) in req


@pytest.mark.parametrize("n", range(1, 5))
def test_system_prompt_names_lists_but_never_contents(n):
    tasks = [TaskKind.SHOPPING, TaskKind.SCHEDULE,
             TaskKind.SHOPPING, TaskKind.SCHEDULE][:n]
    sp = system_prompt(tasks)
    from taskswitch.ops import assign_slots
    for t, s in zip(tasks, assign_slots(tasks)):
        assert slot_label(t, s) in sp
    # Naming any entity would leak part of the answer.
    for item in SLOT_ITEMS[0][:5] + SLOT_TITLES[0][:5]:
        assert item not in sp


def test_vocabulary_wraps_rather_than_raising():
    """Slot indices beyond the pool count must degrade, not crash -- the generator
    guards the cap, but this must stay total."""
    assert vocabulary(TaskKind.SHOPPING, 99)
