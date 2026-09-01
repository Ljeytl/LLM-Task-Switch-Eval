"""Tests for op sampling, ordering and pairing.

The first three tests here are the ones that keep the headline comparison honest:
the two orderings must carry identical text, identical ground truth, and genuinely
different task orderings.
"""

import pytest

from taskswitch.generator import (Ordering, TokenMatchError, build_pair, order_ops,
                                  sample_ops)
from taskswitch.ops import OpKind, TaskKind

T2 = [TaskKind.SHOPPING, TaskKind.SCHEDULE]
CONFIGS = [(12, 2, 0), (24, 3, 0), (24, 3, 40), (24, 3, 120), (36, 4, 20)]


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("n_ops,n_false,n_noise", CONFIGS)
def test_pair_is_token_matched(seed, n_ops, n_false, n_noise):
    """The load-bearing assertion: same strings, same count, same answer key."""
    b, i = build_pair(seed, T2, n_ops, n_false, n_noise)
    assert sorted(b.turns) == sorted(i.turns)
    assert b.token_count == i.token_count
    assert b.expected == i.expected
    assert len(b.turns) == len(i.turns) == n_ops + n_false + n_noise


@pytest.mark.parametrize("seed", range(25))
def test_orderings_actually_differ(seed):
    """Guards against a silent bug that made both orderings the same sequence -- the
    pair would still pass every equality check above while measuring nothing."""
    b, i = build_pair(seed, T2, 24, 3, 10)
    assert [o.task for o in b.ops] != [o.task for o in i.ops]


@pytest.mark.parametrize("seed", range(25))
def test_within_task_order_is_identical(seed):
    """Both orderings must present each task's ops in the same relative sequence.
    If they did not, the two conversations would encode different operations and the
    ground truth could legitimately differ."""
    b, i = build_pair(seed, T2, 24, 3, 10)
    for t in T2:
        assert [o.idx for o in b.ops if o.task == t] == [o.idx for o in i.ops if o.task == t]


def test_blocked_groups_and_interleaved_alternates():
    b, i = build_pair(3, T2, 24, 0, 0)
    blocked_tasks = [o.task for o in b.ops]
    # Exactly one task transition in blocked ordering with two tasks.
    assert sum(a != c for a, c in zip(blocked_tasks, blocked_tasks[1:])) == 1
    inter_tasks = [o.task for o in i.ops]
    assert all(a != c for a, c in zip(inter_tasks, inter_tasks[1:]))


@pytest.mark.parametrize("seed", range(15))
def test_reproducible_from_seed(seed):
    a1, b1 = build_pair(seed, T2, 24, 3, 10)
    a2, b2 = build_pair(seed, T2, 24, 3, 10)
    assert a1.turns == a2.turns and b1.turns == b2.turns
    assert a1.expected == a2.expected


@pytest.mark.parametrize("seed", range(15))
def test_n_ops_is_total_not_per_task(seed):
    """Raising the task count must NOT lengthen the conversation.

    This is the fix for task count being confounded with context length. If n_ops were
    per-task, four tasks would produce roughly twice the conversation of two, and any
    'more tasks is harder' result would be partly a 'longer is harder' result.
    """
    two, _ = build_pair(seed, T2, 24, 0, 0)
    assert len(two.turns) == 24


@pytest.mark.parametrize("seed", range(15))
def test_noise_lengthens_context_without_changing_the_answer(seed):
    """Validates the length lever: noise adds turns, never state.

    This is what lets context length be varied independently of state-update load --
    and, because noise never enters the final answer, it costs only fast prefill and
    never slow generation.
    """
    short, _ = build_pair(seed, T2, 24, 3, 0)
    long_, _ = build_pair(seed, T2, 24, 3, 120)
    assert short.expected == long_.expected
    assert len(long_.turns) == len(short.turns) + 120
    assert long_.token_count > short.token_count


@pytest.mark.parametrize("seed", range(15))
def test_removes_and_updates_target_live_entities(seed):
    """The generator must never ask the model to remove something that was never added.
    Testing on dangling references would conflate state tracking with a guess at our
    edge-case policy."""
    ops = sample_ops(seed, T2, 36, 4, 10)
    live_items, live_titles = set(), set()
    for o in ops:
        if o.task is TaskKind.SHOPPING:
            if o.kind is OpKind.ADD:
                live_items.add(o.payload["item"])
            elif o.kind is OpKind.REMOVE:
                assert o.payload["item"] in live_items
                live_items.discard(o.payload["item"])
        else:
            if o.kind is OpKind.ADD:
                live_titles.add(o.payload["title"])
            elif o.kind in (OpKind.REMOVE, OpKind.UPDATE):
                assert o.payload["title"] in live_titles
                if o.kind is OpKind.REMOVE:
                    live_titles.discard(o.payload["title"])


def test_order_ops_preserves_all_ops():
    ops = sample_ops(11, T2, 24, 3, 10)
    for ordering in Ordering:
        out = order_ops(ops, ordering, T2)
        assert sorted(o.idx for o in out) == sorted(o.idx for o in ops)


def test_token_match_error_is_available():
    assert issubclass(TokenMatchError, AssertionError)


def test_cli_accepts_config_without_explicit_sweep_flag():
    """Regression: `--config` alone matched the documented usage but argparse rejected
    it, so the first sweep launch silently did nothing for ten minutes."""
    import subprocess, sys
    r = subprocess.run([sys.executable, "run.py"], capture_output=True, text=True)
    assert r.returncode != 0 and "--config" in r.stderr


def test_duplicate_task_kinds_are_rejected():
    """Regression, and an important one. Task identity is TaskKind, so [SHOPPING,
    SCHEDULE, SHOPPING] is not three concurrent tasks -- the duplicate merges into the
    first one's state and order_ops emits its turns twice. A degenerate "3 tasks"
    condition that quietly measured 2 would be a fabricated result, so this raises."""
    with pytest.raises(ValueError, match="duplicate task kinds"):
        build_pair(1, [TaskKind.SHOPPING, TaskKind.SCHEDULE, TaskKind.SHOPPING], 6, 0, 0)


def test_single_task_pair_is_a_null_control():
    """With one task there is nothing to interleave, so both orderings must be the
    identical sequence and the measured delta must be exactly zero. This is the
    harness's negative control: any non-zero switch cost here would mean the pipeline
    manufactures differences on its own."""
    b, i = build_pair(5, [TaskKind.SHOPPING], 6, 2, 4)
    assert b.turns == i.turns
    assert [o.idx for o in b.ops] == [o.idx for o in i.ops]
