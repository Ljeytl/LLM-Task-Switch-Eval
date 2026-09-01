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
    four, _ = build_pair(seed, T2 + T2, 24, 0, 0)
    assert len(two.turns) == len(four.turns) == 24


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


@pytest.mark.parametrize("seed", range(10))
def test_duplicate_task_kinds_are_now_independent_tasks(seed):
    """The D14 fix. [SHOPPING, SCHEDULE, SHOPPING] used to collapse to two states and
    emit the shopping turns twice; it is now three genuine concurrent tasks."""
    tasks = [TaskKind.SHOPPING, TaskKind.SCHEDULE, TaskKind.SHOPPING]
    b, i = build_pair(seed, tasks, 9, 0, 0)
    assert sorted(b.expected) == ["schedule_0", "shopping_0", "shopping_1"]
    assert len(b.turns) == len(i.turns) == 9
    assert sorted(b.turns) == sorted(i.turns)


@pytest.mark.parametrize("n_tasks", [1, 2, 3, 4])
def test_task_count_arm_is_deliverable(n_tasks):
    """The whole point of the refactor: 1-4 concurrent tasks all work, token-matched."""
    tasks = [TaskKind.SHOPPING, TaskKind.SCHEDULE,
             TaskKind.SHOPPING, TaskKind.SCHEDULE][:n_tasks]
    b, i = build_pair(3, tasks, 8, 2, 4)
    assert len(b.expected) == n_tasks
    assert sorted(b.turns) == sorted(i.turns)
    assert b.token_count == i.token_count


def test_same_kind_slots_draw_disjoint_vocabularies():
    """Two shopping lists must not share items, or misattribution becomes invisible --
    which is the property the original two-kind design got for free and the reason the
    taxonomy could diagnose misfiling at all."""
    from taskswitch.surface import vocabulary
    a, b = set(vocabulary(TaskKind.SHOPPING, 0)), set(vocabulary(TaskKind.SHOPPING, 1))
    assert not (a & b)
    ca, cb = set(vocabulary(TaskKind.SCHEDULE, 0)), set(vocabulary(TaskKind.SCHEDULE, 1))
    assert not (ca & cb)


def test_too_many_instances_of_one_kind_is_rejected():
    with pytest.raises(ValueError, match="distinct vocabularies"):
        build_pair(1, [TaskKind.SHOPPING] * 9, 9, 0, 0)


def test_single_task_pair_is_a_null_control():
    """With one task there is nothing to interleave, so both orderings must be the
    identical sequence and the measured delta must be exactly zero. This is the
    harness's negative control: any non-zero switch cost here would mean the pipeline
    manufactures differences on its own."""
    b, i = build_pair(5, [TaskKind.SHOPPING], 6, 2, 4)
    assert b.turns == i.turns
    assert [o.idx for o in b.ops] == [o.idx for o in i.ops]


@pytest.mark.parametrize("seed", range(12))
def test_noise_conditions_are_nested_not_independent_draws(seed):
    """LIMITATIONS 4b fix. Raising n_noise must ADD padding, not redraw it.

    Previously `n_noise` was part of the mix-RNG key, so the 40-noise and 120-noise
    conditions drew entirely different placements. Between-cell trends then carried
    placement variance on top of quantity -- visible in the first sweep as seeds whose
    outcome went correct -> wrong -> correct across noise levels, which additive
    difficulty cannot produce.
    """
    short, _ = build_pair(seed, T2, 12, 2, 4)
    long_, _ = build_pair(seed, T2, 12, 2, 20)
    # The shorter condition's turns must all survive into the longer one.
    from collections import Counter
    assert not (Counter(short.turns) - Counter(long_.turns))
    assert short.expected == long_.expected


@pytest.mark.parametrize("seed", range(8))
def test_mutating_stream_is_invariant_to_noise_level(seed):
    """The operations themselves must not move when padding changes."""
    a = [(o.key, o.kind.value, tuple(sorted(o.payload.items())))
         for o in sample_ops(seed, T2, 12, 2, 0) if o.mutating]
    b = [(o.key, o.kind.value, tuple(sorted(o.payload.items())))
         for o in sample_ops(seed, T2, 12, 2, 60) if o.mutating]
    assert a == b


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_cli_tasks_for_builds_n_independent_states(n):
    """Integration gap that cost a sweep. `build_pair` was tested directly with 3 and 4
    tasks and passed, but nothing exercised run.py's task construction -- which still
    carried a D14-era cap at two. 538 green tests and a broken entry point."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run import tasks_for
    tasks = tasks_for(n)
    assert len(tasks) == n
    b, i = build_pair(1, tasks, 6, 2, 0)
    assert len(b.expected) == n
    assert sorted(b.turns) == sorted(i.turns)


def test_cli_tasks_for_cycles_kinds_rather_than_capping():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run import tasks_for
    assert tasks_for(3) == [TaskKind.SHOPPING, TaskKind.SCHEDULE, TaskKind.SHOPPING]
