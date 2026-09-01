"""Tests for the diff and the failure taxonomy, against handmade cases.

The taxonomy is the part of the pipeline with genuine judgement in it, so each bucket
is pinned by an example that could plausibly be argued the other way.
"""

import pytest

from taskswitch.generator import build_pair
from taskswitch.ops import Op, OpKind, TaskKind
from taskswitch.runner import FinalState, ModelSpec, RunResult
from taskswitch.scorer import Failure, score

T2 = [TaskKind.SHOPPING, TaskKind.SCHEDULE]
MODEL = ModelSpec(name="test")


def _result(expected, reported, ops=None, parse_ok=True):
    conv, _ = build_pair(0, T2, 12, 1, 0)
    conv.expected = expected
    if ops is not None:
        conv.ops = ops
    # model_construct skips validation on purpose: the malformed-payload test needs to
    # reach the scorer with a shape pydantic would normally reject, so that we are
    # testing the scorer's own defences rather than pydantic's.
    parsed = FinalState.model_construct(**reported) if parse_ok else None
    return RunResult(conversation=conv, model=MODEL, raw_final="", parsed=parsed,
                     parse_ok=parse_ok, n_retries=0, wall_seconds=0.0)


def test_exact_match_is_joint_correct():
    s = score(_result({"shopping": ["milk"], "schedule": [["09:00", "standup"]]},
                      {"shopping": ["milk"], "schedule": [["09:00", "standup"]]}))
    assert s.joint_correct and s.failures == []


def test_normalisation_ignores_case_and_time_format():
    """Case and `9:00` vs `09:00` are formatting, not tracking. Grading them would
    inflate the headline number with noise."""
    s = score(_result({"shopping": ["milk"], "schedule": [["09:00", "standup"]]},
                      {"shopping": [" Milk "], "schedule": [["9:00", "Standup"]]}))
    assert s.joint_correct


def test_dropped_when_item_vanishes():
    s = score(_result({"shopping": ["milk", "eggs"], "schedule": []},
                      {"shopping": ["milk"], "schedule": []}))
    assert (Failure.DROPPED, "shopping:eggs") in s.failures
    assert not s.joint_correct


def test_misattributed_when_item_lands_in_the_other_task():
    """A grocery item appearing in the schedule is unambiguous -- which is exactly why
    these two task types were chosen over two lists."""
    s = score(_result({"shopping": ["milk"], "schedule": [["09:00", "standup"]]},
                      {"shopping": [], "schedule": [["09:00", "standup"], ["10:00", "milk"]]}))
    kinds = s.failure_kinds
    assert "misattributed" in kinds
    assert "dropped" not in kinds


def test_absorbed_flags_a_planted_false_assertion():
    ops = [Op(TaskKind.SHOPPING, OpKind.FALSE_ASSERT, {"item": "caviar"}, 0)]
    s = score(_result({"shopping": ["milk"], "schedule": []},
                      {"shopping": ["milk", "caviar"], "schedule": []}, ops=ops))
    assert any(f is Failure.ABSORBED and "false_assert" in d for f, d in s.failures)


def test_absorbed_distinguishes_hallucination_from_false_assertion():
    """Both are ungrounded content entering state, so they share a bucket -- but only
    one was deliberately injected, so the detail keeps them separable."""
    s = score(_result({"shopping": ["milk"], "schedule": []},
                      {"shopping": ["milk", "truffles"], "schedule": []}, ops=[]))
    assert any(f is Failure.ABSORBED and "hallucination" in d for f, d in s.failures)


def test_wrong_time_is_one_failure_not_two():
    """A meeting at the wrong hour is one tracking error. Counting it as both a
    disappearance and an invention would double-count a single mistake."""
    s = score(_result({"shopping": [], "schedule": [["09:00", "standup"]]},
                      {"shopping": [], "schedule": [["15:00", "standup"]]}))
    assert len(s.failures) == 1
    assert s.failures[0][0] is Failure.DROPPED and "wrong value" in s.failures[0][1]


def test_format_failure_excludes_task_scores():
    """parse_ok=False must yield FORMAT and per_task_correct=None, so a formatting
    problem can be excluded from state accuracy rather than reported as a tracking one."""
    s = score(_result({"shopping": ["milk"], "schedule": []}, {}, parse_ok=False))
    assert s.per_task_correct is None
    assert not s.joint_correct
    assert s.failure_kinds == ["format"]


def test_joint_requires_every_task():
    s = score(_result({"shopping": ["milk"], "schedule": [["09:00", "standup"]]},
                      {"shopping": ["milk"], "schedule": []}))
    assert s.per_task_correct == {"shopping": True, "schedule": False}
    assert not s.joint_correct


def test_score_is_pure_and_repeatable():
    r = _result({"shopping": ["milk"], "schedule": []}, {"shopping": [], "schedule": []})
    assert score(r).failures == score(r).failures


@pytest.mark.parametrize("garbage", [{"shopping": "not a list"}, {"shopping": [None]},
                                     {"schedule": [["09:00"]]}, {"schedule": ["nope"]}])
def test_scorer_survives_malformed_but_parseable_payloads(garbage):
    """Constrained decoding plus pydantic validation should make these unreachable, but
    the scorer must not crash on a shape it did not expect -- a crash would be recorded
    as a model failure, which would be a lie about the model."""
    score(_result({"shopping": ["milk"], "schedule": [["09:00", "standup"]]}, garbage))


def test_positional_pair_is_read_by_content_not_position():
    """Regression: the first live run returned [title, time] in one condition and
    [time, title] in the other, and the scorer counted the difference as six tracking
    failures. Schema ambiguity that correlates with the condition would manufacture a
    switch cost, so the parser now decides by which element looks like a clock time."""
    exp = {"shopping": [], "schedule": [["09:00", "standup"]]}
    assert score(_result(exp, {"shopping": [], "schedule": [["09:00", "standup"]]})).joint_correct
    assert score(_result(exp, {"shopping": [], "schedule": [["standup", "09:00"]]})).joint_correct


def test_schedule_accepts_named_keys():
    exp = {"shopping": [], "schedule": [["09:00", "standup"]]}
    got = {"shopping": [], "schedule": [{"time": "9:00", "title": "Standup"}]}
    assert score(_result(exp, got)).joint_correct


def test_stale_duplicate_is_not_called_a_hallucination():
    """Found by the grounding audit, not by reading accuracy numbers.

    A meeting was added at 11:30, removed, then re-added at 14:30. The model reported
    BOTH. The 11:30 entry is ungrounded in the final state, but the model did not invent
    it -- it failed to retract it, which is a dropped REMOVE. Labelling that a
    hallucination would point anyone debugging it at entirely the wrong problem.
    """
    exp = {"shopping": [], "schedule": [["14:30", "one-on-one"]]}
    got = {"shopping": [], "schedule": [{"time": "11:30", "title": "one-on-one"},
                                        {"time": "14:30", "title": "one-on-one"}]}
    s = score(_result(exp, got))
    assert any(f is Failure.ABSORBED and d.endswith("stale") for f, d in s.failures)
    assert not any("hallucination" in d for _, d in s.failures)


def test_true_hallucination_still_reads_as_hallucination():
    """The stale case must not swallow genuine invention."""
    exp = {"shopping": [], "schedule": [["14:30", "one-on-one"]]}
    got = {"shopping": [], "schedule": [{"time": "14:30", "title": "one-on-one"},
                                        {"time": "09:00", "title": "board meeting"}]}
    s = score(_result(exp, got))
    assert any("hallucination" in d for _, d in s.failures)
