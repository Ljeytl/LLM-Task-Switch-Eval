"""Tests for the estimators, checked against closed-form values and known properties."""

import math

import pytest

from taskswitch.stats import (bonferroni, clustered_se, mcnemar, mcnemar_table,
                              naive_se, paired_bootstrap, wilson)


def test_wilson_stays_inside_unit_interval_at_the_boundaries():
    """The reason for using Wilson at all: Wald would give a zero-width interval at
    p=0 and can range outside [0,1] near it."""
    lo, hi = wilson(0, 20)
    assert lo == 0.0 and 0 < hi < 1
    lo, hi = wilson(20, 20)
    assert 0 < lo < 1 and hi == 1.0


def test_wilson_matches_published_value():
    """n=10, x=8 -> approximately (0.490, 0.943) at 95%."""
    lo, hi = wilson(8, 10)
    assert lo == pytest.approx(0.490, abs=0.005)
    assert hi == pytest.approx(0.943, abs=0.005)


def test_wilson_narrows_as_n_grows():
    w_small = wilson(50, 100); w_large = wilson(500, 1000)
    assert (w_large[1] - w_large[0]) < (w_small[1] - w_small[0])


def test_mcnemar_ignores_concordant_pairs():
    """Only discordant pairs carry information about ordering. Adding pairs where both
    orderings agree must not move the p-value."""
    b = [True] * 5 + [False] * 5
    i = [False] * 5 + [False] * 5
    p1 = mcnemar(b, i)[1]
    p2 = mcnemar(b + [True] * 20, i + [True] * 20)[1]
    assert p1 == pytest.approx(p2)


def test_mcnemar_table_partitions_every_pair():
    b = [True, True, False, False, True]
    i = [True, False, True, False, False]
    t = mcnemar_table(b, i)
    assert t.both_correct + t.b + t.c + t.both_wrong == len(b)
    assert (t.both_correct, t.b, t.c, t.both_wrong) == (1, 2, 1, 1)


def test_mcnemar_null_when_no_discordance():
    assert mcnemar([True] * 10, [True] * 10) == (0.0, 1.0)


def test_mcnemar_detects_a_one_sided_effect():
    """20 conversations broken by interleaving, none fixed: unambiguous."""
    stat, p = mcnemar([True] * 20, [False] * 20)
    assert stat == 20.0 and p < 1e-5


def test_mcnemar_exact_matches_hand_computation():
    """b=7, c=1 -> two-sided exact binomial on 8 trials = 2 * P(X<=1) = 0.0703125."""
    b = [True] * 7 + [False]
    i = [False] * 7 + [True]
    assert mcnemar(b, i)[1] == pytest.approx(0.0703125, abs=1e-9)


def test_mcnemar_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])


def test_paired_bootstrap_ci_brackets_the_point_estimate():
    b = [True] * 60 + [False] * 40
    i = [True] * 40 + [False] * 60
    d, lo, hi = paired_bootstrap(b, i, n_boot=2000, seed=1)
    assert d == pytest.approx(-0.20)
    assert lo < d < hi


def test_paired_bootstrap_is_deterministic_given_a_seed():
    b = [True, False] * 50; i = [False, True] * 50
    assert paired_bootstrap(b, i, 1000, 7) == paired_bootstrap(b, i, 1000, 7)


def test_clustered_se_exceeds_naive_when_clusters_are_correlated():
    """The point of clustering. Two tasks per conversation that always agree carry the
    information of one observation, not two -- the naive SE does not know that."""
    per_task, clusters = [], []
    for c in range(50):
        v = c % 2 == 0
        per_task += [v, v]           # perfectly correlated within conversation
        clusters += [c, c]
    assert clustered_se(per_task, clusters) > naive_se(per_task) * 1.3


def test_clustered_se_approaches_naive_when_clusters_are_singletons():
    per_task = [True, False] * 40
    clusters = list(range(80))
    assert clustered_se(per_task, clusters) == pytest.approx(naive_se(per_task), rel=0.05)


def test_clustered_se_handles_degenerate_input():
    assert clustered_se([], []) == 0.0
    assert clustered_se([True], [0]) == 0.0


def test_bonferroni_caps_at_one():
    assert bonferroni([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]
    assert bonferroni([0.01, 0.02])[0] == pytest.approx(0.02)
