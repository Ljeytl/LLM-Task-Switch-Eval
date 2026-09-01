"""Inference for a paired binary experiment.

Each estimator here is chosen over a more obvious alternative, and the reason is
recorded next to it, because "why this test?" is the question this analysis invites.

  Wilson, not Wald        Wald intervals overshoot past 0 and 1 near the boundaries and
                          collapse to zero width at exactly 0 or 1 -- precisely where
                          small models land.
  Exact binomial McNemar  The chi-square approximation is unreliable when the discordant
                          count is small, which is the regime a real effect produces.
  Paired bootstrap        Resamples *conversations*, not tasks, so the CI on the delta
                          inherits the pairing rather than assuming independence.
  Clustered SE            Tasks inside one conversation share context and are not
                          independent; naive SEs on per-task outcomes can be several
                          times too narrow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as _sps


def wilson(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

        centre = (p + z^2/2n) / (1 + z^2/n)
        half   = z/(1 + z^2/n) * sqrt( p(1-p)/n + z^2/4n^2 )

    Derived by inverting the score test rather than the Wald test, which is why it
    stays inside [0, 1] and keeps sensible width at p = 0 and p = 1.
    """
    if n == 0:
        return (0.0, 1.0)
    z = _sps.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class McNemarTable:
    """The 2x2 of paired outcomes. Only the off-diagonal carries information.

        b = blocked correct, interleaved wrong   (interleaving broke it)
        c = blocked wrong,   interleaved correct (interleaving fixed it)

    Conversations where both orderings agree tell you nothing about the ordering,
    which is the whole reason the paired analysis is more powerful than two
    independent proportions at the same n.
    """

    both_correct: int
    b: int
    c: int
    both_wrong: int

    @property
    def n_discordant(self) -> int:
        return self.b + self.c


def mcnemar_table(blocked: list[bool], interleaved: list[bool]) -> McNemarTable:
    if len(blocked) != len(interleaved):
        raise ValueError("paired lists must be the same length and in the same order")
    both = sum(1 for x, y in zip(blocked, interleaved) if x and y)
    b = sum(1 for x, y in zip(blocked, interleaved) if x and not y)
    c = sum(1 for x, y in zip(blocked, interleaved) if not x and y)
    neither = sum(1 for x, y in zip(blocked, interleaved) if not x and not y)
    return McNemarTable(both, b, c, neither)


def mcnemar(blocked: list[bool], interleaved: list[bool]) -> tuple[float, float]:
    """Paired test on the same conversations, in the same order.

    Returns `(b - c, p)`. The statistic is the raw discordant difference because it is
    directly interpretable -- how many more conversations interleaving broke than it
    fixed. The p-value is the **exact** two-sided binomial test of b ~ Binomial(b+c, 1/2)
    under the null that ordering does not matter, rather than the chi-square
    approximation, which misbehaves exactly where a real effect lives (small b+c).
    """
    t = mcnemar_table(blocked, interleaved)
    if t.n_discordant == 0:
        return (0.0, 1.0)
    p = _sps.binomtest(t.b, t.n_discordant, 0.5, alternative="two-sided").pvalue
    return (float(t.b - t.c), float(p))


def paired_bootstrap(blocked: list[bool], interleaved: list[bool],
                     n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Percentile CI for the accuracy delta (interleaved - blocked).

    Resamples **conversations**, carrying both orderings together. Resampling the two
    arms independently would discard the pairing and inflate the interval; resampling
    tasks would treat non-independent observations as independent.
    """
    if not blocked:
        return (0.0, 0.0, 0.0)
    b = np.asarray(blocked, dtype=float)
    i = np.asarray(interleaved, dtype=float)
    delta = float(i.mean() - b.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), size=(n_boot, len(b)))
    deltas = i[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return (delta, float(lo), float(hi))


def clustered_se(per_task: list[bool], cluster_ids: list[int]) -> float:
    """Cluster-robust standard error of a mean, clustered on conversation.

        SE = sqrt( (G/(G-1)) * sum_g ( sum_{i in g} (y_i - ybar) )^2 ) / n

    Summing residuals *within* a cluster before squaring is what accounts for
    correlation: if two tasks in one conversation fail together, their residuals
    reinforce rather than cancel, and the SE widens to reflect it.
    """
    if not per_task:
        return 0.0
    y = np.asarray(per_task, dtype=float)
    g = np.asarray(cluster_ids)
    n, ybar = len(y), y.mean()
    groups = np.unique(g)
    G = len(groups)
    if G < 2:
        return float(y.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    meat = sum(float(np.sum(y[g == k] - ybar)) ** 2 for k in groups)
    return float(math.sqrt(meat * (G / (G - 1))) / n)


def naive_se(per_task: list[bool]) -> float:
    """Unclustered SE, computed only so the README can show how much too narrow it is."""
    if len(per_task) < 2:
        return 0.0
    y = np.asarray(per_task, dtype=float)
    return float(y.std(ddof=1) / math.sqrt(len(y)))


def bonferroni(p_values: list[float]) -> list[float]:
    """Family-wise correction, capped at 1. Applied only to the secondary comparisons;
    the primary contrast is pre-registered and reported uncorrected."""
    m = len(p_values)
    return [min(1.0, p * m) for p in p_values]
