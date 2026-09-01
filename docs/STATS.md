# Statistics

Every estimator here was chosen over a more obvious alternative. This document records
which alternative, and why — so the choices can be defended rather than cited.

Implementation: `taskswitch/stats.py`. Tests: `tests/test_stats.py`, which check the
values against closed-form results and hand computations rather than against
themselves.

---

## 1. Why the design is paired

Each seeded operation sequence is run in **both** orderings. That single choice is the
highest-leverage statistical decision available.

Consider what an unpaired design throws away. If conversation 47 happens to sample a
hard operation sequence, it is hard in *both* orderings. Comparing two independent
groups makes that difficulty show up as variance between groups. Comparing the same
sequence against itself removes it entirely.

Formally, for a difference of means, `Var(X̄ - Ȳ) = Var(X̄) + Var(Ȳ) - 2·Cov(X̄, Ȳ)`.
Independent sampling forces the covariance term to zero. Pairing keeps it, and at a
correlation of 0.5 it cuts estimator variance by roughly a third — the same precision
for about a third fewer model calls.

---

## 2. Wilson, not Wald, for a proportion

The Wald interval — `p̂ ± z·√(p̂(1-p̂)/n)` — is the one everyone learns, and it fails
exactly where this experiment lives.

At `p̂ = 0` it produces `[0, 0]`: a claim of *perfect certainty* from a sample that
merely never succeeded. Small models on 4-task conditions land there. Near the
boundary it also ranges outside `[0, 1]`, which is not a probability.

Wilson inverts the **score** test instead of the Wald test. Solve for the values of `p`
that the data would not reject:

```
        p̂ + z²/2n                z              ⎧ p̂(1-p̂)     z²  ⎫
centre = ──────────      half = ───────── · sqrt ⎨ ─────── + ───── ⎬
        1 + z²/n                1 + z²/n         ⎩    n      4n²  ⎭
```

The `z²/2n` term pulls the centre toward ½, and the interval keeps a sensible width at
0 and 1. `tests/test_stats.py` pins it against the published value for n=10, x=8:
approximately (0.490, 0.943).

---

## 3. McNemar, exact rather than chi-square

**Be able to derive this from the 2×2 table.** Lay the paired outcomes out:

|  | interleaved ✓ | interleaved ✗ |
|---|---|---|
| **blocked ✓** | a | **b** |
| **blocked ✗** | **c** | d |

- `a` — both orderings right
- `b` — blocked right, interleaved wrong → *interleaving broke it*
- `c` — blocked wrong, interleaved right → *interleaving fixed it*
- `d` — both wrong

`a` and `d` carry **no information about ordering**. A conversation the model got right
both ways, or wrong both ways, tells you nothing about which ordering is harder. Only
the off-diagonal speaks.

Under the null hypothesis that ordering does not matter, each discordant pair is a coin
flip: it is equally likely to land in `b` or in `c`. So

```
b ~ Binomial(b + c, ½)
```

and the test is simply: is `b` surprising for a fair coin tossed `b + c` times?

The textbook form uses `χ² = (|b - c| - 1)² / (b + c)` with a continuity correction,
which is a *normal approximation* to that binomial. It is unreliable when `b + c` is
small — which is precisely the regime a real but modest effect produces. So this
implementation computes the **exact two-sided binomial p-value** via
`scipy.stats.binomtest` and never approximates.

`mcnemar()` returns `(b - c, p)`. The statistic is the raw discordant difference
because it is directly interpretable: how many more conversations interleaving broke
than it fixed.

*Worked check, pinned in the tests:* b=7, c=1 gives a two-sided exact p of
`2 · P(X ≤ 1)` on 8 trials = `2 · (1 + 8)/256` = **0.0703125**.

---

## 4. Paired bootstrap for a confidence interval on the delta

McNemar gives a p-value; it does not give an interval on the effect size, and the
effect size is what a reader actually wants.

The bootstrap resamples **conversations**, carrying both orderings together:

```python
idx    = rng.integers(0, n, size=(n_boot, n))   # resample conversation indices
deltas = interleaved[idx].mean(axis=1) - blocked[idx].mean(axis=1)
lo, hi = np.percentile(deltas, [2.5, 97.5])
```

The subtlety is in `idx` being drawn **once per replicate and applied to both arms**.
Resampling the arms independently would discard the pairing and inflate the interval.
Resampling *tasks* rather than conversations would treat non-independent observations
as independent — the error described next.

This is a simple percentile interval. With n=25 and very few discordant pairs it can be
non-consonant with the exact McNemar test—for example, an interval may exclude zero while
the exact p-value remains above 0.05. The interval is therefore reported as an
exploratory effect-size summary, not an exact-test confidence set.

---

## 5. Clustered standard errors

The primary metric, joint goal accuracy, is one binary outcome per conversation, so it
has no clustering problem. The **secondary** per-task metric does.

Tasks inside one conversation share a context window, a system prompt, and a single
model call. If the model loses the thread, it plausibly loses several tasks at once.
Treating those as independent observations overstates the effective sample size, and
unclustered standard errors can be several times too narrow.

The cluster-robust estimator sums residuals **within** a cluster before squaring:

```
         ⎧   G      ⎛           ⎞² ⎫
SE = sqrt⎨ ───── · Σ ⎜  Σ (yᵢ-ȳ) ⎟  ⎬  /  n
         ⎩ G - 1   g ⎝ i∈g       ⎠  ⎭
```

That inner sum is the whole idea. If two tasks in one conversation fail together their
residuals reinforce instead of cancelling, and the SE widens to reflect the correlation.
`G/(G-1)` is a small-sample correction for the number of clusters.

`naive_se()` exists purely so the README can print both and show the ratio. Reporting
"the naive SE is 1.8× too narrow" is more persuasive than asserting that clustering
matters.

---

## 6. Multiple comparisons

The **primary comparison was prespecified in the repository before the v2 sweep**: 2
tasks, medium length, `qwen2.5-coder:7b`, joint goal accuracy, McNemar. There was no
external registration. It is reported uncorrected because it is one prespecified test.

The one-task negative controls are validation checks outside the inferential family. The
remaining 11 model-condition comparisons are secondary and exploratory. Both raw and
Bonferroni-adjusted p-values are reported; the smallest raw value, 0.039 for qwen
`tasks_4`, adjusts to 0.430. Correction does not turn an exploratory hypothesis into a
confirmatory one.

---

## 7. What is deliberately *not* claimed

**Bit-exact reproducibility.** Batch-invariant kernels do not exist on Metal, so
identical inputs can produce slightly different logits depending on batching. Seeds and
model digests are logged; results are reported as intervals rather than point
estimates, and no claim of exact reproduction is made.

**Greedy decoding is not a variance-reduction trick.** Temperature 0 with top_k 1 is
used because Renze and Guven (EMNLP 2024) found no statistically significant effect of
temperature on problem-solving accuracy across 0–1.0. Sampling would buy variance
without signal. It also means a **parse-failure retry would be a no-op** — re-asking
reproduces the same tokens — which is why `run_conversation` retries transport errors
only and records `parse_ok=False` immediately.


---

## 8. Observed-data sensitivity, not prospective power

`tools/power_analysis.py` simulates sensitivity using the mean discordance rate observed
across heterogeneous cells. McNemar sensitivity is driven by the discordant count, since
concordant pairs carry no information about ordering. Because this assumption comes from
the analyzed data, the output is post-hoc guidance for future design, not prospective
power for this sweep.

The result (`results/POWER.md`) is that **n=25** per cell — the sweep's actual size — has
roughly 4% power against a 7-point effect and 9% against an 11-point one. **Every
non-significant result in this project should be read through that table**: it is
inconclusive at this sample size, not evidence of no effect.

This also explains the shape of the primary cell's null: b=6, c=3. The design did detect
interleaving changing outcomes on nine of twenty-five conversations — it simply changed
them in both directions in a sample too small to resolve the imbalance.

The grid used to open at n=30, which no cell ran. A power table whose smallest column is
larger than the study it qualifies overstates that study, and the fix was to derive both
the grid's first column and the summary sentence from the data rather than write them by
hand.
