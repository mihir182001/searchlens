"""
ab_test.py -- Week 8: a statistically rigorous A/B test between two
retrieval configurations, instead of eyeballing two aggregate MRR@10
numbers and calling whichever is higher "better".

Why this matters for THIS project specifically: hybrid_rrf.py's own
docstring already flags an open question from Week 4 -- the best hybrid
config found (dense_weight=10) scored MRR@10 0.9022, actually BELOW Dense
alone's 0.9130. Is Hybrid genuinely worse on this corpus, or is a ~1-point
gap within the noise you'd expect from ordinary query-to-query variance?
Two aggregate numbers can't answer that on their own -- you need a PAIRED
test across the same queries. (See run_ab_test.py, which runs exactly this
comparison by default.)

Method: paired bootstrap significance test (the percentile-shift method
from Efron & Tibshirani's "An Introduction to the Bootstrap"; this is also
the standard approach for comparing IR system runs, e.g. Smucker, Allan &
Carterette 2007, "A comparison of statistical significance tests for
information retrieval evaluation"). Unlike a paired t-test, it makes no
assumption that per-query score differences are normally distributed --
worth avoiding here since reciprocal rank is a bounded, lumpy quantity
(0, 1, 0.5, 0.333, 0.25, ...), not a smooth continuous measurement.

    1. Every query gets a score under config A and under config B (e.g.
       its reciprocal rank, from per_query_reciprocal_rank in metrics.py).
    2. diff[qid] = score_B[qid] - score_A[qid], for every query both
       configs were evaluated on.
    3. observed_mean_diff = mean(diff over all queries).
    4. Resample the diffs (with replacement, same size as the original) a
       large number of times. Each resample's mean approximates the
       sampling distribution of the mean difference:
         - the resampled means' 2.5th/97.5th percentiles give a 95%
           confidence interval on the true mean difference.
         - re-centering that same distribution at zero (the null
           hypothesis "no real difference between A and B") and asking how
           often a re-centered resample is at least as extreme as what was
           actually observed gives a p-value.

No numpy/scipy dependency -- same as pearson_correlation in metrics.py,
pure Python and the standard library's `random`, so this is fully testable
in the sandbox with nothing installed and no real model or API needed.
"""
import random
from dataclasses import dataclass


@dataclass
class ABTestResult:
    n_queries: int
    mean_a: float
    mean_b: float
    mean_diff: float  # mean_b - mean_a: positive means B scored higher on average
    ci_low: float
    ci_high: float
    confidence: float
    p_value: float


def paired_bootstrap_test(
    scores_a: dict,
    scores_b: dict,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> ABTestResult:
    """scores_a, scores_b: {qid: float} per-query metric scores (e.g. from
    per_query_reciprocal_rank) for two retrieval configurations, over the
    SAME set of queries.

    Mismatched qid sets are rejected rather than silently intersected --
    comparing two configs over different query sets would confound "which
    config is better" with "which query set happened to be easier", which
    defeats the entire point of a paired test.
    """
    if set(scores_a.keys()) != set(scores_b.keys()):
        only_a = set(scores_a.keys()) - set(scores_b.keys())
        only_b = set(scores_b.keys()) - set(scores_a.keys())
        raise ValueError(
            "scores_a and scores_b must cover the exact same queries "
            f"(only in A: {len(only_a)}, only in B: {len(only_b)}) -- "
            "run both configs over the same query set before comparing."
        )

    qids = sorted(scores_a.keys())
    n = len(qids)
    if n < 2:
        raise ValueError("Need at least 2 shared queries to run a paired test.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1.")

    diffs = [scores_b[qid] - scores_a[qid] for qid in qids]
    observed_mean_diff = sum(diffs) / n

    rng = random.Random(seed)
    bootstrap_means = []
    for _ in range(n_resamples):
        resample_sum = 0.0
        for _ in range(n):
            resample_sum += diffs[rng.randrange(n)]
        bootstrap_means.append(resample_sum / n)
    bootstrap_means.sort()

    alpha = 1.0 - confidence
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)
    ci_low = bootstrap_means[lo_idx]
    ci_high = bootstrap_means[hi_idx]

    # Percentile-shift p-value: re-center the bootstrap distribution at the
    # null hypothesis (zero difference) by subtracting the OBSERVED mean
    # diff from every resample, then measure how often a re-centered
    # resample is at least as extreme (in absolute value) as the actual
    # observation. This uses the real shape of the bootstrap distribution
    # instead of assuming a normal curve.
    extreme_count = sum(
        1 for bm in bootstrap_means if abs(bm - observed_mean_diff) >= abs(observed_mean_diff)
    )
    p_value = extreme_count / n_resamples

    return ABTestResult(
        n_queries=n,
        mean_a=sum(scores_a[qid] for qid in qids) / n,
        mean_b=sum(scores_b[qid] for qid in qids) / n,
        mean_diff=observed_mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        p_value=p_value,
    )


def is_significant(result: ABTestResult, alpha: float = 0.05) -> bool:
    """True if the paired bootstrap test rejects "no difference" at the
    given significance level (default: the conventional 0.05). This will
    usually agree with "does the confidence interval exclude zero" -- if
    the two ever disagree, it's a sign n_resamples was too low for how
    close p_value sits to alpha, and it's worth re-running with more
    resamples.
    """
    return result.p_value < alpha
