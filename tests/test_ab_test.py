"""
test suite for the paired bootstrap
significance test in src/evaluation/ab_test.py. Fully offline -- pure
Python arithmetic and stdlib `random`, no model or API involved.
"""
import pytest

from src.evaluation.ab_test import paired_bootstrap_test, is_significant, ABTestResult


def test_identical_scores_show_zero_difference_and_p_value_of_one():
    # If A and B are literally the same scores on every query, every
    # possible resample also has mean_diff == 0, so the observed
    # difference (0) is never "more extreme" than any resample -- p must
    # be exactly 1.0, not just "high".
    scores = {f"q{i}": 0.5 for i in range(20)}
    result = paired_bootstrap_test(scores, dict(scores), n_resamples=2000, seed=1)
    assert result.mean_diff == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert not is_significant(result)


def test_constant_improvement_on_every_query_is_significant():
    # B beats A by exactly 0.2 on every single query -- every resample of
    # these diffs is also exactly 0.2 (resampling a constant array can't
    # produce anything else), so the CI collapses to a point at 0.2 and
    # the null hypothesis of "no difference" is rejected outright.
    scores_a = {f"q{i}": 0.3 for i in range(15)}
    scores_b = {f"q{i}": 0.5 for i in range(15)}
    result = paired_bootstrap_test(scores_a, scores_b, n_resamples=2000, seed=1)
    assert result.mean_diff == pytest.approx(0.2)
    assert result.ci_low == pytest.approx(0.2)
    assert result.ci_high == pytest.approx(0.2)
    assert result.p_value == pytest.approx(0.0)
    assert is_significant(result)


def test_strong_majority_improvement_with_one_outlier_is_still_significant():
    # B beats A on 9/10 queries by a wide margin (+0.3) and loses on just
    # 1/10 by a smaller margin (-0.2) -- the improvement is consistent
    # enough that the effect should still be clearly detectable.
    scores_a = {f"q{i}": 0.0 for i in range(10)}
    scores_b = {f"q{i}": 0.3 for i in range(9)}
    scores_b["q9"] = -0.2
    result = paired_bootstrap_test(scores_a, scores_b, n_resamples=5000, seed=7)
    assert result.mean_diff > 0
    assert result.ci_low > 0
    assert is_significant(result)


def test_mean_a_and_mean_b_match_plain_averages():
    scores_a = {"q0": 1.0, "q1": 0.0, "q2": 0.5}
    scores_b = {"q0": 0.5, "q1": 1.0, "q2": 0.5}
    result = paired_bootstrap_test(scores_a, scores_b, n_resamples=500, seed=1)
    assert result.mean_a == pytest.approx(0.5)
    assert result.mean_b == pytest.approx(2 / 3)
    assert result.n_queries == 3


def test_raises_on_mismatched_query_sets():
    scores_a = {"q0": 1.0, "q1": 0.5}
    scores_b = {"q0": 1.0, "q2": 0.5}
    with pytest.raises(ValueError, match="same queries"):
        paired_bootstrap_test(scores_a, scores_b)


def test_raises_on_fewer_than_two_shared_queries():
    with pytest.raises(ValueError, match="at least 2"):
        paired_bootstrap_test({"q0": 1.0}, {"q0": 0.5})


def test_raises_on_nonpositive_n_resamples():
    with pytest.raises(ValueError, match="n_resamples"):
        paired_bootstrap_test({"q0": 1.0, "q1": 0.0}, {"q0": 0.5, "q1": 0.5}, n_resamples=0)


def test_result_is_deterministic_given_same_seed():
    scores_a = {f"q{i}": (i % 3) / 3 for i in range(20)}
    scores_b = {f"q{i}": ((i + 1) % 3) / 3 for i in range(20)}
    r1 = paired_bootstrap_test(scores_a, scores_b, n_resamples=1000, seed=99)
    r2 = paired_bootstrap_test(scores_a, scores_b, n_resamples=1000, seed=99)
    assert r1 == r2  # dataclass equality -- same seed must give an identical result


def test_mean_diff_does_not_depend_on_bootstrap_seed():
    # The OBSERVED mean difference comes straight from the data, not the
    # resampling -- different seeds can shift the CI/p-value slightly but
    # must never change mean_diff itself.
    scores_a = {f"q{i}": (i % 4) / 4 for i in range(30)}
    scores_b = {f"q{i}": ((i + 2) % 4) / 4 for i in range(30)}
    r1 = paired_bootstrap_test(scores_a, scores_b, n_resamples=3000, seed=1)
    r2 = paired_bootstrap_test(scores_a, scores_b, n_resamples=3000, seed=2)
    assert r1.mean_diff == pytest.approx(r2.mean_diff)


def test_is_significant_thresholds_on_alpha():
    result = ABTestResult(
        n_queries=100, mean_a=0.5, mean_b=0.6, mean_diff=0.1,
        ci_low=0.01, ci_high=0.19, confidence=0.95, p_value=0.03,
    )
    assert is_significant(result, alpha=0.05) is True
    assert is_significant(result, alpha=0.01) is False