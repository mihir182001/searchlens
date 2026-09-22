"""
 Test suite for pearson_correlation, added to src/evaluation/metrics.py.
"""
import pytest

from src.evaluation.metrics import pearson_correlation


def test_perfect_positive_correlation():
    xs = [1, 2, 3, 4, 5]
    ys = [2, 4, 6, 8, 10]
    assert pearson_correlation(xs, ys) == pytest.approx(1.0)


def test_perfect_negative_correlation():
    xs = [1, 2, 3, 4, 5]
    ys = [10, 8, 6, 4, 2]
    assert pearson_correlation(xs, ys) == pytest.approx(-1.0)


def test_binary_label_correlation_matches_manual_expectation():
    # Scores clearly track the binary label -- relevant (1) passages
    # scored high, non-relevant (0) passages scored low. Correlation
    # should be strongly positive, matching Week 6's actual use case
    # (LLM 1-5 score vs. binary qrel label).
    scores = [5, 4, 1, 2, 5, 1]
    labels = [1, 1, 0, 0, 1, 0]
    correlation = pearson_correlation(scores, labels)
    assert correlation > 0.8


def test_no_correlation_when_scores_dont_track_labels():
    scores = [3, 3, 3, 3]
    labels = [1, 0, 1, 0]
    with pytest.raises(ValueError):
        # zero variance in scores -- undefined correlation, must raise
        pearson_correlation(scores, labels)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        pearson_correlation([1, 2, 3], [1, 2])


def test_rejects_fewer_than_two_pairs():
    with pytest.raises(ValueError):
        pearson_correlation([1], [1])


def test_rejects_zero_variance_labels():
    scores = [1, 2, 3, 4]
    labels = [1, 1, 1, 1]  # every label identical -- zero variance
    with pytest.raises(ValueError):
        pearson_correlation(scores, labels)