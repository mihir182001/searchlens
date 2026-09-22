"""
tests/tes test suite for
per_query_reciprocal_rank, added to src/evaluation/metrics.py so
ab_test.py's paired significance test has per-query scores to work with
(mrr_at_k only ever returned the mean, not the per-qid breakdown).
"""
import pytest

from src.evaluation.metrics import per_query_reciprocal_rank, mrr_at_k


class R:
    def __init__(self, pid):
        self.pid = pid


def test_per_query_reciprocal_rank_manual_example():
    results = {
        "q0": [R("p1"), R("p2")],  # relevant doc (p1) ranked 1st -> RR = 1.0
        "q1": [R("p2"), R("p1")],  # relevant doc (p1) ranked 2nd -> RR = 0.5
    }
    qrels = {"q0": {"p1"}, "q1": {"p1"}}
    scores = per_query_reciprocal_rank(results, qrels, k=10)
    assert scores == {"q0": pytest.approx(1.0), "q1": pytest.approx(0.5)}


def test_per_query_reciprocal_rank_missing_query_scores_zero():
    results = {"q0": [R("p1")]}  # q1 has no results at all
    qrels = {"q0": {"p1"}, "q1": {"p2"}}
    scores = per_query_reciprocal_rank(results, qrels, k=10)
    assert scores == {"q0": pytest.approx(1.0), "q1": 0.0}


def test_per_query_reciprocal_rank_not_found_is_zero():
    results = {"q0": [R("p9"), R("p8")]}
    qrels = {"q0": {"p1"}}
    scores = per_query_reciprocal_rank(results, qrels, k=10)
    assert scores == {"q0": 0.0}


def test_per_query_reciprocal_rank_respects_k():
    results = {"q0": [R("p9"), R("p1")]}  # relevant doc at rank 2
    qrels = {"q0": {"p1"}}
    assert per_query_reciprocal_rank(results, qrels, k=1) == {"q0": 0.0}
    assert per_query_reciprocal_rank(results, qrels, k=2) == {"q0": pytest.approx(0.5)}


def test_per_query_reciprocal_rank_raises_on_empty_qrels():
    with pytest.raises(ValueError):
        per_query_reciprocal_rank({}, {}, k=10)


def test_mrr_at_k_equals_mean_of_per_query_scores():
    results = {
        "q0": [R("p1"), R("p2")],
        "q1": [R("p2"), R("p1")],
        "q2": [R("p9")],
    }
    qrels = {"q0": {"p1"}, "q1": {"p1"}, "q2": {"p1"}}
    per_query = per_query_reciprocal_rank(results, qrels, k=10)
    assert mrr_at_k(results, qrels, k=10) == pytest.approx(sum(per_query.values()) / len(per_query))