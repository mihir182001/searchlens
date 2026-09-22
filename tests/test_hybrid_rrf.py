"""
RRF needs no model, so unlike Week 2/3's tests this validates the REAL
production logic directly no offline stand-in required.
"""
import pytest

from src.retrieval.hybrid_rrf import reciprocal_rank_fusion, fuse_many, SearchResult


def _results(*pids):
    return [SearchResult(pid=pid, score=0.0) for pid in pids]


def test_rrf_rewards_agreement_between_lists():
    # p2 is #1 in both lists -- should come out #1 fused, ahead of p1
    # (which is #1 in only one list).
    bm25 = _results("p1", "p2", "p3")
    dense = _results("p2", "p4", "p1")
    fused = reciprocal_rank_fusion([bm25, dense], top_k=10)
    assert fused[0].pid == "p2"


def test_rrf_single_list_preserves_relative_order():
    bm25 = _results("p1", "p2", "p3")
    fused = reciprocal_rank_fusion([bm25], top_k=10)
    assert [r.pid for r in fused] == ["p1", "p2", "p3"]


def test_rrf_document_in_only_one_list_is_still_included():
    bm25 = _results("p1", "p2")
    dense = _results("p3", "p4")
    fused = reciprocal_rank_fusion([bm25, dense], top_k=10)
    assert {r.pid for r in fused} == {"p1", "p2", "p3", "p4"}


def test_rrf_respects_top_k():
    bm25 = _results("p1", "p2", "p3", "p4", "p5")
    fused = reciprocal_rank_fusion([bm25], top_k=2)
    assert len(fused) == 2


def test_rrf_manual_score_calculation():
    # p1 is rank 1 in list A (score contribution 1/(60+1)) and rank 2 in
    # list B (contribution 1/(60+2)) -- verify the exact fused score.
    list_a = _results("p1", "p2")
    list_b = _results("p2", "p1")
    fused = reciprocal_rank_fusion([list_a, list_b], k=60, top_k=10)
    fused_by_pid = {r.pid: r.score for r in fused}
    expected_p1 = 1 / (60 + 1) + 1 / (60 + 2)
    expected_p2 = 1 / (60 + 2) + 1 / (60 + 1)
    assert fused_by_pid["p1"] == pytest.approx(expected_p1)
    assert fused_by_pid["p2"] == pytest.approx(expected_p2)
    assert fused_by_pid["p1"] == pytest.approx(fused_by_pid["p2"])  # symmetric agreement


def test_rrf_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([], top_k=10) == []
    assert reciprocal_rank_fusion([[], []], top_k=10) == []


def test_fuse_many_returns_all_qids_across_dicts():
    bm25_results = {"q0": _results("p1", "p2"), "q1": _results("p3")}
    dense_results = {"q0": _results("p2", "p1"), "q2": _results("p4")}  # q2 only in dense
    fused = fuse_many([bm25_results, dense_results], top_k=10)
    assert set(fused.keys()) == {"q0", "q1", "q2"}


def test_fuse_many_handles_qid_missing_from_one_method():
    bm25_results = {"q0": _results("p1", "p2")}
    dense_results = {}  # q0 missing entirely from dense
    fused = fuse_many([bm25_results, dense_results], top_k=10)
    assert [r.pid for r in fused["q0"]] == ["p1", "p2"]


# ---------------------------------------------------------------------------
# Weighted RRF -- added after the real Week 4 run showed equal-weight RRF
# (0.8394 MRR@10) underperforming Dense alone (0.9130), a known failure
# mode when one retriever is much stronger than the other.
# ---------------------------------------------------------------------------

def test_weighted_rrf_lets_stronger_list_dominate():
    # p1 is BM25's #1 pick; p2 is Dense's #1 pick. With equal weights
    # they'd tie (both rank 1 in their own single-item list); weighting
    # Dense higher should make its pick win outright.
    bm25 = _results("p1")
    dense = _results("p2")

    equal = reciprocal_rank_fusion([bm25, dense], top_k=10, weights=[1.0, 1.0])
    assert equal[0].score == pytest.approx(equal[1].score)  # tied under equal weight

    dense_favored = reciprocal_rank_fusion([bm25, dense], top_k=10, weights=[1.0, 3.0])
    assert dense_favored[0].pid == "p2"  # Dense's pick now clearly wins


def test_weighted_rrf_manual_score_calculation():
    list_a = _results("p1")
    list_b = _results("p1")
    fused = reciprocal_rank_fusion([list_a, list_b], k=60, top_k=10, weights=[1.0, 2.0])
    expected = 1.0 / (60 + 1) + 2.0 / (60 + 1)
    assert fused[0].score == pytest.approx(expected)


def test_weighted_rrf_defaults_to_equal_weights():
    bm25 = _results("p1", "p2")
    dense = _results("p2", "p1")
    unweighted = reciprocal_rank_fusion([bm25, dense], top_k=10)
    explicitly_equal = reciprocal_rank_fusion([bm25, dense], top_k=10, weights=[1.0, 1.0])
    assert [(r.pid, r.score) for r in unweighted] == [(r.pid, r.score) for r in explicitly_equal]


def test_weighted_rrf_rejects_mismatched_weights_length():
    bm25 = _results("p1")
    dense = _results("p2")
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([bm25, dense], weights=[1.0])


def test_fuse_many_passes_weights_through():
    bm25_results = {"q0": _results("p1")}
    dense_results = {"q0": _results("p2")}
    fused = fuse_many([bm25_results, dense_results], top_k=10, weights=[1.0, 3.0])
    assert fused["q0"][0].pid == "p2"