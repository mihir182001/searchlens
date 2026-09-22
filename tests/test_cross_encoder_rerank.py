"""

Uses a deterministic, offline word-overlap scorer instead of a real
cross-encoder model, for the same reason DenseIndex's tests use a hashing
encoder: this sandbox cannot reach huggingface.co to download
cross-encoder/ms-marco-MiniLM-L-6-v2. These tests validate the actual
responsibility of CrossEncoderReranker -- pairing queries with the right
passage text, re-sorting by the new score, respecting top_k, handling
empty candidate lists -- without needing real semantic judgments. The real
re-ranking quality check happens in run_rerank_eval.py on your machine.
"""
import re
import pytest

from src.retrieval.cross_encoder_rerank import CrossEncoderReranker, SearchResult


def word_overlap_scorer(pairs):
    """pairs: list of (query, passage_text) tuples -> list of scores.

    Not a real cross-encoder -- just counts shared tokens between query and
    passage. Enough real signal to verify CrossEncoderReranker's plumbing
    (pairing/scoring/re-sorting) end-to-end without a network call.
    """
    scores = []
    for query, passage in pairs:
        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        p_tokens = set(re.findall(r"[a-z0-9]+", passage.lower()))
        scores.append(float(len(q_tokens & p_tokens)))
    return scores


@pytest.fixture
def passages():
    return {
        "p0": "The cat sat on the mat in the sun.",
        "p1": "Quantum computers use qubits instead of classical bits.",
        "p2": "A dog barked loudly at the mail carrier in the park.",
        "p3": "Superconducting qubits are a leading approach to quantum computing.",
    }


def _candidates(*pids):
    # First-stage results with deliberately WRONG/arbitrary original scores,
    # to make sure reranking actually re-sorts rather than trusting them.
    return [SearchResult(pid=pid, score=0.0) for pid in pids]


def test_rerank_reorders_by_new_score(passages):
    reranker = CrossEncoderReranker(predict_fn=word_overlap_scorer)
    # p3 has the most word overlap with the query, but is listed last in
    # the first-stage candidates -- reranking should move it to the top.
    candidates = _candidates("p0", "p2", "p1", "p3")
    results = reranker.rerank("quantum computing qubits", candidates, passages, top_k=4)
    assert results[0].pid == "p3"


def test_rerank_respects_top_k(passages):
    reranker = CrossEncoderReranker(predict_fn=word_overlap_scorer)
    candidates = _candidates("p0", "p1", "p2", "p3")
    results = reranker.rerank("quantum qubits", candidates, passages, top_k=2)
    assert len(results) == 2


def test_rerank_empty_candidates_returns_empty(passages):
    reranker = CrossEncoderReranker(predict_fn=word_overlap_scorer)
    results = reranker.rerank("anything", [], passages, top_k=10)
    assert results == []


def test_rerank_scores_are_descending(passages):
    reranker = CrossEncoderReranker(predict_fn=word_overlap_scorer)
    candidates = _candidates("p0", "p1", "p2", "p3")
    results = reranker.rerank("quantum qubits park dog", candidates, passages, top_k=4)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_only_scores_given_candidates_not_whole_corpus(passages):
    """A cross-encoder never sees passages outside the first-stage
    candidate set -- if only p0 and p2 were retrieved upstream, p1/p3
    (better matches for this query) must never appear in the output.
    """
    reranker = CrossEncoderReranker(predict_fn=word_overlap_scorer)
    candidates = _candidates("p0", "p2")  # deliberately excludes p1, p3
    results = reranker.rerank("quantum qubits computing", candidates, passages, top_k=10)
    result_pids = {r.pid for r in results}
    assert result_pids == {"p0", "p2"}


def test_rerank_many_returns_one_entry_per_query(passages):
    reranker = CrossEncoderReranker(predict_fn=word_overlap_scorer)
    queries = {"q0": "quantum qubits", "q1": "dog park"}
    first_stage = {
        "q0": _candidates("p0", "p1", "p3"),
        "q1": _candidates("p0", "p2"),
    }
    results = reranker.rerank_many(queries, first_stage, passages, top_k=5)
    assert set(results.keys()) == {"q0", "q1"}
    assert results["q0"][0].pid in {"p1", "p3"}
    assert results["q1"][0].pid == "p2"


def test_rerank_many_handles_query_with_no_first_stage_results(passages):
    reranker = CrossEncoderReranker(predict_fn=word_overlap_scorer)
    queries = {"q0": "quantum qubits", "q1": "nothing retrieved for this one"}
    first_stage = {"q0": _candidates("p1", "p3")}  # q1 missing entirely
    results = reranker.rerank_many(queries, first_stage, passages, top_k=5)
    assert results["q1"] == []