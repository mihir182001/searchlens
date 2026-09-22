"""
Test suite for building the
(query, candidates, binary label) dataset the LLM judge scores. No model
of any kind is needed here -- just fake retrieval results, same
SearchResult dataclass Weeks 1-4 already use.
"""
import pytest

from src.retrieval.hybrid_rrf import SearchResult
from src.evaluation.build_judge_dataset import build_judge_queries, sample_judge_queries, JudgeQuery


def _results(*pids):
    return [SearchResult(pid=pid, score=1.0) for pid in pids]


def test_build_judge_queries_basic_shape():
    queries = {"q0": "what is x"}
    passages = {"p1": "text about x", "p2": "text about y", "p3": "text about z"}
    qrels = {"q0": {"p2"}}
    retrieval_results = {"q0": _results("p1", "p2", "p3")}

    judge_queries = build_judge_queries(queries, passages, qrels, retrieval_results, depth=3)

    assert len(judge_queries) == 1
    jq = judge_queries[0]
    assert isinstance(jq, JudgeQuery)
    assert jq.qid == "q0"
    assert jq.query == "what is x"
    assert jq.pids == ["p1", "p2", "p3"]
    assert jq.passages == ["text about x", "text about y", "text about z"]
    assert jq.labels == [0, 1, 0]  # only p2 is in qrels


def test_build_judge_queries_respects_depth():
    queries = {"q0": "q"}
    passages = {f"p{i}": f"text {i}" for i in range(10)}
    qrels = {"q0": {"p0"}}
    retrieval_results = {"q0": _results(*[f"p{i}" for i in range(10)])}

    judge_queries = build_judge_queries(queries, passages, qrels, retrieval_results, depth=3)

    assert len(judge_queries[0].pids) == 3
    assert judge_queries[0].pids == ["p0", "p1", "p2"]


def test_build_judge_queries_skips_queries_with_no_retrieval_results():
    queries = {"q0": "has results", "q1": "no results"}
    passages = {"p1": "text"}
    qrels = {"q0": {"p1"}, "q1": {"p1"}}
    retrieval_results = {"q0": _results("p1")}  # q1 missing entirely

    judge_queries = build_judge_queries(queries, passages, qrels, retrieval_results, depth=5)

    assert len(judge_queries) == 1
    assert judge_queries[0].qid == "q0"


def test_build_judge_queries_handles_query_with_no_qrels_entry():
    # A query present in retrieval_results but absent from qrels -- every
    # candidate should just get label 0, not raise.
    queries = {"q0": "q"}
    passages = {"p1": "text"}
    qrels = {}  # q0 has no qrels entry at all
    retrieval_results = {"q0": _results("p1")}

    judge_queries = build_judge_queries(queries, passages, qrels, retrieval_results, depth=5)

    assert judge_queries[0].labels == [0]


def test_sample_judge_queries_returns_all_when_max_exceeds_total():
    judge_queries = [
        JudgeQuery(qid=f"q{i}", query="q", pids=["p1"], passages=["t"], labels=[0]) for i in range(3)
    ]
    sampled = sample_judge_queries(judge_queries, max_queries=10)
    assert len(sampled) == 3


def test_sample_judge_queries_respects_max():
    judge_queries = [
        JudgeQuery(qid=f"q{i}", query="q", pids=["p1"], passages=["t"], labels=[0]) for i in range(20)
    ]
    sampled = sample_judge_queries(judge_queries, max_queries=5, seed=42)
    assert len(sampled) == 5


def test_sample_judge_queries_is_deterministic_with_same_seed():
    judge_queries = [
        JudgeQuery(qid=f"q{i}", query="q", pids=["p1"], passages=["t"], labels=[0]) for i in range(20)
    ]
    sampled_a = sample_judge_queries(judge_queries, max_queries=5, seed=7)
    sampled_b = sample_judge_queries(judge_queries, max_queries=5, seed=7)
    assert [jq.qid for jq in sampled_a] == [jq.qid for jq in sampled_b]