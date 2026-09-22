
"""
Week 9 test suite for the FastAPI service in
src/api/app.py. Uses fake BM25Index/DenseIndex/classifier/RAGGenerator
stand-ins (same duck-typed pattern as tests/test_query_router.py) plus
FastAPI's TestClient, which talks to the app in-process over ASGI -- no
real network socket, model, FAISS index, or Anthropic API call anywhere
in this file.
"""
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.retrieval.hybrid_rrf import SearchResult
from src.models.query_intent_classifier import IntentPrediction


class _FakeIndex:
    def __init__(self, name, n_results=3, scores=None):
        self.name = name
        self.n_results = n_results
        # Optional explicit score list, for testing low_confidence flagging
        # -- defaults to a descending 1.0, 0.5, 0.333... series (always
        # comfortably above any reasonable threshold) when not given.
        self.scores = scores
        self.calls = []

    def search(self, query, top_k=10):
        self.calls.append((query, top_k))
        n = min(top_k, self.n_results)
        scores = self.scores[:n] if self.scores is not None else [1.0 / (i + 1) for i in range(n)]
        return [SearchResult(pid=f"{self.name}-p{i}", score=scores[i]) for i in range(n)]


class _EmptyIndex:
    def search(self, query, top_k=10):
        return []


class _FakeClassifier:
    def __init__(self, intent="factual", confidence=0.9):
        self.intent = intent
        self.confidence = confidence
        self.queries_seen = []

    def predict(self, query):
        self.queries_seen.append(query)
        return IntentPrediction(intent=self.intent, confidence=self.confidence)


class _FakeRAGAnswer:
    def __init__(self, answer_text, cited_indices):
        self.answer_text = answer_text
        self.cited_indices = cited_indices


class _FakeRAGGenerator:
    def __init__(self, answer_text="the answer [1]", cited_indices=None):
        self.answer_text = answer_text
        self.cited_indices = cited_indices if cited_indices is not None else [1]
        self.calls = []

    def answer_query(self, query, passages):
        self.calls.append((query, passages))
        return _FakeRAGAnswer(self.answer_text, self.cited_indices)


PASSAGES = {
    "dense-p0": "dense passage 0 text", "dense-p1": "dense passage 1 text", "dense-p2": "dense passage 2 text",
    "bm25-p0": "bm25 passage 0 text", "bm25-p1": "bm25 passage 1 text", "bm25-p2": "bm25 passage 2 text",
}


def make_client(intent="factual", confidence=0.9, rag_generator=None, dense_index=None,
                 bm25_index=None, score_thresholds=None):
    bm25_index = bm25_index or _FakeIndex("bm25")
    dense_index = dense_index or _FakeIndex("dense")
    classifier = _FakeClassifier(intent=intent, confidence=confidence)
    app = create_app(
        bm25_index, dense_index, classifier, PASSAGES,
        rag_generator=rag_generator, score_thresholds=score_thresholds,
    )
    return TestClient(app), bm25_index, dense_index, classifier


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_returns_ok():
    client, _, _, _ = make_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /search
# ---------------------------------------------------------------------------

def test_search_routes_factual_to_dense_and_returns_results():
    client, bm25_index, dense_index, classifier = make_client(intent="factual", confidence=0.87)
    response = client.post("/search", json={"query": "what is the capital of france", "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "factual"
    assert body["confidence"] == pytest.approx(0.87)
    assert body["method"] == "dense"
    assert len(body["results"]) == 2
    assert all(r["pid"].startswith("dense-") for r in body["results"])
    assert body["results"][0]["text"] == "dense passage 0 text"
    assert body["low_confidence"] is False  # top score 1.0, comfortably above the 0.5 dense default
    assert len(bm25_index.calls) == 0
    assert len(dense_index.calls) == 1
    assert classifier.queries_seen == ["what is the capital of france"]


def test_search_routes_navigational_to_bm25():
    client, bm25_index, dense_index, _ = make_client(intent="navigational", confidence=0.99)
    response = client.post("/search", json={"query": "facebook login", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "bm25"
    assert all(r["pid"].startswith("bm25-") for r in body["results"])
    assert len(bm25_index.calls) == 1
    assert len(dense_index.calls) == 0


def test_search_rejects_empty_query():
    client, _, _, _ = make_client()
    response = client.post("/search", json={"query": "", "top_k": 5})
    assert response.status_code == 422


def test_search_rejects_top_k_out_of_range():
    client, _, _, _ = make_client()
    assert client.post("/search", json={"query": "hello", "top_k": 0}).status_code == 422
    assert client.post("/search", json={"query": "hello", "top_k": 1000}).status_code == 422


def test_search_default_top_k_is_ten():
    client, _, dense_index, _ = make_client(intent="factual")
    client.post("/search", json={"query": "hello"})
    assert dense_index.calls[0][1] == 10


def test_search_rejects_missing_query_field():
    client, _, _, _ = make_client()
    response = client.post("/search", json={"top_k": 5})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# low_confidence flagging (added after live-testing "mysql vs postgresql"
# against the real corpus turned up zero matching passages -- see app.py's
# module docstring for the full story)
# ---------------------------------------------------------------------------

def test_search_flags_low_confidence_when_dense_score_below_default_threshold():
    dense_index = _FakeIndex("dense", scores=[0.1, 0.05, 0.02])
    client, _, _, _ = make_client(intent="factual", dense_index=dense_index)
    response = client.post("/search", json={"query": "mysql vs postgresql", "top_k": 3})
    assert response.status_code == 200
    assert response.json()["low_confidence"] is True


def test_search_does_not_flag_bm25_regardless_of_score_by_default():
    # BM25's raw score has no universal, corpus-independent cutoff, so the
    # default threshold is None (feature off) rather than a guessed value
    # -- even a very low BM25 score must NOT be flagged out of the box.
    bm25_index = _FakeIndex("bm25", scores=[0.001])
    client, _, _, _ = make_client(intent="navigational", bm25_index=bm25_index)
    response = client.post("/search", json={"query": "facebook login", "top_k": 1})
    assert response.status_code == 200
    assert response.json()["low_confidence"] is False


def test_search_flags_low_confidence_when_no_results_at_all():
    client, _, _, _ = make_client(intent="factual", dense_index=_EmptyIndex())
    response = client.post("/search", json={"query": "something obscure", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["results"] == []
    assert body["low_confidence"] is True


def test_search_custom_threshold_can_enable_bm25_flagging():
    bm25_index = _FakeIndex("bm25", scores=[0.1])
    client, _, _, _ = make_client(
        intent="navigational", bm25_index=bm25_index, score_thresholds={"bm25": 0.5},
    )
    response = client.post("/search", json={"query": "facebook login", "top_k": 1})
    assert response.json()["low_confidence"] is True


def test_search_custom_bm25_threshold_does_not_disturb_dense_default():
    # A partial override ({"bm25": ...}) must merge over the defaults, not
    # replace them wholesale -- dense should still use its 0.5 default.
    dense_index = _FakeIndex("dense", scores=[0.1, 0.05])
    client, _, _, _ = make_client(
        intent="factual", dense_index=dense_index, score_thresholds={"bm25": 5.0},
    )
    response = client.post("/search", json={"query": "mysql vs postgresql", "top_k": 2})
    assert response.json()["low_confidence"] is True


def test_search_regression_default_threshold_correctly_brackets_real_observed_scores():
    # Locks in the exact bug found by live-testing against the real
    # server: the first version of this threshold (0.35) sat BELOW the
    # real "weak match" score (0.3668, from "mysql vs postgresql" against
    # a corpus with zero passages on either database), so it never
    # flagged the case it was built for. The real "strong match" score
    # (0.753, from "what is the capital of france") must stay unflagged.
    # These are the ACTUAL scores the live server returned, not synthetic
    # round numbers, so this test would have caught the original bug.
    weak_dense_index = _FakeIndex("dense", scores=[0.3667989671230316])
    client, _, _, _ = make_client(intent="factual", dense_index=weak_dense_index)
    weak_response = client.post("/search", json={"query": "mysql vs postgresql", "top_k": 1})
    assert weak_response.json()["low_confidence"] is True

    strong_dense_index = _FakeIndex("dense", scores=[0.752997636795044])
    client, _, _, _ = make_client(intent="factual", dense_index=strong_dense_index)
    strong_response = client.post("/search", json={"query": "what is the capital of france", "top_k": 1})
    assert strong_response.json()["low_confidence"] is False


# ---------------------------------------------------------------------------
# /rag
# ---------------------------------------------------------------------------

def test_rag_returns_503_when_not_configured():
    client, _, _, _ = make_client(rag_generator=None)
    response = client.post("/rag", json={"query": "when was the eiffel tower built"})
    assert response.status_code == 503


def test_rag_returns_answer_and_citations_when_configured():
    rag_generator = _FakeRAGGenerator(answer_text="It was built in 1889 [1].", cited_indices=[1])
    client, _, dense_index, _ = make_client(rag_generator=rag_generator)
    response = client.post("/rag", json={"query": "when was the eiffel tower built"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "It was built in 1889 [1]."
    assert body["cited_indices"] == [1]
    assert len(body["passages"]) == 3  # _FakeIndex's default n_results
    assert body["low_confidence"] is False  # top score 1.0, above the 0.5 dense default
    assert dense_index.calls[0][1] == 5  # RAG always retrieves top-5 (spec depth)
    assert len(rag_generator.calls) == 1


def test_rag_passes_retrieved_passage_text_to_generator():
    rag_generator = _FakeRAGGenerator()
    client, _, _, _ = make_client(rag_generator=rag_generator)
    client.post("/rag", json={"query": "some query"})
    seen_query, seen_passages = rag_generator.calls[0]
    assert seen_query == "some query"
    assert seen_passages == ["dense passage 0 text", "dense passage 1 text", "dense passage 2 text"]


def test_rag_returns_404_when_no_passages_retrieved():
    rag_generator = _FakeRAGGenerator()
    client, _, _, _ = make_client(rag_generator=rag_generator, dense_index=_EmptyIndex())
    response = client.post("/rag", json={"query": "some obscure query"})
    assert response.status_code == 404


def test_rag_rejects_empty_query():
    client, _, _, _ = make_client(rag_generator=_FakeRAGGenerator())
    response = client.post("/rag", json={"query": ""})
    assert response.status_code == 422


def test_rag_flags_low_confidence_when_dense_score_below_threshold():
    # _FakeIndex only ever returns min(top_k, n_results) results, and
    # n_results here stays at the default (3) -- PASSAGES only has
    # dense-p0..dense-p2 defined, and RAG asking for top_k=5 would
    # otherwise try to look up a pid with no passage text behind it.
    rag_generator = _FakeRAGGenerator()
    dense_index = _FakeIndex("dense", scores=[0.1, 0.08, 0.05])
    client, _, _, _ = make_client(rag_generator=rag_generator, dense_index=dense_index)
    response = client.post("/rag", json={"query": "mysql vs postgresql"})
    assert response.status_code == 200
    assert response.json()["low_confidence"] is True