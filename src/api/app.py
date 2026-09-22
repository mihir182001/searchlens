
"""
FastAPI service wrapping Weeks 1-8's retrieval, routing,
and RAG pipeline behind a small HTTP API.

Same dependency-injection discipline as every other model/API-backed piece
of this project: create_app() takes already-constructed components (a
BM25Index, a DenseIndex, a QueryIntentClassifier, a passages dict, and an
optional RAGGenerator) rather than constructing them itself. That means
the ENTIRE HTTP layer -- request validation, intent-based routing
dispatch, error handling, response shaping -- is testable offline with
FastAPI's TestClient and fake stand-ins for every component (see
tests/test_api.py), with no real model, FAISS index, or Anthropic API call
anywhere in the test suite. The real components are only constructed in
run_api_server.py, which you run on your own machine.

Endpoints:
    GET  /health   -- liveness check.
    POST /search   -- classify a query's intent (Week 5), route it via
                       Week 5+8's routing table (BM25 or Dense -- Week 8's
                       A/B test found Dense beats Hybrid on this corpus, so
                       no route currently uses Hybrid), and return the
                       top-k passages plus a `low_confidence` flag.
    POST /rag      -- retrieve the top-5 passages with Dense (matching
                       Week 7's run_rag_eval.py depth and Week 8's
                       evidence that Dense is the strongest single
                       retriever here) and generate a cited answer (Week
                       7's RAGGenerator). Returns 503 if no RAGGenerator
                       was configured (e.g. no ANTHROPIC_API_KEY set) so
                       /search and /health keep working without one.

LOW-CONFIDENCE FLAGGING (added after live-testing against the real corpus):
    Manually testing /search turned up a real gap -- "mysql vs postgresql"
    got confidently routed and searched, but the corpus has ZERO passages
    mentioning either database (verified by grepping the raw corpus file),
    so Dense returned the least-bad generic database passage it could find
    at a score of 0.37, next to nothing distinguishing it as a poor match
    in the response itself. Compare that to "what is the capital of
    france", which correctly top-ranked the right passage at 0.75. The
    score gap between a genuine answer and "nothing relevant exists" is
    real signal -- this adds a `low_confidence` field to both responses so
    callers can tell the difference instead of trusting every returned
    passage equally.

    DENSE_LOW_CONFIDENCE_THRESHOLD (0.5 default) is a heuristic set from
    exactly those two live data points, not a statistically calibrated
    cutoff -- it sits BETWEEN the two observed scores (0.37 confirmed
    weak, 0.75 confirmed strong), so both real examples land on the
    correct side of it: 0.37 < 0.5 (flagged) and 0.75 > 0.5 (not flagged).
    An earlier version of this threshold was set to 0.35 -- BELOW the
    weak example instead of above it -- which meant it could never catch
    the exact case it was built for; caught by actually re-running the
    weak query against the live server after adding the feature, not by
    reasoning about it in the abstract. Revisit this value if you collect
    more labeled examples of "good match" vs "corpus doesn't have this"
    queries; until then, treat it as a reasonable starting point from two
    data points, not a proven value.

    BM25's raw score has no natural bound (it scales with term frequency,
    document length, and corpus statistics), so a single universal cutoff
    is much less meaningful there than Dense's bounded cosine similarity.
    BM25_LOW_CONFIDENCE_THRESHOLD defaults to None (never flags) rather
    than guessing a number with no evidence behind it -- same discipline
    as everywhere else in this project: don't report a heuristic as if
    it were measured.
"""
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.retrieval.query_router import retrieve, route_for_intent, BM25, DENSE, HYBRID

# Per-method score thresholds below which the top result is flagged
# low_confidence. None means "no calibrated threshold for this method --
# don't guess", so that method's results are never flagged on score alone
# (an empty result list is still always flagged, regardless of method).
DEFAULT_SCORE_THRESHOLDS = {
    DENSE: 0.5,
    BM25: None,
    HYBRID: None,
}


def _is_low_confidence(method: str, results: list, score_thresholds: dict) -> bool:
    if not results:
        return True
    threshold = score_thresholds.get(method)
    if threshold is None:
        return False
    return results[0].score < threshold


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)


class PassageResult(BaseModel):
    pid: str
    score: float
    text: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    intent: str
    confidence: float
    method: str
    low_confidence: bool
    results: list[PassageResult]


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1)


class RAGResponse(BaseModel):
    query: str
    answer: str
    cited_indices: list[int]
    low_confidence: bool
    passages: list[PassageResult]


RAG_DEPTH = 5  # matches Week 7's spec-mandated top-5 RAG context


def create_app(
    bm25_index,
    dense_index,
    classifier,
    passages: dict,
    rag_generator=None,
    score_thresholds: dict = None,
) -> FastAPI:
    app = FastAPI(title="SearchLens API")

    # Merge caller-provided thresholds over the defaults so a partial
    # override (e.g. just {"bm25": 5.0}) doesn't silently drop Dense's
    # default.
    thresholds = dict(DEFAULT_SCORE_THRESHOLDS)
    if score_thresholds:
        thresholds.update(score_thresholds)

    # Stashed on app.state mainly so tests / an interactive shell can
    # inspect what a running app was wired up with; the route closures
    # below use the local variables directly.
    app.state.bm25_index = bm25_index
    app.state.dense_index = dense_index
    app.state.classifier = classifier
    app.state.passages = passages
    app.state.rag_generator = rag_generator
    app.state.score_thresholds = thresholds

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest):
        prediction = classifier.predict(req.query)
        config = route_for_intent(prediction.intent)
        results = retrieve(req.query, prediction.intent, bm25_index, dense_index, top_k=req.top_k)
        return SearchResponse(
            query=req.query,
            intent=prediction.intent,
            confidence=prediction.confidence,
            method=config.method,
            low_confidence=_is_low_confidence(config.method, results, thresholds),
            results=[
                PassageResult(pid=r.pid, score=r.score, text=passages.get(r.pid))
                for r in results
            ],
        )

    @app.post("/rag", response_model=RAGResponse)
    def rag(req: RAGRequest):
        if rag_generator is None:
            raise HTTPException(
                status_code=503,
                detail="RAG is not configured on this server (no ANTHROPIC_API_KEY set).",
            )

        candidate_results = dense_index.search(req.query, top_k=RAG_DEPTH)
        if not candidate_results:
            raise HTTPException(status_code=404, detail="No passages retrieved for this query.")

        context_passages = [passages[r.pid] for r in candidate_results]
        answer = rag_generator.answer_query(req.query, context_passages)

        return RAGResponse(
            query=req.query,
            answer=answer.answer_text,
            cited_indices=answer.cited_indices,
            low_confidence=_is_low_confidence(DENSE, candidate_results, thresholds),
            passages=[
                PassageResult(pid=r.pid, score=r.score, text=passages[r.pid])
                for r in candidate_results
            ],
        )

    return app
