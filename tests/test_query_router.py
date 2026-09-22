"""
test suite for intent -> retrieval
method routing. Uses fake BM25/Dense index stand-ins (not real models) so
the dispatch logic is tested fully offline -- reciprocal_rank_fusion's own
arithmetic is already covered by tests/test_hybrid_rrf.py.
"""
import pytest

from src.data.query_intent_dataset import INTENT_LABELS
from src.retrieval.hybrid_rrf import SearchResult
from src.retrieval.query_router import (
    BM25,
    DENSE,
    HYBRID,
    ROUTING_TABLE,
    RouteConfig,
    route_for_intent,
    retrieve,
    validate_routing_table_covers_all_labels,
)


def test_routing_table_covers_every_intent_label():
    assert validate_routing_table_covers_all_labels()
    assert set(ROUTING_TABLE.keys()) == set(INTENT_LABELS)


def test_route_for_intent_matches_expected_methods():
    # exploratory/comparison route to DENSE, not HYBRID, as of Week 8:
    # a paired A/B significance test found Hybrid RRF (even at
    # dense_weight=10) significantly WORSE than Dense alone on this
    # corpus (see query_router.py's module docstring for the numbers).
    assert route_for_intent("factual").method == DENSE
    assert route_for_intent("navigational").method == BM25
    assert route_for_intent("exploratory").method == DENSE
    assert route_for_intent("comparison").method == DENSE
    assert route_for_intent("how-to").method == BM25


def test_route_for_intent_rejects_unknown_intent():
    with pytest.raises(ValueError):
        route_for_intent("not-a-real-intent")


class _FakeIndex:
    """Stand-in for BM25Index/DenseIndex -- returns a fixed, distinguishable
    result list so tests can tell which index actually got called."""

    def __init__(self, name):
        self.name = name
        self.calls = []

    def search(self, query, top_k=10):
        self.calls.append((query, top_k))
        return [SearchResult(pid=f"{self.name}-p{i}", score=1.0 / (i + 1)) for i in range(top_k)]


def test_retrieve_dispatches_navigational_to_bm25_only():
    bm25_index = _FakeIndex("bm25")
    dense_index = _FakeIndex("dense")
    results = retrieve("facebook login", "navigational", bm25_index, dense_index, top_k=5)
    assert len(bm25_index.calls) == 1
    assert len(dense_index.calls) == 0
    assert all(r.pid.startswith("bm25-") for r in results)


def test_retrieve_dispatches_factual_to_dense_only():
    bm25_index = _FakeIndex("bm25")
    dense_index = _FakeIndex("dense")
    results = retrieve("what is the capital of france", "factual", bm25_index, dense_index, top_k=5)
    assert len(bm25_index.calls) == 0
    assert len(dense_index.calls) == 1
    assert all(r.pid.startswith("dense-") for r in results)


def test_retrieve_dispatches_how_to_to_bm25_only():
    bm25_index = _FakeIndex("bm25")
    dense_index = _FakeIndex("dense")
    results = retrieve("how to bake bread", "how-to", bm25_index, dense_index, top_k=5)
    assert len(bm25_index.calls) == 1
    assert len(dense_index.calls) == 0
    assert all(r.pid.startswith("bm25-") for r in results)


def test_retrieve_dispatches_exploratory_to_dense_only():
    # Changed in Week 8: exploratory used to route to hybrid (both
    # indices); a paired A/B significance test found Dense alone
    # significantly beats Hybrid RRF on this corpus, so exploratory routes
    # to dense-only now. See query_router.py's module docstring.
    bm25_index = _FakeIndex("bm25")
    dense_index = _FakeIndex("dense")
    results = retrieve("explain how photosynthesis works", "exploratory", bm25_index, dense_index, top_k=5)
    assert len(bm25_index.calls) == 0
    assert len(dense_index.calls) == 1
    assert all(r.pid.startswith("dense-") for r in results)


def test_retrieve_dispatches_comparison_to_dense_only():
    # Changed in Week 8, same reasoning as exploratory above.
    bm25_index = _FakeIndex("bm25")
    dense_index = _FakeIndex("dense")
    results = retrieve("mysql vs postgresql", "comparison", bm25_index, dense_index, top_k=5)
    assert len(bm25_index.calls) == 0
    assert len(dense_index.calls) == 1
    assert all(r.pid.startswith("dense-") for r in results)


def test_retrieve_still_supports_hybrid_dispatch_via_monkeypatched_route(monkeypatch):
    # No current ROUTING_TABLE entry uses HYBRID any more (Week 8 moved
    # exploratory/comparison to DENSE), but the dispatch branch itself must
    # keep working in case a route is ever switched back once real
    # per-intent-class evidence supports it -- verified here without
    # touching the real table.
    import src.retrieval.query_router as query_router

    monkeypatch.setitem(
        query_router.ROUTING_TABLE, "exploratory",
        RouteConfig(method=HYBRID, bm25_weight=1.0, dense_weight=10.0),
    )
    bm25_index = _FakeIndex("bm25")
    dense_index = _FakeIndex("dense")
    results = retrieve("explain how photosynthesis works", "exploratory", bm25_index, dense_index, top_k=5)
    assert len(bm25_index.calls) == 1
    assert len(dense_index.calls) == 1
    assert bm25_index.calls[0][1] == 100  # fused at depth 100 before trimming to top_k
    assert dense_index.calls[0][1] == 100
    assert len(results) <= 5


def test_retrieve_respects_top_k_for_single_method_routes():
    bm25_index = _FakeIndex("bm25")
    dense_index = _FakeIndex("dense")
    results = retrieve("facebook login", "navigational", bm25_index, dense_index, top_k=3)
    assert len(results) == 3


def test_retrieve_rejects_unknown_intent():
    bm25_index = _FakeIndex("bm25")
    dense_index = _FakeIndex("dense")
    with pytest.raises(ValueError):
        retrieve("some query", "not-a-real-intent", bm25_index, dense_index)