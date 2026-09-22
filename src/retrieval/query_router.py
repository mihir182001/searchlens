"""
 retrieval-method routing logic.

No model of any kind is needed here -- this maps a CLASSIFIED intent label
(a string) to which of Week 1-4's retrieval methods should handle a query
with that intent, plus any method-specific config (e.g. hybrid RRF
weights). Fully offline-testable.

Routing rationale (per the SearchLens spec's own reasoning for each class,
plus what Weeks 1-4-and-8 actually measured on this project's corpus):

    factual       -> dense
        Has one correct, verifiable answer -- usually phrased very
        differently from the passage that states it ("what is the boiling
        point of water" vs. a passage that never uses the phrase "boiling
        point" as a question). Week 2 showed dense retrieval's semantic
        matching clearly outperforms BM25's keyword overlap here
        (MRR@10 0.9130 vs 0.7238 on this project's real corpus).

    navigational  -> bm25
        The user already knows the exact site/page they want ("facebook
        login") -- these queries are themselves close to exact keyword
        matches for the target page's title/URL text, which is BM25's
        strength, not a case needing semantic generalization.

    exploratory   -> dense   (CHANGED after Week 8 -- was hybrid)
    comparison    -> dense   (CHANGED after Week 8 -- was hybrid)
        Originally routed to hybrid on the reasoning that RRF's
        recall-broadening -- combining two retrievers' different notions
        of relevance -- would help both "broad understanding of a topic"
        (exploratory) and "passages about MULTIPLE distinct things"
        (comparison) queries. Week 8's A/B test (src/evaluation/ab_test.py,
        run via run_ab_test.py) checked that reasoning against a paired
        significance test on real aggregate numbers and found the
        opposite: Hybrid RRF at its best-found config (dense_weight=10,
        aggregate MRR@10 0.8980) is SIGNIFICANTLY WORSE than Dense alone
        (0.9130) on this corpus -- 95% CI on the difference
        [-0.0252, -0.0052], p=0.0039, so this is not query-sampling noise.
        The mechanism: RRF still uses BM25's RANK POSITIONS even at a
        heavily downweighted contribution, and BM25 confidently misranking
        a passage can still demote a passage Dense already had correct at
        rank 1 (the known RRF failure mode hybrid_rrf.py's docstring
        already flagged as a risk -- Week 8 confirmed it's what actually
        happens here). Until a genuine PER-INTENT-CLASS evaluation exists
        (see the caveat below) showing exploratory/comparison queries
        specifically behave differently from the aggregate, routing them
        to Dense is the evidence-backed choice, not Hybrid.

    how-to        -> bm25
        Step-by-step instructional content tends to share a lot of literal
        vocabulary with the query itself (titles like "How to Fix a Flat
        Tire" almost restate the query), so BM25's exact-term matching is
        a reasonable, cheap first choice; dense remains a fallback if a
        per-intent eval later shows BM25's recall is insufficient for this
        class.

This mapping is a reasoned STARTING POINT, not a proven-optimal policy --
unlike Weeks 1-4/8's aggregate numbers, there is no per-intent-class
MRR@10 backing it yet (that would need running each retrieval method
separately on queries grouped by TRUE intent, which the current MS MARCO
qrels don't label). Treat ROUTING_TABLE as configurable, and revisit it if/
when per-intent evaluation numbers say otherwise. HYBRID is left fully
implemented and dispatchable (see `retrieve` below) precisely so it can be
switched back in for a specific intent the moment real per-intent evidence
supports it -- it isn't wrong in principle, Week 8 just showed it isn't
winning on THIS corpus's aggregate today.
"""
from dataclasses import dataclass

from src.data.query_intent_dataset import INTENT_LABELS

BM25 = "bm25"
DENSE = "dense"
HYBRID = "hybrid"


@dataclass
class RouteConfig:
    method: str
    # Only meaningful when method == HYBRID -- weights for [bm25, dense] in
    # reciprocal_rank_fusion. dense_weight=10 (vs bm25_weight=1) was Week
    # 4's best-found hybrid config (aggregate MRR@10 0.8980 on Week 8's
    # re-measurement) -- kept as the default here in case a route is ever
    # switched back to HYBRID, even though no current route uses it (see
    # module docstring: Week 8 found Dense alone significantly beats this
    # config on this corpus).
    bm25_weight: float = 1.0
    dense_weight: float = 1.0


ROUTING_TABLE = {
    "factual": RouteConfig(method=DENSE),
    "navigational": RouteConfig(method=BM25),
    "exploratory": RouteConfig(method=DENSE),
    "comparison": RouteConfig(method=DENSE),
    "how-to": RouteConfig(method=BM25),
}


def route_for_intent(intent: str) -> RouteConfig:
    if intent not in ROUTING_TABLE:
        raise ValueError(f"Unknown intent {intent!r}. Known intents: {sorted(ROUTING_TABLE.keys())}")
    return ROUTING_TABLE[intent]


def validate_routing_table_covers_all_labels() -> bool:
    """Sanity check: every label the classifier can predict (INTENT_LABELS)
    must have a route. Called by tests, but also safe to call at import
    time in a script that wires the router up to a live classifier.
    """
    missing = set(INTENT_LABELS) - set(ROUTING_TABLE.keys())
    if missing:
        raise ValueError(f"ROUTING_TABLE is missing routes for: {sorted(missing)}")
    return True


def retrieve(query: str, intent: str, bm25_index, dense_index, top_k: int = 10) -> list:
    """Dispatches a single query to the retrieval method its intent routes
    to. bm25_index / dense_index are BM25Index / DenseIndex instances
    (Week 1/2) -- only the one actually needed for this intent gets called,
    except for HYBRID, which needs both and fuses them with hybrid_rrf
    (Week 4), using this route's configured weights.
    """
    from src.retrieval.hybrid_rrf import reciprocal_rank_fusion

    config = route_for_intent(intent)

    if config.method == BM25:
        return bm25_index.search(query, top_k=top_k)
    if config.method == DENSE:
        return dense_index.search(query, top_k=top_k)
    if config.method == HYBRID:
        bm25_results = bm25_index.search(query, top_k=100)
        dense_results = dense_index.search(query, top_k=100)
        return reciprocal_rank_fusion(
            [bm25_results, dense_results],
            top_k=top_k,
            weights=[config.bm25_weight, config.dense_weight],
        )
    raise ValueError(f"Unhandled method {config.method!r}")  # pragma: no cover