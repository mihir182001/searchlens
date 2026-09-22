"""
IR evaluation metrics used across every retrieval method in
SearchLens (BM25 now; dense, cross-encoder, and hybrid later reuse this
unchanged, the same way PlayerLens's evaluation/metrics.py pattern would
keep one place responsible for scoring).
"""


def reciprocal_rank(ranked_pids: list, relevant_pids: set) -> float:
    """1/rank of the first relevant pid in ranked_pids, else 0.0."""
    for rank, pid in enumerate(ranked_pids, start=1):
        if pid in relevant_pids:
            return 1.0 / rank
    return 0.0


def per_query_reciprocal_rank(results: dict, qrels: dict, k: int = 10) -> dict:
    """Per-query reciprocal rank @ k, keyed by qid -- the individual scores
    that mrr_at_k averages together. Added in Week 8 for A/B significance
    testing (src/evaluation/ab_test.py): a paired significance test needs
    each query's own score under two configurations, not just the two
    configurations' means.

    results: {qid: [SearchResult, ...]} (already sorted best-first, as
        returned by BM25Index.search_many)
    qrels:   {qid: set(relevant_pid)}
    k:       only the top-k of each ranked list counts

    Queries present in qrels but missing from results are treated as a
    reciprocal rank of 0 (never found), which is the standard TREC
    convention and prevents silently dropping hard queries from the
    average.
    """
    if not qrels:
        raise ValueError("qrels is empty -- nothing to evaluate against.")

    scores = {}
    for qid, relevant_pids in qrels.items():
        ranked_pids = [r.pid for r in results.get(qid, [])][:k]
        scores[qid] = reciprocal_rank(ranked_pids, relevant_pids)
    return scores


def mrr_at_k(results: dict, qrels: dict, k: int = 10) -> float:
    """Mean Reciprocal Rank @ k -- the mean of per_query_reciprocal_rank's
    per-query scores. See per_query_reciprocal_rank for the qid-by-qid
    breakdown this averages.
    """
    scores = per_query_reciprocal_rank(results, qrels, k=k)
    return sum(scores.values()) / len(scores)


def recall_at_k(results: dict, qrels: dict, k: int = 100) -> float:
    """Fraction of queries for which at least one relevant pid appears in
    the top-k retrieved results. (With MS MARCO's ~1 relevant pid/query,
    this is equivalent to "hit rate @ k".)
    """
    if not qrels:
        raise ValueError("qrels is empty -- nothing to evaluate against.")

    hits = 0
    for qid, relevant_pids in qrels.items():
        ranked_pids = [r.pid for r in results.get(qid, [])][:k]
        if relevant_pids & set(ranked_pids):
            hits += 1

    return hits / len(qrels)


def pearson_correlation(xs: list, ys: list) -> float:
    """Pearson correlation coefficient between two equal-length numeric
    sequences. Week 6 uses this to correlate the LLM judge's 1-5 relevance
    scores against MS MARCO's binary qrels labels (0/1) -- a valid, if
    unusual, use of Pearson r (correlating a continuous-ish variable
    against a binary one is sometimes called a point-biserial correlation,
    which is mathematically identical to Pearson r computed this way).

    No numpy/scipy dependency -- implemented directly so this stays testable
    with nothing but the standard library.
    """
    if len(xs) != len(ys):
        raise ValueError(f"xs and ys must be the same length, got {len(xs)} and {len(ys)}.")
    n = len(xs)
    if n < 2:
        raise ValueError("Need at least 2 pairs to compute a correlation.")

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    denominator = (var_x * var_y) ** 0.5
    if denominator == 0:
        raise ValueError(
            "Cannot compute correlation: one of xs/ys has zero variance "
            "(e.g. every score is identical, or every label is the same class)."
        )
    return cov / denominator