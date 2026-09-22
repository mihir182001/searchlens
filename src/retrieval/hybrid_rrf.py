"""
Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Combines two (or more) independently-ranked result lists for the same
query into one fused ranking, using each candidate's RANK POSITION in
each list rather than raw scores -- this sidesteps the fact that BM25
scores and dense cosine similarities live on completely different,
non-comparable scales, which is exactly why RRF (not a weighted sum of
raw scores) is the standard way to combine heterogeneous retrievers.

RRF formula (Cormack et al., 2009), matching the SearchLens spec's own
formula:
    score(d) = sum over each ranked list L containing d of  1 / (k + rank_L(d))

where rank_L(d) is d's 1-indexed position in list L (a document not in a
list contributes 0 from that list), and k (default 60, the standard value
from the original paper, also the value the spec cites) discounts how
much any single high rank dominates the fused score.

Needs no model of any kind -- it operates purely on rank positions from
lists BM25Index/DenseIndex already produced -- so unlike Week 2/3 this
runs and is fully tested in the sandbox, no real-model caveat needed.

WEIGHTED RRF (added after the first real run on this project's data):
    Plain, equal-weight RRF assumes both retrievers are roughly comparable
    in quality. When one is substantially stronger than the other --
    which is exactly what happened here (Dense's MRR@10 of 0.9130 was far
    ahead of BM25's 0.7238 on this corpus) -- equal weighting can actually
    HURT the fused MRR@10 relative to just using the stronger retriever
    alone: a passage the weaker retriever ranks confidently (but wrongly)
    can out-vote a passage the stronger retriever had correctly ranked
    #1. This is a documented, known failure mode of naive RRF, not a bug.
    `weights` lets each input list's contribution be scaled up or down to
    reflect its known relative quality, so a much stronger retriever
    isn't diluted by a much weaker one.
"""
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class SearchResult:
    pid: str
    score: float


def reciprocal_rank_fusion(result_lists: list, k: int = 60, top_k: int = 10, weights: list = None) -> list:
    """result_lists: a list of ranked result lists (each a list of objects
    with a `.pid` attribute, best-first) for the SAME query, e.g.
    [bm25_results_for_q, dense_results_for_q]. A pid that appears near the
    top of more than one list gets the highest fused score -- that's the
    whole point: it rewards agreement between different retrieval signals.

    weights: optional list of per-list multipliers, same length as
        result_lists (e.g. [1.0, 2.0] to give the second list -- say,
        Dense -- twice the influence of the first). Defaults to equal
        weight (1.0) for every list, i.e. standard RRF.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError(f"weights has {len(weights)} entries but result_lists has {len(result_lists)}.")

    fused_scores = defaultdict(float)
    for weight, result_list in zip(weights, result_lists):
        for rank, result in enumerate(result_list, start=1):
            fused_scores[result.pid] += weight / (k + rank)

    ranked = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    return [SearchResult(pid=pid, score=score) for pid, score in ranked[:top_k]]


def fuse_many(list_of_result_dicts: list, k: int = 60, top_k: int = 10, weights: list = None) -> dict:
    """list_of_result_dicts: a list of {qid: [SearchResult, ...]} dicts, one
    per retrieval method (e.g. [bm25_results, dense_results]). Query sets
    don't need to match exactly between dicts -- a qid missing from one
    method's results just contributes nothing from that method, rather
    than failing.

    weights: optional list of per-method multipliers, same order and
        length as list_of_result_dicts. See reciprocal_rank_fusion.
    """
    all_qids = set()
    for results_dict in list_of_result_dicts:
        all_qids |= set(results_dict.keys())

    fused = {}
    for qid in all_qids:
        lists_for_this_query = [results_dict.get(qid, []) for results_dict in list_of_result_dicts]
        fused[qid] = reciprocal_rank_fusion(lists_for_this_query, k=k, top_k=top_k, weights=weights)
    return fused