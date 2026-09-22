"""
Builds the (query, candidate passages,
binary relevance label) dataset that the LLM judge will score.

No LLM call happens here -- this only needs an existing retriever's
results (e.g. BM25 or Hybrid RRF from Weeks 1-4), so it's fully
offline-testable. The actual LLM scoring happens in llm_judge.py /
run_llm_judge_eval.py (the latter run on your own machine).

Candidates are drawn from a REAL retriever's top-k results, not sampled
randomly from the corpus -- this matters because it makes the judge's task
realistic (a mix of genuinely relevant and plausible-but-wrong passages,
the way an actual search results page looks), rather than an easy task of
distinguishing relevant passages from random, obviously-unrelated ones.
"""
import random
from dataclasses import dataclass


@dataclass
class JudgeQuery:
    qid: str
    query: str
    pids: list  # candidate passage ids, best-first
    passages: list  # candidate passage texts, same order as pids
    labels: list  # binary relevance labels (1 if pid in qrels[qid], else 0), same order


def build_judge_queries(
    queries: dict, passages: dict, qrels: dict, retrieval_results: dict, depth: int = 5
) -> list:
    """queries: {qid: query text}
    passages: {pid: passage text}
    qrels: {qid: set(relevant pids)}
    retrieval_results: {qid: [SearchResult, ...]} -- e.g. BM25Index or
        fuse_many's output, used to pick realistic candidates.
    depth: how many top candidates per query to include.

    Returns a list of JudgeQuery, one per qid present in both `queries`
    and `retrieval_results` with at least one candidate, skipping the rest.
    """
    judge_queries = []
    for qid, query_text in queries.items():
        results = retrieval_results.get(qid, [])[:depth]
        if not results:
            continue
        relevant_pids = qrels.get(qid, set())
        pids = [r.pid for r in results]
        passage_texts = [passages[pid] for pid in pids]
        labels = [1 if pid in relevant_pids else 0 for pid in pids]
        judge_queries.append(
            JudgeQuery(qid=qid, query=query_text, pids=pids, passages=passage_texts, labels=labels)
        )
    return judge_queries


def sample_judge_queries(judge_queries: list, max_queries: int, seed: int = 42) -> list:
    """Randomly samples up to max_queries JudgeQuery objects, deterministic
    given the same seed -- used to pick e.g. 500 of the full query set
    without biasing toward the front of the list (which could correlate
    with query difficulty/order in the source data).
    """
    if max_queries >= len(judge_queries):
        return list(judge_queries)
    rng = random.Random(seed)
    return rng.sample(judge_queries, max_queries)