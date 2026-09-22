"""
Baseline retriever.

Wraps rank_bm25.BM25Okapi wrapped
lifetimes.utils.summary_data_from_transaction_data: BM25's tokenization and
scoring details are easy to get subtly wrong by hand, so we lean on a tested
library and keep our own code to indexing/search plumbing plus the parts
rank_bm25 does not provide (stable pid <-> row mapping, top-k search API).
"""
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    """Lowercase, strip punctuation, split on whitespace/non-alphanumerics.

    Deliberately simple (no stemming, no stopword removal) so behaviour is
    transparent and easy to reason about; this matches how a first BM25
    baseline is normally built before adding NLP preprocessing as a later
    experiment.
    """
    return _TOKEN_RE.findall(text.lower())


@dataclass
class SearchResult:
    pid: str
    score: float


class BM25Index:
    """BM25 index over a corpus of {pid: text} passages."""

    def __init__(self, passages: dict):
        if not passages:
            raise ValueError("passages dict is empty -- nothing to index.")
        self.pids = list(passages.keys())
        self._tokenized_corpus = [tokenize(passages[pid]) for pid in self.pids]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> list:
        """Return up to top_k SearchResult objects, best score first."""
        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        ranked = sorted(zip(self.pids, scores), key=lambda pair: pair[1], reverse=True)
        return [SearchResult(pid=pid, score=float(score)) for pid, score in ranked[:top_k]]

    def search_many(self, queries: dict, top_k: int = 10) -> dict:
        """queries: {qid: text} -> {qid: [SearchResult, ...]}"""
        return {qid: self.search(text, top_k=top_k) for qid, text in queries.items()}