"""
 cross-encoder re-ranking.

A cross-encoder scores a (query, passage) PAIR directly through one
transformer forward pass, letting the model attend across both texts at
once -- this is why it's more precise than BM25/dense retrieval (which
score query and passage independently, encoded separately) but also far
slower: there's no way to pre-compute a passage's representation ahead of
time the way DenseIndex does, since the score depends on the specific
query it's paired with. That's why re-ranking always operates on a small
candidate SET (BM25 or dense top-k), not the whole corpus.

Same interface shape as BM25Index/DenseIndex's SearchResult, but the entry
point is `rerank`, not `search` -- a cross-encoder has nothing to search
over exhaustively; it only ever re-scores candidates someone else already
retrieved.

`predict_fn` is injectable for the same reason DenseIndex's `encoder` is:
this sandbox cannot download cross-encoder/ms-marco-MiniLM-L-6-v2 from
huggingface.co, so the plumbing (pairing, batching, re-sorting, top-k) is
tested here with an offline stand-in scorer, and the real model only runs
on your machine.
"""
from dataclasses import dataclass


@dataclass
class SearchResult:
    pid: str
    score: float


def _default_predict_factory(model_name: str, batch_size: int):
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)

    def predict(pairs):
        return model.predict(pairs, batch_size=batch_size, show_progress_bar=len(pairs) > 500)

    return predict


class CrossEncoderReranker:
    """Re-scores a set of first-stage candidates for one query using a
    cross-encoder, and returns them re-sorted by the new score.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 predict_fn=None, batch_size: int = 32):
        self.model_name = model_name
        self._predict = predict_fn or _default_predict_factory(model_name, batch_size)

    def rerank(self, query: str, candidates: list, passages: dict, top_k: int = 10) -> list:
        """candidates: a list of objects with a `.pid` attribute (a
        BM25Index/DenseIndex SearchResult, or anything duck-typed the same
        way) -- the first-stage retrieval results to re-score. Their
        original scores are ignored; only the pid ordering they came in
        gives us the corpus lookup.
        """
        if not candidates:
            return []

        pairs = [(query, passages[c.pid]) for c in candidates]
        scores = self._predict(pairs)

        reranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [SearchResult(pid=c.pid, score=float(s)) for c, s in reranked[:top_k]]

    def rerank_many(self, queries: dict, first_stage_results: dict, passages: dict, top_k: int = 10) -> dict:
        """queries: {qid: text}
        first_stage_results: {qid: [SearchResult, ...]} -- e.g. BM25Index
            or DenseIndex's search_many() output, already computed.
        """
        return {
            qid: self.rerank(text, first_stage_results.get(qid, []), passages, top_k=top_k)
            for qid, text in queries.items()
        }