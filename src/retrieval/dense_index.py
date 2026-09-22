"""
dense_index.py -- Week 2: dense bi-encoder retrieval.

Same interface as BM25Index (search / search_many returning SearchResult),
so run_bm25_eval-style scripts and any future FastAPI endpoint can treat
BM25Index and DenseIndex interchangeably -- that interchangeability was the
whole point of keeping BM25Index's interface minimal in Week 1.

Encoder is injectable (the `encoder` parameter) for a deliberate reason:
this project was partly built in a sandbox that cannot reach
huggingface.co, so sentence-transformers model weights cannot be
downloaded there. Everything about DenseIndex EXCEPT "what a real
transformer thinks two pieces of text mean" -- the FAISS indexing, top-k
search, save/load round-trip, score ordering -- can still be tested with a
lightweight offline stand-in encoder (see tests/test_dense_retrieval.py's
`hashing_encoder`). The default encoder (real sentence-transformers) is
only exercised on your own machine, where huggingface.co is reachable.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import faiss


@dataclass
class SearchResult:
    pid: str
    score: float


def _default_encoder_factory(model_name: str, batch_size: int):
    """Lazily imports sentence-transformers only when actually needed, so
    importing this module doesn't require the package (or its model
    download) unless you actually build an index without your own encoder.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def encode(texts):
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 1000,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity via inner product
        )
        return embeddings.astype("float32")

    return encode


class DenseIndex:
    """FAISS-backed dense retriever over a corpus of {pid: text} passages.

    Uses exact inner-product search (IndexFlatIP) over L2-normalized
    embeddings, which is equivalent to cosine similarity ranking. Exact
    search is intentional at this project's scale (hundreds of thousands
    of passages, not the full 8.84M) -- an approximate index (IVF/HNSW)
    is a reasonable Week-3+ optimization once latency matters, not a
    Week-2 requirement.
    """

    def __init__(self, passages: dict, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 encoder=None, batch_size: int = 64):
        if not passages:
            raise ValueError("passages dict is empty -- nothing to index.")

        self.model_name = model_name
        self.pids = list(passages.keys())
        self._encode = encoder or _default_encoder_factory(model_name, batch_size)

        texts = [passages[pid] for pid in self.pids]
        embeddings = self._encode(texts)
        embeddings = self._ensure_normalized(embeddings)

        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

    @staticmethod
    def _ensure_normalized(embeddings: np.ndarray) -> np.ndarray:
        """Belt-and-suspenders L2 normalization. The default encoder already
        normalizes, but a custom `encoder` passed in (e.g. in tests) might
        not -- IndexFlatIP is only equivalent to cosine similarity if the
        vectors are unit-length, so we enforce it here rather than trusting
        every encoder to have done it.
        """
        embeddings = np.asarray(embeddings, dtype="float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid div-by-zero for an all-zero embedding
        return embeddings / norms

    def search(self, query: str, top_k: int = 10) -> list:
        query_embedding = self._ensure_normalized(self._encode([query]))
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.pids)))

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:  # FAISS pads with -1 if top_k > corpus size
                continue
            results.append(SearchResult(pid=self.pids[idx], score=float(score)))
        return results

    def search_many(self, queries: dict, top_k: int = 10) -> dict:
        return {qid: self.search(text, top_k=top_k) for qid, text in queries.items()}

    def save(self, out_dir: Path):
        """Persist the FAISS index + pid mapping so the (slow) encoding
        step doesn't need to be redone on every run. The encoder itself
        (model weights) is NOT saved here -- `load()` re-creates it from
        `model_name`, relying on sentence-transformers' own on-disk cache.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out_dir / "dense_index.faiss"))
        with open(out_dir / "dense_index_meta.json", "w") as f:
            json.dump({"pids": self.pids, "model_name": self.model_name, "dim": self.dim}, f)

    @classmethod
    def load(cls, in_dir: Path, encoder=None, batch_size: int = 64):
        in_dir = Path(in_dir)
        with open(in_dir / "dense_index_meta.json") as f:
            meta = json.load(f)

        obj = cls.__new__(cls)  # bypass __init__ (don't re-encode the corpus)
        obj.model_name = meta["model_name"]
        obj.pids = meta["pids"]
        obj.dim = meta["dim"]
        obj._encode = encoder or _default_encoder_factory(obj.model_name, batch_size)
        obj.index = faiss.read_index(str(in_dir / "dense_index.faiss"))
        return obj