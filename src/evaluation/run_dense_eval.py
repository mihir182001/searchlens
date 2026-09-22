"""
run_dense_eval.py -- Week 2 end-to-end script. RUN THIS ON YOUR OWN MACHINE.

This sandbox can install sentence-transformers and faiss-cpu (both are on
PyPI), but cannot download the actual model weights, since those come from
huggingface.co which this sandbox cannot reach. So this script -- unlike
run_bm25_eval.py -- was never executed here. It was tested indirectly: the
FAISS indexing/search plumbing it depends on (DenseIndex) has its own test
suite using an offline stand-in encoder. This script itself needs to be
run and its real output reported by you.

    python -m src.evaluation.run_dense_eval

What it does:
    1. Loads whatever corpus is currently in data/raw/ (run
       load_real_msmarco.py first if you haven't -- this script does not
       regenerate data, it evaluates whatever's already there, same as
       run_bm25_eval.py).
    2. Builds (or loads a cached) FAISS dense index using
       sentence-transformers/all-MiniLM-L6-v2 -- a small, fast, CPU-friendly
       384-dimensional bi-encoder, the same model size class named in the
       SearchLens spec.
    3. Runs MRR@10 / Recall@100, same metrics as Week 1, so the numbers are
       directly comparable to your BM25 result on the SAME corpus.
    4. Re-runs BM25 on the same corpus for a side-by-side comparison table,
       since the fair comparison is "both methods on the same passages",
       not BM25 on one run's corpus and dense on another's.

First run will download the model (~90MB, one-time, cached by
sentence-transformers afterwards) and then encode every passage in your
corpus -- for ~300k passages this is CPU-bound and will likely take
several minutes (MiniLM-L6 is one of the faster models for this). The
built index is cached to data/processed/dense_index/ so re-running this
script later reuses it instead of re-encoding everything.

Report whatever this actually prints -- do not assume it will match the
spec's aspirational ~0.314 MRR@10 figure, which (like the BM25 baseline)
was measured on the full 8.84M-passage corpus, not your capped subsample.
"""
import time
from pathlib import Path

from src.data.loader import load_passages, load_queries, load_qrels
from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex
from src.evaluation.metrics import mrr_at_k, recall_at_k

DENSE_INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "dense_index"


def main():
    passages = load_passages()
    queries = load_queries()
    qrels = load_qrels()
    print(f"Corpus: {len(passages)} passages, {len(queries)} queries, {len(qrels)} qrels")

    # ---- BM25 (re-run here for a same-corpus, same-run comparison) ----
    print("\n--- BM25 ---")
    t0 = time.time()
    bm25_index = BM25Index(passages)
    bm25_results = bm25_index.search_many(queries, top_k=100)
    bm25_mrr10 = mrr_at_k(bm25_results, qrels, k=10)
    bm25_recall100 = recall_at_k(bm25_results, qrels, k=100)
    print(f"MRR@10:      {bm25_mrr10:.4f}")
    print(f"Recall@100:  {bm25_recall100:.4f}")
    print(f"(took {time.time() - t0:.1f}s)")

    # ---- Dense bi-encoder ----
    print("\n--- Dense (sentence-transformers/all-MiniLM-L6-v2) ---")
    t0 = time.time()
    if (DENSE_INDEX_DIR / "dense_index.faiss").exists():
        print(f"Loading cached dense index from {DENSE_INDEX_DIR} ...")
        dense_index = DenseIndex.load(DENSE_INDEX_DIR)
    else:
        print("No cached index found -- encoding corpus (this is the slow part, one-time cost)...")
        dense_index = DenseIndex(passages)
        dense_index.save(DENSE_INDEX_DIR)
        print(f"Cached index to {DENSE_INDEX_DIR} for next time.")
    print(f"(index ready after {time.time() - t0:.1f}s)")

    t0 = time.time()
    dense_results = dense_index.search_many(queries, top_k=100)
    dense_mrr10 = mrr_at_k(dense_results, qrels, k=10)
    dense_recall100 = recall_at_k(dense_results, qrels, k=100)
    print(f"MRR@10:      {dense_mrr10:.4f}")
    print(f"Recall@100:  {dense_recall100:.4f}")
    print(f"(search took {time.time() - t0:.1f}s for {len(queries)} queries)")

    # ---- Side-by-side ----
    print("\n--- Summary (same corpus, same queries) ---")
    print(f"{'Method':<10} {'MRR@10':>10} {'Recall@100':>12}")
    print(f"{'BM25':<10} {bm25_mrr10:>10.4f} {bm25_recall100:>12.4f}")
    print(f"{'Dense':<10} {dense_mrr10:>10.4f} {dense_recall100:>12.4f}")
    if bm25_mrr10 > 0:
        lift = (dense_mrr10 - bm25_mrr10) / bm25_mrr10 * 100
        print(f"\nDense vs BM25 MRR@10 change: {lift:+.1f}%")


if __name__ == "__main__":
    main()