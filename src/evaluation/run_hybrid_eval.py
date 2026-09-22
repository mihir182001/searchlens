"""
RUN THIS ON YOUR OWN MACHINE.

    python -m src.evaluation.run_hybrid_eval

Unlike Week 2/3's scripts, the RRF fusion step itself needs no model and
is fast (pure Python over rank positions) -- the only slow part here is
building the dense index if you don't already have one cached from Week 2
at data/processed/dense_index/ (this script reuses that cache the same
way run_dense_eval.py does).

What it does:
    1. Runs BM25 and (cached) Dense retrieval, top-100 each.
    2. Fuses them with Reciprocal Rank Fusion (hybrid_rrf.py) -- the
       SearchLens spec's "Approach 4".
    3. Reports MRR@10/Recall@100 for BM25, Dense, and Hybrid side by side.
    4. OPTIONALLY (only with --rerank-dense-depth N) re-ranks DENSE
       retrieval's top-N candidates with the cross-encoder instead of
       BM25's -- testing the finding from the Week 3 run: since dense
       retrieval had much higher first-stage recall than BM25 on this
       corpus, reranking ITS candidates should beat every other result so
       far. This repeats Week 3's slow cross-encoder cost
       (num_queries * N forward passes) so it's opt-in, not automatic --
       time a small --max-queries slice first, same as Week 3.

Report whatever this actually prints. Do not assume hybrid automatically
beats dense alone -- on THIS corpus, dense's first-stage recall was
already close to ceiling (0.999 @ 100), so RRF fusing it with a weaker
BM25 list could help, hurt, or be a wash; that's an empirical question,
not a given, and the honest thing is to report what actually happened.
"""
import argparse
import time
from pathlib import Path

from src.data.loader import load_passages, load_queries, load_qrels
from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex
from src.retrieval.hybrid_rrf import fuse_many
from src.retrieval.cross_encoder_rerank import CrossEncoderReranker
from src.evaluation.metrics import mrr_at_k, recall_at_k

DENSE_INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "dense_index"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank-dense-depth", type=int, default=0,
                         help="If > 0, also re-rank DENSE retrieval's top-N candidates with the "
                              "cross-encoder and report that result too. Slow (same cost profile "
                              "as Week 3) -- 0 (default) skips this.")
    parser.add_argument("--max-queries", type=int, default=-1,
                         help="Cap on queries evaluated, mainly useful for timing --rerank-dense-depth "
                              "before committing to a full run.")
    parser.add_argument("--bm25-weight", type=float, default=1.0,
                         help="RRF weight for the BM25 list. Default 1.0 (standard equal-weight RRF).")
    parser.add_argument("--dense-weight", type=float, default=1.0,
                         help="RRF weight for the Dense list. Raise this above --bm25-weight if Dense is "
                              "the stronger retriever on your corpus (check your BM25 vs Dense MRR@10 "
                              "first) -- equal weighting can dilute a much stronger retriever's confident "
                              "top picks with a much weaker one's, hurting fused MRR@10 even while it "
                              "improves Recall (see module docstring in hybrid_rrf.py).")
    args = parser.parse_args()

    passages = load_passages()
    all_queries = load_queries()
    qrels = load_qrels()

    if args.max_queries != -1 and args.max_queries < len(all_queries):
        kept_qids = sorted(all_queries.keys())[: args.max_queries]
        queries = {qid: all_queries[qid] for qid in kept_qids}
        qrels = {qid: qrels[qid] for qid in kept_qids if qid in qrels}
    else:
        queries = all_queries

    print(f"Corpus: {len(passages)} passages, {len(queries)} queries, {len(qrels)} qrels")

    # ---- BM25 ----
    print("\n--- BM25 ---")
    t0 = time.time()
    bm25_index = BM25Index(passages)
    bm25_results = bm25_index.search_many(queries, top_k=100)
    bm25_mrr10 = mrr_at_k(bm25_results, qrels, k=10)
    bm25_recall100 = recall_at_k(bm25_results, qrels, k=100)
    print(f"MRR@10: {bm25_mrr10:.4f}  Recall@100: {bm25_recall100:.4f}  ({time.time() - t0:.1f}s)")

    # ---- Dense (reuse cached index from Week 2 if present) ----
    print("\n--- Dense ---")
    t0 = time.time()
    if (DENSE_INDEX_DIR / "dense_index.faiss").exists():
        print(f"Loading cached dense index from {DENSE_INDEX_DIR} ...")
        dense_index = DenseIndex.load(DENSE_INDEX_DIR)
    else:
        print("No cached dense index found -- encoding corpus (slow, one-time)...")
        dense_index = DenseIndex(passages)
        dense_index.save(DENSE_INDEX_DIR)
    dense_results = dense_index.search_many(queries, top_k=100)
    dense_mrr10 = mrr_at_k(dense_results, qrels, k=10)
    dense_recall100 = recall_at_k(dense_results, qrels, k=100)
    print(f"MRR@10: {dense_mrr10:.4f}  Recall@100: {dense_recall100:.4f}  ({time.time() - t0:.1f}s)")

    # ---- Hybrid RRF ----
    weights = [args.bm25_weight, args.dense_weight]
    print(f"\n--- Hybrid (RRF: BM25 + Dense, weights={weights}) ---")
    t0 = time.time()
    hybrid_results = fuse_many([bm25_results, dense_results], k=60, top_k=100, weights=weights)
    hybrid_mrr10 = mrr_at_k(hybrid_results, qrels, k=10)
    hybrid_recall100 = recall_at_k(hybrid_results, qrels, k=100)
    print(f"MRR@10: {hybrid_mrr10:.4f}  Recall@100: {hybrid_recall100:.4f}  ({time.time() - t0:.1f}s)")

    # ---- Summary so far ----
    print("\n--- Summary ---")
    print(f"{'Method':<10} {'MRR@10':>10} {'Recall@100':>12}")
    print(f"{'BM25':<10} {bm25_mrr10:>10.4f} {bm25_recall100:>12.4f}")
    print(f"{'Dense':<10} {dense_mrr10:>10.4f} {dense_recall100:>12.4f}")
    print(f"{'Hybrid':<10} {hybrid_mrr10:>10.4f} {hybrid_recall100:>12.4f}")

    # ---- Optional: rerank DENSE's candidates instead of BM25's ----
    if args.rerank_dense_depth > 0:
        depth = args.rerank_dense_depth
        print(f"\n--- Cross-encoder rerank on DENSE's top-{depth} (testing the Week 3 insight) ---")
        print(f"Scoring up to {len(queries) * depth:,} (query, passage) pairs -- this is the slow part.")
        t0 = time.time()
        dense_for_rerank = {qid: results[:depth] for qid, results in dense_results.items()}
        reranker = CrossEncoderReranker()
        dense_reranked = reranker.rerank_many(queries, dense_for_rerank, passages, top_k=10)
        dense_reranked_mrr10 = mrr_at_k(dense_reranked, qrels, k=10)
        dense_reranked_recall_at_depth = recall_at_k(dense_reranked, qrels, k=depth)
        print(f"MRR@10: {dense_reranked_mrr10:.4f}  Recall@{depth}: {dense_reranked_recall_at_depth:.4f}  "
              f"({time.time() - t0:.1f}s)")

        print("\n--- Final summary (all methods) ---")
        print(f"{'Method':<26} {'MRR@10':>10}")
        print(f"{'BM25':<26} {bm25_mrr10:>10.4f}")
        print(f"{'Dense':<26} {dense_mrr10:>10.4f}")
        print(f"{'Hybrid (RRF)':<26} {hybrid_mrr10:>10.4f}")
        print(f"{'Dense + Cross-encoder':<26} {dense_reranked_mrr10:>10.4f}")


if __name__ == "__main__":
    main()