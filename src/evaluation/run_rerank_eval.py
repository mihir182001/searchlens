"""
run_rerank_eval.py -- Week 3 end-to-end script. RUN THIS ON YOUR OWN MACHINE.

    python -m src.evaluation.run_rerank_eval --rerank-depth 50 --max-queries 300

Like run_dense_eval.py, this was never executed in the sandbox -- the real
cross-encoder model comes from huggingface.co, which the sandbox cannot
reach. CrossEncoderReranker's plumbing has its own offline-tested suite
(tests/test_cross_encoder_rerank.py); this script's real numbers need to
come from you.

Why re-ranking is much slower than Week 1/2, and how the flags help:
    BM25 and dense retrieval each score a query against EVERY passage in
    one batched/vectorized pass. A cross-encoder scores one (query,
    passage) PAIR per forward pass, with no way to precompute anything
    ahead of time (the two texts are encoded together, not separately) --
    so re-ranking cost is `num_queries * rerank_depth` forward passes,
    each one roughly as expensive as encoding a full passage was in
    Week 2. Re-ranking all 1,000 queries at depth 100 would be ~100,000
    forward passes -- on a CPU-only laptop that's a genuinely long run.

    --rerank-depth (default 50): how many first-stage BM25 candidates get
        re-scored per query. Recall@100 from Week 1 was already >0.94 on
        your corpus, meaning the true answer is almost always inside the
        top 50-100 BM25 results -- re-ranking only needs to fix the ORDER
        within that set, not search further, so a smaller depth is a
        legitimate speed/thoroughness trade-off, not a shortcut that
        changes what's being measured. State whatever depth you used.
    --max-queries (default: all queries currently in data/raw/): lets you
        sanity-check on a small slice (e.g. 100-200) before committing to
        a run across all 1,000 -- given how long Week 2's dense encoding
        took on your machine, it's worth timing a small run first and
        extrapolating before starting the full one.

What it does:
    1. Loads the corpus already in data/raw/ (same one Week 1/2 used).
    2. Runs BM25 to get first-stage top-`rerank-depth` candidates per query
       (fast -- this is the same BM25Index from Week 1).
    3. Re-ranks those candidates with cross-encoder/ms-marco-MiniLM-L-6-v2.
    4. Reports MRR@10/Recall@100 for: BM25 alone, and BM25+re-ranked --
       the fair comparison, since re-ranking can only ever help within
       what BM25 already retrieved (a re-ranked Recall@100 can never
       exceed the first-stage Recall@100 it started from).

Report the real numbers this prints, at whatever --rerank-depth and
--max-queries you actually ran -- do not assume it matches the spec's
aspirational ~0.372 MRR@10 (also a full-8.84M-corpus figure, same caveat
as Weeks 1 and 2).
"""
import argparse
import time

from src.data.loader import load_passages, load_queries, load_qrels
from src.retrieval.bm25_index import BM25Index
from src.retrieval.cross_encoder_rerank import CrossEncoderReranker
from src.evaluation.metrics import mrr_at_k, recall_at_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank-depth", type=int, default=50,
                         help="How many first-stage BM25 candidates to re-rank per query.")
    parser.add_argument("--max-queries", type=int, default=-1,
                         help="Cap on queries evaluated, for a quick timing sanity-check before a full run.")
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

    print(f"Corpus: {len(passages)} passages, {len(queries)} queries evaluated, "
          f"rerank_depth={args.rerank_depth}")

    # ---- Stage 1: BM25 first-stage retrieval ----
    print("\n--- Stage 1: BM25 (first-stage retrieval) ---")
    t0 = time.time()
    bm25_index = BM25Index(passages)
    bm25_results_full = bm25_index.search_many(queries, top_k=100)  # for a fair Recall@100 baseline
    bm25_results_for_rerank = {qid: results[: args.rerank_depth] for qid, results in bm25_results_full.items()}

    bm25_mrr10 = mrr_at_k(bm25_results_full, qrels, k=10)
    bm25_recall100 = recall_at_k(bm25_results_full, qrels, k=100)
    print(f"MRR@10:      {bm25_mrr10:.4f}")
    print(f"Recall@100:  {bm25_recall100:.4f}")
    print(f"(took {time.time() - t0:.1f}s)")

    # ---- Stage 2: cross-encoder re-ranking ----
    print(f"\n--- Stage 2: Cross-encoder re-rank (cross-encoder/ms-marco-MiniLM-L-6-v2, depth={args.rerank_depth}) ---")
    print(f"Scoring up to {len(queries) * args.rerank_depth:,} (query, passage) pairs -- this is the slow part.")
    t0 = time.time()

    reranker = CrossEncoderReranker()
    reranked_results = reranker.rerank_many(queries, bm25_results_for_rerank, passages, top_k=10)

    # Recall@100 after reranking is capped by what BM25 handed it (rerank_depth,
    # not 100) -- report Recall@rerank_depth instead so the number means what
    # it says, and note MRR@10 is directly comparable to Stage 1's since both
    # cap at k=10.
    reranked_mrr10 = mrr_at_k(reranked_results, qrels, k=10)
    reranked_recall_at_depth = recall_at_k(reranked_results, qrels, k=args.rerank_depth)
    print(f"MRR@10:              {reranked_mrr10:.4f}")
    print(f"Recall@{args.rerank_depth} (post-rerank): {reranked_recall_at_depth:.4f}")
    print(f"(took {time.time() - t0:.1f}s for {len(queries)} queries x depth {args.rerank_depth})")

    # ---- Summary ----
    print("\n--- Summary ---")
    print(f"{'Stage':<24} {'MRR@10':>10}")
    print(f"{'BM25 (first-stage)':<24} {bm25_mrr10:>10.4f}")
    print(f"{'+ Cross-encoder rerank':<24} {reranked_mrr10:>10.4f}")
    if bm25_mrr10 > 0:
        lift = (reranked_mrr10 - bm25_mrr10) / bm25_mrr10 * 100
        print(f"\nRe-ranking vs BM25-alone MRR@10 change: {lift:+.1f}%")


if __name__ == "__main__":
    main()