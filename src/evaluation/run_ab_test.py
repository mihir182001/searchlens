"""
 RUN THIS ON YOUR OWN MACHINE.

    python -m src.evaluation.run_ab_test
    python -m src.evaluation.run_ab_test --config-a bm25 --config-b hybrid
    python -m src.evaluation.run_ab_test --dense-weight 5 --n-resamples 20000

Answers the question hybrid_rrf.py's own docstring left open from Week 4:
the best hybrid config found there (dense_weight=10) scored MRR@10 0.9022,
BELOW Dense alone's 0.9130. Is Hybrid actually worse on this corpus, or is
a ~1-point gap just noise from ordinary query-to-query variance? Two
aggregate numbers can never answer that by themselves -- this script runs
a PAIRED significance test (src/evaluation/ab_test.py's paired-bootstrap
method) across the exact same queries to find out.

Default comparison: dense (config A, the "control") vs hybrid (config B,
the "treatment") -- exactly the open question above. Pass --config-a /
--config-b to compare any other pair of {bm25, dense, hybrid}.

What it does:
    1. Builds/loads BM25 and (cached) Dense results, top-10, over the SAME
       query set for both configs -- required for a paired test; comparing
       different query sets would confound "which config is better" with
       "which query set happened to be easier" (see ab_test.py's
       docstring).
    2. If either config is "hybrid", fuses BM25+Dense via RRF using
       --bm25-weight/--dense-weight (defaults 1.0/10.0 -- Week 4's own
       best found config).
    3. Computes each query's reciprocal rank under both configs
       (per_query_reciprocal_rank, metrics.py).
    4. Runs the paired bootstrap test and reports each config's mean
       MRR@10, the difference, its confidence interval, the p-value, and
       whether it's statistically significant at --alpha (default 0.05).

Report whatever this actually finds, INCLUDING "not significant" -- that
is itself a real, useful finding: it means a headline aggregate number
(like Week 4's) shouldn't be treated as a real winner without this caveat,
and the honest conclusion is "we can't tell these two configs apart on
this corpus," not "the higher number wins."

Needs a cached dense index at data/processed/dense_index/ (from Week 2/4)
if either config is "dense" or "hybrid" -- if it's missing this will build
one, which is slow the first time (a real embedding model), same as
run_dense_eval.py / run_hybrid_eval.py.
"""
import argparse
import time
from pathlib import Path

from src.data.loader import load_passages, load_queries, load_qrels
from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex
from src.retrieval.hybrid_rrf import fuse_many
from src.evaluation.metrics import per_query_reciprocal_rank
from src.evaluation.ab_test import paired_bootstrap_test, is_significant

DENSE_INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "dense_index"
CONFIG_CHOICES = ["bm25", "dense", "hybrid"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-a", choices=CONFIG_CHOICES, default="dense")
    parser.add_argument("--config-b", choices=CONFIG_CHOICES, default="hybrid")
    parser.add_argument("--bm25-weight", type=float, default=1.0,
                         help="RRF weight for BM25, only used if 'hybrid' is one of the configs.")
    parser.add_argument("--dense-weight", type=float, default=10.0,
                         help="RRF weight for Dense, only used if 'hybrid' is one of the configs. "
                              "Default 10.0 matches Week 4's own best found config.")
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05,
                         help="Significance threshold for the 'is this a real difference' verdict.")
    parser.add_argument("--max-queries", type=int, default=-1,
                         help="Cap on queries evaluated, mainly for a quick timing sanity-check.")
    args = parser.parse_args()

    if args.config_a == args.config_b:
        raise SystemExit("--config-a and --config-b must be different -- nothing to compare.")

    passages = load_passages()
    all_queries = load_queries()
    qrels = load_qrels()

    if args.max_queries != -1 and args.max_queries < len(all_queries):
        kept_qids = sorted(all_queries.keys())[: args.max_queries]
        queries = {qid: all_queries[qid] for qid in kept_qids}
        qrels = {qid: qrels[qid] for qid in kept_qids if qid in qrels}
    else:
        queries = all_queries

    needed = {args.config_a, args.config_b}
    print(f"Corpus: {len(passages)} passages, {len(queries)} queries, {len(qrels)} qrels")
    print(f"Comparing: A={args.config_a}  vs  B={args.config_b}")

    results_by_config = {}

    # BM25 is cheap and needed either directly or as hybrid's first input.
    bm25_results = None
    if "bm25" in needed or "hybrid" in needed:
        print("\n--- BM25 ---")
        t0 = time.time()
        bm25_index = BM25Index(passages)
        bm25_results = bm25_index.search_many(queries, top_k=10)
        print(f"(done in {time.time() - t0:.1f}s)")
        if "bm25" in needed:
            results_by_config["bm25"] = bm25_results

    dense_results = None
    if "dense" in needed or "hybrid" in needed:
        print("\n--- Dense ---")
        t0 = time.time()
        if (DENSE_INDEX_DIR / "dense_index.faiss").exists():
            print(f"Loading cached dense index from {DENSE_INDEX_DIR} ...")
            dense_index = DenseIndex.load(DENSE_INDEX_DIR)
        else:
            print("No cached dense index found -- encoding corpus (slow, one-time)...")
            dense_index = DenseIndex(passages)
            dense_index.save(DENSE_INDEX_DIR)
        dense_results = dense_index.search_many(queries, top_k=10)
        print(f"(done in {time.time() - t0:.1f}s)")
        if "dense" in needed:
            results_by_config["dense"] = dense_results

    if "hybrid" in needed:
        weights = [args.bm25_weight, args.dense_weight]
        print(f"\n--- Hybrid (RRF: BM25 + Dense, weights={weights}) ---")
        results_by_config["hybrid"] = fuse_many([bm25_results, dense_results], top_k=10, weights=weights)

    # ---- Per-query scores + paired significance test ----
    scores_a = per_query_reciprocal_rank(results_by_config[args.config_a], qrels, k=10)
    scores_b = per_query_reciprocal_rank(results_by_config[args.config_b], qrels, k=10)

    print(f"\nRunning paired bootstrap test ({args.n_resamples} resamples, seed={args.seed})...")
    t0 = time.time()
    result = paired_bootstrap_test(scores_a, scores_b, n_resamples=args.n_resamples, seed=args.seed)
    print(f"(done in {time.time() - t0:.1f}s)")

    print("\n--- Results ---")
    print(f"Config A ({args.config_a}):  MRR@10 = {result.mean_a:.4f}")
    print(f"Config B ({args.config_b}):  MRR@10 = {result.mean_b:.4f}")
    print(f"Difference (B - A):     {result.mean_diff:+.4f}")
    print(f"{int(result.confidence * 100)}% CI on the difference: "
          f"[{result.ci_low:+.4f}, {result.ci_high:+.4f}]")
    print(f"p-value: {result.p_value:.4f}")

    significant = is_significant(result, alpha=args.alpha)
    print(f"\nStatistically significant at alpha={args.alpha}: {significant}")
    if significant:
        direction = f"{args.config_b} beats {args.config_a}" if result.mean_diff > 0 else f"{args.config_a} beats {args.config_b}"
        print(f"-> {direction}, and this is unlikely to be query-sampling noise.")
    else:
        print("-> The observed difference is NOT distinguishable from noise at this "
              "significance level -- do not treat the higher aggregate MRR@10 as "
              "a proven winner without this caveat.")


if __name__ == "__main__":
    main()