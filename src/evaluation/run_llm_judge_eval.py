"""
RUN THIS ON YOUR OWN
MACHINE with your own Anthropic API key.

    (PowerShell) $env:ANTHROPIC_API_KEY="sk-ant-..."
    pip install anthropic
    python -m src.evaluation.run_llm_judge_eval --max-queries 20   (time it first)
    python -m src.evaluation.run_llm_judge_eval --max-queries 500 --depth 5

What it does:
    1. Loads the real corpus/queries/qrels from Weeks 1-4.
    2. Retrieves top-`depth` candidates per query with the best method
       found so far (Hybrid RRF, dense_weight=10 per Week 4) if a cached
       dense index exists at data/processed/dense_index/, else falls back
       to BM25-only candidates.
    3. Samples up to `--max-queries` queries (default 500, matching the
       SearchLens spec).
    4. For each sampled query, sends ONE Claude API call with the query +
       its `depth` candidate passages, asking Claude to rate each
       passage's relevance 1-5 (src/evaluation/llm_judge.py) -- one call
       per QUERY, not per passage, to keep the call count reasonable.
    5. Computes Pearson correlation between Claude's 1-5 scores and the
       real MS MARCO qrels' binary relevance labels (1 = relevant,
       0 = not), across every (query, passage) pair scored.
    6. Reports the correlation, plus the mean LLM score for actually-
       relevant vs actually-non-relevant passages (a more interpretable
       sanity check than the correlation number alone), and saves every
       raw judgment to data/processed/llm_judge_results.csv for audit.

Cost/time note: this makes ~`--max-queries` API calls (one per query, not
per passage), each scoring `depth` passages together -- 500 queries at
depth 5 is 500 calls, each a few hundred to a couple thousand tokens.
Time a small --max-queries slice first (e.g. 20-50) before committing to
the full run, same pattern as every other week's expensive script.

Report whatever correlation this actually prints -- do not assume it
matches the spec's target of 0.81. A 1-5 LLM judgment correlated against
a BINARY qrels label (not a graded human relevance judgment) is a
different, harder comparison than whatever MS MARCO's own creators used
to arrive at any published number, so a lower correlation here does not
by itself mean the judge is bad -- explain the discrepancy rather than
hiding it, consistent with every other week's numbers in this project.
"""
import argparse
import csv
import time
from pathlib import Path

from src.data.loader import load_passages, load_queries, load_qrels
from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex
from src.retrieval.hybrid_rrf import fuse_many
from src.evaluation.build_judge_dataset import build_judge_queries, sample_judge_queries
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.metrics import pearson_correlation

DENSE_INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "dense_index"
RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "llm_judge_results.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--depth", type=int, default=5, help="Candidates per query shown to the judge.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929")
    args = parser.parse_args()

    passages = load_passages()
    queries = load_queries()
    qrels_raw = load_qrels()
    qrels = {qid: set(pids) for qid, pids in qrels_raw.items()}

    print(f"Corpus: {len(passages)} passages, {len(queries)} queries")

    # ---- First-stage retrieval: Hybrid RRF (Week 4's best config) if a
    # cached dense index exists, else fall back to BM25 alone. ----
    bm25_index = BM25Index(passages)
    bm25_results = bm25_index.search_many(queries, top_k=args.depth)

    if (DENSE_INDEX_DIR / "dense_index.faiss").exists():
        print("\nCached dense index found -- using Hybrid RRF (dense_weight=10, Week 4's best config).")
        dense_index = DenseIndex.load(DENSE_INDEX_DIR)
        dense_results = dense_index.search_many(queries, top_k=args.depth)
        retrieval_results = fuse_many(
            [bm25_results, dense_results], top_k=args.depth, weights=[1.0, 10.0]
        )
    else:
        print("\nNo cached dense index found -- using BM25-only candidates.")
        retrieval_results = bm25_results

    # ---- Build + sample judge queries ----
    judge_queries = build_judge_queries(queries, passages, qrels, retrieval_results, depth=args.depth)
    judge_queries = sample_judge_queries(judge_queries, args.max_queries, seed=args.seed)
    print(
        f"Judging {len(judge_queries)} queries x up to {args.depth} candidates each "
        f"= up to {len(judge_queries) * args.depth} (query, passage) pairs."
    )

    # ---- Score with Claude-as-judge ----
    judge = LLMJudge(model=args.model)
    all_scores, all_labels, rows = [], [], []
    t0 = time.time()
    for i, jq in enumerate(judge_queries, start=1):
        scores = judge.score_query(jq.query, jq.passages)
        for pid, passage, label, score in zip(jq.pids, jq.passages, jq.labels, scores):
            all_scores.append(score)
            all_labels.append(label)
            rows.append(
                {"qid": jq.qid, "pid": pid, "query": jq.query, "llm_score": score, "qrel_label": label}
            )
        if i % 50 == 0 or i == len(judge_queries):
            print(f"  {i}/{len(judge_queries)} queries judged ({time.time() - t0:.1f}s elapsed)")

    # ---- Save raw judgments for audit ----
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "pid", "query", "llm_score", "qrel_label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} raw judgments to {RESULTS_PATH}")

    # ---- Correlation + interpretable summary ----
    correlation = pearson_correlation(all_scores, all_labels)
    relevant_scores = [s for s, l in zip(all_scores, all_labels) if l == 1]
    non_relevant_scores = [s for s, l in zip(all_scores, all_labels) if l == 0]

    print("\n--- Results ---")
    print(f"Pairs judged: {len(all_scores)}")
    print(f"Pearson correlation (LLM score vs. binary qrel label): {correlation:.4f}")
    if relevant_scores:
        print(
            f"Mean LLM score on RELEVANT passages ({len(relevant_scores)}): "
            f"{sum(relevant_scores) / len(relevant_scores):.2f}"
        )
    if non_relevant_scores:
        print(
            f"Mean LLM score on NON-RELEVANT passages ({len(non_relevant_scores)}): "
            f"{sum(non_relevant_scores) / len(non_relevant_scores):.2f}"
        )
    print(f"\n(Took {time.time() - t0:.1f}s total for {len(judge_queries)} API calls.)")


if __name__ == "__main__":
    main()