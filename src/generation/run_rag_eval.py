"""
RUN THIS ON YOUR OWN MACHINE
with your own Anthropic API key.

    (PowerShell) $env:ANTHROPIC_API_KEY="sk-ant-..."
    pip install anthropic
    python -m src.generation.run_rag_eval --max-queries 10   (time it first)
    python -m src.generation.run_rag_eval --max-queries 100

What it does, per sampled query:
    1. Retrieves top-5 passages (Hybrid RRF, Week 4's best config, if a
       cached dense index exists, else BM25-only) as RAG context -- the
       spec's "top-5 passages as context".
    2. Generates an answer with inline citations via Claude
       (src/generation/rag_pipeline.py) -- ONE API call.
    3. Judges whether that answer is fully supported by the SAME 5
       context passages via a SEPARATE Claude call acting as judge
       (src/generation/faithfulness_judge.py) -- a SECOND, independent
       API call, so each query costs 2 calls, not 1. An LLM judging its
       own answer in the same call it wrote it is a much weaker check
       than an independent pass.
    4. Records the answer, which passages were cited, whether every cited
       index is actually within [1, 5] (a citation pointing outside that
       range is a clear generation bug, distinct from an unfaithful
       claim), and the faithfulness verdict + one-sentence reason.

Reports:
    - Faithfulness rate: fraction of answers the judge marked fully
      supported by their context. Compare honestly against the spec's
      92% target -- do not round up or explain away a lower number
      without evidence, same as every other week's numbers in this
      project.
    - Citation stats: average citations per answer, % of answers with
      zero citations (suspicious either way -- the passages may not have
      helped, or the model didn't cite when it should have), % of answers
      with at least one out-of-range citation.

Saves every (query, answer, faithfulness verdict) record to
data/processed/rag_eval_results.csv for audit -- read a few of the
unfaithful ones yourself before trusting the aggregate number. A single
faithfulness percentage hides WHY answers failed (a fabricated detail vs.
citing the wrong passage vs. going slightly beyond what a passage actually
says are very different failure modes with different fixes).

Cost/time note: 2 API calls per query (generation + faithfulness judge),
so 100 queries = 200 calls. Time a small slice first, same pattern as
every other week's expensive script.
"""
import argparse
import csv
import random
import time
from pathlib import Path

from src.data.loader import load_passages, load_queries
from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex
from src.retrieval.hybrid_rrf import fuse_many
from src.generation.rag_pipeline import RAGGenerator
from src.generation.faithfulness_judge import FaithfulnessJudge

DENSE_INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "dense_index"
RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "rag_eval_results.csv"

DEPTH = 5  # spec-mandated top-5 passages as RAG context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929")
    args = parser.parse_args()

    passages = load_passages()
    all_queries = load_queries()
    print(f"Corpus: {len(passages)} passages, {len(all_queries)} queries available")

    # ---- Retrieval: same Hybrid RRF config as Week 6, top-5 ----
    bm25_index = BM25Index(passages)
    bm25_results = bm25_index.search_many(all_queries, top_k=DEPTH)

    if (DENSE_INDEX_DIR / "dense_index.faiss").exists():
        print("Cached dense index found -- using Hybrid RRF (dense_weight=10, Week 4's best config).")
        dense_index = DenseIndex.load(DENSE_INDEX_DIR)
        dense_results = dense_index.search_many(all_queries, top_k=DEPTH)
        retrieval_results = fuse_many([bm25_results, dense_results], top_k=DEPTH, weights=[1.0, 10.0])
    else:
        print("No cached dense index found -- using BM25-only candidates.")
        retrieval_results = bm25_results

    # ---- Sample queries that actually have candidates ----
    eligible_qids = [qid for qid, results in retrieval_results.items() if results]
    rng = random.Random(args.seed)
    if args.max_queries < len(eligible_qids):
        sampled_qids = rng.sample(eligible_qids, args.max_queries)
    else:
        sampled_qids = eligible_qids
    print(
        f"Running RAG + faithfulness check on {len(sampled_qids)} queries "
        f"({len(sampled_qids) * 2} API calls total)."
    )

    # ---- Generate + judge ----
    generator = RAGGenerator(model=args.model)
    judge = FaithfulnessJudge(model=args.model)

    rows = []
    faithful_count = 0
    citation_counts = []
    zero_citation_count = 0
    invalid_citation_count = 0

    t0 = time.time()
    for i, qid in enumerate(sampled_qids, start=1):
        query_text = all_queries[qid]
        candidate_results = retrieval_results[qid][:DEPTH]
        pids = [r.pid for r in candidate_results]
        context_passages = [passages[pid] for pid in pids]

        answer = generator.answer_query(query_text, context_passages)
        verdict = judge.judge(query_text, answer.answer_text, context_passages)

        num_citations = len(answer.cited_indices)
        citation_counts.append(num_citations)
        if num_citations == 0:
            zero_citation_count += 1
        has_invalid_citation = any(idx < 1 or idx > len(context_passages) for idx in answer.cited_indices)
        if has_invalid_citation:
            invalid_citation_count += 1
        if verdict.faithful:
            faithful_count += 1

        rows.append(
            {
                "qid": qid,
                "query": query_text,
                "answer": answer.answer_text,
                "cited_indices": ",".join(str(idx) for idx in answer.cited_indices),
                "has_invalid_citation": has_invalid_citation,
                "faithful": verdict.faithful,
                "reason": verdict.reason,
            }
        )

        if i % 20 == 0 or i == len(sampled_qids):
            print(f"  {i}/{len(sampled_qids)} queries processed ({time.time() - t0:.1f}s elapsed)")

    # ---- Save raw results for audit ----
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["qid", "query", "answer", "cited_indices", "has_invalid_citation", "faithful", "reason"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} records to {RESULTS_PATH}")

    # ---- Summary ----
    n = len(sampled_qids)
    faithfulness_rate = faithful_count / n if n else 0.0
    avg_citations = sum(citation_counts) / n if n else 0.0

    print("\n--- Results ---")
    print(f"Queries evaluated: {n}")
    print(f"Faithfulness rate: {faithfulness_rate:.4f} ({faithful_count}/{n})")
    print(f"Average citations per answer: {avg_citations:.2f}")
    print(f"Answers with zero citations: {zero_citation_count}/{n} ({zero_citation_count / n:.1%})")
    print(
        f"Answers with an out-of-range citation: {invalid_citation_count}/{n} "
        f"({invalid_citation_count / n:.1%})"
    )
    print(f"\n(Took {time.time() - t0:.1f}s total for {n * 2} API calls.)")


if __name__ == "__main__":
    main()