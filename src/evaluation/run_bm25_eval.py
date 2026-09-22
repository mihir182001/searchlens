"""

    python -m src.evaluation.run_bm25_eval

Loads the corpus (generating the synthetic one first if it doesn't exist
yet), builds a BM25Index, runs every query, and prints MRR@10 and
Recall@100. This is the script whose numbers get quoted as the BM25
baseline -- and, on your own machine with real MS MARCO swapped in
(see README), the real MRR@10 to report on the CV/portfolio.
"""
from pathlib import Path

from src.data.generate_synthetic_msmarco import generate_corpus, save_corpus, RAW_DIR
from src.data.loader import load_passages, load_queries, load_qrels
from src.retrieval.bm25_index import BM25Index
from src.evaluation.metrics import mrr_at_k, recall_at_k


def main():
    if not (RAW_DIR / "passages.jsonl").exists():
        print("No corpus found -- generating synthetic corpus (seed=42)...")
        passages, queries, qrels = generate_corpus()
        save_corpus(passages, queries, qrels)

    passages = load_passages()
    queries = load_queries()
    qrels = load_qrels()

    print(f"Corpus: {len(passages)} passages, {len(queries)} queries, {len(qrels)} qrels")

    index = BM25Index(passages)
    results = index.search_many(queries, top_k=100)

    mrr10 = mrr_at_k(results, qrels, k=10)
    recall100 = recall_at_k(results, qrels, k=100)

    print(f"MRR@10:      {mrr10:.4f}")
    print(f"Recall@100:  {recall100:.4f}")

    return mrr10, recall100


if __name__ == "__main__":
    main()