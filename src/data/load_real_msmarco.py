"""
load_real_msmarco.py -- run this on YOUR OWN MACHINE, not in the sandbox.

v2: pulls the REAL MS MARCO Passage Ranking benchmark (the one the
published "BM25 MRR@10 ~ 0.167" number in the SearchLens spec refers to),
via BeIR's standardized version on Hugging Face -- not the QnA-formatted
`microsoft/ms_marco` dataset the first version of this script used, which
only ships ~8 pre-selected candidate passages per query and silently
inflates BM25's score because it isn't really doing full-corpus retrieval.

    BeIR/msmarco          -- corpus config: 8,841,823 real passages
                             queries config: 509,962 real queries
    BeIR/msmarco-qrels    -- validation split: the standard MS MARCO
                             "dev" relevance judgments (~6,980 queries)

Why we don't index all 8.84M passages:
    rank_bm25 (pure Python + numpy) has to hold every tokenized passage in
    memory and does a full pass over the corpus per query. On 8.84M
    passages that's tens of GB of RAM just for tokenized text -- not
    something a laptop can do. So this script downloads the full corpus
    via streaming (so we never hold all 8.84M in RAM at once) and keeps:
      1. EVERY passage that is an actual relevant answer for a query we
         kept (so the evaluation is never unfairly impossible), plus
      2. A random sample of additional passages as realistic distractors,
         up to --corpus-size total.

    This is still real MS MARCO text, real queries, and real human
    relevance judgments -- it is a size-capped subset of the true
    benchmark, not a fabricated one. Report it as such (see README): it
    is a meaningfully larger and harder retrieval task than the previous
    16k-passage run, but it is not the full 8.84M-passage number the
    published 0.167 baseline was measured on, and the honest way to say
    so is to state --corpus-size and --max-queries alongside the result.

Setup (on your machine):
    pip install datasets

Usage:
    python -m src.data.load_real_msmarco --corpus-size 300000 --max-queries 1000

Increase --corpus-size if your machine has plenty of RAM (16GB+ can
usually handle 500k-1,000,000); lower it if the index build is too slow
or you run out of memory.
"""
import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-size", type=int, default=300_000,
                         help="Total passages to keep (all true-positive passages are always "
                              "kept on top of this; the rest is a random distractor sample). "
                              "Full real corpus is 8,841,823 -- this is a capped subset, see docstring.")
    parser.add_argument("--max-queries", type=int, default=1000,
                         help="Cap on dev queries evaluated. The real MS MARCO dev-small set has "
                              "~6,980 queries; use -1 to keep all of them (slower).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(RAW_DIR))
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The `datasets` package isn't installed. Run `pip install datasets` "
            "on your own machine (this won't work inside the sandbox -- "
            "huggingface.co isn't reachable from there)."
        )

    rng = random.Random(args.seed)

    # ------------------------------------------------------------------
    # 1. Real dev qrels (the standard MS MARCO dev-small relevance judgments)
    # ------------------------------------------------------------------
    print("Loading MS MARCO dev qrels (BeIR/msmarco-qrels, validation split)...")
    qrels_ds = load_dataset("BeIR/msmarco-qrels", split="validation")

    full_qrels = {}
    for row in qrels_ds:
        if row["score"] <= 0:
            continue
        qid = str(row["query-id"])
        pid = str(row["corpus-id"])
        full_qrels.setdefault(qid, set()).add(pid)

    print(f"  {len(full_qrels)} real dev queries have at least one relevant passage")

    # ------------------------------------------------------------------
    # 2. Real dev queries -- filter the 509,962-query pool down to just
    #    the ones that are actually in the dev qrels set above
    # ------------------------------------------------------------------
    print("Loading MS MARCO queries (BeIR/msmarco, queries config)...")
    queries_ds = load_dataset("BeIR/msmarco", name="queries", split="queries")

    needed_qids = set(full_qrels.keys())
    dev_queries = {}
    for row in queries_ds:
        qid = str(row["_id"])
        if qid in needed_qids:
            dev_queries[qid] = row["text"]

    print(f"  matched {len(dev_queries)} of {len(full_qrels)} dev qrels queries to query text")

    # Cap the query count if requested (random sample for reproducibility)
    all_qids = sorted(dev_queries.keys())
    if args.max_queries != -1 and args.max_queries < len(all_qids):
        rng.shuffle(all_qids)
        kept_qids = set(all_qids[: args.max_queries])
    else:
        kept_qids = set(all_qids)

    queries = {qid: dev_queries[qid] for qid in kept_qids}
    qrels = {qid: full_qrels[qid] for qid in kept_qids}

    needed_pids = set()
    for pids in qrels.values():
        needed_pids |= pids

    print(f"  keeping {len(queries)} queries, requiring {len(needed_pids)} true-positive passages")

    # ------------------------------------------------------------------
    # 3. Stream the full 8.84M-passage corpus once. Always keep a passage
    #    if it's a needed true-positive; otherwise keep it as a random
    #    distractor with probability chosen to hit ~corpus_size total.
    # ------------------------------------------------------------------
    distractor_budget = max(args.corpus_size - len(needed_pids), 0)
    approx_total_corpus = 8_841_823
    keep_prob = distractor_budget / approx_total_corpus

    print(f"Streaming full corpus (8,841,823 passages) once, sampling ~{args.corpus_size} total "
          f"(keep_prob={keep_prob:.5f} for non-relevant passages)...")
    print("This downloads several GB and can take a while depending on your connection.")

    corpus_ds = load_dataset("BeIR/msmarco", name="corpus", split="corpus", streaming=True)

    passages = {}
    seen = 0
    for row in corpus_ds:
        seen += 1
        pid = str(row["_id"])
        if pid in needed_pids or rng.random() < keep_prob:
            text = row["text"]
            if row.get("title"):
                text = row["title"] + ". " + text
            passages[pid] = text
        if seen % 1_000_000 == 0:
            print(f"  ...scanned {seen:,} passages, kept {len(passages):,} so far")

    print(f"Done streaming. Scanned {seen:,} passages total, kept {len(passages):,}.")

    missing = needed_pids - set(passages.keys())
    if missing:
        # Should be rare/never (every needed pid should appear in the corpus
        # stream), but if it happens, drop those queries rather than silently
        # scoring them as guaranteed misses against a passage that isn't
        # even in the pool.
        print(f"  Warning: {len(missing)} true-positive passages were not found in the corpus "
              f"stream (unexpected) -- dropping the affected queries.")
        queries = {qid: q for qid, q in queries.items() if not (qrels[qid] & missing)}
        qrels = {qid: p for qid, p in qrels.items() if qid in queries}

    # ------------------------------------------------------------------
    # 4. Save in the same schema everything else already expects
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "passages.jsonl", "w", encoding="utf-8") as f:
        for pid, text in passages.items():
            f.write(json.dumps({"pid": pid, "text": text}) + "\n")

    with open(out_dir / "queries.jsonl", "w", encoding="utf-8") as f:
        for qid, text in queries.items():
            f.write(json.dumps({"qid": qid, "text": text}) + "\n")

    with open(out_dir / "qrels.json", "w", encoding="utf-8") as f:
        json.dump({qid: sorted(pids) for qid, pids in qrels.items()}, f, indent=2)

    print(f"\nWrote {len(passages)} passages, {len(queries)} queries, {len(qrels)} qrels to {out_dir}")
    print(f"(corpus_size={args.corpus_size}, max_queries={args.max_queries}, seed={args.seed} -- "
          f"record these alongside any MRR@10 you report from this run)")
    print("Now run: python -m src.evaluation.run_bm25_eval")


if __name__ == "__main__":
    main()