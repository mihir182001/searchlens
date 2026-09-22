"""
Loads a corpus (passages/queries/qrels) from disk into the
in-memory dicts the rest of the pipeline expects.

Same file layout for the synthetic corpus AND for a real MS MARCO dump the
user prepares on their own machine (see README's "Running on real MS MARCO"
section), so BM25Index and the evaluator never need to know which one they
are given.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"


def load_passages(path: Path = None) -> dict:
    path = path or (RAW_DIR / "passages.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"No passages file at {path}. Run "
            "`python -m src.data.generate_synthetic_msmarco` first to create "
            "the synthetic corpus, or point `path` at your own passages.jsonl."
        )
    passages = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            passages[row["pid"]] = row["text"]
    return passages


def load_queries(path: Path = None) -> dict:
    path = path or (RAW_DIR / "queries.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"No queries file at {path}.")
    queries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            queries[row["qid"]] = row["text"]
    return queries


def load_qrels(path: Path = None) -> dict:
    path = path or (RAW_DIR / "qrels.json")
    if not path.exists():
        raise FileNotFoundError(f"No qrels file at {path}.")
    with open(path) as f:
        raw = json.load(f)
    return {qid: set(pids) for qid, pids in raw.items()}