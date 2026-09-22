"""
 Writes the seed query-intent dataset
(src/data/query_intent_examples.py) out to disk as a train/val split.

    python -m src.data.generate_query_intent_data

Splits each class independently (stratified) so a rare-ish class can't end
up entirely in one split by bad luck -- with only 40 examples/class this
matters a lot more than it would on a large dataset.

Writes:
    data/raw/query_intent_train.csv
    data/raw/query_intent_val.csv
each with columns: query, intent
"""
import argparse
import csv
import random
from pathlib import Path

from src.data.query_intent_examples import INTENT_EXAMPLES

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def stratified_split(intent_examples: dict, val_fraction: float = 0.2, seed: int = 42):
    """Returns (train_rows, val_rows), each a list of (query, intent) tuples.
    Splits WITHIN each class so every class is represented in both splits
    proportionally, rather than one global shuffle-then-split (which, with
    only 40 examples/class, could easily leave a class thin or absent from
    val by chance).
    """
    rng = random.Random(seed)
    train_rows, val_rows = [], []
    for intent, examples in intent_examples.items():
        shuffled = list(examples)
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction))
        val_examples = shuffled[:n_val]
        train_examples = shuffled[n_val:]
        train_rows.extend((q, intent) for q in train_examples)
        val_rows.extend((q, intent) for q in val_examples)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def write_csv(rows: list, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "intent"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_rows, val_rows = stratified_split(
        INTENT_EXAMPLES, val_fraction=args.val_fraction, seed=args.seed
    )

    train_path = RAW_DIR / "query_intent_train.csv"
    val_path = RAW_DIR / "query_intent_val.csv"
    write_csv(train_rows, train_path)
    write_csv(val_rows, val_path)

    print(f"Wrote {len(train_rows)} train rows -> {train_path}")
    print(f"Wrote {len(val_rows)} val rows -> {val_path}")

    counts = {}
    for _, intent in train_rows + val_rows:
        counts[intent] = counts.get(intent, 0) + 1
    print("\nPer-class totals (train+val):")
    for intent, count in sorted(counts.items()):
        print(f"  {intent:<14} {count}")


if __name__ == "__main__":
    main()