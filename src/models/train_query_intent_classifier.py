"""
. RUN THIS ON
YOUR OWN MACHINE (needs `transformers`, `torch`, `scikit-learn`, and real
Hugging Face model weights for distilbert-base-uncased).

    python -m src.data.generate_query_intent_data      (if not already run)
    python -m src.models.train_query_intent_classifier

What it does:
    1. Loads data/raw/query_intent_{train,val}.csv.
    2. Fine-tunes distilbert-base-uncased as a 5-class sequence classifier
       (labels: INTENT_LABELS, in src/data/query_intent_dataset.py).
    3. Evaluates on the val split each epoch (accuracy + macro F1).
    4. Saves the fine-tuned model + tokenizer to
       data/processed/query_intent_model/ so
       src/models/query_intent_classifier.py's QueryIntentClassifier can
       load it directly from that path.

Caveats to report honestly, matching this whole project's pattern:
    - This dataset is small, and before the Claude-API expansion was also
      stylistically narrow -- don't assume val accuracy alone tells you
      how well the model generalizes. That's exactly what
      src/models/sanity_check_intent_classifier.py is for -- it tests
      genuinely novel phrasings, not the val split.
    - This is a SEPARATE metric from MS MARCO's MRR@10 from Weeks 1-4 --
      intent classification accuracy and retrieval quality are two
      different things being tracked in this project. Do not conflate or
      compare them directly.
    - If your installed `transformers` version is older and raises a
      TypeError on `eval_strategy`, rename that argument to
      `evaluation_strategy` below (the kwarg was renamed in a fairly
      recent transformers release).
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np

from src.data.query_intent_dataset import (
    INTENT_LABELS,
    LabelEncoder,
    load_intent_train,
    load_intent_val,
    encode_rows,
)

MODEL_OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "query_intent_model"

# Per-epoch training checkpoints (which include the full optimizer state --
# much larger than the final model alone) are written OUTSIDE the project
# folder, in the OS temp dir, to avoid OneDrive sync interference. Even so,
# this project's real run showed the crash isn't OneDrive-specific -- it's
# a flaky Windows I/O issue (torch.save's write stream getting locked
# mid-write, most likely by antivirus real-time scanning). save_strategy
# is set to "no" below for that reason -- see the comment there.
CHECKPOINT_DIR = Path(tempfile.gettempdir()) / "searchlens_query_intent_checkpoints"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    from sklearn.metrics import accuracy_score, f1_score

    encoder = LabelEncoder()
    train_rows = load_intent_train()
    val_rows = load_intent_val()
    train_queries, train_labels = encode_rows(train_rows, encoder)
    val_queries, val_labels = encode_rows(val_rows, encoder)

    print(
        f"Train: {len(train_queries)} examples, Val: {len(val_queries)} examples, "
        f"{encoder.num_labels} classes: {INTENT_LABELS}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    class IntentDataset(Dataset):
        def __init__(self, queries, labels):
            self.encodings = tokenizer(queries, truncation=True, padding=True)
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_dataset = IntentDataset(train_queries, train_labels)
    val_dataset = IntentDataset(val_queries, val_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=encoder.num_labels
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        # Per-epoch checkpointing was hitting a flaky Windows I/O failure
        # (torch.save on the optimizer state getting its write stream
        # locked mid-write, likely by antivirus real-time scanning --
        # confirmed NOT OneDrive-specific, since it recurred after moving
        # output_dir off OneDrive entirely). This dataset is tiny and
        # converges to perfect val metrics by epoch 3-4, so there's
        # nothing worth resuming from -- the fix is to stop writing
        # per-epoch checkpoints (which include the large optimizer state)
        # altogether, and only write to disk ONCE, at the very end, with
        # the final trained weights (no optimizer state at all -- see
        # model.save_pretrained below). One write is far less likely to
        # collide with whatever was intermittently locking the file.
        save_strategy="no",
        load_best_model_at_end=False,
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n--- Final val evaluation ---")
    metrics = trainer.evaluate()
    print(metrics)

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_OUT_DIR)
    tokenizer.save_pretrained(MODEL_OUT_DIR)
    print(f"\nSaved fine-tuned model + tokenizer to {MODEL_OUT_DIR}")
    print("Load it with: QueryIntentClassifier(model_dir=str(MODEL_OUT_DIR))")


if __name__ == "__main__":
    main()