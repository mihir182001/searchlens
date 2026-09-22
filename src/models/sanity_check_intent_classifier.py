"""
quick manual generalization
check. RUN ON YOUR OWN MACHINE, after train_query_intent_classifier.py has
saved a model to data/processed/query_intent_model/.

    python -m src.models.sanity_check_intent_classifier

The train/val split's 1.0 macro F1 is expected given how small and
stylistically narrow the 200-example dataset is -- see
train_query_intent_classifier.py's docstring. That number does NOT by
itself tell you whether the model generalizes to queries phrased
differently from the training templates (the val set is drawn from the
exact same 5 template styles as train, so it can't catch overfitting to
surface patterns like "how to X" or "X vs Y").

This script feeds it a small, hand-picked set of queries that are NOT in
query_intent_examples.py -- some close to the training style, and some
deliberately off it (no "how to" prefix, informal phrasing, a topic never
seen in training) -- to get an honest read on generalization before
trusting this model in the retrieval pipeline.
"""
from pathlib import Path

from src.models.query_intent_classifier import QueryIntentClassifier

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "query_intent_model"

TEST_QUERIES = [
    ("what is the tallest mountain in africa", "factual"),
    ("how many moons does jupiter have", "factual"),
    ("reddit sign in", "navigational"),
    ("spotify login page", "navigational"),
    ("how does weather forecasting work", "exploratory"),
    ("tell me about the history of the olympics", "exploratory"),
    ("postgres vs mongodb", "comparison"),
    ("should i get a macbook or a windows laptop", "comparison"),
    ("how to change a tire on a motorcycle", "how-to"),
    ("how do i fix my wifi", "how-to"),  # no "how to" prefix -- tests real generalization
]


def main():
    classifier = QueryIntentClassifier(model_dir=str(MODEL_DIR))
    correct = 0
    print(f"{'Query':<45} {'Predicted':<14} {'Expected':<14} {'Confidence':>10}")
    for query, expected in TEST_QUERIES:
        result = classifier.predict(query)
        is_correct = result.intent == expected
        correct += is_correct
        marker = "OK" if is_correct else "MISS"
        print(f"{query:<45} {result.intent:<14} {expected:<14} {result.confidence:>10.3f}  {marker}")
    print(f"\n{correct}/{len(TEST_QUERIES)} correct on held-out, hand-picked queries not in training data.")


if __name__ == "__main__":
    main()