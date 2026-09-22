"""
 inference wrapper for a fine-tuned
DistilBERT query intent classifier.

Same constraint as Weeks 2/3: the real model (distilbert-base-uncased) and
tokenizer come from huggingface.co. As with DenseIndex/CrossEncoderReranker,
QueryIntentClassifier takes an injectable `predict_proba_fn` so its
plumbing (label decoding, batching, confidence extraction, error handling)
can be unit-tested offline with a fake predictor -- see
tests/test_query_intent_classifier.py -- while the real fine-tuned model
runs on your machine.

Training the actual model is a separate script:
    src/models/train_query_intent_classifier.py  (RUN ON YOUR OWN MACHINE)
"""
from dataclasses import dataclass

from src.data.query_intent_dataset import LabelEncoder


@dataclass
class IntentPrediction:
    intent: str
    confidence: float


def _default_predict_proba_factory(model_dir: str):
    """Lazy-imports torch/transformers and loads a fine-tuned model from
    `model_dir` (produced by train_query_intent_classifier.py). Returns a
    predict_proba(queries: list[str]) -> list[list[float]] function, one
    probability vector per query, in INTENT_LABELS order.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    def predict_proba(queries):
        inputs = tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return probs.tolist()

    return predict_proba


class QueryIntentClassifier:
    """Wraps a fine-tuned DistilBERT sequence-classification model for
    5-class query intent prediction.

    predict_proba_fn: injectable stand-in for testing (see
        tests/test_query_intent_classifier.py). Defaults to loading the
        real model from `model_dir` via _default_predict_proba_factory
        (needs transformers + torch + actual fine-tuned weights on disk).
    """

    def __init__(self, model_dir: str = None, predict_proba_fn=None, encoder: LabelEncoder = None):
        self.encoder = encoder or LabelEncoder()
        if predict_proba_fn is not None:
            self._predict_proba = predict_proba_fn
        else:
            if model_dir is None:
                raise ValueError("model_dir is required unless predict_proba_fn is provided.")
            self._predict_proba = _default_predict_proba_factory(model_dir)

    def predict(self, query: str) -> IntentPrediction:
        return self.predict_many([query])[0]

    def predict_many(self, queries: list) -> list:
        if not queries:
            return []
        probs_batch = self._predict_proba(queries)
        predictions = []
        for probs in probs_batch:
            if len(probs) != self.encoder.num_labels:
                raise ValueError(
                    f"predict_proba_fn returned {len(probs)} probabilities but "
                    f"encoder has {self.encoder.num_labels} labels -- label order mismatch."
                )
            best_id = max(range(len(probs)), key=lambda i: probs[i])
            predictions.append(
                IntentPrediction(intent=self.encoder.decode(best_id), confidence=probs[best_id])
            )
        return predictions