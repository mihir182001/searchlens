"""
tests/test_query_intent_classifier.py -- Week 5 test suite for
QueryIntentClassifier's plumbing (label decoding, batching, confidence
extraction) using a fake predict_proba_fn so it runs fully offline -- no
torch/transformers/HF weights needed, same pattern as
tests/test_dense_retrieval.py and tests/test_cross_encoder_rerank.py.
"""
import pytest

from src.data.query_intent_dataset import LabelEncoder
from src.models.query_intent_classifier import QueryIntentClassifier, IntentPrediction


def fake_predict_proba_factory(fixed_label: str):
    """Returns a predict_proba_fn that's always maximally confident about
    `fixed_label`, regardless of the query -- simplest possible fake."""
    encoder = LabelEncoder()
    target_id = encoder.encode(fixed_label)

    def predict_proba(queries):
        probs = [0.0] * encoder.num_labels
        probs[target_id] = 1.0
        return [probs for _ in queries]

    return predict_proba


def test_predict_returns_intent_prediction():
    classifier = QueryIntentClassifier(predict_proba_fn=fake_predict_proba_factory("factual"))
    result = classifier.predict("what is the capital of france")
    assert isinstance(result, IntentPrediction)
    assert result.intent == "factual"
    assert result.confidence == pytest.approx(1.0)


def test_predict_many_handles_multiple_queries():
    classifier = QueryIntentClassifier(predict_proba_fn=fake_predict_proba_factory("how-to"))
    results = classifier.predict_many(["how to bake bread", "how to fix a tire"])
    assert len(results) == 2
    assert all(r.intent == "how-to" for r in results)


def test_predict_many_empty_list_returns_empty():
    classifier = QueryIntentClassifier(predict_proba_fn=fake_predict_proba_factory("factual"))
    assert classifier.predict_many([]) == []


def test_predict_picks_argmax_not_just_first_class():
    encoder = LabelEncoder()

    def predict_proba(queries):
        # Deliberately put the highest probability on a class that is NOT
        # index 0, to make sure argmax logic (not "always pick class 0")
        # is actually what's running.
        probs = [0.1] * encoder.num_labels
        probs[encoder.encode("navigational")] = 0.6
        return [probs for _ in queries]

    classifier = QueryIntentClassifier(predict_proba_fn=predict_proba)
    result = classifier.predict("facebook login")
    assert result.intent == "navigational"
    assert result.confidence == pytest.approx(0.6)


def test_requires_model_dir_or_predict_proba_fn():
    with pytest.raises(ValueError):
        QueryIntentClassifier()


def test_mismatched_label_count_raises():
    def bad_predict_proba(queries):
        return [[0.5, 0.5] for _ in queries]  # only 2 probs, but 5 labels expected

    classifier = QueryIntentClassifier(predict_proba_fn=bad_predict_proba)
    with pytest.raises(ValueError, match="label order mismatch"):
        classifier.predict("test query")