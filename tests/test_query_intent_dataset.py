"""
test suite for LabelEncoder
and CSV loading. Pure Python, no model needed.
"""
import pytest

from src.data.query_intent_dataset import (
    INTENT_LABELS,
    LabelEncoder,
    load_intent_csv,
    encode_rows,
)


def test_label_encoder_round_trip():
    encoder = LabelEncoder()
    for label in INTENT_LABELS:
        encoded = encoder.encode(label)
        assert encoder.decode(encoded) == label


def test_label_encoder_num_labels():
    encoder = LabelEncoder()
    assert encoder.num_labels == len(INTENT_LABELS)


def test_label_encoder_rejects_unknown_label():
    encoder = LabelEncoder()
    with pytest.raises(ValueError):
        encoder.encode("not-a-real-intent")


def test_label_encoder_decode_out_of_range_raises():
    encoder = LabelEncoder()
    with pytest.raises(ValueError):
        encoder.decode(999)


def test_label_encoder_is_deterministic_across_instances():
    a = LabelEncoder()
    b = LabelEncoder()
    for label in INTENT_LABELS:
        assert a.encode(label) == b.encode(label)


def test_load_intent_csv_missing_file_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_query_intent_data"):
        load_intent_csv(tmp_path / "does_not_exist.csv")


def test_load_intent_csv_reads_query_intent_columns(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("query,intent\nwhat is x,factual\nhow to y,how-to\n", encoding="utf-8")
    rows = load_intent_csv(path)
    assert rows == [("what is x", "factual"), ("how to y", "how-to")]


def test_encode_rows_produces_parallel_lists():
    rows = [("what is x", "factual"), ("how to y", "how-to")]
    queries, label_ids = encode_rows(rows)
    encoder = LabelEncoder()
    assert queries == ["what is x", "how to y"]
    assert label_ids == [encoder.encode("factual"), encoder.encode("how-to")]


def test_encode_rows_uses_provided_encoder():
    custom_encoder = LabelEncoder(labels=["how-to", "factual"])  # reversed order
    rows = [("what is x", "factual"), ("how to y", "how-to")]
    _, label_ids = encode_rows(rows, encoder=custom_encoder)
    assert label_ids == [1, 0]