"""
tests/test_generate_query_intent_data.py -- Week 5 test suite for the
stratified train/val split and CSV writer. No model needed -- pure Python.
"""
import pytest

from src.data.generate_query_intent_data import stratified_split, write_csv
from src.data.query_intent_examples import INTENT_EXAMPLES
from src.data.query_intent_dataset import load_intent_csv


def test_stratified_split_covers_every_class_in_both_splits():
    train_rows, val_rows = stratified_split(INTENT_EXAMPLES, val_fraction=0.2, seed=42)
    train_intents = {intent for _, intent in train_rows}
    val_intents = {intent for _, intent in val_rows}
    assert train_intents == set(INTENT_EXAMPLES.keys())
    assert val_intents == set(INTENT_EXAMPLES.keys())


def test_stratified_split_preserves_total_count_per_class():
    train_rows, val_rows = stratified_split(INTENT_EXAMPLES, val_fraction=0.2, seed=42)
    for intent, examples in INTENT_EXAMPLES.items():
        train_count = sum(1 for _, i in train_rows if i == intent)
        val_count = sum(1 for _, i in val_rows if i == intent)
        assert train_count + val_count == len(examples)


def test_stratified_split_no_query_in_both_splits():
    train_rows, val_rows = stratified_split(INTENT_EXAMPLES, val_fraction=0.2, seed=42)
    train_queries = {q for q, _ in train_rows}
    val_queries = {q for q, _ in val_rows}
    assert train_queries.isdisjoint(val_queries)


def test_stratified_split_is_reproducible_with_same_seed():
    train_a, val_a = stratified_split(INTENT_EXAMPLES, seed=7)
    train_b, val_b = stratified_split(INTENT_EXAMPLES, seed=7)
    assert train_a == train_b
    assert val_a == val_b


def test_stratified_split_val_fraction_respected_for_uniform_classes():
    # Every class in INTENT_EXAMPLES has 40 examples, so a 20% split should
    # give exactly 8 val / 32 train per class => 40 val / 160 train total.
    train_rows, val_rows = stratified_split(INTENT_EXAMPLES, val_fraction=0.2, seed=42)
    assert len(val_rows) == 40
    assert len(train_rows) == 160


def test_write_csv_round_trips_rows(tmp_path):
    rows = [("what is x", "factual"), ("how to y", "how-to")]
    out_path = tmp_path / "test.csv"
    write_csv(rows, out_path)
    loaded = load_intent_csv(out_path)
    assert loaded == rows


def test_write_csv_creates_parent_directories(tmp_path):
    out_path = tmp_path / "nested" / "dir" / "test.csv"
    write_csv([("q", "factual")], out_path)
    assert out_path.exists()