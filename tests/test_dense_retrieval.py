"""
Uses a lightweight, dependency-free hashing encoder instead of a real
sentence-transformers model. This sandbox cannot reach huggingface.co to
download model weights, so these tests validate everything DenseIndex is
actually responsible for -- FAISS indexing, top-k search, score ordering,
save/load round-trips, interface parity with BM25Index -- without needing
real semantic embeddings. The real semantic-quality check (does it
actually beat BM25 on meaning, not just plumbing) happens in
run_dense_eval.py on your own machine with the real model.
"""
import re
import zlib
import numpy as np
import pytest

from src.retrieval.dense_index import DenseIndex


def hashing_encoder(texts, dim=64):
    """Deterministic, offline bag-of-words hashing embedding.

    Not a real semantic encoder -- it has no notion that "heart attack" and
    "myocardial infarction" mean the same thing. It DOES give two pieces of
    text with overlapping words a higher cosine similarity than two with no
    overlap, which is enough real signal to test that DenseIndex's FAISS
    plumbing (ranking, top-k, normalization) behaves correctly end-to-end,
    without depending on network access or a multi-hundred-MB model.

    Uses zlib.crc32 rather than Python's built-in hash() -- hash() is
    randomized per-process (PYTHONHASHSEED) as a security feature, which
    made this fixture's bucket assignment (and therefore which passages
    look "more similar") change nondeterministically between runs,
    occasionally flipping a test's outcome. crc32 is stable across runs
    and processes, so a given token always lands in the same bucket.
    """
    vectors = np.zeros((len(texts), dim), dtype="float32")
    for i, text in enumerate(texts):
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = zlib.crc32(token.encode("utf-8")) % dim
            vectors[i, bucket] += 1.0
    return vectors


@pytest.fixture
def small_passages():
    return {
        "p0": "The cat sat on the mat in the sun.",
        "p1": "Quantum computers use qubits instead of classical bits.",
        "p2": "A dog barked loudly at the mail carrier in the park.",
        "p3": "Superconducting qubits are a leading approach to quantum computing.",
    }


def test_dense_index_raises_on_empty_corpus():
    with pytest.raises(ValueError):
        DenseIndex({}, encoder=hashing_encoder)


def test_dense_index_finds_word_overlap_match(small_passages):
    index = DenseIndex(small_passages, encoder=hashing_encoder)
    results = index.search("qubits in quantum computing", top_k=4)
    top_pids = {results[0].pid, results[1].pid}
    assert top_pids == {"p1", "p3"}  # the two quantum-computing passages


def test_dense_index_respects_top_k(small_passages):
    index = DenseIndex(small_passages, encoder=hashing_encoder)
    results = index.search("cat dog", top_k=2)
    assert len(results) == 2


def test_dense_index_scores_are_descending(small_passages):
    index = DenseIndex(small_passages, encoder=hashing_encoder)
    results = index.search("cat dog park", top_k=4)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_dense_index_scores_are_bounded_cosine_similarity(small_passages):
    index = DenseIndex(small_passages, encoder=hashing_encoder)
    results = index.search("cat dog park", top_k=4)
    for r in results:
        assert -1.0001 <= r.score <= 1.0001  # cosine similarity range


def test_search_many_returns_one_entry_per_query(small_passages):
    index = DenseIndex(small_passages, encoder=hashing_encoder)
    queries = {"q0": "cat", "q1": "quantum"}
    results = index.search_many(queries, top_k=2)
    assert set(results.keys()) == {"q0", "q1"}


def test_ensure_normalized_handles_unnormalized_encoder(small_passages):
    """A custom encoder that returns non-unit-length vectors should still
    produce valid cosine-similarity search results -- DenseIndex normalizes
    internally rather than trusting every encoder to have done it.
    """
    def unnormalized_encoder(texts):
        return hashing_encoder(texts) * 37.0  # arbitrary non-unit scale

    index = DenseIndex(small_passages, encoder=unnormalized_encoder)
    results = index.search("qubits quantum", top_k=4)
    for r in results:
        assert -1.0001 <= r.score <= 1.0001


def test_save_and_load_round_trip_preserves_search(tmp_path, small_passages):
    index = DenseIndex(small_passages, encoder=hashing_encoder)
    original_results = index.search("qubits quantum computing", top_k=4)

    index.save(tmp_path)
    loaded = DenseIndex.load(tmp_path, encoder=hashing_encoder)
    loaded_results = loaded.search("qubits quantum computing", top_k=4)

    assert [r.pid for r in original_results] == [r.pid for r in loaded_results]
    for original, loaded_r in zip(original_results, loaded_results):
        assert original.score == pytest.approx(loaded_r.score, abs=1e-4)


def test_save_creates_expected_files(tmp_path, small_passages):
    index = DenseIndex(small_passages, encoder=hashing_encoder)
    index.save(tmp_path)
    assert (tmp_path / "dense_index.faiss").exists()
    assert (tmp_path / "dense_index_meta.json").exists()