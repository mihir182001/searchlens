"""
Test suite for RAG prompt building,
citation parsing, and RAGGenerator's dispatch logic, using a fake
raw_response_fn so it runs fully offline -- no anthropic package or API
key needed, same pattern as tests/test_llm_judge.py.
"""
import pytest

from src.generation.rag_pipeline import build_rag_prompt, parse_citations, RAGGenerator, RAGAnswer


def test_build_rag_prompt_includes_query_and_all_passages():
    prompt = build_rag_prompt("what is photosynthesis", ["passage one text", "passage two text"])
    assert "what is photosynthesis" in prompt
    assert "passage one text" in prompt
    assert "passage two text" in prompt
    assert "Passage 1:" in prompt
    assert "Passage 2:" in prompt


def test_parse_citations_single():
    assert parse_citations("The tower was built in 1889 [2].") == [2]


def test_parse_citations_multiple_separate_brackets():
    assert parse_citations("Fact one [1]. Fact two [3].") == [1, 3]


def test_parse_citations_comma_separated_in_one_bracket():
    assert parse_citations("This is supported by two sources [1, 3].") == [1, 3]


def test_parse_citations_no_citations_returns_empty():
    assert parse_citations("The passages don't contain enough information to answer.") == []


def test_parse_citations_deduplicates():
    assert parse_citations("Stated here [2]. Restated here [2].") == [2]


def test_parse_citations_sorts_out_of_order_citations():
    assert parse_citations("Later point [3]. Earlier point [1].") == [1, 3]


def test_rag_generator_answer_query_returns_parsed_answer():
    def fake_raw_response_fn(prompt):
        return "The Eiffel Tower was completed in 1889 [2]."

    generator = RAGGenerator(raw_response_fn=fake_raw_response_fn)
    answer = generator.answer_query("when was the eiffel tower built", ["p1 text", "p2 text"])
    assert isinstance(answer, RAGAnswer)
    assert answer.cited_indices == [2]
    assert "1889" in answer.answer_text


def test_rag_generator_raises_on_empty_passages():
    def fake_raw_response_fn(prompt):
        return "should never be called"

    generator = RAGGenerator(raw_response_fn=fake_raw_response_fn)
    with pytest.raises(ValueError, match="zero passages"):
        generator.answer_query("some query", [])


def test_rag_generator_passes_built_prompt_to_raw_response_fn():
    seen_prompts = []

    def fake_raw_response_fn(prompt):
        seen_prompts.append(prompt)
        return "answer text [1]"

    generator = RAGGenerator(raw_response_fn=fake_raw_response_fn)
    generator.answer_query("unique query text", ["unique passage text"])
    assert len(seen_prompts) == 1
    assert "unique query text" in seen_prompts[0]
    assert "unique passage text" in seen_prompts[0]


def test_requires_api_key_when_no_raw_response_fn_and_no_key_set(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        RAGGenerator()