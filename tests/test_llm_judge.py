"""
Test suite for prompt building, response
parsing, and LLMJudge's dispatch logic, using a fake raw_response_fn so it
runs fully offline -- no anthropic package or API key needed, same pattern
as tests/test_dense_retrieval.py and tests/test_cross_encoder_rerank.py.
"""
import pytest

from src.evaluation.llm_judge import build_judge_prompt, parse_judge_response, LLMJudge


def test_build_judge_prompt_includes_query_and_all_passages():
    prompt = build_judge_prompt("what is photosynthesis", ["passage one text", "passage two text"])
    assert "what is photosynthesis" in prompt
    assert "passage one text" in prompt
    assert "passage two text" in prompt
    assert "Passage 1:" in prompt
    assert "Passage 2:" in prompt


def test_build_judge_prompt_asks_for_correct_count():
    prompt = build_judge_prompt("q", ["a", "b", "c"])
    assert "JSON array of 3 integers" in prompt


def test_parse_judge_response_plain_json():
    scores = parse_judge_response("[3, 5, 1]", expected_count=3)
    assert scores == [3, 5, 1]


def test_parse_judge_response_strips_code_fence():
    scores = parse_judge_response("```json\n[2, 4]\n```", expected_count=2)
    assert scores == [2, 4]


def test_parse_judge_response_strips_bare_code_fence_no_json_label():
    scores = parse_judge_response("```\n[1, 1, 1]\n```", expected_count=3)
    assert scores == [1, 1, 1]


def test_parse_judge_response_rejects_wrong_count():
    with pytest.raises(ValueError, match="Expected 3 scores"):
        parse_judge_response("[1, 2]", expected_count=3)


def test_parse_judge_response_rejects_out_of_range_score():
    with pytest.raises(ValueError, match="out of range"):
        parse_judge_response("[1, 6, 3]", expected_count=3)


def test_parse_judge_response_rejects_non_list():
    with pytest.raises(ValueError, match="Expected a JSON array"):
        parse_judge_response('{"score": 3}', expected_count=1)


def test_parse_judge_response_rejects_invalid_json():
    with pytest.raises(Exception):
        parse_judge_response("not json at all", expected_count=1)


def test_llm_judge_score_query_returns_parsed_scores():
    def fake_raw_response_fn(prompt):
        return "[4, 2, 5]"

    judge = LLMJudge(raw_response_fn=fake_raw_response_fn)
    scores = judge.score_query("some query", ["p1", "p2", "p3"])
    assert scores == [4, 2, 5]


def test_llm_judge_score_query_empty_passages_returns_empty_without_calling_model():
    calls = []

    def fake_raw_response_fn(prompt):
        calls.append(prompt)
        return "[]"

    judge = LLMJudge(raw_response_fn=fake_raw_response_fn)
    scores = judge.score_query("some query", [])
    assert scores == []
    assert calls == []  # never called the model for an empty candidate list


def test_llm_judge_passes_built_prompt_to_raw_response_fn():
    seen_prompts = []

    def fake_raw_response_fn(prompt):
        seen_prompts.append(prompt)
        return "[3]"

    judge = LLMJudge(raw_response_fn=fake_raw_response_fn)
    judge.score_query("unique query text", ["unique passage text"])
    assert len(seen_prompts) == 1
    assert "unique query text" in seen_prompts[0]
    assert "unique passage text" in seen_prompts[0]


def test_requires_api_key_when_no_raw_response_fn_and_no_key_set(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        LLMJudge()