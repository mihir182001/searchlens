"""
Test suite for the
faithfulness-judging prompt, response parsing, and FaithfulnessJudge's
dispatch logic, using a fake raw_response_fn so it runs fully offline --
no anthropic package or API key needed.
"""
import pytest

from src.generation.faithfulness_judge import (
    build_faithfulness_prompt,
    parse_faithfulness_response,
    FaithfulnessJudge,
    FaithfulnessVerdict,
)


def test_build_faithfulness_prompt_includes_query_answer_and_passages():
    prompt = build_faithfulness_prompt(
        "what is x", "x is the answer [1]", ["passage about x"]
    )
    assert "what is x" in prompt
    assert "x is the answer [1]" in prompt
    assert "passage about x" in prompt
    assert "Passage 1:" in prompt


def test_parse_faithfulness_response_true():
    verdict = parse_faithfulness_response('{"faithful": true, "reason": "fully supported"}')
    assert verdict.faithful is True
    assert verdict.reason == "fully supported"


def test_parse_faithfulness_response_false():
    verdict = parse_faithfulness_response('{"faithful": false, "reason": "fabricated a date"}')
    assert verdict.faithful is False
    assert verdict.reason == "fabricated a date"


def test_parse_faithfulness_response_strips_code_fence():
    verdict = parse_faithfulness_response('```json\n{"faithful": true, "reason": "ok"}\n```')
    assert verdict.faithful is True


def test_parse_faithfulness_response_missing_key_raises():
    with pytest.raises(ValueError, match="faithful"):
        parse_faithfulness_response('{"reason": "no faithful key here"}')


def test_parse_faithfulness_response_non_dict_raises():
    with pytest.raises(ValueError, match="JSON object"):
        parse_faithfulness_response("[true, false]")


def test_parse_faithfulness_response_missing_reason_defaults_to_empty_string():
    verdict = parse_faithfulness_response('{"faithful": true}')
    assert verdict.reason == ""


def test_faithfulness_judge_returns_verdict():
    def fake_raw_response_fn(prompt):
        return '{"faithful": false, "reason": "unsupported claim"}'

    judge = FaithfulnessJudge(raw_response_fn=fake_raw_response_fn)
    verdict = judge.judge("q", "some answer", ["passage text"])
    assert isinstance(verdict, FaithfulnessVerdict)
    assert verdict.faithful is False
    assert verdict.reason == "unsupported claim"


def test_faithfulness_judge_passes_all_inputs_into_prompt():
    seen_prompts = []

    def fake_raw_response_fn(prompt):
        seen_prompts.append(prompt)
        return '{"faithful": true, "reason": "ok"}'

    judge = FaithfulnessJudge(raw_response_fn=fake_raw_response_fn)
    judge.judge("unique query", "unique answer text", ["unique passage text"])
    assert len(seen_prompts) == 1
    assert "unique query" in seen_prompts[0]
    assert "unique answer text" in seen_prompts[0]
    assert "unique passage text" in seen_prompts[0]


def test_requires_api_key_when_no_raw_response_fn_and_no_key_set(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        FaithfulnessJudge()