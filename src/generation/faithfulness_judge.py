"""
LLM-as-judge check for whether a
generated RAG answer is faithful to (fully supported by) the passages it
was given as context.

Same injectable pattern as llm_judge.py (Week 6) -- fully offline-tested
via a fake raw_response_fn; the real Claude API call only happens when no
raw_response_fn is injected, exercised by run_rag_eval.py on your own
machine.
"""
import json
import os
from dataclasses import dataclass


@dataclass
class FaithfulnessVerdict:
    faithful: bool
    reason: str


def build_faithfulness_prompt(query: str, answer_text: str, passages: list) -> str:
    passage_lines = "\n\n".join(f"Passage {i + 1}:\n{passage}" for i, passage in enumerate(passages))
    return f"""You are checking whether an AI-generated answer is fully supported by the passages it was given -- not whether the answer is well-written, complete, or the best possible answer, only whether every factual claim it makes can be traced back to the passages below (no outside knowledge, no unsupported claims, no fabricated details).

Question: {query}

Passages:
{passage_lines}

Generated answer:
{answer_text}

Return ONLY a JSON object of the form {{"faithful": true or false, "reason": "one short sentence"}}. Mark faithful=false if the answer states anything not supported by the passages above, even if that claim happens to be true in general -- it must be traceable to these specific passages.
"""


def parse_faithfulness_response(text: str) -> FaithfulnessVerdict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
        if text.lower().startswith("json"):
            text = text[len("json") :]

    data = json.loads(text)
    if not isinstance(data, dict) or "faithful" not in data:
        raise ValueError(f"Expected a JSON object with a 'faithful' key, got: {data}")
    faithful = bool(data["faithful"])
    reason = str(data.get("reason", ""))
    return FaithfulnessVerdict(faithful=faithful, reason=reason)


def _default_raw_response_fn_factory(model: str):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set. Set it in your environment before running this.")

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def raw_response_fn(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return raw_response_fn


class FaithfulnessJudge:
    """Judges whether a generated answer is fully supported by the
    passages it was given, via a SEPARATE LLM call from the one that
    generated the answer (an LLM judging its own answer in the same call
    is a much weaker check than an independent judging pass).

    raw_response_fn: injectable stand-in for testing -- a function
        prompt(str) -> raw response text (str). Defaults to a real Claude
        API call (needs `anthropic` + ANTHROPIC_API_KEY).
    """

    def __init__(self, raw_response_fn=None, model: str = "claude-sonnet-4-5-20250929"):
        self._raw_response_fn = raw_response_fn or _default_raw_response_fn_factory(model)

    def judge(self, query: str, answer_text: str, passages: list) -> FaithfulnessVerdict:
        prompt = build_faithfulness_prompt(query, answer_text, passages)
        raw_text = self._raw_response_fn(prompt)
        return parse_faithfulness_response(raw_text)