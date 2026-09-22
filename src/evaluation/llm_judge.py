"""
Claude-as-judge scoring of (query, passage)
relevance on a 1-5 scale.

Same dependency-injection pattern as Weeks 2/3/5's model-backed
components: LLMJudge takes an injectable `raw_response_fn` (a function
prompt(str) -> raw response text(str)) so prompt-building and
response-parsing -- the actual logic that can have bugs -- are fully
unit-tested offline with a fake, canned response, without ever calling
the real Claude API. See tests/test_llm_judge.py.

The real API call (via the `anthropic` package + your ANTHROPIC_API_KEY)
only happens when no raw_response_fn is injected; that path is exercised
by run_llm_judge_eval.py, which you run on your own machine.
"""
import json
import os
from dataclasses import dataclass


@dataclass
class Judgment:
    passage_index: int  # position within the batch passed to score_query
    score: int  # 1-5


def build_judge_prompt(query: str, passages: list) -> str:
    passage_lines = "\n\n".join(f"Passage {i + 1}:\n{passage}" for i, passage in enumerate(passages))
    return f"""You are judging search result relevance. Rate how relevant each passage is to the query, on a scale of 1-5:
1 = not relevant at all
2 = slightly relevant
3 = somewhat relevant
4 = relevant
5 = highly relevant, directly answers the query

Query: {query}

{passage_lines}

Return ONLY a JSON array of {len(passages)} integers (one score per passage, in the same order as above), nothing else. Example format: [3, 5, 1, 2, 4]
"""


def parse_judge_response(text: str, expected_count: int) -> list:
    """Parses the model's raw reply into a list of int scores, validating
    both the count (must match expected_count) and the range (1-5 each).
    Strips a code fence if the model wraps its JSON in one despite
    instructions not to -- same defensive pattern as
    expand_query_intent_data_with_claude.py's JSON parsing.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
        if text.lower().startswith("json"):
            text = text[len("json") :]

    scores = json.loads(text)
    if not isinstance(scores, list):
        raise ValueError(f"Expected a JSON array of scores, got: {type(scores)}")
    if len(scores) != expected_count:
        raise ValueError(f"Expected {expected_count} scores, got {len(scores)}: {scores}")

    result = []
    for raw_score in scores:
        score = int(raw_score)
        if not (1 <= score <= 5):
            raise ValueError(f"Score {score} out of range [1, 5]: {scores}")
        result.append(score)
    return result


def _default_raw_response_fn_factory(model: str):
    # Check for the API key BEFORE importing `anthropic` -- this package
    # isn't installed in the sandbox (or possibly not yet on your machine
    # either), and checking the key first gives a clear, actionable error
    # instead of a confusing ModuleNotFoundError when neither is set up.
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set. Set it in your environment before running this.")

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def raw_response_fn(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return raw_response_fn


class LLMJudge:
    """Scores a query's candidate passages for relevance via an LLM, one
    API call per query (all of that query's candidates scored together in
    a single prompt, not one call per passage -- see the module docstring
    of run_llm_judge_eval.py for why that matters for cost/time).

    raw_response_fn: injectable stand-in for testing -- a function
        prompt(str) -> raw response text (str). Defaults to a real Claude
        API call (needs `anthropic` + ANTHROPIC_API_KEY -- not available
        in the sandbox).
    """

    def __init__(self, raw_response_fn=None, model: str = "claude-sonnet-4-5-20250929"):
        self._raw_response_fn = raw_response_fn or _default_raw_response_fn_factory(model)

    def score_query(self, query: str, passages: list) -> list:
        if not passages:
            return []
        prompt = build_judge_prompt(query, passages)
        raw_text = self._raw_response_fn(prompt)
        return parse_judge_response(raw_text, expected_count=len(passages))