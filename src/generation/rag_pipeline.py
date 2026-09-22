"""
RAG answer generation with citation tracking.

Same dependency-injection pattern as every other LLM-backed component in
this project (Weeks 5/6): RAGGenerator takes an injectable
`raw_response_fn` (prompt(str) -> raw response text(str)) so prompt
building and citation parsing -- the logic that can actually have bugs --
are fully unit-tested offline with a fake, canned response. The real
Claude API call only happens when no raw_response_fn is injected; see
run_rag_eval.py, which you run on your own machine.
"""
import os
import re
from dataclasses import dataclass


@dataclass
class RAGAnswer:
    answer_text: str  # the model's raw answer, citations left in-line
    cited_indices: list  # sorted, de-duplicated list of 1-indexed passage numbers actually cited


def build_rag_prompt(query: str, passages: list) -> str:
    passage_lines = "\n\n".join(f"Passage {i + 1}:\n{passage}" for i, passage in enumerate(passages))
    return f"""Answer the question using ONLY the information in the passages below. Cite every passage you rely on by its number in square brackets right after the relevant statement, e.g. "The Eiffel Tower was completed in 1889 [2]." If the passages don't contain enough information to answer, say so plainly instead of guessing or using outside knowledge.

Question: {query}

{passage_lines}

Answer:"""


_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_citations(answer_text: str) -> list:
    """Extracts every passage number cited in square brackets, e.g. "[2]"
    or "[1, 3]", and returns a sorted, de-duplicated list of ints. Does
    NOT validate that a cited number is actually a valid passage index --
    that's a separate, deliberate check made downstream in run_rag_eval.py
    (an out-of-range citation is itself a meaningful failure to detect,
    not something to silently filter out here).
    """
    cited = set()
    for match in _CITATION_PATTERN.finditer(answer_text):
        for num in match.group(1).split(","):
            cited.add(int(num.strip()))
    return sorted(cited)


def _default_raw_response_fn_factory(model: str):
    # Check for the API key BEFORE importing `anthropic` -- gives a clear,
    # actionable error instead of a confusing ModuleNotFoundError when
    # neither is set up yet (same pattern as llm_judge.py).
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set. Set it in your environment before running this.")

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def raw_response_fn(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return raw_response_fn


class RAGGenerator:
    """Generates a cited answer for a query given retrieved passages.

    raw_response_fn: injectable stand-in for testing -- a function
        prompt(str) -> raw response text (str). Defaults to a real Claude
        API call (needs `anthropic` + ANTHROPIC_API_KEY -- not available
        in the sandbox).
    """

    def __init__(self, raw_response_fn=None, model: str = "claude-sonnet-4-5-20250929"):
        self._raw_response_fn = raw_response_fn or _default_raw_response_fn_factory(model)

    def answer_query(self, query: str, passages: list) -> RAGAnswer:
        if not passages:
            raise ValueError("Cannot generate an answer with zero passages.")
        prompt = build_rag_prompt(query, passages)
        answer_text = self._raw_response_fn(prompt)
        cited_indices = parse_citations(answer_text)
        return RAGAnswer(answer_text=answer_text, cited_indices=cited_indices)