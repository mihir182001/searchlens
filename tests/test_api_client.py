"""
tests/test_api_client.py -- Week 10 test suite for the HTTP client in
src/ui/api_client.py. Uses a fake requests-like session (duck-typed the
same way every other injectable dependency in this project is faked), so
this runs fully offline -- no real network call, no running API server.
"""
import pytest
import requests

from src.ui.api_client import check_health, search, ask_rag, APIError


class _FakeResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._json_data


class _FakeSession:
    """Stand-in for `requests` (or a requests.Session) -- records calls and
    returns a canned response, or raises a canned connection error.
    """

    def __init__(self, get_response=None, post_response=None, raise_exc=None):
        self.get_response = get_response
        self.post_response = post_response
        self.raise_exc = raise_exc
        self.get_calls = []
        self.post_calls = []

    def get(self, url, timeout=None):
        if self.raise_exc:
            raise self.raise_exc
        self.get_calls.append((url, timeout))
        return self.get_response

    def post(self, url, json=None, timeout=None):
        if self.raise_exc:
            raise self.raise_exc
        self.post_calls.append((url, json, timeout))
        return self.post_response


# ---------------------------------------------------------------------------
# check_health
# ---------------------------------------------------------------------------

def test_check_health_returns_true_on_200():
    session = _FakeSession(get_response=_FakeResponse(200))
    assert check_health("http://x", session=session) is True
    assert session.get_calls == [("http://x/health", 5.0)]


def test_check_health_returns_false_on_non_200():
    session = _FakeSession(get_response=_FakeResponse(500))
    assert check_health("http://x", session=session) is False


def test_check_health_raises_api_error_on_connection_failure():
    session = _FakeSession(raise_exc=requests.exceptions.ConnectionError("refused"))
    with pytest.raises(APIError, match="Could not reach"):
        check_health("http://x", session=session)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_posts_correct_payload_and_returns_json():
    body = {"query": "q", "intent": "factual", "confidence": 0.9, "method": "dense",
             "low_confidence": False, "results": []}
    session = _FakeSession(post_response=_FakeResponse(200, json_data=body))
    result = search("http://x", "q", top_k=7, session=session)
    assert result == body
    assert session.post_calls == [("http://x/search", {"query": "q", "top_k": 7}, 30.0)]


def test_search_raises_on_422():
    session = _FakeSession(post_response=_FakeResponse(422, text="bad request"))
    with pytest.raises(APIError, match="rejected"):
        search("http://x", "", session=session)


def test_search_raises_on_other_non_ok_status():
    session = _FakeSession(post_response=_FakeResponse(500, text="boom"))
    with pytest.raises(APIError, match="HTTP 500"):
        search("http://x", "q", session=session)


def test_search_raises_api_error_on_connection_failure():
    session = _FakeSession(raise_exc=requests.exceptions.Timeout("slow"))
    with pytest.raises(APIError, match="Could not reach"):
        search("http://x", "q", session=session)


# ---------------------------------------------------------------------------
# ask_rag
# ---------------------------------------------------------------------------

def test_ask_rag_posts_correct_payload_and_returns_json():
    body = {"query": "q", "answer": "a", "cited_indices": [1], "low_confidence": False, "passages": []}
    session = _FakeSession(post_response=_FakeResponse(200, json_data=body))
    result = ask_rag("http://x", "q", session=session)
    assert result == body
    assert session.post_calls == [("http://x/rag", {"query": "q"}, 60.0)]


def test_ask_rag_raises_on_503_with_helpful_message():
    session = _FakeSession(post_response=_FakeResponse(503))
    with pytest.raises(APIError, match="ANTHROPIC_API_KEY"):
        ask_rag("http://x", "q", session=session)


def test_ask_rag_raises_on_404():
    session = _FakeSession(post_response=_FakeResponse(404))
    with pytest.raises(APIError, match="No passages"):
        ask_rag("http://x", "q", session=session)


def test_ask_rag_raises_on_422():
    session = _FakeSession(post_response=_FakeResponse(422))
    with pytest.raises(APIError, match="rejected"):
        ask_rag("http://x", "", session=session)


def test_ask_rag_raises_on_other_non_ok_status():
    session = _FakeSession(post_response=_FakeResponse(500, text="boom"))
    with pytest.raises(APIError, match="HTTP 500"):
        ask_rag("http://x", "q", session=session)


def test_ask_rag_raises_api_error_on_connection_failure():
    session = _FakeSession(raise_exc=requests.exceptions.ConnectionError("refused"))
    with pytest.raises(APIError, match="Could not reach"):
        ask_rag("http://x", "q", session=session)
