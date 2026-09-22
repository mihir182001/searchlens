"""
tests/test_streamlit_app.py -- Week 10 test suite for the Streamlit UI in
src/ui/streamlit_app.py. Uses Streamlit's own AppTest framework, which
runs the script in-process and lets tests inspect the resulting widget
tree -- no browser, no `streamlit run` subprocess.

api_client's functions (check_health/search/ask_rag) are patched with
unittest.mock so this never makes a real HTTP call or needs a running API
server -- consistent with every other offline-tested component in this
project. streamlit_app.py imports `api_client` as a module (not
`from api_client import search`), specifically so patching
`src.ui.api_client.search` etc. actually takes effect when the script
looks up the attribute at call time.
"""
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.ui.api_client import APIError

# Resolved as an absolute path -- AppTest.from_file resolves a relative
# path against the file that CALLS it (this test file, in tests/), not
# against the project root, so a bare "src/ui/streamlit_app.py" would
# incorrectly look for tests/src/ui/streamlit_app.py.
APP_PATH = Path(__file__).resolve().parents[1] / "src" / "ui" / "streamlit_app.py"


def make_app():
    at = AppTest.from_file(str(APP_PATH))
    at.run()
    return at


# ---------------------------------------------------------------------------
# Sidebar: connection check
# ---------------------------------------------------------------------------

def test_check_connection_success_shows_success_message():
    at = make_app()
    with patch("src.ui.api_client.check_health", return_value=True):
        at.button(key="check_health_button").click().run()
    assert any("reachable" in s.value for s in at.success)
    assert len(at.error) == 0


def test_check_connection_failure_shows_error_message():
    at = make_app()
    with patch("src.ui.api_client.check_health", side_effect=APIError("Could not reach the API at http://x: refused")):
        at.button(key="check_health_button").click().run()
    assert any("Could not reach" in e.value for e in at.error)


# ---------------------------------------------------------------------------
# Search mode (default mode)
# ---------------------------------------------------------------------------

def test_search_empty_query_shows_error():
    at = make_app()
    at.button(key="search_button").click().run()
    assert any("enter a query" in e.value for e in at.error)


def test_search_success_shows_metrics_and_results():
    at = make_app()
    at.text_input(key="search_query").set_value("what is the capital of france").run()

    fake_response = {
        "query": "what is the capital of france",
        "intent": "factual",
        "confidence": 0.9881821274757385,
        "method": "dense",
        "low_confidence": False,
        "results": [
            {"pid": "4295731", "score": 0.752997636795044, "text": "Paris is the capital of France."},
            {"pid": "7763891", "score": 0.5816375017166138, "text": "Paris, France Lat Long Coordinates Info."},
        ],
    }
    with patch("src.ui.api_client.search", return_value=fake_response) as mock_search:
        at.button(key="search_button").click().run()

    mock_search.assert_called_once()
    call_args = mock_search.call_args
    assert call_args.args[1] == "what is the capital of france" or call_args.kwargs.get("query") == "what is the capital of france"

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Intent"] == "Factual"
    assert metrics["Confidence"] == "98.8%"
    assert metrics["Retrieval method"] == "DENSE"

    assert len(at.warning) == 0  # not low_confidence
    assert any("Results (2)" in s.value for s in at.subheader)
    assert any("4295731" in m.value for m in at.markdown)
    assert any("Paris is the capital of France." in m.value for m in at.markdown)


def test_search_low_confidence_shows_warning_banner():
    at = make_app()
    at.text_input(key="search_query").set_value("mysql vs postgresql").run()

    fake_response = {
        "query": "mysql vs postgresql", "intent": "comparison", "confidence": 0.99,
        "method": "dense", "low_confidence": True,
        "results": [{"pid": "3549616", "score": 0.3668, "text": "Database languages..."}],
    }
    with patch("src.ui.api_client.search", return_value=fake_response):
        at.button(key="search_button").click().run()

    assert any("Low confidence" in w.value for w in at.warning)


def test_search_no_results_shows_info_message():
    at = make_app()
    at.text_input(key="search_query").set_value("something obscure").run()

    fake_response = {
        "query": "something obscure", "intent": "factual", "confidence": 0.5,
        "method": "dense", "low_confidence": True, "results": [],
    }
    with patch("src.ui.api_client.search", return_value=fake_response):
        at.button(key="search_button").click().run()

    assert any("No results found" in i.value for i in at.info)


def test_search_api_error_shows_error_message():
    at = make_app()
    at.text_input(key="search_query").set_value("anything").run()

    with patch("src.ui.api_client.search", side_effect=APIError("Could not reach the API at http://x: refused")):
        at.button(key="search_button").click().run()

    assert any("Could not reach" in e.value for e in at.error)


def test_search_top_k_slider_is_passed_through():
    at = make_app()
    at.text_input(key="search_query").set_value("hello").run()
    at.slider(key="search_top_k").set_value(25).run()

    fake_response = {
        "query": "hello", "intent": "factual", "confidence": 0.9,
        "method": "dense", "low_confidence": False, "results": [],
    }
    with patch("src.ui.api_client.search", return_value=fake_response) as mock_search:
        at.button(key="search_button").click().run()

    _, kwargs = mock_search.call_args
    assert kwargs.get("top_k") == 25


# ---------------------------------------------------------------------------
# RAG mode
# ---------------------------------------------------------------------------

def switch_to_rag_mode(at):
    at.radio(key="mode").set_value("Ask (RAG)").run()
    return at


def test_rag_empty_query_shows_error():
    at = make_app()
    switch_to_rag_mode(at)
    at.button(key="rag_button").click().run()
    assert any("enter a question" in e.value for e in at.error)


def test_rag_success_shows_answer_and_source_passages():
    at = make_app()
    switch_to_rag_mode(at)
    at.text_input(key="rag_query").set_value("when was the eiffel tower built").run()

    fake_response = {
        "query": "when was the eiffel tower built",
        "answer": "It was built in 1889 [1].",
        "cited_indices": [1],
        "low_confidence": False,
        "passages": [
            {"pid": "p1", "score": 0.9, "text": "The Eiffel Tower was completed in 1889."},
            {"pid": "p2", "score": 0.4, "text": "Paris landmarks include the Eiffel Tower."},
        ],
    }
    with patch("src.ui.api_client.ask_rag", return_value=fake_response):
        at.button(key="rag_button").click().run()

    assert any("It was built in 1889 [1]." in i.value for i in at.info)
    assert any("Cited passage numbers: 1" in c.value for c in at.caption)
    assert any("Source passages (2)" in s.value for s in at.subheader)
    assert len(at.warning) == 0


def test_rag_low_confidence_shows_warning_banner():
    at = make_app()
    switch_to_rag_mode(at)
    at.text_input(key="rag_query").set_value("mysql vs postgresql").run()

    fake_response = {
        "query": "mysql vs postgresql", "answer": "The passages do not cover this.",
        "cited_indices": [], "low_confidence": True,
        "passages": [{"pid": "p1", "score": 0.1, "text": "unrelated text"}],
    }
    with patch("src.ui.api_client.ask_rag", return_value=fake_response):
        at.button(key="rag_button").click().run()

    assert any("Low confidence" in w.value for w in at.warning)
    assert any("did not cite" in c.value for c in at.caption)


def test_rag_service_unavailable_shows_helpful_error():
    at = make_app()
    switch_to_rag_mode(at)
    at.text_input(key="rag_query").set_value("anything").run()

    with patch(
        "src.ui.api_client.ask_rag",
        side_effect=APIError("RAG isn't enabled on this server (no ANTHROPIC_API_KEY was set when it started)."),
    ):
        at.button(key="rag_button").click().run()

    assert any("ANTHROPIC_API_KEY" in e.value for e in at.error)


def test_switching_modes_hides_the_other_mode_widgets():
    at = make_app()
    assert any(w.key == "search_query" for w in at.text_input)
    switch_to_rag_mode(at)
    assert any(w.key == "rag_query" for w in at.text_input)
    # The Search-mode widgets should not be rendered while in RAG mode.
    assert not any(w.key == "search_query" for w in at.text_input)
