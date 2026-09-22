"""
streamlit_app.py -- Week 10: Streamlit front end for the Week 9 FastAPI
service (src/api/app.py). RUN THIS ON YOUR OWN MACHINE, alongside a
running API server:

    (terminal 1) python -m src.api.run_api_server
    (terminal 2) streamlit run src/ui/streamlit_app.py

This is a thin presentation layer only -- it never touches BM25Index,
DenseIndex, the classifier, or the RAG generator directly. Every piece of
retrieval/routing/generation logic already lives behind the Week 9 API,
and this app just calls it over HTTP through api_client.py. That keeps
retrieval/generation logic in exactly one place (the API) instead of
duplicated between the API and the UI, and it keeps the UI itself testable
offline: tests/test_streamlit_app.py drives this file with Streamlit's own
AppTest framework, with api_client's functions replaced by fakes, so no
real server, model, or network call is needed to test the UI's logic
(what renders, what's shown on error, whether the low-confidence banner
appears).

`streamlit run` executes this file directly, putting only this file's own
folder (src/ui) on sys.path -- not the project root. Without the block
below, `from src.ui import api_client` fails with
`ModuleNotFoundError: No module named 'src'` the moment Streamlit (rather
than pytest, which adds the project root itself) is the one running this
file. Inserting the project root at the top, before that import, makes
this script work no matter how it's launched.

Week 11: the default API URL is now read from an API_BASE_URL environment
variable (falling back to the same 127.0.0.1:8000 as before). Running the
API and UI directly on your own machine, neither container, needs nothing
set -- the fallback still applies. Running both under docker-compose,
127.0.0.1:8000 inside the UI container's own network namespace does NOT
reach the API container; docker-compose.yml sets API_BASE_URL=http://api:8000
so the UI finds the API by its Compose service name instead.
"""
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.ui import api_client

st.set_page_config(page_title="SearchLens", page_icon="🔍", layout="wide")

DEFAULT_API_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL


def render_low_confidence_warning():
    st.warning(
        "Low confidence: the retrieved passages don't look strongly relevant to "
        "this query, possibly because the corpus doesn't cover this topic well. "
        "Treat the results below with that in mind rather than as a confident answer.",
        icon="⚠️",
    )


def render_search_result(result: dict, rank: int):
    score = result["score"]
    pid = result["pid"]
    text = result.get("text") or "(no text available for this passage)"
    st.markdown(f"**{rank}. Score: {score:.3f}**  ·  passage `{pid}`")
    st.write(text)
    st.divider()


# ---------------------------------------------------------------------------
# Sidebar: API connection + mode
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔍 SearchLens")
    st.caption("Hybrid retrieval, query intent routing, and RAG over MS MARCO.")

    st.session_state.api_url = st.text_input(
        "API base URL",
        value=st.session_state.api_url,
        key="api_url_input",
        help="Where the Week 9 FastAPI server (run_api_server.py) is running.",
    )

    if st.button("Check connection", key="check_health_button"):
        try:
            healthy = api_client.check_health(st.session_state.api_url)
        except api_client.APIError as exc:
            st.session_state.health_status = ("error", str(exc))
        else:
            st.session_state.health_status = (
                ("ok", None) if healthy else ("error", "API responded, but the health check did not return OK.")
            )

    status = st.session_state.get("health_status")
    if status:
        kind, message = status
        if kind == "ok":
            st.success("API is reachable.")
        else:
            st.error(message)

    st.divider()
    mode = st.radio("Mode", ["Search", "Ask (RAG)"], key="mode")

    st.divider()
    st.caption(
        "Search classifies your query's intent and routes it to whichever "
        "retrieval method the project's own A/B testing found works best. "
        "Ask (RAG) additionally generates a cited answer from the top passages."
    )


# ---------------------------------------------------------------------------
# Search mode
# ---------------------------------------------------------------------------
if mode == "Search":
    st.header("Search")
    query = st.text_input("Enter a query", key="search_query")
    top_k = st.slider("Number of results", min_value=1, max_value=50, value=10, key="search_top_k")

    if st.button("Search", key="search_button", type="primary"):
        if not query.strip():
            st.error("Please enter a query.")
        else:
            try:
                response = api_client.search(st.session_state.api_url, query, top_k=top_k)
            except api_client.APIError as exc:
                st.error(str(exc))
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Intent", response["intent"].replace("-", " ").capitalize())
                col2.metric("Confidence", f"{response['confidence']:.1%}")
                col3.metric("Retrieval method", response["method"].upper())

                if response.get("low_confidence"):
                    render_low_confidence_warning()

                results = response.get("results", [])
                if not results:
                    st.info("No results found.")
                else:
                    st.subheader(f"Results ({len(results)})")
                    for rank, result in enumerate(results, start=1):
                        render_search_result(result, rank)

# ---------------------------------------------------------------------------
# RAG mode
# ---------------------------------------------------------------------------
else:
    st.header("Ask (RAG)")
    st.caption("Answers are generated only from retrieved passages, with inline citations.")
    rag_query = st.text_input("Ask a question", key="rag_query")

    if st.button("Ask", key="rag_button", type="primary"):
        if not rag_query.strip():
            st.error("Please enter a question.")
        else:
            try:
                response = api_client.ask_rag(st.session_state.api_url, rag_query)
            except api_client.APIError as exc:
                st.error(str(exc))
            else:
                if response.get("low_confidence"):
                    render_low_confidence_warning()

                st.subheader("Answer")
                st.info(response["answer"])

                cited_indices = response.get("cited_indices", [])
                passages = response.get("passages", [])

                if cited_indices:
                    st.caption(f"Cited passage numbers: {', '.join(str(i) for i in cited_indices)}")
                else:
                    st.caption("The answer did not cite any specific passage.")

                st.subheader(f"Source passages ({len(passages)})")
                for i, passage in enumerate(passages, start=1):
                    marker = "cited" if i in cited_indices else "not cited"
                    st.markdown(f"**Passage {i}** ({marker})  ·  score {passage['score']:.3f}")
                    st.write(passage.get("text") or "(no text available for this passage)")
                    st.divider()
