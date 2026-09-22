"""
api_client.py -- Week 10: thin HTTP client wrapping SearchLens's Week 9
FastAPI service (src/api/app.py), used by the Streamlit UI
(streamlit_app.py).

Kept as its own module, separate from streamlit_app.py, for the same
reason every other API-backed component in this project is split this
way: the actual HTTP logic (building requests, handling connection
failures, turning non-2xx responses into clear, user-facing messages) is
unit-tested offline here (tests/test_api_client.py) with a fake
requests-like session, independent of Streamlit's own rendering, which is
tested separately with Streamlit's AppTest framework
(tests/test_streamlit_app.py) against a fake api_client.

Every function takes an explicit `base_url` and an injectable `session`
(defaults to the `requests` module itself, which exposes .get/.post the
same way a requests.Session does) so tests never make a real network call
and the real `requests` package is only exercised when you actually run
this against a live server.
"""
import requests


class APIError(Exception):
    """Raised for anything that isn't a clean, expected JSON response --
    connection failure, timeout, or a non-2xx status the UI should show
    to the person using it rather than let bubble up as a raw traceback.
    """


def check_health(base_url: str, session=requests, timeout: float = 5.0) -> bool:
    try:
        response = session.get(f"{base_url}/health", timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise APIError(f"Could not reach the API at {base_url}: {exc}") from exc
    return response.status_code == 200


def search(base_url: str, query: str, top_k: int = 10, session=requests, timeout: float = 30.0) -> dict:
    try:
        response = session.post(
            f"{base_url}/search", json={"query": query, "top_k": top_k}, timeout=timeout,
        )
    except requests.exceptions.RequestException as exc:
        raise APIError(f"Could not reach the API at {base_url}: {exc}") from exc

    if response.status_code == 422:
        raise APIError("The API rejected this request (empty query or invalid top_k).")
    if not response.ok:
        raise APIError(f"API returned HTTP {response.status_code}: {response.text}")
    return response.json()


def ask_rag(base_url: str, query: str, session=requests, timeout: float = 60.0) -> dict:
    try:
        response = session.post(f"{base_url}/rag", json={"query": query}, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise APIError(f"Could not reach the API at {base_url}: {exc}") from exc

    if response.status_code == 503:
        raise APIError(
            "RAG isn't enabled on this server (no ANTHROPIC_API_KEY was set when it started)."
        )
    if response.status_code == 404:
        raise APIError("No passages were found for this query.")
    if response.status_code == 422:
        raise APIError("The API rejected this request (empty query).")
    if not response.ok:
        raise APIError(f"API returned HTTP {response.status_code}: {response.text}")
    return response.json()
