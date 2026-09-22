"""
run_api_server.py -- Week 9 end-to-end script. RUN THIS ON YOUR OWN MACHINE.

    uvicorn src.api.run_api_server:app --reload
    (or)
    python -m src.api.run_api_server

Wires up the REAL components -- BM25Index, your cached DenseIndex, your
fine-tuned QueryIntentClassifier (Week 5), and (if ANTHROPIC_API_KEY is
set) a RAGGenerator (Week 7) -- behind the FastAPI app defined in
src/api/app.py. None of this is testable in the sandbox: it needs real
HF model weights, your actual trained classifier checkpoint on disk, and
optionally a live Anthropic API call -- the same caveat as every other
run_*.py script in this project. The HTTP layer itself (routing,
validation, error handling) is already fully tested offline in
tests/test_api.py with fake stand-ins for every component.

Once running, try it at http://127.0.0.1:8000/docs -- FastAPI's automatic
interactive API docs (Swagger UI), where you can fire requests from the
browser without curl.

    curl http://127.0.0.1:8000/health

    curl -X POST http://127.0.0.1:8000/search ^
      -H "Content-Type: application/json" ^
      -d "{\"query\": \"what is the capital of france\", \"top_k\": 5}"

    curl -X POST http://127.0.0.1:8000/rag ^
      -H "Content-Type: application/json" ^
      -d "{\"query\": \"when was the eiffel tower built\"}"

(The ^ line-continuation above is PowerShell/cmd syntax -- use \ instead on
macOS/Linux, or just paste each curl command as one line.)

RAG needs ANTHROPIC_API_KEY set in your environment before starting the
server (same as Week 7's run_rag_eval.py). If it's not set, /rag returns
HTTP 503 rather than crashing at startup, so /search and /health still
work without an API key -- start the server without one first to confirm
routing works, then set the key and restart for /rag.
"""
import os
from pathlib import Path

from src.data.loader import load_passages
from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex
from src.models.query_intent_classifier import QueryIntentClassifier
from src.api.app import create_app

DENSE_INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "dense_index"
INTENT_MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "query_intent_model"


def build_app():
    passages = load_passages()
    print(f"Loaded {len(passages)} passages.")

    print("Building BM25 index...")
    bm25_index = BM25Index(passages)

    if (DENSE_INDEX_DIR / "dense_index.faiss").exists():
        print(f"Loading cached dense index from {DENSE_INDEX_DIR} ...")
        dense_index = DenseIndex.load(DENSE_INDEX_DIR)
    else:
        print("No cached dense index found -- encoding corpus (slow, one-time)...")
        dense_index = DenseIndex(passages)
        dense_index.save(DENSE_INDEX_DIR)

    if not INTENT_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"No trained query intent model at {INTENT_MODEL_DIR} -- run "
            "`python -m src.models.train_query_intent_classifier` (Week 5) first."
        )
    print(f"Loading query intent classifier from {INTENT_MODEL_DIR} ...")
    classifier = QueryIntentClassifier(model_dir=str(INTENT_MODEL_DIR))

    rag_generator = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        from src.generation.rag_pipeline import RAGGenerator
        rag_generator = RAGGenerator()
        print("ANTHROPIC_API_KEY found -- /rag endpoint enabled.")
    else:
        print("No ANTHROPIC_API_KEY set -- /rag endpoint will return 503 until you set one.")

    return create_app(bm25_index, dense_index, classifier, passages, rag_generator=rag_generator)


app = build_app()  # module-level `app` -- what `uvicorn src.api.run_api_server:app` expects


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
