# SearchLens: A Neural Information Retrieval and Ranking Platform

A neural information retrieval and ranking platform: BM25 and dense (embedding) retrieval, hybrid fusion, a learned query-intent router, retrieval-augmented generation with citations, and a real A/B testing framework, wrapped behind a FastAPI service and a Streamlit UI, with Docker images and CI.

Built incrementally over 11 weeks, each one adding a real, tested capability on top of the last. The interesting part isn't the feature list. It's that several of this project's design decisions changed because of what its own evaluation code found, not because of how the spec originally described them. Those findings are in the section below.

## What It Does

Given a query, SearchLens:

1. Classifies its **intent** (factual, navigational, exploratory, comparison, or how-to) with a fine-tuned DistilBERT classifier.
2. **Routes** it to whichever retrieval method that intent has been measured to work best with (BM25 lexical retrieval or dense embedding retrieval), rather than always using one method or an untested guess.
3. Retrieves the top-k passages, and flags the result as **low confidence** if the retrieval scores suggest the corpus doesn't actually cover the topic.
4. Optionally answers the query in natural language via **RAG** (retrieval-augmented generation), with inline citations back to the specific passages used, and refuses to answer when the passages don't support one.

All of it is reachable through a FastAPI service (`/health`, `/search`, `/rag`) and a Streamlit UI on top of it.

## What This Project Actually Found

Most of this project's more interesting decisions came from running its own evaluation code and believing the result, even when that meant reversing an earlier assumption.

**Hybrid retrieval (RRF fusion of BM25 and dense) is significantly worse than dense retrieval alone on this corpus.** This isn't a guess; it comes from a paired bootstrap significance test (`src/evaluation/ab_test.py`) run over the same queries: MRR@10 of 0.9130 for dense versus 0.8980 for hybrid, with a 95% confidence interval on the difference of [-0.0252, -0.0052] and p=0.0039. The root cause is that Reciprocal Rank Fusion still incorporates a low-weighted BM25 vote's rank position, so a confidently wrong lexical match can demote a passage dense retrieval already ranked correctly. The routing table (`src/retrieval/query_router.py`) was rewritten to route exploratory and comparison queries to dense-only, not hybrid, as a direct result. The hybrid code path itself is kept and still tested, in case future per-intent evaluation ever justifies switching back.

**RAG faithfulness measured at 98%** over a 100-query real evaluation run, using an LLM-as-judge (`src/generation/faithfulness_judge.py`) rather than a spot check. The raw failure records were inspected by hand rather than only trusting the aggregate score.

**A real corpus coverage gap, found by actually using the API.** A live query for "mysql vs postgresql" returned only weak, generic results (top score 0.37, a passage about "database languages" that wasn't really an answer), because the corpus genuinely contains zero passages mentioning either term, confirmed by grepping the raw corpus rather than assumed. This became the `low_confidence` field on `/search` and `/rag`: a dense-retrieval score threshold calibrated from real observed scores (0.37 as a confirmed weak match, 0.75 as a confirmed strong match), explicitly not set for BM25, whose raw score has no natural bound to calibrate against. The first version of that threshold, 0.35, was a genuine bug: it sat below the 0.37 weak-match score it was built to catch, so it could never fire. It was caught only by re-running the live query after shipping the feature, not by re-reading the code. It's now fixed to 0.5, with a regression test built from the exact real floating-point scores involved so this can't silently regress.

**`requirements.txt` had drifted from what the project actually needed.** It listed `pandas` (grepped across the whole codebase; never imported anywhere) but not `faiss-cpu`, `sentence-transformers`, `torch`, `transformers`, or `anthropic`, even though the API's real startup path genuinely needs all five. They'd been installed by hand over several weeks and never written back to the file, invisible until Docker's from-scratch build environment would have hit it immediately. This was fixed by tracing the actual import chain and splitting dependencies per service (`requirements-api.txt` and `requirements-ui.txt`; the Streamlit UI needs neither PyTorch nor FAISS, since it only ever calls the API over HTTP), with every version pinned against a real from-scratch install rather than copied from a working machine's `pip freeze` (an earlier pass of those pins was generated under Python 3.11 and silently included a NumPy version that doesn't exist for Python 3.10 at all).


Everything above is served by **FastAPI** (`src/api/`, Week 9) and consumed by a **Streamlit** UI (`src/ui/`, Week 10) that talks to it only over HTTP. The UI never imports a model, an index, or the classifier directly, so retrieval and generation logic lives in exactly one place.

## What Each Week Actually Built

| Week | What it added | Files |
|---|---|---|
| 1 | BM25 baseline over a synthetic MS MARCO style corpus, MRR@10/Recall@k metrics | `src/retrieval/bm25_index.py`, `src/data/generate_synthetic_msmarco.py`, `src/evaluation/metrics.py` |
| 2 | Dense (embedding) retrieval with FAISS | `src/retrieval/dense_index.py` |
| 3 | Cross-encoder reranking | `src/retrieval/cross_encoder_rerank.py` |
| 4 | Hybrid retrieval via Reciprocal Rank Fusion | `src/retrieval/hybrid_rrf.py` |
| 5 | Fine-tuned DistilBERT query-intent classifier and routing table | `src/models/query_intent_classifier.py`, `src/retrieval/query_router.py` |
| 6 | LLM-as-judge relevance evaluation | `src/evaluation/llm_judge.py`, `src/evaluation/build_judge_dataset.py` |
| 7 | RAG pipeline with citations and a faithfulness judge | `src/generation/rag_pipeline.py`, `src/generation/faithfulness_judge.py` |
| 8 | Paired bootstrap A/B significance testing, used to fix Week 5's routing table | `src/evaluation/ab_test.py`, `src/evaluation/run_ab_test.py` |
| 9 | FastAPI service (`/health`, `/search`, `/rag`) and low confidence flagging | `src/api/app.py`, `src/api/run_api_server.py` |
| 10 | Streamlit UI as a thin HTTP client over the API | `src/ui/streamlit_app.py`, `src/ui/api_client.py` |
| 11 | Docker images (API and UI split, matching the thin-client architecture) and GitHub Actions CI | `Dockerfile.api`, `Dockerfile.ui`, `docker-compose.yml`, `.github/workflows/ci.yml` |

## Why a Synthetic Corpus

This project was built where the retrieval and model code couldn't reach `huggingface.co` or the real Anthropic API, so `src/data/generate_synthetic_msmarco.py` builds a small corpus that mimics MS MARCO's structure (passages, queries, qrels) with deliberate vocabulary mismatch (for example "heart attack" in a passage, "myocardial infarction" in the paraphrased query that's relevant to it). That's enough to make BM25's lexical-matching weakness and dense retrieval's semantic-matching strength both real and measurable, without needing real MS MARCO. `src/data/load_real_msmarco.py` exists to swap in the real dataset on a machine that can reach Hugging Face.

Every model-training and live-API script in this repo is explicit about this in its own docstring, and reports what it actually measured rather than a target number.

## Running It

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

(Use `source .venv/bin/activate` instead on macOS/Linux.)

### Tests

Fully offline; no data, model weights, or API key needed.

```bash
python -m pytest tests/ -v
```

189 tests, all passing against injected fakes for every real model or API call (`DenseIndex`'s encoder, `QueryIntentClassifier`'s `predict_proba_fn`, `RAGGenerator`'s `raw_response_fn`, the HTTP layer via FastAPI's `TestClient`, the Streamlit UI via `streamlit.testing.v1.AppTest`). Verified to still pass from a completely empty `data/` directory, which is exactly what GitHub Actions CI runs on every push.

### Training the Real Models

One-time, needs Hugging Face access.

```bash
python -m src.models.train_query_intent_classifier
```

### Running the API and UI for Real

```bash
python -m src.api.run_api_server
```

In a second terminal:

```bash
streamlit run src/ui/streamlit_app.py
```

Set `ANTHROPIC_API_KEY` in your environment before starting the API to enable `/rag`. Without it, `/search` still works and `/rag` returns a clean 503 rather than crashing.

### Running It With Docker

```bash
docker compose up --build
```

Builds and runs the API and UI as separate containers. The UI's image has no PyTorch or FAISS in it at all, since it only ever calls the API over HTTP. Mount your own `data/` for the corpus, cached dense index, and trained classifier; see `Dockerfile.api`'s comments for why those aren't baked into the image itself.

## Tech Stack

Python, `rank_bm25`, FAISS, `sentence-transformers`, PyTorch, `transformers` (DistilBERT), the Anthropic API for RAG generation and LLM judging, FastAPI, Streamlit, Docker, GitHub Actions.