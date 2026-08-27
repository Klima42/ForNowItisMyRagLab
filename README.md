# Atlas RAG

Advanced RAG platform with a React dashboard and FastAPI API. The first slice runs locally with an in-memory deterministic retrieval adapter, so the workflow is usable before connecting Qdrant, embeddings, or an LLM.

## Run locally

Backend (PowerShell):

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Frontend in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Upload PDF, DOCX, Markdown, or TXT files, select vector, lexical, or hybrid retrieval, then ask a question. The API docs are at `http://localhost:8000/docs`.

## Docker

```powershell
docker compose up --build
```

## API surface

- `POST /api/documents`: parse and index files
- `POST /api/search`: retrieve ranked passages with optional metadata filters
- `POST /api/answer`: return an answer plus source citations
- `GET /api/health`: report indexed documents and passages

The adapter boundary in `backend/app/main.py` is intentionally small: replace `InMemoryStore` with Qdrant plus embedding and reranking clients as the project grows, without changing the frontend contract.

The V3 `rerank` option retrieves a wider candidate set, then performs a second deterministic pass that boosts query-term coverage and exact phrase matches before returning the final top K passages.

The V4 `rewrite` option expands a small set of vague questions before retrieval and returns the effective query in the answer payload, making query rewriting inspectable. It is ready to be replaced by an Ollama or cloud LLM adapter later.

The V5 filters let you restrict retrieval to a document type (`pdf`, `docx`, `md`, or `txt`) and/or a year present in the filename or extracted text.

The V6 citations panel makes provenance explicit: each answer includes numbered citations with source filename, section, optional page, and a short supporting quote.

V7 adds `POST /api/evaluate`. Send cases shaped like `{ "question": "...", "relevant_sources": ["security.md"] }` to measure Recall@K, Precision@K, MRR, and average latency for the selected retrieval strategy. The **Run benchmark** button in the dashboard runs a quick benchmark against the current query and top result.

The answer endpoint now uses Ollama when available. Install Ollama, pull a model such as `qwen2.5:3b`, then set `OLLAMA_MODEL=qwen2.5:3b` if needed. Without Ollama, the local grounded evidence fallback remains active.

## Test

```powershell
cd backend
pytest
```
