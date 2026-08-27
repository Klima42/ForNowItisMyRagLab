from fastapi.testclient import TestClient

from app.main import app, store

client = TestClient(app)


def setup_function() -> None:
    store.chunks.clear()


def test_ingest_search_and_citations() -> None:
    response = client.post("/api/documents", files={"files": ("security.md", b"Authentication tokens expire after 24 hours.\n\nAdmins can revoke tokens.", "text/markdown")})
    assert response.status_code == 200
    assert response.json()["total_chunks"] == 2

    search = client.post("/api/search", json={"query": "token expire", "mode": "hybrid", "top_k": 3})
    assert search.status_code == 200
    assert search.json()[0]["source"] == "security.md"

    answer = client.post("/api/answer", json={"query": "token expire", "mode": "hybrid"})
    assert answer.json()["citations"][0]["source"] == "security.md"


def test_metadata_filter() -> None:
    client.post("/api/documents", files={"files": ("guide.txt", b"Public deployment guide", "text/plain")})
    response = client.post("/api/search", json={"query": "deployment", "document_type": "pdf"})
    assert response.json() == []


def test_hybrid_fuses_rankings_and_normalizes_score() -> None:
    client.post("/api/documents", files=[
        ("files", ("exact.md", b"JWT token rotation policy for administrators", "text/markdown")),
        ("files", ("related.md", b"Administrators can manage token access", "text/markdown")),
    ])
    response = client.post("/api/search", json={"query": "JWT token rotation", "mode": "hybrid", "top_k": 2})
    results = response.json()
    assert results[0]["source"] == "exact.md"
    assert results[0]["score"] == 1
    assert len(results) == 2


def test_rerank_returns_only_requested_top_k() -> None:
    client.post("/api/documents", files=[
        ("files", ("one.md", b"Authentication token expires after one day", "text/markdown")),
        ("files", ("two.md", b"Authentication token can be revoked by admins", "text/markdown")),
    ])
    response = client.post("/api/search", json={"query": "authentication token", "mode": "hybrid", "rerank": True, "top_k": 1})
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_rewrite_expands_vague_query() -> None:
    client.post("/api/documents", files={"files": ("admin.md", b"Administrators have permissions to manage access.", "text/markdown")})
    response = client.post("/api/answer", json={"query": "what about admins", "rewrite": True})
    payload = response.json()
    assert "administrator" in payload["rewritten_query"]
    assert payload["citations"][0]["source"] == "admin.md"


def test_metadata_filters_select_matching_year_and_type() -> None:
    client.post("/api/documents", files=[
        ("files", ("manual-2026.pdf", b"Authentication policy for 2026", "application/pdf")),
        ("files", ("manual-2025.pdf", b"Authentication policy for 2025", "application/pdf")),
    ])
    response = client.post("/api/search", json={"query": "authentication policy", "document_type": "pdf", "year": 2026})
    results = response.json()
    assert len(results) == 1
    assert results[0]["source"] == "manual-2026.pdf"


def test_citations_include_number_section_and_quote() -> None:
    client.post("/api/documents", files={"files": ("security.md", b"Tokens expire after 24 hours.", "text/markdown")})
    payload = client.post("/api/answer", json={"query": "tokens expire"}).json()
    citation = payload["citations"][0]
    assert citation["citation_id"] == "[1]"
    assert citation["section"] == "Block 1"
    assert citation["quote"] == "Tokens expire after 24 hours."


def test_evaluation_returns_retrieval_metrics() -> None:
    client.post("/api/documents", files={"files": ("security.md", b"Authentication tokens expire after 24 hours.", "text/markdown")})
    payload = client.post("/api/evaluate", json={"mode": "hybrid", "top_k": 3, "cases": [{"question": "authentication tokens", "relevant_sources": ["security.md"]}]}).json()
    assert payload["cases"] == 1
    assert payload["recall_at_k"] == 1
    assert payload["precision_at_k"] == 1
    assert payload["mrr"] == 1
    assert payload["average_latency_ms"] >= 0


def test_answer_falls_back_when_ollama_is_unavailable(monkeypatch) -> None:
    async def unavailable(_question: str, _evidence: str) -> None:
        return None

    monkeypatch.setattr("app.main.generator.generate", unavailable)
    client.post("/api/documents", files={"files": ("guide.txt", b"Local fallback evidence.", "text/plain")})
    response = client.post("/api/answer", json={"query": "fallback evidence"})
    assert response.status_code == 200
    assert response.json()["answer"].startswith("Based on the indexed sources:")


def test_overview_question_returns_document_context() -> None:
    client.post("/api/documents", files={"files": ("guide.txt", b"This guide explains document ingestion.", "text/plain")})
    response = client.post("/api/search", json={"query": "what is this doc ?"})
    assert response.status_code == 200
    assert response.json()[0]["source"] == "guide.txt"
