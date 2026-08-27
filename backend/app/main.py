from __future__ import annotations

import re
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field

from app.generation import generator


app = FastAPI(title="Advanced RAG Platform", version="0.1.0")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None = None
    section: str | None = None
    document_type: str = "text"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal["vector", "lexical", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = False
    rewrite: bool = False
    document_type: str | None = None
    year: int | None = None


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    source: str
    page: int | None = None
    section: str | None = None
    score: float


class AnswerRequest(SearchRequest):
    pass


class Citation(BaseModel):
    citation_id: str
    source: str
    page: int | None = None
    section: str | None = None
    quote: str


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[SearchResult]
    strategy: str
    rewritten_query: str | None = None


class EvaluationCase(BaseModel):
    question: str = Field(min_length=1)
    relevant_sources: list[str] = Field(min_length=1)


class EvaluationRequest(BaseModel):
    cases: list[EvaluationCase] = Field(min_length=1)
    mode: Literal["vector", "lexical", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = False
    rewrite: bool = False


class EvaluationResponse(BaseModel):
    cases: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    average_latency_ms: float
    mode: str


class InMemoryStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

    def search(self, request: SearchRequest) -> list[SearchResult]:
        candidates = [chunk for chunk in self.chunks if self._matches(chunk, request)]
        query_terms = self._terms(request.query)
        candidate_limit = min(request.top_k * 4 if request.rerank else request.top_k, 20)
        lexical = self._rank(candidates, query_terms, lambda terms: sum(term in terms for term in query_terms) / max(len(query_terms), 1))
        vector = self._rank(candidates, query_terms, lambda terms: self._cosine_like(query_terms, terms))
        if request.mode == "hybrid":
            lexical_positions = {item.chunk_id: position for position, item in enumerate(lexical, start=1)}
            vector_positions = {item.chunk_id: position for position, item in enumerate(vector, start=1)}
            fused = []
            for chunk in candidates:
                if chunk.id not in lexical_positions and chunk.id not in vector_positions:
                    continue
                reciprocal_score = (1 / (60 + lexical_positions.get(chunk.id, 1000))) + (1 / (60 + vector_positions.get(chunk.id, 1000)))
                fused.append((chunk, reciprocal_score))
            maximum = max((score for _, score in fused), default=1)
            results = [self._result(chunk, score / maximum) for chunk, score in sorted(fused, key=lambda item: item[1], reverse=True)[:candidate_limit]]
        else:
            results = (vector if request.mode == "vector" else lexical)[:candidate_limit]
        if not results and self._is_overview_query(request.query) and candidates:
            results = [self._result(chunk, 0.1) for chunk in candidates[:candidate_limit]]
        return self._rerank(results, request.query, request.top_k) if request.rerank else results

    @staticmethod
    def _is_overview_query(query: str) -> bool:
        normalized = " ".join(query.lower().replace("?", "").split())
        return normalized in {"what is this doc", "what is this document", "summarize this document", "what does this document contain"}

    @classmethod
    def _rerank(cls, results: list[SearchResult], query: str, top_k: int) -> list[SearchResult]:
        query_terms = cls._terms(query)
        normalized_query = " ".join(query.lower().split())
        rescored = []
        for result in results:
            normalized_text = " ".join(result.text.lower().split())
            coverage = len(query_terms & cls._terms(result.text)) / max(len(query_terms), 1)
            phrase_bonus = 1.0 if normalized_query in normalized_text else 0.0
            score = result.score * 0.35 + coverage * 0.5 + phrase_bonus * 0.15
            rescored.append((result, score))
        maximum = max((score for _, score in rescored), default=1)
        return [result.model_copy(update={"score": round(score / maximum, 4)}) for result, score in sorted(rescored, key=lambda item: item[1], reverse=True)[:top_k]]

    @staticmethod
    def _rank(candidates: list[Chunk], query_terms: set[str], scorer) -> list[SearchResult]:
        ranked = []
        for chunk in candidates:
            score = scorer(InMemoryStore._terms(chunk.text))
            if score > 0:
                ranked.append((chunk, score))
        return [InMemoryStore._result(chunk, score) for chunk, score in sorted(ranked, key=lambda item: item[1], reverse=True)]

    @staticmethod
    def _result(chunk: Chunk, score: float) -> SearchResult:
        return SearchResult(chunk_id=chunk.id, text=chunk.text, source=chunk.source, page=chunk.page, section=chunk.section, score=round(score, 4))

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-ZÀ-ÿ0-9_]{3,}", text.lower()))

    @classmethod
    def _cosine_like(cls, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / ((len(left) * len(right)) ** 0.5)

    @staticmethod
    def _matches(chunk: Chunk, request: SearchRequest) -> bool:
        if request.document_type and chunk.document_type != request.document_type:
            return False
        if request.year and str(request.year) not in chunk.text and str(request.year) not in chunk.source:
            return False
        return True


store = InMemoryStore()


def rewrite_query(query: str) -> str:
    """Expand common vague questions while keeping specific queries unchanged."""
    normalized = " ".join(query.lower().split())
    expansions = {
        "how does it work": "functionality process workflow",
        "how does this work": "functionality process workflow",
        "what about admins": "administrator features permissions access",
        "what about the admins": "administrator features permissions access",
        "what are the rules": "requirements policy rules configuration",
    }
    for phrase, expansion in expansions.items():
        if phrase in normalized:
            return f"{query} {expansion}"
    return query


def chunk_text(text: str, source: str, document_type: str) -> list[Chunk]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[Chunk] = []
    for index, paragraph in enumerate(paragraphs or [text.strip()]):
        if not paragraph:
            continue
        chunks.append(Chunk(id=f"{source}:{index + 1}", text=paragraph[:1400], source=source, section=f"Block {index + 1}", document_type=document_type))
    return chunks


def extract_text(filename: str, content: bytes) -> tuple[str, str]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(content))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages), "pdf"
        except Exception:
            return content.decode("utf-8", errors="replace"), "pdf"
    if suffix == ".docx":
        try:
            from docx import Document
            import io
            document = Document(io.BytesIO(content))
            return "\n\n".join(paragraph.text for paragraph in document.paragraphs), "docx"
        except Exception:
            return content.decode("utf-8", errors="replace"), "docx"
    return content.decode("utf-8", errors="replace"), suffix.lstrip(".") or "text"


@app.get("/api/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "documents": len({chunk.source for chunk in store.chunks}), "chunks": len(store.chunks)}


@app.post("/api/documents")
async def ingest(files: list[UploadFile] = File(...)) -> dict[str, object]:
    ingested = []
    for upload in files:
        text, document_type = extract_text(upload.filename or "document.txt", await upload.read())
        chunks = chunk_text(text, upload.filename or "document.txt", document_type)
        store.add(chunks)
        ingested.append({"source": upload.filename, "type": document_type, "chunks": len(chunks)})
    return {"ingested": ingested, "total_chunks": len(store.chunks)}


@app.post("/api/search", response_model=list[SearchResult])
def search(request: SearchRequest) -> list[SearchResult]:
    effective_request = request.model_copy(update={"query": rewrite_query(request.query) if request.rewrite else request.query})
    return store.search(effective_request)


@app.post("/api/answer", response_model=AnswerResponse)
async def answer(request: AnswerRequest) -> AnswerResponse:
    effective_query = rewrite_query(request.query) if request.rewrite else request.query
    effective_request = request.model_copy(update={"query": effective_query})
    results = store.search(effective_request)
    if not results:
        return AnswerResponse(answer="I could not find supporting evidence in the indexed documents.", citations=[], retrieved_chunks=[], strategy=request.mode, rewritten_query=effective_query if request.rewrite else None)
    evidence = " ".join(result.text for result in results[:3])
    citations = [Citation(citation_id=f"[{index}]", source=result.source, page=result.page, section=result.section, quote=result.text[:240]) for index, result in enumerate(results, start=1)]
    generated = await generator.generate(request.query, evidence)
    response_text = generated or f"Based on the indexed sources: {evidence}"
    return AnswerResponse(answer=response_text, citations=citations, retrieved_chunks=results, strategy=request.mode, rewritten_query=effective_query if request.rewrite else None)


@app.post("/api/evaluate", response_model=EvaluationResponse)
def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    for case in request.cases:
        started = perf_counter()
        effective_query = rewrite_query(case.question) if request.rewrite else case.question
        search_request = SearchRequest(query=effective_query, mode=request.mode, top_k=request.top_k, rerank=request.rerank)
        results = store.search(search_request)
        latencies.append((perf_counter() - started) * 1000)
        expected = set(case.relevant_sources)
        retrieved = [result.source for result in results]
        hits = sum(source in expected for source in retrieved)
        recalls.append(hits / len(expected))
        precisions.append(hits / max(len(retrieved), 1))
        reciprocal_ranks.append(next((1 / (index + 1) for index, source in enumerate(retrieved) if source in expected), 0.0))
    return EvaluationResponse(cases=len(request.cases), recall_at_k=round(sum(recalls) / len(recalls), 4), precision_at_k=round(sum(precisions) / len(precisions), 4), mrr=round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4), average_latency_ms=round(sum(latencies) / len(latencies), 2), mode=request.mode)
