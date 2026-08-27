from __future__ import annotations

import os

import httpx


class Generator:
    def __init__(self) -> None:
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

    async def generate(self, question: str, evidence: str) -> str | None:
        prompt = (
            "Answer the question using only the evidence below. "
            "If the evidence is insufficient, say so. Be concise and answer in the question's language.\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence}"
        )
        try:
            async with httpx.AsyncClient(timeout=45) as client:
                response = await client.post(f"{self.url}/api/generate", json={"model": self.model, "prompt": prompt, "stream": False})
                response.raise_for_status()
                generated = response.json().get("response", "").strip()
                return generated or None
        except (httpx.HTTPError, ValueError):
            return None


generator = Generator()
