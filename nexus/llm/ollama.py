"""Ollama LLM provider."""

from __future__ import annotations

import time

import httpx
import ollama as ollama_client

from nexus.config import LLMProviderConfig
from nexus.llm.provider import LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    """Local Ollama provider for generation and embeddings."""

    def __init__(self, config: LLMProviderConfig, embed_model: str = "nomic-embed-text"):
        self._config = config
        self._embed_model = embed_model
        self._base_url = config.base_url or "http://localhost:11434"
        self._client = ollama_client.AsyncClient(host=self._base_url)

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        start = time.perf_counter_ns()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat(
            model=self._config.model,
            messages=messages,
            options={"num_predict": max_tokens, "temperature": temperature},
        )

        duration_ms = (time.perf_counter_ns() - start) // 1_000_000
        tokens = (
            response.get("eval_count", 0) + response.get("prompt_eval_count", 0)
        )

        return LLMResponse(
            text=response["message"]["content"],
            tokens_used=tokens,
            model=self._config.model,
            provider="ollama",
            duration_ms=duration_ms,
        )

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embed(model=self._embed_model, input=text)
        embeddings = response.get("embeddings", [])
        if embeddings:
            return embeddings[0]
        raise RuntimeError("Ollama returned no embeddings")

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/version")
                return resp.status_code == 200
        except Exception:
            return False
