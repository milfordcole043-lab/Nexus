"""Groq LLM provider."""

from __future__ import annotations

import time

from groq import AsyncGroq

from nexus.config import LLMProviderConfig
from nexus.llm.provider import LLMProvider, LLMResponse


class GroqProvider(LLMProvider):
    """Groq cloud provider for fast generation (no embeddings)."""

    def __init__(self, config: LLMProviderConfig):
        self._config = config
        api_key = config.api_key
        if not api_key:
            raise ValueError("Groq API key not found. Set GROQ_API_KEY env var.")
        self._client = AsyncGroq(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "groq"

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

        response = await self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        duration_ms = (time.perf_counter_ns() - start) // 1_000_000
        usage = response.usage
        tokens = (usage.total_tokens if usage else 0)

        return LLMResponse(
            text=response.choices[0].message.content or "",
            tokens_used=tokens,
            model=self._config.model,
            provider="groq",
            duration_ms=duration_ms,
        )

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Groq does not support embeddings")

    async def is_available(self) -> bool:
        try:
            # Light check — just verify client can reach Groq
            await self._client.models.list()
            return True
        except Exception:
            return False
