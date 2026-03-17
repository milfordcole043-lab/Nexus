"""LLM cascade manager — tries providers in order with fallback."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from nexus.llm.provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class AllProvidersFailedError(Exception):
    """Raised when all providers in the cascade have failed."""


@dataclass
class CascadeStats:
    """Tracks cascade usage statistics."""

    total_requests: int = 0
    total_tokens: int = 0
    failures: int = 0
    fallbacks: int = 0
    provider_usage: dict[str, int] = field(default_factory=dict)


class CascadeManager:
    """Manages an ordered list of LLM providers with fallback."""

    def __init__(
        self,
        providers: list[LLMProvider],
        max_retries: int = 2,
        base_delay: float = 1.0,
    ):
        if not providers:
            raise ValueError("At least one provider is required")
        self.providers = providers
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.stats = CascadeStats()

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Try each provider in order until one succeeds."""
        self.stats.total_requests += 1
        errors: list[str] = []

        for i, provider in enumerate(self.providers):
            if i > 0:
                self.stats.fallbacks += 1
                logger.warning(
                    "Falling back to %s", provider.provider_name
                )

            for attempt in range(self.max_retries + 1):
                try:
                    response = await provider.generate(
                        prompt, system_prompt, max_tokens, temperature
                    )
                    self.stats.total_tokens += response.tokens_used
                    self.stats.provider_usage[provider.provider_name] = (
                        self.stats.provider_usage.get(provider.provider_name, 0) + 1
                    )
                    return response
                except Exception as e:
                    errors.append(f"{provider.provider_name} (attempt {attempt + 1}): {e}")
                    logger.warning(
                        "Provider %s attempt %d failed: %s",
                        provider.provider_name,
                        attempt + 1,
                        e,
                    )
                    if attempt < self.max_retries:
                        delay = self.base_delay * (2**attempt)
                        await asyncio.sleep(delay)

            self.stats.failures += 1

        raise AllProvidersFailedError(
            f"All providers failed:\n" + "\n".join(errors)
        )

    async def embed(self, text: str) -> list[float]:
        """Try each provider that supports embeddings."""
        errors: list[str] = []

        for provider in self.providers:
            try:
                return await provider.embed(text)
            except NotImplementedError:
                continue
            except Exception as e:
                errors.append(f"{provider.provider_name}: {e}")
                logger.warning(
                    "Embed failed for %s: %s", provider.provider_name, e
                )

        raise AllProvidersFailedError(
            f"No provider could generate embeddings:\n" + "\n".join(errors)
        )
