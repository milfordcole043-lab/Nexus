"""Shared test fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio

from nexus.db.database import DatabaseManager
from nexus.llm.cascade import CascadeManager
from nexus.llm.provider import LLMProvider, LLMResponse


class MockProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, name: str = "mock", embed_dim: int = 768, should_fail: bool = False):
        self._name = name
        self._embed_dim = embed_dim
        self._should_fail = should_fail
        self.generate_calls: list[str] = []
        self.embed_calls: list[str] = []

    @property
    def provider_name(self) -> str:
        return self._name

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        if self._should_fail:
            raise RuntimeError(f"{self._name} is unavailable")
        self.generate_calls.append(prompt)
        return LLMResponse(
            text=f"Mock response to: {prompt[:50]}",
            tokens_used=10,
            model="mock-model",
            provider=self._name,
            duration_ms=5,
        )

    async def embed(self, text: str) -> list[float]:
        if self._should_fail:
            raise RuntimeError(f"{self._name} embed unavailable")
        self.embed_calls.append(text)
        # Deterministic embeddings based on text hash
        rng = np.random.RandomState(hash(text) % (2**31))
        vec = rng.randn(self._embed_dim).astype(np.float32)
        # Normalize
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    async def is_available(self) -> bool:
        return not self._should_fail


class NoEmbedProvider(LLMProvider):
    """Mock provider that doesn't support embeddings (like Groq)."""

    @property
    def provider_name(self) -> str:
        return "no-embed"

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(text="response", tokens_used=5, model="m", provider="no-embed")

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError("No embeddings")

    async def is_available(self) -> bool:
        return True


@pytest_asyncio.fixture
async def db(tmp_path: Path):
    """Provide a temporary database."""
    db_manager = DatabaseManager(tmp_path / "test.db")
    await db_manager.initialize()
    yield db_manager
    await db_manager.close()


@pytest.fixture
def mock_provider():
    """Provide a mock LLM provider."""
    return MockProvider()


@pytest.fixture
def failing_provider():
    """Provide a provider that always fails."""
    return MockProvider(name="failing", should_fail=True)


@pytest.fixture
def cascade(mock_provider):
    """Provide a cascade with a single mock provider."""
    return CascadeManager([mock_provider])
