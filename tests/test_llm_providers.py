"""Tests for LLM providers and cascade manager."""

from __future__ import annotations

import pytest

from nexus.llm.cascade import AllProvidersFailedError, CascadeManager

from .conftest import MockProvider, NoEmbedProvider


class TestCascadeManager:
    async def test_single_provider_success(self, mock_provider):
        cascade = CascadeManager([mock_provider])
        resp = await cascade.generate("Hello")
        assert "Mock response" in resp.text
        assert resp.provider == "mock"
        assert cascade.stats.total_requests == 1

    async def test_fallback_on_failure(self):
        failing = MockProvider(name="primary", should_fail=True)
        backup = MockProvider(name="backup")

        cascade = CascadeManager([failing, backup], max_retries=0, base_delay=0.01)
        resp = await cascade.generate("Hello")
        assert resp.provider == "backup"
        assert cascade.stats.fallbacks == 1
        assert cascade.stats.failures == 1

    async def test_all_providers_fail(self):
        p1 = MockProvider(name="a", should_fail=True)
        p2 = MockProvider(name="b", should_fail=True)

        cascade = CascadeManager([p1, p2], max_retries=0, base_delay=0.01)
        with pytest.raises(AllProvidersFailedError):
            await cascade.generate("Hello")

    async def test_retry_then_succeed(self):
        """Provider fails first attempt, succeeds on retry."""
        provider = MockProvider(name="flaky")
        call_count = 0
        original_generate = provider.generate

        async def flaky_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Temporary failure")
            return await original_generate(*args, **kwargs)

        provider.generate = flaky_generate

        cascade = CascadeManager([provider], max_retries=2, base_delay=0.01)
        resp = await cascade.generate("Hello")
        assert "Mock response" in resp.text
        assert call_count == 2

    async def test_embed_skips_unsupported(self):
        no_embed = NoEmbedProvider()
        with_embed = MockProvider(name="embedder")

        cascade = CascadeManager([no_embed, with_embed])
        result = await cascade.embed("test text")
        assert len(result) == 768

    async def test_embed_all_fail(self):
        no_embed = NoEmbedProvider()
        cascade = CascadeManager([no_embed])
        with pytest.raises(AllProvidersFailedError):
            await cascade.embed("test")

    async def test_stats_tracking(self, mock_provider):
        cascade = CascadeManager([mock_provider])
        await cascade.generate("one")
        await cascade.generate("two")
        assert cascade.stats.total_requests == 2
        assert cascade.stats.provider_usage["mock"] == 2
        assert cascade.stats.total_tokens == 20

    async def test_requires_at_least_one_provider(self):
        with pytest.raises(ValueError):
            CascadeManager([])
