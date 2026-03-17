"""Embedding convenience wrapper for agents."""

from nexus.db.vectors import EmbeddingPipeline

# Re-export for convenience — agents import from here
__all__ = ["EmbeddingPipeline"]
