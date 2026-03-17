"""Tests for the embedding pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from nexus.db.database import DatabaseManager
from nexus.db.models import Document
from nexus.db.vectors import EmbeddingPipeline
from nexus.llm.cascade import CascadeManager


class TestEmbeddingPipeline:
    async def test_embed_text(self, db, mock_provider):
        cascade = CascadeManager([mock_provider])
        pipeline = EmbeddingPipeline(cascade, db)

        vec = await pipeline.embed_text("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert len(vec) == 768

    async def test_embed_and_store(self, db, mock_provider):
        cascade = CascadeManager([mock_provider])
        pipeline = EmbeddingPipeline(cascade, db)

        doc = Document(title="Test", content="Short text for testing")
        doc_id = await db.insert_document(doc)

        emb_ids = await pipeline.embed_and_store(doc_id, "Short text for testing")
        assert len(emb_ids) >= 1

        stored = await db.get_embeddings_for_document(doc_id)
        assert len(stored) == len(emb_ids)

    async def test_search_similar(self, db, mock_provider):
        cascade = CascadeManager([mock_provider])
        pipeline = EmbeddingPipeline(cascade, db)

        # Insert 3 documents with different content
        texts = [
            "Python programming language for data science",
            "JavaScript frontend web development",
            "Python machine learning with scikit-learn",
        ]
        for i, text in enumerate(texts):
            doc = Document(title=f"Doc {i}", content=text)
            doc_id = await db.insert_document(doc)
            await pipeline.embed_and_store(doc_id, text)

        # Search for something related to Python
        results = await pipeline.search_similar("Python data science", top_k=3)
        assert len(results) == 3

        # All results should have scores
        for doc, score in results:
            assert isinstance(score, float)
            assert -1.0 <= score <= 1.0

    async def test_search_empty_db(self, db, mock_provider):
        cascade = CascadeManager([mock_provider])
        pipeline = EmbeddingPipeline(cascade, db)

        results = await pipeline.search_similar("anything")
        assert results == []

    async def test_deduplication(self, db, mock_provider):
        """Multiple chunks from same doc should be deduplicated."""
        cascade = CascadeManager([mock_provider])
        pipeline = EmbeddingPipeline(cascade, db, chunk_size=5, chunk_overlap=1)

        long_text = " ".join(["word"] * 20)
        doc = Document(title="Long", content=long_text)
        doc_id = await db.insert_document(doc)
        emb_ids = await pipeline.embed_and_store(doc_id, long_text)

        # Should have multiple chunks
        assert len(emb_ids) > 1

        # But search should deduplicate to 1 document
        results = await pipeline.search_similar("word", top_k=5)
        assert len(results) == 1


class TestChunking:
    def test_short_text_single_chunk(self):
        pipeline = EmbeddingPipeline.__new__(EmbeddingPipeline)
        pipeline.chunk_size = 500
        pipeline.chunk_overlap = 50

        chunks = pipeline._chunk_text("hello world")
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_long_text_multiple_chunks(self):
        pipeline = EmbeddingPipeline.__new__(EmbeddingPipeline)
        pipeline.chunk_size = 5
        pipeline.chunk_overlap = 2

        text = " ".join(f"word{i}" for i in range(15))
        chunks = pipeline._chunk_text(text)
        assert len(chunks) > 1

        # Each chunk should have at most chunk_size words
        for chunk in chunks:
            assert len(chunk.split()) <= 5

    def test_overlap(self):
        pipeline = EmbeddingPipeline.__new__(EmbeddingPipeline)
        pipeline.chunk_size = 5
        pipeline.chunk_overlap = 2

        text = " ".join(f"w{i}" for i in range(10))
        chunks = pipeline._chunk_text(text)

        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            words_a = set(chunks[i].split())
            words_b = set(chunks[i + 1].split())
            assert len(words_a & words_b) > 0


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert EmbeddingPipeline._cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert EmbeddingPipeline._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert EmbeddingPipeline._cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.0], dtype=np.float32)
        assert EmbeddingPipeline._cosine_similarity(a, b) == 0.0
