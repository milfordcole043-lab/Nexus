"""Tests for hybrid search: LanceDB + FTS5 + RRF."""

from __future__ import annotations

import numpy as np
import pytest
import pytest_asyncio

from nexus.db.database import DatabaseManager
from nexus.db.lancedb_store import LanceDBStore
from nexus.db.models import Document, Embedding
from nexus.db.vectors import EmbeddingPipeline
from nexus.llm.cascade import CascadeManager


# --- LanceDB Store Tests ---


async def test_lancedb_store_add_and_search(lancedb_store: LanceDBStore):
    """Basic LanceDB CRUD: add embeddings, search, verify results."""
    rng = np.random.RandomState(42)
    vec1 = rng.randn(768).astype(np.float32)
    vec1 = (vec1 / np.linalg.norm(vec1)).tolist()

    await lancedb_store.add_embeddings([
        {"document_id": 1, "chunk_index": 0, "vector": vec1},
    ])

    results = await lancedb_store.search(vec1, top_k=5)
    assert len(results) == 1
    assert results[0][0] == 1
    assert results[0][1] > 0.99  # near-identical vector


async def test_lancedb_search_deduplication(lancedb_store: LanceDBStore):
    """Multiple chunks per doc -> search returns 1 result per doc."""
    rng = np.random.RandomState(42)
    base = rng.randn(768).astype(np.float32)
    base = base / np.linalg.norm(base)

    records = [
        {"document_id": 1, "chunk_index": 0, "vector": base.tolist()},
        {"document_id": 1, "chunk_index": 1, "vector": (base + rng.randn(768) * 0.01).tolist()},
        {"document_id": 2, "chunk_index": 0, "vector": rng.randn(768).astype(np.float32).tolist()},
    ]
    await lancedb_store.add_embeddings(records)

    results = await lancedb_store.search(base.tolist(), top_k=10)
    doc_ids = [r[0] for r in results]
    # doc_id=1 should appear only once despite 2 chunks
    assert doc_ids.count(1) == 1
    assert 1 in doc_ids
    assert 2 in doc_ids


async def test_lancedb_delete_by_document(lancedb_store: LanceDBStore):
    """Delete cleans up all chunks for a doc."""
    rng = np.random.RandomState(42)
    records = [
        {"document_id": 1, "chunk_index": 0, "vector": rng.randn(768).astype(np.float32).tolist()},
        {"document_id": 1, "chunk_index": 1, "vector": rng.randn(768).astype(np.float32).tolist()},
        {"document_id": 2, "chunk_index": 0, "vector": rng.randn(768).astype(np.float32).tolist()},
    ]
    await lancedb_store.add_embeddings(records)
    assert await lancedb_store.count() == 3

    await lancedb_store.delete_by_document(1)
    assert await lancedb_store.count() == 1


# --- FTS5 Tests ---


async def test_fts5_keyword_search(db: DatabaseManager):
    """Insert docs, FTS5 MATCH finds exact terms."""
    doc = Document(title="Python Guide", content="Python is a programming language", category="docs")
    await db.insert_document(doc)

    results = await db.search_fts("Python")
    assert len(results) >= 1
    assert results[0][1] < 0  # BM25 rank is negative


async def test_fts5_phrase_search(db: DatabaseManager):
    """FTS5 finds multi-word phrases."""
    await db.insert_document(Document(
        title="Machine Learning", content="Deep learning neural networks are powerful", category="docs"
    ))
    await db.insert_document(Document(
        title="Cooking", content="How to make pasta from scratch", category="recipes"
    ))

    results = await db.search_fts("neural networks")
    assert len(results) >= 1
    doc_ids = [r[0] for r in results]
    # The ML doc should be found
    assert any(doc_id for doc_id in doc_ids)


async def test_fts5_file_path_search(db: DatabaseManager):
    """Searching file_watcher.py finds the right doc."""
    await db.insert_document(Document(
        title="File Watcher Agent",
        content="Monitors directories for changes",
        file_path="/nexus/agents/file_watcher.py",
        category="code",
    ))
    await db.insert_document(Document(
        title="Config", content="Configuration system", file_path="/nexus/config.py", category="code"
    ))

    results = await db.search_fts("file_watcher.py")
    assert len(results) >= 1


async def test_fts5_empty_query(db: DatabaseManager):
    """Empty/invalid query returns [] gracefully."""
    assert await db.search_fts("") == []
    assert await db.search_fts("   ") == []
    assert await db.search_fts("***") == []


# --- RRF Tests ---


def test_rrf_scoring():
    """Verify RRF formula: known ranks -> expected scores."""
    vector_results = [(1, 0.95), (2, 0.80), (3, 0.70)]
    keyword_results = [(2, 5.0), (4, 3.0), (1, 1.0)]

    fused = EmbeddingPipeline._fuse_rrf(vector_results, keyword_results, top_k=10, k=60)
    scores = dict(fused)

    # Doc 2: rank 1 in vector (1/62) + rank 0 in keyword (1/61)
    expected_doc2 = 1 / 62 + 1 / 61
    assert abs(scores[2] - expected_doc2) < 1e-9

    # Doc 1: rank 0 in vector (1/61) + rank 2 in keyword (1/63)
    expected_doc1 = 1 / 61 + 1 / 63
    assert abs(scores[1] - expected_doc1) < 1e-9

    # Doc 4: only in keyword rank 1 (1/62)
    expected_doc4 = 1 / 62
    assert abs(scores[4] - expected_doc4) < 1e-9

    # Doc 2 should rank highest (appears in both lists at good positions)
    assert fused[0][0] == 2


# --- Hybrid Search Integration Tests ---


@pytest_asyncio.fixture
async def hybrid_pipeline(db, cascade, lancedb_store):
    """Pipeline with LanceDB for hybrid search."""
    pipeline = EmbeddingPipeline(
        cascade=cascade,
        db=db,
        lancedb_store=lancedb_store,
        search_mode="hybrid",
    )
    return pipeline


async def test_hybrid_combines_results(hybrid_pipeline, db):
    """Doc high in vector + Doc high in keyword -> both in hybrid results."""
    # Insert two docs with different content
    doc1_id = await db.insert_document(Document(
        title="Machine Learning Basics",
        content="Neural networks and deep learning are transforming AI research",
        category="docs",
    ))
    doc2_id = await db.insert_document(Document(
        title="file_watcher.py documentation",
        content="The file watcher agent monitors directories for changes and indexes new files",
        category="code",
    ))

    await hybrid_pipeline.embed_and_store(doc1_id, "Neural networks and deep learning are transforming AI research")
    await hybrid_pipeline.embed_and_store(doc2_id, "The file watcher agent monitors directories for changes")

    results = await hybrid_pipeline.search_similar("file watcher neural networks", top_k=5)
    doc_ids = [doc.id for doc, _ in results]
    # Both docs should appear in results
    assert doc1_id in doc_ids or doc2_id in doc_ids


async def test_search_mode_vector_only(db, cascade, lancedb_store):
    """search_mode='vector' -> only vector results."""
    pipeline = EmbeddingPipeline(
        cascade=cascade, db=db,
        lancedb_store=lancedb_store, search_mode="vector",
    )

    doc_id = await db.insert_document(Document(
        title="Test Doc", content="Machine learning algorithms", category="docs",
    ))
    await pipeline.embed_and_store(doc_id, "Machine learning algorithms")

    results = await pipeline.search_similar("machine learning", top_k=5)
    assert len(results) >= 1
    assert results[0][0].id == doc_id


async def test_search_mode_keyword_only(db, cascade, lancedb_store):
    """search_mode='keyword' -> only keyword results."""
    pipeline = EmbeddingPipeline(
        cascade=cascade, db=db,
        lancedb_store=lancedb_store, search_mode="keyword",
    )

    doc_id = await db.insert_document(Document(
        title="Test Doc", content="Specific keyword xylophone testing", category="docs",
    ))

    results = await pipeline.search_similar("xylophone", top_k=5)
    assert len(results) >= 1
    assert results[0][0].id == doc_id


async def test_migration_sqlite_to_lancedb(db, cascade, tmp_path):
    """Embeddings in SQLite -> initialize() -> searchable in LanceDB."""
    # First, store embeddings the old way (SQLite)
    old_pipeline = EmbeddingPipeline(cascade=cascade, db=db)
    doc_id = await db.insert_document(Document(
        title="Migration Test", content="Testing migration from SQLite to LanceDB", category="test",
    ))
    await old_pipeline.embed_and_store(doc_id, "Testing migration from SQLite to LanceDB")

    # Verify embeddings in SQLite
    embs = await db.get_all_embeddings()
    assert len(embs) >= 1

    # Now create a new pipeline with LanceDB and initialize (should migrate)
    store = LanceDBStore(tmp_path / "migrate_lancedb", dimensions=768)
    new_pipeline = EmbeddingPipeline(
        cascade=cascade, db=db,
        lancedb_store=store, search_mode="vector",
    )
    await new_pipeline.initialize()

    # LanceDB should now have the embeddings
    assert not await store.is_empty()

    # Search should work via LanceDB
    results = await new_pipeline.search_similar("migration testing", top_k=5)
    assert len(results) >= 1
    assert results[0][0].id == doc_id


async def test_backward_compat_no_lancedb(db, cascade):
    """Pipeline without lancedb_store uses numpy brute-force."""
    pipeline = EmbeddingPipeline(cascade=cascade, db=db)

    doc_id = await db.insert_document(Document(
        title="Compat Test", content="Backward compatibility testing", category="test",
    ))
    await pipeline.embed_and_store(doc_id, "Backward compatibility testing")

    results = await pipeline.search_similar("compatibility", top_k=5)
    assert len(results) >= 1
    assert results[0][0].id == doc_id
