"""Tests for MCP server tools."""

from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from nexus.agents.memory import MemoryAgent
from nexus.agents.project_context import ProjectContextAgent
from nexus.db.database import DatabaseManager
from nexus.db.models import Document
from nexus.db.vectors import EmbeddingPipeline
import nexus.mcp.server as srv


async def _index_doc(db: DatabaseManager, pipeline: EmbeddingPipeline,
                     title: str, content: str, file_path: str, category: str = "docs") -> int:
    """Helper: insert document + embed."""
    doc_id = await db.insert_document(Document(
        title=title, content=content, file_path=file_path, category=category,
    ))
    await pipeline.embed_and_store(doc_id, content)
    return doc_id


@pytest_asyncio.fixture
async def mcp_env(db, cascade):
    """Set up MCP server module globals with test instances."""
    pipeline = EmbeddingPipeline(
        cascade=cascade,
        db=db,
        model_name="test-model",
        dimensions=768,
        chunk_size=200,
        chunk_overlap=50,
    )
    memory = MemoryAgent(
        name="memory",
        description="test memory",
        cascade=cascade,
        db=db,
        pipeline=pipeline,
    )
    project_ctx = ProjectContextAgent(
        name="project_context",
        description="test project context",
        cascade=cascade,
        db=db,
        memory=memory,
    )
    srv._db = db
    srv._cascade = cascade
    srv._pipeline = pipeline
    srv._memory = memory
    srv._project_context = project_ctx
    srv._initialized = True
    yield
    srv._initialized = False
    srv._db = None
    srv._cascade = None
    srv._pipeline = None
    srv._memory = None
    srv._project_context = None


async def test_nexus_search_returns_results(mcp_env, db, cascade):
    """Index docs and verify search returns results with expected fields."""
    await _index_doc(db, srv._pipeline,
        title="Python Guide",
        content="Python is a programming language used for web development and data science.",
        file_path="/docs/python.md",
    )
    await _index_doc(db, srv._pipeline,
        title="JavaScript Guide",
        content="JavaScript is used for frontend and backend web development with Node.js.",
        file_path="/docs/javascript.md",
    )

    results = await srv.nexus_search("programming language")
    assert isinstance(results, list)
    assert len(results) > 0
    first = results[0]
    assert "title" in first
    assert "file_path" in first
    assert "score" in first
    assert "snippet" in first


async def test_nexus_search_empty_db(mcp_env):
    """Empty database returns empty list, not an error."""
    results = await srv.nexus_search("anything")
    assert isinstance(results, list)
    assert len(results) == 0


async def test_nexus_ask_returns_answer(mcp_env, db):
    """Index a doc and ask a question — returns dict with answer and sources."""
    await _index_doc(db, srv._pipeline,
        title="Architecture Doc",
        content="Nexus uses SQLite for storage with WAL mode and numpy for vector similarity.",
        file_path="/docs/arch.md",
    )

    result = await srv.nexus_ask("What database does Nexus use?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert result["answer"] is not None


async def test_nexus_context_returns_block(mcp_env, tmp_path):
    """Project context returns a string containing the context header."""
    readme = tmp_path / "README.md"
    readme.write_text("# Test Project\nA test project for MCP tools.")

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 128  # not a git repo
        mock_exec.return_value = mock_proc

        result = await srv.nexus_context(str(tmp_path))
    assert isinstance(result, str)
    assert "=== NEXUS PROJECT CONTEXT ===" in result


async def test_nexus_entities_with_filter(mcp_env, db):
    """Insert entities, filter by type, verify correct subset returned."""
    from nexus.db.models import Entity

    await db.insert_entity(Entity(name="Python", type="technology"))
    await db.insert_entity(Entity(name="FastAPI", type="technology"))
    await db.insert_entity(Entity(name="Alice", type="person"))

    all_entities = await srv.nexus_entities()
    assert len(all_entities) == 3

    tech_only = await srv.nexus_entities(type="technology")
    assert len(tech_only) == 2
    assert all(e["type"] == "technology" for e in tech_only)

    people_only = await srv.nexus_entities(type="person")
    assert len(people_only) == 1
    assert people_only[0]["name"] == "Alice"
    assert "document_count" in people_only[0]


async def test_nexus_stats_returns_counts(mcp_env, db):
    """Stats returns dict with expected keys and non-negative values."""
    await _index_doc(db, srv._pipeline,
        title="Test Doc",
        content="Some content for testing stats endpoint.",
        file_path="/test/stats.md",
        category="test",
    )

    result = await srv.nexus_stats()
    assert isinstance(result, dict)
    assert result["total_documents"] > 0
    assert result["total_embeddings"] >= 0
    assert result["total_entities"] >= 0
    assert result["db_size_mb"] >= 0


async def test_search_error_handling(mcp_env):
    """When pipeline raises, tool returns error dict instead of crashing."""
    original_query = srv._memory.query
    srv._memory.query = AsyncMock(side_effect=RuntimeError("Ollama is down"))
    try:
        result = await srv.nexus_search("test query")
        assert isinstance(result, list)
        assert len(result) == 1
        assert "error" in result[0]
        assert "Ollama is down" in result[0]["error"]
    finally:
        srv._memory.query = original_query
