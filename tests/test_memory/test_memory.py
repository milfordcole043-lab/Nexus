"""Tests for the memory agent, entity extraction, and RAG context assembly."""

from __future__ import annotations

import numpy as np
import pytest
import pytest_asyncio

from nexus.agents.memory.agent import MemoryAgent, QueryMode, MemoryResponse
from nexus.agents.memory.context import RAGContextAssembler, SourceDocument
from nexus.agents.memory.entities import EntityExtractor, ExtractedEntity
from nexus.agents.memory.prompts import MEMORY_SYSTEM_PROMPT, build_answer_prompt
from nexus.db.database import DatabaseManager
from nexus.db.models import Document, Entity
from nexus.db.vectors import EmbeddingPipeline
from nexus.llm.cascade import CascadeManager


# --- Fixtures ---


@pytest_asyncio.fixture
async def pipeline(db, cascade):
    """Provide an embedding pipeline with mock provider."""
    return EmbeddingPipeline(cascade=cascade, db=db, chunk_size=500, chunk_overlap=50)


@pytest_asyncio.fixture
async def memory_agent(db, cascade, pipeline):
    """Provide a MemoryAgent instance."""
    return MemoryAgent(
        name="memory",
        description="RAG knowledge retrieval",
        cascade=cascade,
        db=db,
        pipeline=pipeline,
    )


@pytest_asyncio.fixture
async def assembler(pipeline, db):
    """Provide a RAGContextAssembler."""
    return RAGContextAssembler(pipeline=pipeline, db=db)


@pytest.fixture
def extractor():
    """Provide an EntityExtractor."""
    return EntityExtractor()


async def _index_doc(db: DatabaseManager, pipeline: EmbeddingPipeline, title: str, content: str, file_path: str | None = None) -> int:
    """Helper: insert a document and embed it."""
    doc = Document(title=title, content=content, file_path=file_path, source_agent="test")
    doc_id = await db.insert_document(doc)
    await pipeline.embed_and_store(doc_id, content)
    return doc_id


# ===========================================================================
# Entity Extraction Tests
# ===========================================================================


class TestEntityExtractor:

    def test_extract_languages(self, extractor):
        """Extract Python/JavaScript from code content."""
        content = "This project uses Python for the backend and JavaScript for the frontend."
        entities = extractor.extract(content)
        names = {e.name for e in entities}
        assert "Python" in names
        assert "JavaScript" in names
        assert all(e.type == "language" for e in entities if e.name in ("Python", "JavaScript"))

    def test_extract_libraries(self, extractor):
        """Extract library names from text."""
        content = "Built with FastAPI and React, using numpy for vector math."
        entities = extractor.extract(content)
        names = {e.name for e in entities}
        assert "FastAPI" in names
        assert "React" in names
        assert "numpy" in names

    def test_extract_file_references(self, extractor):
        """Extract file references from content."""
        content = "The main entry point is src/main.py and config lives in config.yaml."
        entities = extractor.extract(content)
        names = {e.name for e in entities}
        assert "src/main.py" in names
        assert "config.yaml" in names
        assert all(e.type == "file_ref" for e in entities if e.name in ("src/main.py", "config.yaml"))

    def test_empty_content(self, extractor):
        """Unknown/empty content returns empty list."""
        assert extractor.extract("") == []
        assert extractor.extract("just some random words here") == []

    def test_deduplication(self, extractor):
        """Deduplication within single document."""
        content = "Python is great. I love Python. Python Python Python."
        entities = extractor.extract(content)
        python_entities = [e for e in entities if e.name == "Python"]
        assert len(python_entities) == 1

    def test_extract_tools(self, extractor):
        """Extract tool names from content."""
        content = "Deployed via Docker on AWS with Terraform infrastructure."
        entities = extractor.extract(content)
        names = {e.name for e in entities}
        assert "Docker" in names
        assert "AWS" in names
        assert "Terraform" in names

    def test_confidence_levels(self, extractor):
        """Different entity types have different confidence levels."""
        content = "Python uses FastAPI deployed on Docker."
        entities = extractor.extract(content)
        by_name = {e.name: e for e in entities}
        assert by_name["Python"].confidence == 0.95  # language
        assert by_name["FastAPI"].confidence == 0.9  # library
        assert by_name["Docker"].confidence == 0.85  # tool

    def test_context_snippet_included(self, extractor):
        """Each extracted entity includes a context snippet."""
        content = "We use Python for the backend API server."
        entities = extractor.extract(content)
        python_ent = next(e for e in entities if e.name == "Python")
        assert "Python" in python_ent.context_snippet
        assert len(python_ent.context_snippet) > 0


# ===========================================================================
# Entity DB Integration Tests
# ===========================================================================


class TestEntityDB:

    @pytest.mark.asyncio
    async def test_store_and_retrieve_entity_links(self, db, memory_agent, pipeline):
        """Store entities + links in DB, verify retrieval."""
        doc_id = await _index_doc(db, pipeline, "Test Doc", "Using Python and FastAPI for the API.")
        entity_ids = await memory_agent.extract_entities(doc_id, "Using Python and FastAPI for the API.")

        assert len(entity_ids) > 0

        # Verify entities are in DB
        entities = await db.get_entities_for_document(doc_id)
        names = {e.name for e in entities}
        assert "Python" in names
        assert "FastAPI" in names

    @pytest.mark.asyncio
    async def test_get_documents_for_entity(self, db, memory_agent, pipeline):
        """Documents can be retrieved by entity."""
        doc_id = await _index_doc(db, pipeline, "Python Guide", "Learn Python programming basics.")
        await memory_agent.extract_entities(doc_id, "Learn Python programming basics.")

        entities = await db.get_entity_by_name("Python")
        assert len(entities) > 0
        docs = await db.get_documents_for_entity(entities[0].id)
        assert any(d.id == doc_id for d in docs)

    @pytest.mark.asyncio
    async def test_delete_entity_links(self, db, memory_agent, pipeline):
        """Entity links are deleted when requested."""
        doc_id = await _index_doc(db, pipeline, "Test", "Python and FastAPI project.")
        await memory_agent.extract_entities(doc_id, "Python and FastAPI project.")

        entities_before = await db.get_entities_for_document(doc_id)
        assert len(entities_before) > 0

        await db.delete_entity_links_for_document(doc_id)
        entities_after = await db.get_entities_for_document(doc_id)
        assert len(entities_after) == 0


# ===========================================================================
# RAG Context Assembly Tests
# ===========================================================================


class TestRAGContextAssembler:

    @pytest.mark.asyncio
    async def test_short_doc_full_content(self, db, pipeline, assembler):
        """Short doc → full content used as snippet."""
        doc_id = await _index_doc(db, pipeline, "Short Doc", "This is a short document about testing.")
        doc = await db.get_document(doc_id)
        query_vec = await pipeline.embed_text("testing")

        sources = await assembler.build_sources(query_vec, [(doc, 0.9)])
        assert len(sources) == 1
        assert sources[0].snippet == "This is a short document about testing."

    @pytest.mark.asyncio
    async def test_long_doc_best_chunk(self, db, pipeline, assembler):
        """Long doc (>500 words) → best chunk selected as snippet."""
        # Create content with >500 words
        words = ["word"] * 600
        # Put distinctive content in a later section
        words[300:310] = ["Python"] * 10
        content = " ".join(words)

        doc_id = await _index_doc(db, pipeline, "Long Doc", content)
        doc = await db.get_document(doc_id)
        query_vec = await pipeline.embed_text("Python")

        sources = await assembler.build_sources(query_vec, [(doc, 0.8)])
        assert len(sources) == 1
        # Snippet should be a chunk, not the full content
        assert len(sources[0].snippet.split()) <= 500

    @pytest.mark.asyncio
    async def test_format_context_numbered(self, assembler):
        """format_context produces numbered source blocks."""
        sources = [
            SourceDocument(doc_id=1, title="Doc A", file_path="a.py", score=0.9, snippet="Content of doc A."),
            SourceDocument(doc_id=2, title="Doc B", file_path="b.py", score=0.8, snippet="Content of doc B."),
        ]
        context = assembler.format_context(sources)
        assert "[Source 1] Doc A (a.py)" in context
        assert "[Source 2] Doc B (b.py)" in context
        assert "Content of doc A." in context
        assert "Content of doc B." in context

    @pytest.mark.asyncio
    async def test_format_context_token_budget(self, assembler):
        """Token budget truncation works correctly."""
        long_snippet = " ".join(["word"] * 3000)
        sources = [
            SourceDocument(doc_id=1, title="Doc A", file_path=None, score=0.9, snippet=long_snippet),
            SourceDocument(doc_id=2, title="Doc B", file_path=None, score=0.8, snippet="Should not appear."),
        ]
        context = assembler.format_context(sources, token_budget=100)
        assert "[Source 1]" in context
        # Second source should be cut off due to budget
        assert "[Source 2]" not in context


# ===========================================================================
# Memory Agent Query Tests
# ===========================================================================


class TestMemoryAgentQuery:

    @pytest.mark.asyncio
    async def test_search_mode_no_llm(self, db, pipeline, memory_agent, mock_provider):
        """Search mode: returns ranked results without calling LLM."""
        await _index_doc(db, pipeline, "Python Guide", "Learn Python programming basics.")
        await _index_doc(db, pipeline, "FastAPI Docs", "FastAPI is a modern Python web framework.")
        await _index_doc(db, pipeline, "Cooking Tips", "How to make a perfect omelette.")

        result = await memory_agent.query("Python programming", mode=QueryMode.SEARCH)

        assert result.answer is None
        assert result.mode == "search"
        assert len(result.sources) > 0
        assert result.tokens_used == 0
        # LLM should not have been called
        assert len(mock_provider.generate_calls) == 0

    @pytest.mark.asyncio
    async def test_answer_mode_calls_llm(self, db, pipeline, memory_agent, mock_provider):
        """Answer mode: LLM called, answer returned with sources."""
        await _index_doc(db, pipeline, "Python Guide", "Python is a high-level programming language.")

        result = await memory_agent.query("What is Python?", mode=QueryMode.ANSWER)

        assert result.answer is not None
        assert result.mode == "answer"
        assert len(result.sources) > 0
        assert result.tokens_used > 0
        assert len(mock_provider.generate_calls) > 0

    @pytest.mark.asyncio
    async def test_auto_mode_search(self, memory_agent):
        """Auto mode detection: 'find files about X' → search."""
        mode = memory_agent._resolve_mode("find files about Python", QueryMode.AUTO)
        assert mode == QueryMode.SEARCH

        mode = memory_agent._resolve_mode("list documents", QueryMode.AUTO)
        assert mode == QueryMode.SEARCH

        mode = memory_agent._resolve_mode("show me configs", QueryMode.AUTO)
        assert mode == QueryMode.SEARCH

    @pytest.mark.asyncio
    async def test_auto_mode_answer(self, memory_agent):
        """Auto mode detection: 'what is...' → answer."""
        mode = memory_agent._resolve_mode("what is the relationship between X and Y?", QueryMode.AUTO)
        assert mode == QueryMode.ANSWER

        mode = memory_agent._resolve_mode("how does the cascade work?", QueryMode.AUTO)
        assert mode == QueryMode.ANSWER

        mode = memory_agent._resolve_mode("explain embeddings", QueryMode.AUTO)
        assert mode == QueryMode.ANSWER

    @pytest.mark.asyncio
    async def test_auto_mode_question_mark(self, memory_agent):
        """Question ending with ? → answer."""
        mode = memory_agent._resolve_mode("is this working?", QueryMode.AUTO)
        assert mode == QueryMode.ANSWER

    @pytest.mark.asyncio
    async def test_auto_mode_short_query(self, memory_agent):
        """Short query without ? → search."""
        mode = memory_agent._resolve_mode("Python", QueryMode.AUTO)
        assert mode == QueryMode.SEARCH

        mode = memory_agent._resolve_mode("FastAPI docs", QueryMode.AUTO)
        assert mode == QueryMode.SEARCH

    @pytest.mark.asyncio
    async def test_empty_results_graceful(self, memory_agent):
        """Empty results: graceful response."""
        result = await memory_agent.query("something that doesn't exist")
        assert result.sources == []
        assert result.answer == "No relevant documents found."

    @pytest.mark.asyncio
    async def test_source_attribution(self, db, pipeline, memory_agent):
        """Sources include doc_id, title, score, snippet."""
        await _index_doc(db, pipeline, "Test Doc", "Important test content.", file_path="/test.txt")

        result = await memory_agent.query("test content", mode=QueryMode.SEARCH)
        assert len(result.sources) > 0
        src = result.sources[0]
        assert src.doc_id is not None
        assert src.title == "Test Doc"
        assert isinstance(src.score, float)
        assert len(src.snippet) > 0

    @pytest.mark.asyncio
    async def test_entity_enhanced_search(self, db, pipeline, memory_agent):
        """Doc linked via entity found even with low vector score."""
        # Index a doc and link it to "FastAPI" entity
        doc_id = await _index_doc(db, pipeline, "API Reference", "REST endpoint documentation for the web framework.")
        # Manually extract entities so FastAPI gets linked
        entity = Entity(name="FastAPI", type="library")
        entity_id = await db.insert_entity(entity)
        await db.link_entity_to_document(doc_id, entity_id)

        # Query by entity name — should find via entity graph
        result = await memory_agent.query("FastAPI", mode=QueryMode.SEARCH)
        doc_ids = [s.doc_id for s in result.sources]
        assert doc_id in doc_ids


# ===========================================================================
# Agent Integration Tests
# ===========================================================================


class TestMemoryAgentIntegration:

    @pytest.mark.asyncio
    async def test_query_logging(self, db, pipeline, memory_agent):
        """Queries are logged to agent_logs."""
        await _index_doc(db, pipeline, "Test", "Content for logging test.")
        await memory_agent.query("logging test", mode=QueryMode.SEARCH)

        logs = await db.get_agent_logs(agent_name="memory")
        assert len(logs) > 0
        assert logs[0].action == "query"
        assert "logging test" in (logs[0].input_summary or "")

    @pytest.mark.asyncio
    async def test_entity_extraction_hook(self, db, pipeline, memory_agent):
        """Entity extraction hook stores entities in DB."""
        doc_id = await _index_doc(db, pipeline, "Hook Test", "Using Python with FastAPI and Docker.")
        entity_ids = await memory_agent.extract_entities(doc_id, "Using Python with FastAPI and Docker.")

        assert len(entity_ids) >= 3  # Python, FastAPI, Docker at minimum
        entities = await db.get_entities_for_document(doc_id)
        names = {e.name for e in entities}
        assert "Python" in names
        assert "FastAPI" in names
        assert "Docker" in names

    @pytest.mark.asyncio
    async def test_full_pipeline(self, db, pipeline, memory_agent):
        """Full pipeline: index doc → extract entities → query by entity name → doc found."""
        # Index and extract
        doc_id = await _index_doc(db, pipeline, "Rust Guide", "A comprehensive guide to Rust programming language.")
        await memory_agent.extract_entities(doc_id, "A comprehensive guide to Rust programming language.")

        # Query by entity name
        result = await memory_agent.query("Rust", mode=QueryMode.SEARCH)
        doc_ids = [s.doc_id for s in result.sources]
        assert doc_id in doc_ids

    @pytest.mark.asyncio
    async def test_execute_delegates_to_query(self, db, pipeline, memory_agent):
        """_execute() delegates to query() in auto mode."""
        await _index_doc(db, pipeline, "Execute Test", "Testing the execute path.")
        result = await memory_agent.run("what is testing?")
        assert result.success

    @pytest.mark.asyncio
    async def test_execute_empty_input(self, memory_agent):
        """_execute() with empty input returns ready message."""
        result = await memory_agent.run("")
        assert result.success
        assert "ready" in result.output.lower()


# ===========================================================================
# Prompts Tests
# ===========================================================================


class TestPrompts:

    def test_system_prompt_exists(self):
        assert "Nexus" in MEMORY_SYSTEM_PROMPT
        assert "Source" in MEMORY_SYSTEM_PROMPT

    def test_build_answer_prompt(self):
        prompt = build_answer_prompt("What is Python?", "Python is a language.")
        assert "What is Python?" in prompt
        assert "Python is a language." in prompt
        assert "Context from your knowledge base:" in prompt


# ===========================================================================
# DB Migration Tests
# ===========================================================================


class TestDBMigration:

    @pytest.mark.asyncio
    async def test_document_entities_table_exists(self, db):
        """Migration 2 creates document_entities table."""
        cursor = await db.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='document_entities'"
        )
        row = await cursor.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_schema_version_is_2(self, db):
        """Schema version is 2 after migrations."""
        cursor = await db.db.execute("SELECT MAX(version) FROM schema_version")
        row = await cursor.fetchone()
        assert row[0] == 2

    @pytest.mark.asyncio
    async def test_stats_include_document_entities(self, db):
        """get_stats() includes document_entities count."""
        stats = await db.get_stats()
        assert "document_entities" in stats
