"""Nexus MCP server — exposes knowledge graph tools for Claude Code."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="nexus")

# Module globals — initialized lazily on first tool call
_db = None
_cascade = None
_pipeline = None
_memory = None
_project_context = None
_init_lock = asyncio.Lock()
_initialized = False

logger = logging.getLogger(__name__)


async def _ensure_initialized():
    """Lazy init: load config, DB, cascade, agents on first tool call."""
    global _db, _cascade, _pipeline, _memory, _project_context, _initialized
    if _initialized:
        return
    async with _init_lock:
        if _initialized:
            return

        from nexus.agents.memory import MemoryAgent
        from nexus.agents.project_context import ProjectContextAgent
        from nexus.config import NexusConfig
        from nexus.db.database import DatabaseManager
        from nexus.db.vectors import EmbeddingPipeline
        from nexus.llm.cascade import build_cascade

        config_path = Path(os.environ.get("NEXUS_CONFIG", "config.yaml"))
        config = NexusConfig.from_yaml(config_path)
        logging.basicConfig(level=config.log_level, stream=sys.stderr)

        _db = DatabaseManager(config.resolved_db_path)
        await _db.initialize()

        _cascade = build_cascade(config)
        _pipeline = EmbeddingPipeline(
            cascade=_cascade,
            db=_db,
            model_name=config.embedding.model,
            dimensions=config.embedding.dimensions,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        _memory = MemoryAgent(
            name="memory",
            description="RAG knowledge retrieval",
            cascade=_cascade,
            db=_db,
            pipeline=_pipeline,
        )
        _project_context = ProjectContextAgent(
            name="project_context",
            description="Project context assembler",
            cascade=_cascade,
            db=_db,
            memory=_memory,
        )
        _initialized = True
        logger.info("Nexus MCP server initialized")


@mcp.tool()
async def nexus_search(query: str, top_k: int = 5) -> list[dict]:
    """Search the Nexus knowledge graph for documents matching a query. Returns ranked results with titles, file paths, scores, and snippets."""
    try:
        await _ensure_initialized()
        from nexus.agents.memory.agent import QueryMode

        response = await _memory.query(question=query, mode=QueryMode.SEARCH, top_k=top_k)
        return [
            {
                "title": s.title,
                "file_path": s.file_path,
                "score": round(s.score, 4),
                "snippet": s.snippet[:500],
            }
            for s in response.sources
        ]
    except Exception as e:
        logger.error("nexus_search failed: %s", e)
        return [{"error": str(e)}]


@mcp.tool()
async def nexus_ask(question: str) -> dict:
    """Ask a question and get an LLM-synthesized answer from the Nexus knowledge graph, with source citations."""
    try:
        await _ensure_initialized()
        from nexus.agents.memory.agent import QueryMode

        response = await _memory.query(question=question, mode=QueryMode.ANSWER, top_k=10)
        return {
            "answer": response.answer,
            "sources": [
                {
                    "title": s.title,
                    "file_path": s.file_path,
                    "score": round(s.score, 4),
                    "snippet": s.snippet[:500],
                }
                for s in response.sources
            ],
        }
    except Exception as e:
        logger.error("nexus_ask failed: %s", e)
        return {"error": str(e)}


@mcp.tool()
async def nexus_context(project_path: str) -> str:
    """Generate a project context block for a given directory path. Includes git info, key files, and related knowledge from Nexus."""
    try:
        await _ensure_initialized()
        ctx = await _project_context.get_context(project_path)
        return ctx.context_block
    except Exception as e:
        logger.error("nexus_context failed: %s", e)
        return f"Error: {e}"


@mcp.tool()
async def nexus_entities(type: str | None = None) -> list[dict]:
    """List entities in the Nexus knowledge graph, optionally filtered by type (e.g. 'person', 'project', 'technology')."""
    try:
        await _ensure_initialized()
        entities = await _db.search_entities(type_=type)
        results = []
        for e in entities:
            doc_count = 0
            if e.id is not None:
                docs = await _db.get_documents_for_entity(e.id)
                doc_count = len(docs)
            results.append({
                "name": e.name,
                "type": e.type,
                "document_count": doc_count,
            })
        return results
    except Exception as e:
        logger.error("nexus_entities failed: %s", e)
        return [{"error": str(e)}]


@mcp.tool()
async def nexus_stats() -> dict:
    """Get Nexus knowledge graph statistics: document, embedding, and entity counts plus database size."""
    try:
        await _ensure_initialized()
        stats = await _db.get_stats()
        db_size_mb = 0.0
        try:
            db_size_mb = round(os.path.getsize(_db.db_path) / (1024 * 1024), 2)
        except OSError:
            pass
        return {
            "total_documents": stats.get("documents", 0),
            "total_embeddings": stats.get("embeddings", 0),
            "total_entities": stats.get("entities", 0),
            "db_size_mb": db_size_mb,
        }
    except Exception as e:
        logger.error("nexus_stats failed: %s", e)
        return {"error": str(e)}
