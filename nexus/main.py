"""Nexus — FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from nexus.config import NexusConfig
from nexus.db.database import DatabaseManager
from nexus.db.vectors import EmbeddingPipeline
from nexus.llm.cascade import CascadeManager
from nexus.llm.ollama import OllamaProvider

logger = logging.getLogger(__name__)

# Global state — set during lifespan
config: NexusConfig | None = None
db: DatabaseManager | None = None
cascade: CascadeManager | None = None
pipeline: EmbeddingPipeline | None = None


def _build_cascade(cfg: NexusConfig) -> CascadeManager:
    """Build the LLM cascade from config. Lazy-imports to avoid missing deps."""
    providers = []
    for p in cfg.llm_cascade:
        if p.type == "ollama":
            providers.append(
                OllamaProvider(p, embed_model=cfg.embedding.model)
            )
        elif p.type == "groq":
            try:
                from nexus.llm.groq import GroqProvider
                providers.append(GroqProvider(p))
            except (ValueError, ImportError) as e:
                logger.warning("Skipping Groq provider: %s", e)
        elif p.type == "claude":
            # Claude provider not implemented in Phase 1
            logger.info("Claude provider not yet implemented, skipping")
    if not providers:
        raise RuntimeError("No LLM providers could be initialized")
    return CascadeManager(providers)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — init config, DB, cascade."""
    global config, db, cascade, pipeline

    config_path = Path("config.yaml")
    config = NexusConfig.from_yaml(config_path)

    logging.basicConfig(level=config.log_level)
    logger.info("Nexus starting up")

    db = DatabaseManager(config.resolved_db_path)
    await db.initialize()

    cascade = _build_cascade(config)
    pipeline = EmbeddingPipeline(
        cascade=cascade,
        db=db,
        model_name=config.embedding.model,
        dimensions=config.embedding.dimensions,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    logger.info("Nexus ready")
    yield

    await db.close()
    logger.info("Nexus shut down")


app = FastAPI(title="Nexus", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    """Health check endpoint."""
    ollama_ok = False
    if cascade:
        for p in cascade.providers:
            if p.provider_name == "ollama":
                ollama_ok = await p.is_available()
                break

    return {
        "status": "ok" if db else "error",
        "ollama": "available" if ollama_ok else "unavailable",
        "db": "connected" if db else "disconnected",
    }


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/query")
async def query(req: QueryRequest):
    """Semantic search endpoint."""
    if not pipeline:
        return {"error": "Pipeline not initialized"}

    results = await pipeline.search_similar(req.query, top_k=req.top_k)
    return {
        "results": [
            {"title": doc.title, "score": round(score, 4), "id": doc.id}
            for doc, score in results
        ]
    }


@app.get("/status")
async def status():
    """System status endpoint."""
    stats = await db.get_stats() if db else {}
    return {
        "version": "0.1.0",
        "db_stats": stats,
        "cascade_stats": {
            "total_requests": cascade.stats.total_requests,
            "total_tokens": cascade.stats.total_tokens,
            "failures": cascade.stats.failures,
            "fallbacks": cascade.stats.fallbacks,
            "provider_usage": cascade.stats.provider_usage,
        }
        if cascade
        else None,
    }
