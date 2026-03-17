"""Nexus — FastAPI application."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from nexus.agents.briefing import BriefingAgent
from nexus.agents.file_watcher import FileWatcherAgent
from nexus.agents.memory import MemoryAgent
from nexus.agents.memory.agent import QueryMode
from nexus.agents.project_context import ProjectContextAgent
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
watcher: FileWatcherAgent | None = None
memory: MemoryAgent | None = None
briefing_agent: BriefingAgent | None = None
project_context_agent: ProjectContextAgent | None = None


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
    """Application lifespan — init config, DB, cascade, watcher, memory."""
    global config, db, cascade, pipeline, watcher, memory, briefing_agent, project_context_agent

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

    # Initialize memory agent
    memory = MemoryAgent(
        name="memory",
        description="RAG-based knowledge retrieval",
        cascade=cascade,
        db=db,
        pipeline=pipeline,
    )

    # Initialize briefing agent
    briefing_agent = BriefingAgent(
        name="briefing",
        description="Daily briefing generator",
        cascade=cascade,
        db=db,
        config=config,
        memory=memory,
    )

    # Initialize project context agent
    project_context_agent = ProjectContextAgent(
        name="project_context",
        description="Project context assembler",
        cascade=cascade,
        db=db,
        memory=memory,
    )

    # Start scheduler if briefing enabled
    scheduler = None
    if config.briefing.enabled:
        try:
            from apscheduler import AsyncScheduler
            from apscheduler.triggers.cron import CronTrigger

            hour, minute = config.briefing.schedule.split(":")
            scheduler = AsyncScheduler()
            await scheduler.__aenter__()
            await scheduler.add_schedule(
                briefing_agent.generate_briefing,
                CronTrigger(hour=int(hour), minute=int(minute), timezone=config.briefing.timezone),
                id="daily-briefing",
            )
            await scheduler.start_in_background()
            logger.info("Briefing scheduler started: %s %s", config.briefing.schedule, config.briefing.timezone)
        except Exception as e:
            logger.warning("Failed to start briefing scheduler: %s", e)
            scheduler = None

    # Start file watcher if enabled
    watcher_task = None
    if config.file_watcher.enabled:
        watcher = FileWatcherAgent(
            name="file_watcher",
            description="Monitors directories for file changes",
            cascade=cascade,
            db=db,
            pipeline=pipeline,
            config=config,
        )
        # Wire entity extraction hook
        watcher.entity_hook = memory.extract_entities
        watcher_task = asyncio.create_task(watcher.start())

    logger.info("Nexus ready")
    yield

    # Shutdown
    if watcher:
        await watcher.stop()
    if watcher_task and not watcher_task.done():
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass

    if scheduler:
        try:
            await scheduler.__aexit__(None, None, None)
        except Exception as e:
            logger.warning("Scheduler shutdown error: %s", e)

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
    question: str
    mode: str = "auto"
    top_k: int = 10


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    """Knowledge query endpoint — semantic + entity search with optional LLM synthesis."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory agent not initialized")

    try:
        mode = QueryMode(req.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {req.mode}")

    result = await memory.query(question=req.question, mode=mode, top_k=req.top_k)
    return result.to_dict()


@app.get("/query/history")
async def query_history(limit: int = 20):
    """Get recent query history from agent logs."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    logs = await db.get_agent_logs(agent_name="memory", limit=limit)
    return {
        "history": [
            {
                "id": log.id,
                "action": log.action,
                "question": log.input_summary,
                "result": log.output_summary,
                "tokens_used": log.tokens_used,
                "duration_ms": log.duration_ms,
                "status": log.status,
                "created_at": log.created_at,
            }
            for log in logs
        ]
    }


@app.get("/entities")
async def list_entities(type: str | None = None, limit: int = 100):
    """List entities with optional type filter."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    entities = await db.search_entities(type_=type)
    return {
        "entities": [
            {
                "id": e.id,
                "name": e.name,
                "type": e.type,
                "created_at": e.created_at,
            }
            for e in entities[:limit]
        ]
    }


@app.get("/entities/{name}")
async def get_entity_details(name: str):
    """Get entity details + related documents."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    entities = await db.get_entity_by_name(name)
    if not entities:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")

    result = []
    for entity in entities:
        docs = await db.get_documents_for_entity(entity.id) if entity.id else []
        result.append({
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "created_at": entity.created_at,
            "documents": [
                {
                    "id": d.id,
                    "title": d.title,
                    "file_path": d.file_path,
                    "category": d.category,
                }
                for d in docs
            ],
        })
    return {"entities": result}


@app.get("/stats")
async def stats():
    """Enhanced statistics — document/embedding/entity counts + DB file size."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    row_counts = await db.get_stats()

    db_size_bytes = 0
    try:
        db_size_bytes = os.path.getsize(db.db_path)
    except OSError:
        pass

    return {
        "version": "0.1.0",
        "db_stats": row_counts,
        "db_size_bytes": db_size_bytes,
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


@app.get("/status")
async def status():
    """System status endpoint."""
    stats_data = await db.get_stats() if db else {}
    return {
        "version": "0.1.0",
        "db_stats": stats_data,
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


@app.get("/watcher/status")
async def watcher_status():
    """File watcher status endpoint."""
    if not watcher:
        return {"running": False, "status": "disabled"}
    return watcher.get_status()


@app.get("/files")
async def list_files(category: str | None = None, limit: int = 100):
    """List recently indexed documents."""
    if not db:
        return {"error": "Database not initialized"}
    docs = await db.list_documents(category=category, limit=limit)
    return {
        "documents": [
            {
                "id": d.id,
                "title": d.title,
                "file_path": d.file_path,
                "file_type": d.file_type,
                "category": d.category,
                "created_at": d.created_at,
                "updated_at": d.updated_at,
            }
            for d in docs
        ]
    }


@app.get("/files/{doc_id}")
async def get_file(doc_id: int):
    """Get a single document by ID."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": doc.id,
        "title": doc.title,
        "content": doc.content[:5000],
        "file_path": doc.file_path,
        "file_type": doc.file_type,
        "category": doc.category,
        "hash": doc.hash,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
    }


# --- Briefing Endpoints ---


@app.post("/briefing/generate")
async def generate_briefing():
    """Trigger briefing generation manually."""
    if not briefing_agent:
        raise HTTPException(status_code=503, detail="Briefing agent not initialized")
    result = await briefing_agent.generate_briefing()
    return {
        "content": result.content,
        "summary": result.summary,
        "sections": result.sections,
        "generated_at": result.generated_at,
    }


@app.get("/briefing/latest")
async def get_latest_briefing():
    """Get the most recent briefing."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    briefing = await db.get_latest_briefing()
    if not briefing:
        raise HTTPException(status_code=404, detail="No briefings found")
    return {
        "id": briefing.id,
        "content": briefing.content,
        "summary": briefing.summary,
        "delivered": briefing.delivered,
        "delivered_at": briefing.delivered_at,
        "created_at": briefing.created_at,
    }


@app.get("/briefing/history")
async def briefing_history(limit: int = 10):
    """Get recent briefing history."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    briefings = await db.list_briefings(limit=limit)
    return {
        "briefings": [
            {
                "id": b.id,
                "summary": b.summary,
                "delivered": b.delivered,
                "created_at": b.created_at,
            }
            for b in briefings
        ]
    }


# --- Project Context Endpoints ---


@app.get("/context/{project_path:path}")
async def get_project_context(project_path: str):
    """Generate project context for a given path."""
    if not project_context_agent:
        raise HTTPException(status_code=503, detail="Project context agent not initialized")

    resolved = Path(project_path).resolve()
    if not resolved.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {project_path}")

    ctx = await project_context_agent.get_context(str(resolved))
    return {
        "project_name": ctx.project_name,
        "branch": ctx.branch,
        "recent_commits": ctx.recent_commits,
        "uncommitted_changes": ctx.uncommitted_changes,
        "key_files": ctx.key_files,
        "related_knowledge": [
            {"title": s.title, "score": round(s.score, 4)} for s in ctx.related_knowledge
        ],
        "context_block": ctx.context_block,
        "generated_at": ctx.generated_at,
    }
