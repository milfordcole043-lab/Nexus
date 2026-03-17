# Nexus — Project Instructions

## What is Nexus?
Local-first Personal AI OS with 4 agents: File Watcher, Memory, Briefing, Project Context.

## Tech Stack
- **Runtime**: Python 3.11+
- **API**: FastAPI + uvicorn
- **Database**: SQLite (async via aiosqlite, WAL mode)
- **Vector search**: numpy cosine similarity (Phase 1), LanceDB planned for Phase 2+
- **LLM cascade**: Ollama → Groq → Claude
- **Embeddings**: Ollama nomic-embed-text (768 dims)
- **Config**: Pydantic + YAML + env vars

## Key Commands
```bash
# Install
pip install -e ".[dev]"

# Run
uvicorn nexus.main:app --reload

# Test
pytest tests/ -v
```

## Architecture
- `nexus/config.py` — Pydantic config, loads from config.yaml
- `nexus/db/database.py` — DatabaseManager with async SQLite, WAL, migrations
- `nexus/db/vectors.py` — EmbeddingPipeline, cosine similarity search
- `nexus/llm/cascade.py` — CascadeManager tries providers in order
- `nexus/agents/base.py` — BaseAgent ABC with lifecycle management
- `nexus/main.py` — FastAPI app with /health, /query, /status endpoints

## Database
- 6 tables: documents, embeddings, entities, entity_relations, briefings, agent_logs
- Schema versioned via schema_version table + sequential migrations
- Embeddings stored as numpy BLOB in SQLite

## Rules
- All timestamps UTC ISO 8601
- Content hashes are SHA-256
- Embeddings are float32 numpy arrays stored via tobytes()
- Chunking is word-based with configurable size/overlap
- Agent results always logged to agent_logs table
