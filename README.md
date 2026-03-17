# Nexus

Local-first Personal AI OS — a knowledge graph that watches your files, extracts entities, answers questions, and injects context into Claude Code sessions.

## Architecture

```
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │ File Watcher │────▶│   SQLite DB  │◀────│ Memory Agent │
 │  (watchdog)  │     │  (WAL mode)  │     │  (RAG + LLM) │
 └──────────────┘     └──────┬───────┘     └──────▲───────┘
                             │                     │
                     ┌───────┴───────┐     ┌───────┴───────┐
                     │  Embeddings   │     │  MCP Server   │
                     │ (numpy cosim) │     │  (5 tools)    │
                     └───────────────┘     └───────┬───────┘
                                                   │
 ┌──────────────┐     ┌──────────────┐     ┌───────┴───────┐
 │   Briefing   │     │   Project    │     │  Claude Code  │
 │    Agent     │     │   Context    │     │   (client)    │
 └──────────────┘     └──────────────┘     └───────────────┘
```

## Features

- **File Watcher** — monitors directories for changes, indexes content with SHA-256 dedup
- **Memory Agent** — RAG queries with semantic search + entity graph traversal + LLM synthesis
- **Briefing Agent** — scheduled daily briefings summarizing recent activity
- **Project Context** — generates context blocks with git info, key files, and related knowledge
- **MCP Server** — 5 tools exposing the knowledge graph directly to Claude Code
- **Entity Extraction** — automatic extraction of people, projects, technologies from indexed files
- **LLM Cascade** — Ollama → Groq → Claude fallback chain

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Configure (edit config.yaml for your paths and LLM providers)
cp config.yaml.example config.yaml  # if needed

# Run the FastAPI server
uvicorn nexus.main:app --reload

# Run as MCP server (for Claude Code integration)
python -m nexus.mcp

# Run tests
pytest tests/ -v
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (DB + Ollama status) |
| POST | `/query` | Knowledge query (search/answer/auto modes) |
| GET | `/query/history` | Recent query history |
| GET | `/entities` | List entities (optional `?type=` filter) |
| GET | `/entities/{name}` | Entity details + linked documents |
| GET | `/stats` | Document/embedding/entity counts + DB size |
| GET | `/status` | System status + cascade stats |
| GET | `/watcher/status` | File watcher status |
| GET | `/files` | List indexed documents (optional `?category=` filter) |
| GET | `/files/{doc_id}` | Get document by ID |
| POST | `/briefing/generate` | Trigger manual briefing generation |
| GET | `/briefing/latest` | Get most recent briefing |
| GET | `/briefing/history` | Briefing history |
| GET | `/context/{project_path}` | Generate project context block |

## MCP Server

The MCP server exposes 5 tools to Claude Code:

| Tool | Description |
|------|-------------|
| `nexus_search` | Semantic search over the knowledge graph |
| `nexus_ask` | Ask a question, get an LLM-synthesized answer with sources |
| `nexus_context` | Generate project context block for a directory |
| `nexus_entities` | List entities, optionally filtered by type |
| `nexus_stats` | Knowledge graph statistics |

### Setup

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "nexus": {
      "command": "python",
      "args": ["-m", "nexus.mcp"],
      "cwd": "/path/to/Nexus"
    }
  }
}
```

## Tech Stack

- **Python 3.11+** with async/await throughout
- **FastAPI** + uvicorn for the HTTP API
- **SQLite** (async via aiosqlite) in WAL mode
- **numpy** for cosine similarity vector search
- **Ollama** for local LLM + embeddings (nomic-embed-text, 768 dims)
- **Groq** as fallback LLM provider
- **watchdog** for filesystem monitoring
- **MCP** (Model Context Protocol) for Claude Code integration

## Project Structure

```
nexus/
├── config.py              # Pydantic config from YAML + env vars
├── main.py                # FastAPI app with all endpoints
├── agents/
│   ├── base.py            # BaseAgent ABC with lifecycle
│   ├── briefing.py        # Daily briefing generator
│   ├── file_watcher.py    # Directory monitoring + indexing
│   ├── project_context.py # Project context assembler
│   └── memory/
│       ├── agent.py       # RAG query engine
│       ├── context.py     # Context assembly + source ranking
│       ├── entities.py    # Entity extraction
│       └── prompts.py     # LLM prompt templates
├── db/
│   ├── database.py        # DatabaseManager with migrations
│   ├── models.py          # Pydantic/dataclass models
│   └── vectors.py         # EmbeddingPipeline + cosine search
├── llm/
│   ├── cascade.py         # CascadeManager + build_cascade()
│   ├── provider.py        # LLMProvider ABC
│   ├── ollama.py          # Ollama provider
│   └── groq.py            # Groq provider
├── mcp/
│   ├── __init__.py
│   ├── __main__.py        # Entry point: python -m nexus.mcp
│   └── server.py          # FastMCP server + 5 tools
├── scripts/
│   └── nexus-context.sh   # Session-start hook for Claude Code
└── tests/
    ├── conftest.py        # Shared fixtures
    ├── test_mcp/
    │   └── test_mcp_tools.py
    └── ...
```

## License

MIT
