"""Async SQLite database manager for Nexus."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import aiosqlite

from nexus.db.models import (
    AgentLog,
    Briefing,
    Document,
    Embedding,
    Entity,
    EntityRelation,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        file_path TEXT UNIQUE,
        file_type TEXT,
        category TEXT,
        source_agent TEXT,
        hash TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT
    );

    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        vector BLOB NOT NULL,
        model_name TEXT NOT NULL,
        dimensions INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        metadata_json TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(name, type)
    );

    CREATE TABLE IF NOT EXISTS entity_relations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
        target_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
        relation_type TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 1.0,
        metadata_json TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS briefings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        summary TEXT,
        delivered INTEGER NOT NULL DEFAULT 0,
        delivered_at TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS agent_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agent_name TEXT NOT NULL,
        action TEXT NOT NULL,
        input_summary TEXT,
        output_summary TEXT,
        tokens_used INTEGER NOT NULL DEFAULT 0,
        duration_ms INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'success',
        error_message TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_documents_file_path ON documents(file_path);
    CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
    CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents(file_type);
    CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
    CREATE INDEX IF NOT EXISTS idx_embeddings_document_id ON embeddings(document_id);
    CREATE INDEX IF NOT EXISTS idx_entities_name_type ON entities(name, type);
    CREATE INDEX IF NOT EXISTS idx_entity_relations_source ON entity_relations(source_id);
    CREATE INDEX IF NOT EXISTS idx_entity_relations_target ON entity_relations(target_id);
    CREATE INDEX IF NOT EXISTS idx_agent_logs_agent ON agent_logs(agent_name);
    CREATE INDEX IF NOT EXISTS idx_agent_logs_status ON agent_logs(status);
    CREATE INDEX IF NOT EXISTS idx_agent_logs_created ON agent_logs(created_at);
    """,
    2: """
    CREATE TABLE IF NOT EXISTS document_entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
        confidence REAL NOT NULL DEFAULT 1.0,
        context_snippet TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(document_id, entity_id)
    );

    CREATE INDEX IF NOT EXISTS idx_doc_entities_doc ON document_entities(document_id);
    CREATE INDEX IF NOT EXISTS idx_doc_entities_entity ON document_entities(entity_id);
    """,
}


class DatabaseManager:
    """Async SQLite database manager with WAL mode and migrations."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._db

    async def initialize(self) -> None:
        """Open connection, enable WAL mode, run migrations."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row

        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        await self._run_migrations()
        logger.info("Database initialized at %s", self.db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def _run_migrations(self) -> None:
        """Run pending schema migrations."""
        # Check if schema_version table exists
        cursor = await self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        table_exists = await cursor.fetchone()

        current_version = 0
        if table_exists:
            cursor = await self.db.execute(
                "SELECT MAX(version) FROM schema_version"
            )
            row = await cursor.fetchone()
            if row and row[0] is not None:
                current_version = row[0]

        for version in sorted(MIGRATIONS.keys()):
            if version > current_version:
                logger.info("Applying migration %d", version)
                await self.db.executescript(MIGRATIONS[version])
                await self.db.execute(
                    "INSERT INTO schema_version (version) VALUES (?)", (version,)
                )
                await self.db.commit()

    # --- Documents ---

    async def insert_document(self, doc: Document) -> int:
        """Insert a document and return its ID."""
        content_hash = hashlib.sha256(doc.content.encode()).hexdigest()
        cursor = await self.db.execute(
            """INSERT INTO documents (title, content, file_path, file_type, category, source_agent, hash, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc.title,
                doc.content,
                doc.file_path,
                doc.file_type,
                doc.category,
                doc.source_agent,
                content_hash,
                doc.created_at,
                doc.updated_at,
            ),
        )
        await self.db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_document(self, doc_id: int) -> Document | None:
        """Get a document by ID."""
        cursor = await self.db.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return Document(**dict(row))

    async def get_document_by_path(self, file_path: str) -> Document | None:
        """Get a document by file path."""
        cursor = await self.db.execute(
            "SELECT * FROM documents WHERE file_path = ?", (file_path,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return Document(**dict(row))

    async def update_document(self, doc_id: int, **kwargs: object) -> None:
        """Update document fields."""
        if "content" in kwargs and isinstance(kwargs["content"], str):
            kwargs["hash"] = hashlib.sha256(kwargs["content"].encode()).hexdigest()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [doc_id]
        await self.db.execute(
            f"UPDATE documents SET {sets} WHERE id = ?", values  # noqa: S608
        )
        await self.db.commit()

    async def delete_document(self, doc_id: int) -> None:
        """Delete a document (cascades to embeddings)."""
        await self.db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        await self.db.commit()

    async def list_documents(
        self, category: str | None = None, limit: int = 100
    ) -> list[Document]:
        """List documents, optionally filtered by category."""
        if category:
            cursor = await self.db.execute(
                "SELECT * FROM documents WHERE category = ? ORDER BY created_at DESC LIMIT ?",
                (category, limit),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM documents ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        rows = await cursor.fetchall()
        return [Document(**dict(r)) for r in rows]

    async def delete_document_by_path(self, file_path: str) -> bool:
        """Delete a document by file path. Returns True if a row was deleted."""
        cursor = await self.db.execute(
            "DELETE FROM documents WHERE file_path = ?", (file_path,)
        )
        await self.db.commit()
        return cursor.rowcount > 0

    # --- Embeddings ---

    async def insert_embedding(self, emb: Embedding) -> int:
        """Insert an embedding and return its ID."""
        cursor = await self.db.execute(
            """INSERT INTO embeddings (document_id, vector, model_name, dimensions, chunk_index, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                emb.document_id,
                emb.vector,
                emb.model_name,
                emb.dimensions,
                emb.chunk_index,
                emb.created_at,
            ),
        )
        await self.db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_embeddings_for_document(self, doc_id: int) -> list[Embedding]:
        """Get all embeddings for a document."""
        cursor = await self.db.execute(
            "SELECT * FROM embeddings WHERE document_id = ? ORDER BY chunk_index",
            (doc_id,),
        )
        rows = await cursor.fetchall()
        return [Embedding(**dict(r)) for r in rows]

    async def delete_embeddings_for_document(self, doc_id: int) -> None:
        """Delete all embeddings for a document."""
        await self.db.execute(
            "DELETE FROM embeddings WHERE document_id = ?", (doc_id,)
        )
        await self.db.commit()

    async def get_all_embeddings(self) -> list[Embedding]:
        """Get all embeddings (for brute-force search)."""
        cursor = await self.db.execute("SELECT * FROM embeddings")
        rows = await cursor.fetchall()
        return [Embedding(**dict(r)) for r in rows]

    # --- Entities ---

    async def insert_entity(self, entity: Entity) -> int:
        """Insert an entity (upsert on name+type)."""
        cursor = await self.db.execute(
            """INSERT INTO entities (name, type, metadata_json, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(name, type) DO UPDATE SET metadata_json = excluded.metadata_json""",
            (entity.name, entity.type, entity.metadata_json, entity.created_at),
        )
        await self.db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_entity(self, entity_id: int) -> Entity | None:
        """Get an entity by ID."""
        cursor = await self.db.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return Entity(**dict(row))

    async def search_entities(self, name: str | None = None, type_: str | None = None) -> list[Entity]:
        """Search entities by name and/or type."""
        conditions = []
        params: list[object] = []
        if name:
            conditions.append("name LIKE ?")
            params.append(f"%{name}%")
        if type_:
            conditions.append("type = ?")
            params.append(type_)
        where = " AND ".join(conditions) if conditions else "1=1"
        cursor = await self.db.execute(
            f"SELECT * FROM entities WHERE {where}", params  # noqa: S608
        )
        rows = await cursor.fetchall()
        return [Entity(**dict(r)) for r in rows]

    # --- Document-Entity Links ---

    async def link_entity_to_document(
        self,
        doc_id: int,
        entity_id: int,
        confidence: float = 1.0,
        context_snippet: str | None = None,
    ) -> int:
        """Link an entity to a document."""
        cursor = await self.db.execute(
            """INSERT INTO document_entities (document_id, entity_id, confidence, context_snippet)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(document_id, entity_id) DO UPDATE SET
                   confidence = excluded.confidence,
                   context_snippet = excluded.context_snippet""",
            (doc_id, entity_id, confidence, context_snippet),
        )
        await self.db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_entities_for_document(self, doc_id: int) -> list[Entity]:
        """Get all entities linked to a document."""
        cursor = await self.db.execute(
            """SELECT e.* FROM entities e
               JOIN document_entities de ON e.id = de.entity_id
               WHERE de.document_id = ?""",
            (doc_id,),
        )
        rows = await cursor.fetchall()
        return [Entity(**dict(r)) for r in rows]

    async def get_documents_for_entity(self, entity_id: int) -> list[Document]:
        """Get all documents linked to an entity."""
        cursor = await self.db.execute(
            """SELECT d.* FROM documents d
               JOIN document_entities de ON d.id = de.document_id
               WHERE de.entity_id = ?""",
            (entity_id,),
        )
        rows = await cursor.fetchall()
        return [Document(**dict(r)) for r in rows]

    async def delete_entity_links_for_document(self, doc_id: int) -> None:
        """Delete all entity links for a document."""
        await self.db.execute(
            "DELETE FROM document_entities WHERE document_id = ?", (doc_id,)
        )
        await self.db.commit()

    async def get_entity_by_name(self, name: str) -> list[Entity]:
        """Get entities by exact name match."""
        cursor = await self.db.execute(
            "SELECT * FROM entities WHERE name = ?", (name,)
        )
        rows = await cursor.fetchall()
        return [Entity(**dict(r)) for r in rows]

    # --- Entity Relations ---

    async def insert_relation(self, rel: EntityRelation) -> int:
        """Insert an entity relation."""
        cursor = await self.db.execute(
            """INSERT INTO entity_relations (source_id, target_id, relation_type, confidence, metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                rel.source_id,
                rel.target_id,
                rel.relation_type,
                rel.confidence,
                rel.metadata_json,
                rel.created_at,
            ),
        )
        await self.db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_relations_for_entity(self, entity_id: int) -> list[EntityRelation]:
        """Get all relations where entity is source or target."""
        cursor = await self.db.execute(
            "SELECT * FROM entity_relations WHERE source_id = ? OR target_id = ?",
            (entity_id, entity_id),
        )
        rows = await cursor.fetchall()
        return [EntityRelation(**dict(r)) for r in rows]

    # --- Briefings ---

    async def insert_briefing(self, briefing: Briefing) -> int:
        """Insert a briefing."""
        cursor = await self.db.execute(
            """INSERT INTO briefings (content, summary, delivered, delivered_at, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                briefing.content,
                briefing.summary,
                int(briefing.delivered),
                briefing.delivered_at,
                briefing.created_at,
            ),
        )
        await self.db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_latest_briefing(self) -> Briefing | None:
        """Get the most recent briefing."""
        cursor = await self.db.execute(
            "SELECT * FROM briefings ORDER BY created_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return Briefing(**dict(row))

    async def mark_briefing_delivered(self, briefing_id: int) -> None:
        """Mark a briefing as delivered."""
        from datetime import UTC, datetime

        await self.db.execute(
            "UPDATE briefings SET delivered = 1, delivered_at = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), briefing_id),
        )
        await self.db.commit()

    # --- Agent Logs ---

    async def insert_agent_log(self, log: AgentLog) -> int:
        """Insert an agent log entry."""
        cursor = await self.db.execute(
            """INSERT INTO agent_logs (agent_name, action, input_summary, output_summary, tokens_used, duration_ms, status, error_message, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                log.agent_name,
                log.action,
                log.input_summary,
                log.output_summary,
                log.tokens_used,
                log.duration_ms,
                log.status,
                log.error_message,
                log.created_at,
            ),
        )
        await self.db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_agent_logs(
        self, agent_name: str | None = None, limit: int = 50
    ) -> list[AgentLog]:
        """Get recent agent logs."""
        if agent_name:
            cursor = await self.db.execute(
                "SELECT * FROM agent_logs WHERE agent_name = ? ORDER BY created_at DESC LIMIT ?",
                (agent_name, limit),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM agent_logs ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        rows = await cursor.fetchall()
        return [AgentLog(**dict(r)) for r in rows]

    # --- Stats ---

    async def get_stats(self) -> dict[str, int]:
        """Get row counts for all tables."""
        stats = {}
        for table in ["documents", "embeddings", "entities", "entity_relations", "document_entities", "briefings", "agent_logs"]:
            cursor = await self.db.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
            row = await cursor.fetchone()
            stats[table] = row[0] if row else 0
        return stats
