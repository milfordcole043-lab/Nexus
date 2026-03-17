"""Tests for the database manager."""

from __future__ import annotations

import pytest

from nexus.db.database import DatabaseManager
from nexus.db.models import (
    AgentLog,
    Briefing,
    Document,
    Embedding,
    Entity,
    EntityRelation,
)


class TestDatabaseInit:
    async def test_creates_tables(self, db: DatabaseManager):
        cursor = await db.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in await cursor.fetchall()]
        assert "documents" in tables
        assert "embeddings" in tables
        assert "entities" in tables
        assert "entity_relations" in tables
        assert "briefings" in tables
        assert "agent_logs" in tables
        assert "schema_version" in tables

    async def test_wal_mode(self, db: DatabaseManager):
        cursor = await db.db.execute("PRAGMA journal_mode")
        row = await cursor.fetchone()
        assert row[0] == "wal"

    async def test_foreign_keys_enabled(self, db: DatabaseManager):
        cursor = await db.db.execute("PRAGMA foreign_keys")
        row = await cursor.fetchone()
        assert row[0] == 1

    async def test_schema_version(self, db: DatabaseManager):
        cursor = await db.db.execute("SELECT MAX(version) FROM schema_version")
        row = await cursor.fetchone()
        assert row[0] == 2

    async def test_idempotent_init(self, tmp_path):
        """Running initialize twice should not fail."""
        db = DatabaseManager(tmp_path / "test2.db")
        await db.initialize()
        await db.initialize()  # Second call should be safe
        await db.close()


class TestDocuments:
    async def test_insert_and_get(self, db: DatabaseManager):
        doc = Document(title="Test", content="Hello world", file_path="/tmp/test.txt")
        doc_id = await db.insert_document(doc)
        assert doc_id > 0

        fetched = await db.get_document(doc_id)
        assert fetched is not None
        assert fetched.title == "Test"
        assert fetched.content == "Hello world"
        assert fetched.hash is not None

    async def test_get_by_path(self, db: DatabaseManager):
        doc = Document(title="Test", content="content", file_path="/tmp/unique.txt")
        await db.insert_document(doc)

        fetched = await db.get_document_by_path("/tmp/unique.txt")
        assert fetched is not None
        assert fetched.title == "Test"

    async def test_update(self, db: DatabaseManager):
        doc = Document(title="Old", content="old content", file_path="/tmp/upd.txt")
        doc_id = await db.insert_document(doc)

        await db.update_document(doc_id, title="New", content="new content")
        fetched = await db.get_document(doc_id)
        assert fetched.title == "New"
        assert fetched.content == "new content"

    async def test_delete_cascades_embeddings(self, db: DatabaseManager):
        doc = Document(title="Del", content="content")
        doc_id = await db.insert_document(doc)

        emb = Embedding(document_id=doc_id, vector=b"\x00" * 8, model_name="test", dimensions=2)
        await db.insert_embedding(emb)

        await db.delete_document(doc_id)
        embeddings = await db.get_embeddings_for_document(doc_id)
        assert len(embeddings) == 0

    async def test_list_documents(self, db: DatabaseManager):
        for i in range(3):
            await db.insert_document(
                Document(title=f"Doc {i}", content=f"content {i}", category="test")
            )

        docs = await db.list_documents(category="test")
        assert len(docs) == 3

    async def test_unique_file_path(self, db: DatabaseManager):
        doc = Document(title="A", content="a", file_path="/tmp/dup.txt")
        await db.insert_document(doc)

        with pytest.raises(Exception):
            await db.insert_document(doc)


class TestEntities:
    async def test_insert_and_search(self, db: DatabaseManager):
        entity = Entity(name="Python", type="language")
        eid = await db.insert_entity(entity)
        assert eid > 0

        results = await db.search_entities(name="Python")
        assert len(results) == 1
        assert results[0].name == "Python"

    async def test_upsert(self, db: DatabaseManager):
        e1 = Entity(name="FastAPI", type="framework", metadata_json='{"v": 1}')
        e2 = Entity(name="FastAPI", type="framework", metadata_json='{"v": 2}')
        await db.insert_entity(e1)
        await db.insert_entity(e2)  # Should upsert

        results = await db.search_entities(name="FastAPI")
        assert len(results) == 1
        assert '"v": 2' in results[0].metadata_json


class TestEntityRelations:
    async def test_insert_and_get(self, db: DatabaseManager):
        e1 = await db.insert_entity(Entity(name="Nexus", type="project"))
        e2 = await db.insert_entity(Entity(name="FastAPI", type="framework"))

        rel = EntityRelation(source_id=e1, target_id=e2, relation_type="uses")
        rid = await db.insert_relation(rel)
        assert rid > 0

        rels = await db.get_relations_for_entity(e1)
        assert len(rels) == 1
        assert rels[0].relation_type == "uses"


class TestBriefings:
    async def test_insert_and_get_latest(self, db: DatabaseManager):
        b = Briefing(content="Good morning", summary="Daily update")
        bid = await db.insert_briefing(b)
        assert bid > 0

        latest = await db.get_latest_briefing()
        assert latest is not None
        assert latest.content == "Good morning"
        assert latest.delivered is False

    async def test_mark_delivered(self, db: DatabaseManager):
        b = Briefing(content="Test")
        bid = await db.insert_briefing(b)
        await db.mark_briefing_delivered(bid)

        latest = await db.get_latest_briefing()
        assert latest.delivered is True
        assert latest.delivered_at is not None


class TestAgentLogs:
    async def test_insert_and_get(self, db: DatabaseManager):
        log = AgentLog(
            agent_name="test-agent",
            action="run",
            tokens_used=100,
            duration_ms=500,
            status="success",
        )
        lid = await db.insert_agent_log(log)
        assert lid > 0

        logs = await db.get_agent_logs(agent_name="test-agent")
        assert len(logs) == 1
        assert logs[0].tokens_used == 100


class TestStats:
    async def test_get_stats(self, db: DatabaseManager):
        await db.insert_document(Document(title="A", content="a"))
        stats = await db.get_stats()
        assert stats["documents"] == 1
        assert stats["embeddings"] == 0
