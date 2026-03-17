"""Tests for the file watcher agent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from nexus.agents.base import AgentResult, AgentStatus
from nexus.agents.file_watcher.agent import FileWatcherAgent
from nexus.agents.file_watcher.categories import classify_file, get_extractor_key
from nexus.agents.file_watcher.extractors import (
    CodeExtractor,
    ImageExtractor,
    ProcessedFile,
    TextExtractor,
)
from nexus.agents.file_watcher.handler import FileEvent, FileEventHandler
from nexus.agents.file_watcher.processor import FileProcessor
from nexus.config import FileWatcherConfig, NexusConfig
from nexus.db.database import DatabaseManager
from nexus.db.models import Document
from nexus.db.vectors import EmbeddingPipeline
from nexus.llm.cascade import CascadeManager


# --- Category classification ---


class TestCategories:
    def test_pdf_is_documents(self):
        assert classify_file(Path("report.pdf")) == "documents"

    def test_txt_is_documents(self):
        assert classify_file(Path("notes.txt")) == "documents"

    def test_md_is_documents(self):
        assert classify_file(Path("README.md")) == "documents"

    def test_py_is_code(self):
        assert classify_file(Path("main.py")) == "code"

    def test_js_is_code(self):
        assert classify_file(Path("app.js")) == "code"

    def test_png_is_images(self):
        assert classify_file(Path("photo.png")) == "images"

    def test_csv_is_data(self):
        assert classify_file(Path("data.csv")) == "data"

    def test_unknown_is_other(self):
        assert classify_file(Path("mystery.xyz")) == "other"

    def test_extractor_key_pdf(self):
        assert get_extractor_key(Path("doc.pdf")) == "pdf"

    def test_extractor_key_py(self):
        assert get_extractor_key(Path("main.py")) == "code"

    def test_extractor_key_png(self):
        assert get_extractor_key(Path("img.png")) == "image"

    def test_extractor_key_unknown(self):
        assert get_extractor_key(Path("file.xyz")) == "text"


# --- Text extraction ---


class TestTextExtractor:
    @pytest.mark.asyncio
    async def test_extract_text_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("Hello, world!", encoding="utf-8")

        extractor = TextExtractor()
        result = await extractor.extract(f)

        assert result.content == "Hello, world!"
        assert result.title == "hello"
        assert result.file_type == "txt"
        assert result.metadata["encoding"] is not None

    @pytest.mark.asyncio
    async def test_extract_handles_binary_gracefully(self, tmp_path):
        f = tmp_path / "binary.dat"
        f.write_bytes(b"\x80\x81\x82\x83 some text \xff\xfe")

        extractor = TextExtractor()
        result = await extractor.extract(f)

        assert "some text" in result.content


# --- Code extraction ---


class TestCodeExtractor:
    @pytest.mark.asyncio
    async def test_extract_python_file(self, tmp_path):
        f = tmp_path / "example.py"
        f.write_text("def foo():\n    return 42\n", encoding="utf-8")

        extractor = CodeExtractor()
        result = await extractor.extract(f)

        assert "def foo():" in result.content
        assert result.file_type == "py"
        assert result.category == "code"
        assert result.metadata["line_count"] == 2
        assert result.metadata["language"] == "py"


# --- PDF extraction ---


class TestPDFExtractor:
    @pytest.mark.asyncio
    async def test_extract_pdf(self, tmp_path):
        """Create a small PDF with PyMuPDF and verify extraction."""
        import pymupdf

        pdf_path = tmp_path / "test.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello PDF World")
        doc.save(str(pdf_path))
        doc.close()

        from nexus.agents.file_watcher.extractors import PDFExtractor

        extractor = PDFExtractor()
        result = await extractor.extract(pdf_path)

        assert "Hello PDF World" in result.content
        assert result.file_type == "pdf"
        assert result.category == "documents"
        assert result.metadata["page_count"] == 1


# --- Image extraction (no tesseract) ---


class TestImageExtractor:
    @pytest.mark.asyncio
    async def test_extract_without_tesseract(self, tmp_path):
        img_path = tmp_path / "test.png"
        # Write a minimal 1x1 PNG
        import struct
        import zlib

        def _minimal_png() -> bytes:
            sig = b"\x89PNG\r\n\x1a\n"
            ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
            ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc & 0xFFFFFFFF)
            raw = b"\x00\x00\x00\x00"
            idat_data = zlib.compress(raw)
            idat_crc = zlib.crc32(b"IDAT" + idat_data)
            idat = struct.pack(">I", len(idat_data)) + b"IDAT" + idat_data + struct.pack(">I", idat_crc & 0xFFFFFFFF)
            iend_crc = zlib.crc32(b"IEND")
            iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc & 0xFFFFFFFF)
            return sig + ihdr + idat + iend

        img_path.write_bytes(_minimal_png())

        with patch("shutil.which", return_value=None):
            extractor = ImageExtractor()
            result = await extractor.extract(img_path)

        assert result.category == "images"
        assert result.metadata["ocr"] is False
        assert "[Image: test.png]" in result.content


# --- Processor routing ---


class TestProcessor:
    @pytest.mark.asyncio
    async def test_routes_txt_to_text_extractor(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content", encoding="utf-8")

        processor = FileProcessor()
        result = await processor.process(f)
        assert result.content == "content"

    @pytest.mark.asyncio
    async def test_routes_py_to_code_extractor(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n", encoding="utf-8")

        processor = FileProcessor()
        result = await processor.process(f)
        assert result.category == "code"
        assert result.metadata.get("line_count") == 1


# --- Ignore patterns ---


class TestIgnorePatterns:
    def test_tmp_ignored(self):
        config = FileWatcherConfig()
        import fnmatch
        assert any(fnmatch.fnmatch("download.tmp", p) for p in config.ignore_patterns)

    def test_thumbs_db_ignored(self):
        config = FileWatcherConfig()
        import fnmatch
        assert any(fnmatch.fnmatch("Thumbs.db", p) for p in config.ignore_patterns)

    def test_normal_txt_not_ignored(self):
        config = FileWatcherConfig()
        import fnmatch
        assert not any(fnmatch.fnmatch("notes.txt", p) for p in config.ignore_patterns)


# --- File size check ---


class TestFileSizeCheck:
    @pytest.mark.asyncio
    async def test_oversized_file_skipped(self, tmp_path, db, cascade):
        """Files over max_file_size_mb should be skipped."""
        config = NexusConfig(
            watch_directories=[str(tmp_path)],
            db_path=str(tmp_path / "test.db"),
            file_watcher=FileWatcherConfig(max_file_size_mb=0),  # 0 MB = skip everything
        )
        pipeline = EmbeddingPipeline(cascade=cascade, db=db)

        agent = FileWatcherAgent(
            name="test_watcher",
            description="test",
            cascade=cascade,
            db=db,
            pipeline=pipeline,
            config=config,
        )

        f = tmp_path / "big.txt"
        f.write_text("some content")

        await agent._process_file(f)
        assert agent._stats["skipped"] == 1


# --- Dedup ---


class TestDedup:
    @pytest.mark.asyncio
    async def test_unchanged_file_skipped(self, tmp_path, db, cascade):
        """Processing the same unchanged file twice should skip on second run."""
        config = NexusConfig(
            watch_directories=[str(tmp_path)],
            db_path=str(tmp_path / "test.db"),
            file_watcher=FileWatcherConfig(
                summary_enabled=False,
                notification_enabled=False,
            ),
        )
        pipeline = EmbeddingPipeline(cascade=cascade, db=db)

        agent = FileWatcherAgent(
            name="test_watcher",
            description="test",
            cascade=cascade,
            db=db,
            pipeline=pipeline,
            config=config,
        )

        f = tmp_path / "notes.txt"
        f.write_text("hello world", encoding="utf-8")

        await agent._process_file(f)
        assert agent._stats["processed"] == 1

        # Process same file again — should skip
        await agent._process_file(f)
        assert agent._stats["processed"] == 1
        assert agent._stats["skipped"] == 1

    @pytest.mark.asyncio
    async def test_changed_file_updated(self, tmp_path, db, cascade):
        """Modifying file content should trigger an update."""
        config = NexusConfig(
            watch_directories=[str(tmp_path)],
            db_path=str(tmp_path / "test.db"),
            file_watcher=FileWatcherConfig(
                summary_enabled=False,
                notification_enabled=False,
            ),
        )
        pipeline = EmbeddingPipeline(cascade=cascade, db=db)

        agent = FileWatcherAgent(
            name="test_watcher",
            description="test",
            cascade=cascade,
            db=db,
            pipeline=pipeline,
            config=config,
        )

        f = tmp_path / "notes.txt"
        f.write_text("version 1", encoding="utf-8")
        await agent._process_file(f)
        assert agent._stats["processed"] == 1

        f.write_text("version 2", encoding="utf-8")
        await agent._process_file(f)
        assert agent._stats["processed"] == 2

        # Verify document was updated, not duplicated
        docs = await db.list_documents()
        assert len(docs) == 1


# --- Full pipeline mock ---


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_process_stores_and_embeds(self, tmp_path, db, cascade):
        """Full pipeline: extract → store → embed."""
        config = NexusConfig(
            watch_directories=[str(tmp_path)],
            db_path=str(tmp_path / "test.db"),
            file_watcher=FileWatcherConfig(
                summary_enabled=False,
                notification_enabled=False,
            ),
        )
        pipeline = EmbeddingPipeline(cascade=cascade, db=db)

        agent = FileWatcherAgent(
            name="test_watcher",
            description="test",
            cascade=cascade,
            db=db,
            pipeline=pipeline,
            config=config,
        )

        f = tmp_path / "test.txt"
        f.write_text("Important document content", encoding="utf-8")

        await agent._process_file(f)

        docs = await db.list_documents()
        assert len(docs) == 1
        assert docs[0].title == "test"
        assert docs[0].source_agent == "test_watcher"

        embeds = await db.get_embeddings_for_document(docs[0].id)
        assert len(embeds) > 0


# --- Error isolation ---


class TestErrorIsolation:
    @pytest.mark.asyncio
    async def test_extraction_error_logged(self, tmp_path, db, cascade):
        """Processor errors should be logged, not crash the agent."""
        config = NexusConfig(
            watch_directories=[str(tmp_path)],
            db_path=str(tmp_path / "test.db"),
            file_watcher=FileWatcherConfig(
                summary_enabled=False,
                notification_enabled=False,
            ),
        )
        pipeline = EmbeddingPipeline(cascade=cascade, db=db)

        agent = FileWatcherAgent(
            name="test_watcher",
            description="test",
            cascade=cascade,
            db=db,
            pipeline=pipeline,
            config=config,
        )

        # Patch processor to raise
        agent._processor = MagicMock()
        agent._processor.process = AsyncMock(side_effect=RuntimeError("extraction failed"))

        f = tmp_path / "bad.txt"
        f.write_text("data", encoding="utf-8")

        # _delayed_process catches errors
        await agent._delayed_process(f, "created")
        assert agent._stats["errors"] == 1

        logs = await db.get_agent_logs(agent_name="test_watcher")
        error_logs = [l for l in logs if l.status == "error"]
        assert len(error_logs) >= 1


# --- Handler bridge ---


class TestHandlerBridge:
    def test_on_created_pushes_event(self):
        """FileEventHandler should push events into the queue via the loop."""
        loop = MagicMock()
        queue = MagicMock()
        handler = FileEventHandler(loop=loop, queue=queue, ignore_patterns=["*.tmp"])

        event = MagicMock()
        event.is_directory = False
        event.src_path = "/tmp/test.txt"

        handler.on_created(event)

        loop.call_soon_threadsafe.assert_called_once()
        args = loop.call_soon_threadsafe.call_args
        assert args[0][0] == queue.put_nowait

    def test_ignored_file_not_pushed(self):
        """Ignored files should not be pushed to the queue."""
        loop = MagicMock()
        queue = MagicMock()
        handler = FileEventHandler(loop=loop, queue=queue, ignore_patterns=["*.tmp"])

        event = MagicMock()
        event.is_directory = False
        event.src_path = "/tmp/download.tmp"

        handler.on_created(event)

        loop.call_soon_threadsafe.assert_not_called()

    def test_directory_events_ignored(self):
        """Directory events should be ignored."""
        loop = MagicMock()
        queue = MagicMock()
        handler = FileEventHandler(loop=loop, queue=queue, ignore_patterns=[])

        event = MagicMock()
        event.is_directory = True
        event.src_path = "/tmp/somedir"

        handler.on_created(event)
        loop.call_soon_threadsafe.assert_not_called()


# --- Debounce ---


class TestDebounce:
    @pytest.mark.asyncio
    async def test_rapid_events_debounced(self, tmp_path, db, cascade):
        """Multiple rapid events for the same path should result in a single process."""
        config = NexusConfig(
            watch_directories=[str(tmp_path)],
            db_path=str(tmp_path / "test.db"),
            file_watcher=FileWatcherConfig(
                debounce_seconds=0.1,  # Short debounce for test
                summary_enabled=False,
                notification_enabled=False,
            ),
        )
        pipeline = EmbeddingPipeline(cascade=cascade, db=db)

        agent = FileWatcherAgent(
            name="test_watcher",
            description="test",
            cascade=cascade,
            db=db,
            pipeline=pipeline,
            config=config,
        )

        f = tmp_path / "rapid.txt"
        f.write_text("final content", encoding="utf-8")

        # Fire 5 rapid events
        for _ in range(5):
            agent._debounce(f, "modified")

        # Only 1 pending task (last one wins)
        assert len(agent._pending) == 1

        # Wait for debounce + processing
        await asyncio.sleep(0.3)

        assert agent._stats["processed"] == 1

        # Clean up pending tasks
        for task in agent._pending.values():
            task.cancel()


# --- DB additions ---


class TestDatabaseAdditions:
    @pytest.mark.asyncio
    async def test_delete_embeddings_for_document(self, db):
        doc = Document(title="t", content="c", file_path="/tmp/t.txt")
        doc_id = await db.insert_document(doc)

        from nexus.db.models import Embedding
        import numpy as np

        vec = np.zeros(768, dtype=np.float32).tobytes()
        emb = Embedding(document_id=doc_id, vector=vec, model_name="test", dimensions=768)
        await db.insert_embedding(emb)

        embeds = await db.get_embeddings_for_document(doc_id)
        assert len(embeds) == 1

        await db.delete_embeddings_for_document(doc_id)
        embeds = await db.get_embeddings_for_document(doc_id)
        assert len(embeds) == 0

    @pytest.mark.asyncio
    async def test_delete_document_by_path(self, db):
        doc = Document(title="t", content="c", file_path="/tmp/del.txt")
        await db.insert_document(doc)

        result = await db.delete_document_by_path("/tmp/del.txt")
        assert result is True

        result = await db.delete_document_by_path("/tmp/nonexistent.txt")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_document_by_path_not_found(self, db):
        result = await db.delete_document_by_path("/tmp/nope.txt")
        assert result is False


# --- Config ---


class TestFileWatcherConfig:
    def test_default_config(self):
        config = FileWatcherConfig()
        assert config.enabled is True
        assert config.debounce_seconds == 2.0
        assert config.max_file_size_mb == 50
        assert "*.tmp" in config.ignore_patterns

    def test_config_in_nexus_config(self):
        config = NexusConfig()
        assert config.file_watcher.enabled is True
