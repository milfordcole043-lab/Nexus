"""File watcher agent — monitors directories and indexes file changes."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import UTC, datetime
from collections.abc import Awaitable, Callable
from pathlib import Path

from watchdog.observers import Observer

from nexus.agents.base import AgentResult, AgentStatus, BaseAgent
from nexus.agents.file_watcher.handler import FileEvent, FileEventHandler
from nexus.agents.file_watcher.processor import FileProcessor
from nexus.config import NexusConfig
from nexus.db.database import DatabaseManager
from nexus.db.models import AgentLog, Document
from nexus.db.vectors import EmbeddingPipeline
from nexus.llm.cascade import CascadeManager
from nexus.tools.notifications import send_notification

logger = logging.getLogger(__name__)


class FileWatcherAgent(BaseAgent):
    """Watches directories for file changes, extracts content, embeds, and notifies."""

    def __init__(
        self,
        name: str,
        description: str,
        cascade: CascadeManager,
        db: DatabaseManager,
        pipeline: EmbeddingPipeline,
        config: NexusConfig,
    ):
        super().__init__(name, description, cascade, db)
        self.pipeline = pipeline
        self.config = config
        self._observer: Observer | None = None
        self._queue: asyncio.Queue[FileEvent] | None = None
        self._pending: dict[Path, asyncio.Task] = {}  # type: ignore[type-arg]
        self._consumer_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._processor = FileProcessor()
        self._stats = {"processed": 0, "errors": 0, "skipped": 0}
        self.entity_hook: Callable[[int, str], Awaitable[list[int]]] | None = None

    async def _execute(self, input_data: str) -> AgentResult:
        """BaseAgent lifecycle hook — returns immediately for long-running agent."""
        return AgentResult(success=True, output="file watcher started")

    async def start(self) -> None:
        """Start watching directories."""
        # Log startup via BaseAgent lifecycle
        await self.run("start")

        self._queue = asyncio.Queue(maxsize=1000)
        loop = asyncio.get_running_loop()

        handler = FileEventHandler(
            loop=loop,
            queue=self._queue,
            ignore_patterns=self.config.file_watcher.ignore_patterns,
        )

        self._observer = Observer()
        for watch_dir in self.config.resolved_watch_directories:
            if watch_dir.exists():
                self._observer.schedule(handler, str(watch_dir), recursive=False)
                logger.info("Watching directory: %s", watch_dir)
            else:
                logger.warning("Watch directory does not exist: %s", watch_dir)

        self._observer.start()
        self.status = AgentStatus.RUNNING
        self._consumer_task = asyncio.create_task(self._consumer_loop())
        logger.info("File watcher agent started")

    async def stop(self) -> None:
        """Stop the watcher and clean up."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        for task in self._pending.values():
            task.cancel()
        self._pending.clear()

        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None

        self.status = AgentStatus.DONE
        logger.info("File watcher agent stopped")

    async def _consumer_loop(self) -> None:
        """Consume events from the queue and route them."""
        assert self._queue is not None
        try:
            while True:
                event = await self._queue.get()
                if event.event_type == "deleted":
                    await self._handle_delete(event.path)
                else:
                    self._debounce(event.path, event.event_type)
        except asyncio.CancelledError:
            return

    def _debounce(self, path: Path, event_type: str) -> None:
        """Cancel any pending task for this path, schedule a new delayed process."""
        if path in self._pending:
            self._pending[path].cancel()
        self._pending[path] = asyncio.create_task(
            self._delayed_process(path, event_type)
        )

    async def _delayed_process(self, path: Path, event_type: str) -> None:
        """Wait for debounce period, then process the file."""
        try:
            await asyncio.sleep(self.config.file_watcher.debounce_seconds)
            await self._process_file(path)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("Error processing %s: %s", path, e)
            self._stats["errors"] += 1
            await self._log_action(
                "process_file", str(path), status="error", error=str(e)
            )
        finally:
            self._pending.pop(path, None)

    async def _process_file(self, path: Path) -> None:
        """Full processing pipeline for a single file."""
        # 1. Validate
        if not path.exists():
            logger.debug("File no longer exists: %s", path)
            self._stats["skipped"] += 1
            return

        if self._should_ignore(path):
            self._stats["skipped"] += 1
            return

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.file_watcher.max_file_size_mb:
            logger.info("Skipping oversized file (%0.1f MB): %s", size_mb, path)
            self._stats["skipped"] += 1
            return

        # 2. Hash for dedup
        file_hash = await self._hash_file(path)
        file_path_str = str(path)

        existing = await self.db.get_document_by_path(file_path_str)
        if existing and existing.hash == file_hash:
            logger.debug("File unchanged (hash match): %s", path)
            self._stats["skipped"] += 1
            return

        # 3. Extract content
        try:
            processed = await self._processor.process(path)
        except PermissionError:
            # Retry once after a short wait (file may still be locked)
            await asyncio.sleep(2.0)
            processed = await self._processor.process(path)

        # 4. Store document
        now = datetime.now(UTC).isoformat()
        if existing:
            await self.pipeline.delete_embeddings(existing.id)  # type: ignore[arg-type]
            await self.db.delete_entity_links_for_document(existing.id)  # type: ignore[arg-type]
            await self.db.update_document(
                existing.id,  # type: ignore[arg-type]
                title=processed.title,
                content=processed.content,
                file_type=processed.file_type,
                category=processed.category,
                updated_at=now,
            )
            doc_id = existing.id
            action = "update_file"
        else:
            doc = Document(
                title=processed.title,
                content=processed.content,
                file_path=file_path_str,
                file_type=processed.file_type,
                category=processed.category,
                source_agent=self.name,
                created_at=now,
            )
            doc_id = await self.db.insert_document(doc)
            action = "index_file"

        # 5. Embed
        await self.pipeline.embed_and_store(doc_id, processed.content)  # type: ignore[arg-type]

        # 5.5 Extract entities (if hook available)
        if self.entity_hook:
            try:
                await self.entity_hook(doc_id, processed.content)  # type: ignore[arg-type]
            except Exception as e:
                logger.warning("Entity extraction failed for %s: %s", path, e)

        # 6. Summarize (best effort)
        summary = None
        if self.config.file_watcher.summary_enabled and processed.content.strip():
            try:
                resp = await self.cascade.generate(
                    prompt=f"Summarize this file in 1-2 sentences:\n\n{processed.content[:3000]}",
                    max_tokens=150,
                    temperature=0.3,
                )
                summary = resp.text
            except Exception as e:
                logger.warning("Summary generation failed for %s: %s", path, e)

        # 7. Notify
        if self.config.file_watcher.notification_enabled:
            title = f"Nexus: {action.replace('_', ' ').title()}"
            msg = f"{processed.title} ({processed.category})"
            if summary:
                msg += f"\n{summary}"
            await send_notification(
                topic=self.config.briefing.ntfy_topic,
                title=title,
                message=msg,
            )

        # 8. Log
        self._stats["processed"] += 1
        await self._log_action(
            action,
            file_path_str,
            output=f"{processed.category}/{processed.file_type} — {len(processed.content)} chars",
        )

    async def _handle_delete(self, path: Path) -> None:
        """Handle a file deletion event."""
        file_path_str = str(path)
        deleted = await self.db.delete_document_by_path(file_path_str)
        if deleted:
            logger.info("Removed indexed document: %s", path)
            await self._log_action("delete_file", file_path_str)

    def _should_ignore(self, path: Path) -> bool:
        """Check if a file should be ignored based on patterns."""
        import fnmatch

        name = path.name
        return any(fnmatch.fnmatch(name, pat) for pat in self.config.file_watcher.ignore_patterns)

    @staticmethod
    async def _hash_file(path: Path) -> str:
        """Compute SHA-256 hash of a file in a thread (chunked read)."""
        def _compute() -> str:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    h.update(chunk)
            return h.hexdigest()

        return await asyncio.to_thread(_compute)

    async def _log_action(
        self,
        action: str,
        input_summary: str,
        output: str | None = None,
        status: str = "success",
        error: str | None = None,
    ) -> None:
        """Insert an agent log entry."""
        await self.db.insert_agent_log(
            AgentLog(
                agent_name=self.name,
                action=action,
                input_summary=input_summary[:200],
                output_summary=output[:200] if output else None,
                status=status,
                error_message=error,
            )
        )

    def get_status(self) -> dict:
        """Return current watcher status."""
        return {
            "running": self.status == AgentStatus.RUNNING,
            "status": self.status.value,
            "processed": self._stats["processed"],
            "errors": self._stats["errors"],
            "skipped": self._stats["skipped"],
            "watched_dirs": [str(d) for d in self.config.resolved_watch_directories if d.exists()],
        }
