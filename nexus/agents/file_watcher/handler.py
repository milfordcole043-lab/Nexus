"""Watchdog event handler — bridges sync watchdog thread to async queue."""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from watchdog.events import FileSystemEvent, FileSystemEventHandler

logger = logging.getLogger(__name__)


@dataclass
class FileEvent:
    """A file system event to be processed asynchronously."""

    path: Path
    event_type: Literal["created", "modified", "deleted"]
    timestamp: float


class FileEventHandler(FileSystemEventHandler):
    """Bridges watchdog (sync/threaded) events into an asyncio.Queue."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[FileEvent],
        ignore_patterns: list[str],
    ) -> None:
        super().__init__()
        self._loop = loop
        self._queue = queue
        self._ignore_patterns = ignore_patterns

    def _should_ignore(self, path: Path) -> bool:
        name = path.name
        return any(fnmatch.fnmatch(name, pat) for pat in self._ignore_patterns)

    def _push(self, event: FileSystemEvent, event_type: Literal["created", "modified", "deleted"]) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path).resolve()
        if self._should_ignore(path):
            return
        fe = FileEvent(path=path, event_type=event_type, timestamp=time.time())
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, fe)
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping event for %s", path)

    def on_created(self, event: FileSystemEvent) -> None:
        self._push(event, "created")

    def on_modified(self, event: FileSystemEvent) -> None:
        self._push(event, "modified")

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._push(event, "deleted")
