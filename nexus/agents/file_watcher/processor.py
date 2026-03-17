"""File processor — routes files to the correct extractor."""

from __future__ import annotations

from pathlib import Path

from nexus.agents.file_watcher.categories import get_extractor_key
from nexus.agents.file_watcher.extractors import (
    BaseExtractor,
    CodeExtractor,
    ImageExtractor,
    PDFExtractor,
    ProcessedFile,
    TextExtractor,
)


class FileProcessor:
    """Routes files to the appropriate content extractor."""

    def __init__(self) -> None:
        self._extractors: dict[str, BaseExtractor] = {
            "pdf": PDFExtractor(),
            "code": CodeExtractor(),
            "image": ImageExtractor(),
            "text": TextExtractor(),
        }

    async def process(self, path: Path) -> ProcessedFile:
        """Extract content from a file using the appropriate extractor."""
        key = get_extractor_key(path)
        extractor = self._extractors.get(key, self._extractors["text"])
        return await extractor.extract(path)
