"""Content extractors for different file types."""

from __future__ import annotations

import asyncio
import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexus.agents.file_watcher.categories import classify_file

logger = logging.getLogger(__name__)


@dataclass
class ProcessedFile:
    """Result of extracting content from a file."""

    title: str
    content: str
    file_type: str
    category: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseExtractor(ABC):
    """Abstract base for file content extractors."""

    @abstractmethod
    async def extract(self, path: Path) -> ProcessedFile:
        ...


class PDFExtractor(BaseExtractor):
    """Extract text from PDF files using PyMuPDF."""

    async def extract(self, path: Path) -> ProcessedFile:
        def _extract() -> tuple[str, dict[str, Any]]:
            import pymupdf

            doc = pymupdf.open(str(path))
            pages_text = []
            for page in doc:
                pages_text.append(page.get_text())
            metadata = dict(doc.metadata) if doc.metadata else {}
            metadata["page_count"] = len(doc)
            doc.close()
            return "\n".join(pages_text), metadata

        content, metadata = await asyncio.to_thread(_extract)
        metadata["size_bytes"] = path.stat().st_size

        return ProcessedFile(
            title=path.stem,
            content=content.strip(),
            file_type="pdf",
            category="documents",
            metadata=metadata,
        )


class CodeExtractor(BaseExtractor):
    """Extract content from source code files."""

    async def extract(self, path: Path) -> ProcessedFile:
        raw = await asyncio.to_thread(path.read_bytes)

        import chardet

        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"

        try:
            content = raw.decode(encoding, errors="replace")
        except (UnicodeDecodeError, LookupError):
            content = raw.decode("utf-8", errors="replace")

        lines = content.splitlines()
        return ProcessedFile(
            title=path.name,
            content=content,
            file_type=path.suffix.lstrip("."),
            category="code",
            metadata={
                "line_count": len(lines),
                "encoding": encoding,
                "size_bytes": len(raw),
                "language": path.suffix.lstrip("."),
            },
        )


class ImageExtractor(BaseExtractor):
    """Extract text from images via OCR (if tesseract available), else metadata only."""

    def __init__(self) -> None:
        self._has_tesseract = shutil.which("tesseract") is not None
        if not self._has_tesseract:
            logger.info("Tesseract not found — image OCR disabled, metadata only")

    async def extract(self, path: Path) -> ProcessedFile:
        metadata: dict[str, Any] = {
            "size_bytes": path.stat().st_size,
            "format": path.suffix.lstrip(".").lower(),
        }
        content = ""

        if self._has_tesseract:
            try:
                import pytesseract
                from PIL import Image

                def _ocr() -> str:
                    img = Image.open(path)
                    return pytesseract.image_to_string(img)

                content = await asyncio.to_thread(_ocr)
                metadata["ocr"] = True
            except Exception as e:
                logger.warning("OCR failed for %s: %s", path, e)
                metadata["ocr_error"] = str(e)
        else:
            metadata["ocr"] = False

        return ProcessedFile(
            title=path.stem,
            content=content.strip() if content else f"[Image: {path.name}]",
            file_type=path.suffix.lstrip(".").lower(),
            category="images",
            metadata=metadata,
        )


class TextExtractor(BaseExtractor):
    """Fallback extractor for plain text and unknown types."""

    async def extract(self, path: Path) -> ProcessedFile:
        raw = await asyncio.to_thread(path.read_bytes)

        import chardet

        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"

        try:
            content = raw.decode(encoding, errors="replace")
        except (UnicodeDecodeError, LookupError):
            content = raw.decode("utf-8", errors="replace")

        return ProcessedFile(
            title=path.stem,
            content=content,
            file_type=path.suffix.lstrip(".") or "txt",
            category=classify_file(path),
            metadata={
                "encoding": encoding,
                "size_bytes": len(raw),
            },
        )
