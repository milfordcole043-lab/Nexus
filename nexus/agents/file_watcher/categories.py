"""File extension to category mapping."""

from __future__ import annotations

from pathlib import Path

EXTENSION_CATEGORIES: dict[str, str] = {
    # Documents
    ".pdf": "documents", ".txt": "documents", ".md": "documents",
    ".rst": "documents", ".doc": "documents", ".docx": "documents",
    ".rtf": "documents", ".odt": "documents",
    # Code
    ".py": "code", ".js": "code", ".ts": "code", ".jsx": "code",
    ".tsx": "code", ".json": "code", ".yaml": "code", ".yml": "code",
    ".html": "code", ".css": "code", ".scss": "code", ".go": "code",
    ".rs": "code", ".java": "code", ".c": "code", ".cpp": "code",
    ".h": "code", ".hpp": "code", ".sql": "code", ".sh": "code",
    ".bash": "code", ".toml": "code", ".xml": "code", ".ini": "code",
    ".cfg": "code", ".rb": "code", ".php": "code", ".swift": "code",
    ".kt": "code", ".cs": "code",
    # Images
    ".png": "images", ".jpg": "images", ".jpeg": "images",
    ".gif": "images", ".bmp": "images", ".tiff": "images",
    ".webp": "images", ".svg": "images",
    # Data
    ".csv": "data", ".tsv": "data", ".xlsx": "data", ".xls": "data",
    ".parquet": "data", ".sqlite": "data", ".db": "data",
}

EXTRACTOR_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".gif": "image", ".bmp": "image", ".tiff": "image",
    ".webp": "image",
}

# Code extensions use the "code" extractor
for ext, cat in EXTENSION_CATEGORIES.items():
    if cat == "code" and ext not in EXTRACTOR_MAP:
        EXTRACTOR_MAP[ext] = "code"


def classify_file(path: Path) -> str:
    """Return the category for a file based on its extension."""
    return EXTENSION_CATEGORIES.get(path.suffix.lower(), "other")


def get_extractor_key(path: Path) -> str:
    """Return the extractor key for a file based on its extension."""
    return EXTRACTOR_MAP.get(path.suffix.lower(), "text")
