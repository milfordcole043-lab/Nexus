"""Configuration system for Nexus."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    type: Literal["ollama", "groq", "claude"]
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    max_tokens: int = 2048
    temperature: float = 0.7

    @property
    def api_key(self) -> str | None:
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    provider: str = "ollama"
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    dimensions: int = 768


class BriefingConfig(BaseModel):
    """Configuration for daily briefings."""

    schedule: str = "08:00"
    timezone: str = "Europe/Amsterdam"
    ntfy_topic: str = "nexus-briefing"
    enabled: bool = True
    include_git: bool = True
    lookback_hours: int = 24


class FileWatcherConfig(BaseModel):
    """Configuration for the file watcher agent."""

    enabled: bool = True
    debounce_seconds: float = 2.0
    max_file_size_mb: int = 50
    ignore_patterns: list[str] = [
        "*.tmp", "*.crdownload", "*.part", "*.partial",
        "Thumbs.db", ".DS_Store", "desktop.ini",
        "~$*", "*.swp", "*.swo",
    ]
    summary_enabled: bool = True
    notification_enabled: bool = True


class NexusConfig(BaseModel):
    """Root configuration for Nexus."""

    watch_directories: list[str] = Field(default_factory=lambda: ["~/Documents"])
    db_path: str = "./nexus.db"
    log_level: str = "INFO"
    chunk_size: int = 500
    chunk_overlap: int = 50
    search_mode: Literal["hybrid", "vector", "keyword"] = "hybrid"
    lancedb_path: str = "./nexus_lancedb"
    llm_cascade: list[LLMProviderConfig] = Field(default_factory=list)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    briefing: BriefingConfig = Field(default_factory=BriefingConfig)
    file_watcher: FileWatcherConfig = Field(default_factory=FileWatcherConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> NexusConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        return cls.model_validate(data)

    @property
    def resolved_lancedb_path(self) -> Path:
        """Resolve the LanceDB path relative to CWD."""
        return Path(self.lancedb_path).resolve()

    @property
    def resolved_db_path(self) -> Path:
        """Resolve the database path relative to CWD."""
        return Path(self.db_path).resolve()

    @property
    def resolved_watch_directories(self) -> list[Path]:
        """Expand ~ and resolve watch directories."""
        return [Path(d).expanduser().resolve() for d in self.watch_directories]
