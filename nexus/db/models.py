"""Pydantic models for database rows."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A document stored in Nexus."""

    id: int | None = None
    title: str
    content: str
    file_path: str | None = None
    file_type: str | None = None
    category: str | None = None
    source_agent: str | None = None
    hash: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str | None = None


class Embedding(BaseModel):
    """An embedding vector linked to a document."""

    id: int | None = None
    document_id: int
    vector: bytes  # numpy array stored as BLOB
    model_name: str
    dimensions: int
    chunk_index: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class Entity(BaseModel):
    """A named entity extracted from documents."""

    id: int | None = None
    name: str
    type: str
    metadata_json: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class EntityRelation(BaseModel):
    """A relation between two entities."""

    id: int | None = None
    source_id: int
    target_id: int
    relation_type: str
    confidence: float = 1.0
    metadata_json: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class Briefing(BaseModel):
    """A daily briefing."""

    id: int | None = None
    content: str
    summary: str | None = None
    delivered: bool = False
    delivered_at: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class AgentLog(BaseModel):
    """A log entry from an agent execution."""

    id: int | None = None
    agent_name: str
    action: str
    input_summary: str | None = None
    output_summary: str | None = None
    tokens_used: int = 0
    duration_ms: int = 0
    status: str = "success"
    error_message: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
