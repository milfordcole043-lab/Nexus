"""Tests for config loading and validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from nexus.config import NexusConfig


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)


class TestNexusConfig:
    def test_from_yaml_loads_defaults(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        _write_yaml(config_path, {})
        cfg = NexusConfig.from_yaml(config_path)
        assert cfg.db_path == "./nexus.db"
        assert cfg.chunk_size == 500
        assert cfg.log_level == "INFO"

    def test_from_yaml_loads_values(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        _write_yaml(
            config_path,
            {
                "db_path": "/tmp/test.db",
                "chunk_size": 1000,
                "log_level": "DEBUG",
                "watch_directories": ["/home/user/docs"],
            },
        )
        cfg = NexusConfig.from_yaml(config_path)
        assert cfg.db_path == "/tmp/test.db"
        assert cfg.chunk_size == 1000
        assert cfg.log_level == "DEBUG"
        assert "/home/user/docs" in cfg.watch_directories

    def test_from_yaml_missing_file(self):
        with pytest.raises(FileNotFoundError):
            NexusConfig.from_yaml("/nonexistent/config.yaml")

    def test_llm_cascade_config(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        _write_yaml(
            config_path,
            {
                "llm_cascade": [
                    {"type": "ollama", "model": "llama3.2", "base_url": "http://localhost:11434"},
                    {"type": "groq", "model": "llama-3.3-70b-versatile", "api_key_env": "GROQ_API_KEY"},
                ]
            },
        )
        cfg = NexusConfig.from_yaml(config_path)
        assert len(cfg.llm_cascade) == 2
        assert cfg.llm_cascade[0].type == "ollama"
        assert cfg.llm_cascade[1].type == "groq"

    def test_api_key_resolution(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        _write_yaml(
            config_path,
            {
                "llm_cascade": [
                    {"type": "groq", "model": "test", "api_key_env": "TEST_API_KEY"},
                ]
            },
        )
        os.environ["TEST_API_KEY"] = "secret-123"
        try:
            cfg = NexusConfig.from_yaml(config_path)
            assert cfg.llm_cascade[0].api_key == "secret-123"
        finally:
            del os.environ["TEST_API_KEY"]

    def test_embedding_config_defaults(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        _write_yaml(config_path, {})
        cfg = NexusConfig.from_yaml(config_path)
        assert cfg.embedding.model == "nomic-embed-text"
        assert cfg.embedding.dimensions == 768

    def test_briefing_config(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        _write_yaml(
            config_path,
            {"briefing": {"schedule": "09:00", "timezone": "US/Eastern", "ntfy_topic": "test"}},
        )
        cfg = NexusConfig.from_yaml(config_path)
        assert cfg.briefing.schedule == "09:00"
        assert cfg.briefing.timezone == "US/Eastern"

    def test_resolved_watch_directories(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        _write_yaml(config_path, {"watch_directories": ["~/Documents"]})
        cfg = NexusConfig.from_yaml(config_path)
        resolved = cfg.resolved_watch_directories
        assert len(resolved) == 1
        assert "~" not in str(resolved[0])
