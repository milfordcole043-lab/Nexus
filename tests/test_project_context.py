"""Tests for the Project Context Agent."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from nexus.agents.memory.context import SourceDocument
from nexus.agents.project_context import ProjectContext, ProjectContextAgent
from nexus.db.models import Document


@pytest_asyncio.fixture
async def project_context_agent(db, cascade):
    """Provide a project context agent with test DB."""
    return ProjectContextAgent(
        name="project_context",
        description="Test project context",
        cascade=cascade,
        db=db,
    )


@pytest.mark.asyncio
async def test_get_context_with_git(project_context_agent, tmp_path):
    """Temp dir, mock git commands → branch/commits/changes populated."""
    responses = {
        ("rev-parse", "--abbrev-ref", "HEAD"): b"feature/test\n",
        ("log", "--oneline", "-5"): b"abc1234 First commit\ndef5678 Second commit\n",
        ("status", "--porcelain"): b" M file.py\n?? new.txt\n",
    }

    async def mock_exec(*args, **kwargs):
        proc = AsyncMock()
        # Find which git subcommand
        git_args = tuple(a for a in args[2:] if a != "-C" and a != str(tmp_path))
        for key, output in responses.items():
            if all(k in git_args for k in key):
                proc.communicate = AsyncMock(return_value=(output, b""))
                proc.returncode = 0
                return proc
        proc.communicate = AsyncMock(return_value=(b"", b""))
        proc.returncode = 1
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
        ctx = await project_context_agent.get_context(str(tmp_path))

    assert ctx.branch == "feature/test"
    assert len(ctx.recent_commits) == 2
    assert len(ctx.uncommitted_changes) == 2


@pytest.mark.asyncio
async def test_get_context_no_git(project_context_agent, tmp_path):
    """Temp dir, not a git repo → branch=None, commits=[], no error."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"fatal: not a git repo"))
    mock_proc.returncode = 128

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        ctx = await project_context_agent.get_context(str(tmp_path))

    assert ctx.branch is None
    assert ctx.recent_commits == []
    assert ctx.uncommitted_changes == []


@pytest.mark.asyncio
async def test_git_not_installed(project_context_agent, tmp_path):
    """Mock FileNotFoundError → graceful fallback."""
    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("git")):
        ctx = await project_context_agent.get_context(str(tmp_path))

    assert ctx.branch is None
    assert ctx.recent_commits == []


@pytest.mark.asyncio
async def test_key_files_detected(project_context_agent, tmp_path):
    """Create CLAUDE.md + README.md in tmp_path → both in key_files."""
    (tmp_path / "CLAUDE.md").write_text("# Project rules")
    (tmp_path / "README.md").write_text("# My Project\nDescription here")

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 128

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        ctx = await project_context_agent.get_context(str(tmp_path))

    assert "CLAUDE.md" in ctx.key_files
    assert "README.md" in ctx.key_files
    assert "Project rules" in ctx.key_files["CLAUDE.md"]


@pytest.mark.asyncio
async def test_key_files_missing(project_context_agent, tmp_path):
    """Empty tmp dir → key_files empty, no error."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 128

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        ctx = await project_context_agent.get_context(str(tmp_path))

    assert ctx.key_files == {}


@pytest.mark.asyncio
async def test_memory_integration(db, cascade, tmp_path):
    """Mock memory.query → related_knowledge populated."""
    mock_memory = MagicMock()
    mock_response = MagicMock()
    mock_response.sources = [
        SourceDocument(doc_id=1, title="Related Doc", file_path="/test", score=0.9, snippet="test"),
    ]
    mock_memory.query = AsyncMock(return_value=mock_response)

    agent = ProjectContextAgent(
        name="project_context",
        description="Test",
        cascade=cascade,
        db=db,
        memory=mock_memory,
    )

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 128

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        ctx = await agent.get_context(str(tmp_path))

    assert len(ctx.related_knowledge) == 1
    assert ctx.related_knowledge[0].title == "Related Doc"


@pytest.mark.asyncio
async def test_memory_none(project_context_agent, tmp_path):
    """memory=None → related_knowledge empty."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 128

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        ctx = await project_context_agent.get_context(str(tmp_path))

    assert ctx.related_knowledge == []


@pytest.mark.asyncio
async def test_recent_activity(project_context_agent, db, tmp_path):
    """Insert docs with file_path matching project dir → appear in context."""
    for i in range(3):
        await db.insert_document(Document(
            title=f"ProjectFile{i}",
            content=f"Content {i}",
            file_path=f"{tmp_path}/file{i}.py",
        ))

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 128

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        ctx = await project_context_agent.get_context(str(tmp_path))

    assert "Recent Activity" in ctx.context_block
    assert "ProjectFile" in ctx.context_block


@pytest.mark.asyncio
async def test_context_block_delimiters(project_context_agent, tmp_path):
    """Output starts with === NEXUS PROJECT CONTEXT ===, ends with === END NEXUS CONTEXT ===."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 128

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        ctx = await project_context_agent.get_context(str(tmp_path))

    assert ctx.context_block.startswith("=== NEXUS PROJECT CONTEXT ===")
    assert ctx.context_block.rstrip().endswith("=== END NEXUS CONTEXT ===")


@pytest.mark.asyncio
async def test_context_block_plain_text(project_context_agent, tmp_path):
    """No JSON braces, no markdown headers in context_block."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 128

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        ctx = await project_context_agent.get_context(str(tmp_path))

    assert "{" not in ctx.context_block
    assert "}" not in ctx.context_block
    assert "# " not in ctx.context_block


@pytest.mark.asyncio
async def test_execute_delegates(project_context_agent, tmp_path):
    """agent.run("/path") → AgentResult via BaseAgent."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 128

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await project_context_agent.run(str(tmp_path))

    assert result.success
    assert "=== NEXUS PROJECT CONTEXT ===" in result.output


@pytest.mark.asyncio
async def test_project_name_from_dir(project_context_agent, tmp_path):
    """project_name = directory basename."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 128

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        ctx = await project_context_agent.get_context(str(tmp_path))

    assert ctx.project_name == tmp_path.name
