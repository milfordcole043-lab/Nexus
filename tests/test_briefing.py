"""Tests for the Daily Briefing Agent."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from nexus.agents.briefing import BriefingAgent, BriefingResult
from nexus.config import BriefingConfig, NexusConfig
from nexus.db.models import Document, Entity


@pytest.fixture
def test_config(tmp_path):
    """Config with temp watch dirs."""
    return NexusConfig(
        watch_directories=[str(tmp_path)],
        db_path=str(tmp_path / "test.db"),
        briefing=BriefingConfig(
            schedule="08:00",
            timezone="UTC",
            ntfy_topic="test-briefing",
            enabled=True,
            include_git=True,
            lookback_hours=24,
        ),
    )


@pytest_asyncio.fixture
async def briefing_agent(db, cascade, test_config):
    """Provide a briefing agent with test DB."""
    return BriefingAgent(
        name="briefing",
        description="Test briefing",
        cascade=cascade,
        db=db,
        config=test_config,
    )


@pytest.mark.asyncio
@patch("nexus.agents.briefing.send_notification", new_callable=AsyncMock, return_value=True)
async def test_generate_empty_db(mock_notify, briefing_agent, db):
    """No docs/entities → valid markdown with 0 files, stored in DB."""
    result = await briefing_agent.generate_briefing()

    assert isinstance(result, BriefingResult)
    assert "# Daily Briefing" in result.content
    assert "0 files" in result.content
    assert result.summary

    stored = await db.get_latest_briefing()
    assert stored is not None
    assert stored.content == result.content


@pytest.mark.asyncio
@patch("nexus.agents.briefing.send_notification", new_callable=AsyncMock, return_value=True)
async def test_generate_with_recent_docs(mock_notify, briefing_agent, db):
    """Insert 3 docs, briefing mentions count + titles."""
    for i in range(3):
        await db.insert_document(Document(
            title=f"TestDoc{i}",
            content=f"Content {i}",
            category="notes",
        ))

    result = await briefing_agent.generate_briefing()

    assert "3 files" in result.content
    assert "TestDoc0" in result.content


@pytest.mark.asyncio
@patch("nexus.agents.briefing.send_notification", new_callable=AsyncMock, return_value=True)
async def test_generate_with_entities(mock_notify, briefing_agent, db):
    """Insert entities + links, knowledge highlights populated."""
    doc_id = await db.insert_document(Document(title="Entity Doc", content="Test"))
    entity_id = await db.insert_entity(Entity(name="Python", type="technology"))
    await db.link_entity_to_document(doc_id, entity_id)

    result = await briefing_agent.generate_briefing()

    assert result.sections["entity_highlights"]
    assert any(e["name"] == "Python" for e in result.sections["entity_highlights"])


@pytest.mark.asyncio
@patch("nexus.agents.briefing.send_notification", new_callable=AsyncMock, return_value=True)
async def test_git_activity(mock_notify, briefing_agent):
    """Mock subprocess → fake git log, verify git section."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"abc1234 Initial commit\ndef5678 Add feature\n", b""))
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await briefing_agent.generate_briefing()

    assert result.sections["git_activity"]
    assert "abc1234" in result.content


@pytest.mark.asyncio
@patch("nexus.agents.briefing.send_notification", new_callable=AsyncMock, return_value=True)
async def test_git_disabled(mock_notify, briefing_agent):
    """include_git=False → no git section."""
    briefing_agent.config.briefing.include_git = False

    result = await briefing_agent.generate_briefing()

    assert result.sections["git_activity"] == []


@pytest.mark.asyncio
@patch("nexus.agents.briefing.send_notification", new_callable=AsyncMock, return_value=True)
async def test_git_not_installed(mock_notify, briefing_agent):
    """Mock FileNotFoundError → briefing still generates."""
    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("git not found")):
        result = await briefing_agent.generate_briefing()

    assert isinstance(result, BriefingResult)
    assert "# Daily Briefing" in result.content
    assert result.sections["git_activity"] == []


@pytest.mark.asyncio
@patch("nexus.agents.briefing.send_notification", new_callable=AsyncMock, return_value=True)
async def test_stored_in_db(mock_notify, briefing_agent, db):
    """Generate → get_latest_briefing matches."""
    result = await briefing_agent.generate_briefing()

    stored = await db.get_latest_briefing()
    assert stored is not None
    assert stored.content == result.content
    assert stored.summary == result.summary


@pytest.mark.asyncio
async def test_notification_sent(briefing_agent):
    """Mock send_notification → verify called with summary."""
    with patch("nexus.agents.briefing.send_notification", new_callable=AsyncMock, return_value=True) as mock_notify:
        result = await briefing_agent.generate_briefing()

    mock_notify.assert_called_once()
    call_args = mock_notify.call_args
    assert call_args.kwargs["topic"] == "test-briefing" or call_args[1].get("topic") == "test-briefing" or call_args[0][0] == "test-briefing"


@pytest.mark.asyncio
async def test_notification_failure(briefing_agent, db):
    """send_notification returns False → briefing still succeeds."""
    with patch("nexus.agents.briefing.send_notification", new_callable=AsyncMock, return_value=False):
        result = await briefing_agent.generate_briefing()

    assert isinstance(result, BriefingResult)
    assert result.content
    # Should still be stored
    stored = await db.get_latest_briefing()
    assert stored is not None


@pytest.mark.asyncio
@patch("nexus.agents.briefing.send_notification", new_callable=AsyncMock, return_value=True)
async def test_result_structure(mock_notify, briefing_agent):
    """BriefingResult has content, summary, sections, generated_at."""
    result = await briefing_agent.generate_briefing()

    assert isinstance(result.content, str)
    assert isinstance(result.summary, str)
    assert isinstance(result.sections, dict)
    assert isinstance(result.generated_at, str)


@pytest.mark.asyncio
@patch("nexus.agents.briefing.send_notification", new_callable=AsyncMock, return_value=True)
async def test_execute_delegates(mock_notify, briefing_agent):
    """agent.run("") → goes through BaseAgent lifecycle."""
    agent_result = await briefing_agent.run("")

    assert agent_result.success
    assert agent_result.output  # summary string


@pytest.mark.asyncio
@patch("nexus.agents.briefing.send_notification", new_callable=AsyncMock, return_value=True)
async def test_sections_dict(mock_notify, briefing_agent):
    """sections has keys: recent_files, entity_highlights, git_activity, system_status."""
    result = await briefing_agent.generate_briefing()

    expected_keys = {"recent_files", "entity_highlights", "git_activity", "system_status"}
    assert set(result.sections.keys()) == expected_keys
