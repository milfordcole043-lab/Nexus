"""Tests for the base agent class."""

from __future__ import annotations

import pytest

from nexus.agents.base import AgentResult, AgentStatus, BaseAgent
from nexus.db.database import DatabaseManager
from nexus.llm.cascade import CascadeManager


class EchoAgent(BaseAgent):
    """Simple test agent that echoes input."""

    async def _execute(self, input_data: str) -> AgentResult:
        resp = await self.cascade.generate(input_data)
        return AgentResult(
            success=True,
            output=resp.text,
            tokens_used=resp.tokens_used,
        )


class FailingAgent(BaseAgent):
    """Agent that always raises an exception."""

    async def _execute(self, input_data: str) -> AgentResult:
        raise RuntimeError("Something broke")


class TestBaseAgent:
    async def test_successful_run(self, db: DatabaseManager, mock_provider):
        cascade = CascadeManager([mock_provider])
        agent = EchoAgent("echo", "Echoes input", cascade, db)

        assert agent.status == AgentStatus.INIT

        result = await agent.run("Hello")
        assert result.success is True
        assert "Mock response" in result.output
        assert result.duration_ms >= 0
        assert result.tokens_used == 10
        assert agent.status == AgentStatus.DONE

    async def test_failed_run(self, db: DatabaseManager, mock_provider):
        cascade = CascadeManager([mock_provider])
        agent = FailingAgent("fail", "Always fails", cascade, db)

        result = await agent.run("Hello")
        assert result.success is False
        assert result.error == "Something broke"
        assert agent.status == AgentStatus.ERROR

    async def test_logs_success(self, db: DatabaseManager, mock_provider):
        cascade = CascadeManager([mock_provider])
        agent = EchoAgent("echo", "Echoes input", cascade, db)
        await agent.run("Hello")

        logs = await db.get_agent_logs(agent_name="echo")
        assert len(logs) == 1
        assert logs[0].status == "success"
        assert logs[0].tokens_used == 10

    async def test_logs_error(self, db: DatabaseManager, mock_provider):
        cascade = CascadeManager([mock_provider])
        agent = FailingAgent("fail", "Fails", cascade, db)
        await agent.run("Hello")

        logs = await db.get_agent_logs(agent_name="fail")
        assert len(logs) == 1
        assert logs[0].status == "error"
        assert "Something broke" in logs[0].error_message

    async def test_timing(self, db: DatabaseManager, mock_provider):
        cascade = CascadeManager([mock_provider])
        agent = EchoAgent("echo", "Echoes", cascade, db)

        result = await agent.run("Hello")
        assert result.duration_ms >= 0

    async def test_input_truncation_in_log(self, db: DatabaseManager, mock_provider):
        cascade = CascadeManager([mock_provider])
        agent = EchoAgent("echo", "Echoes", cascade, db)

        long_input = "x" * 500
        await agent.run(long_input)

        logs = await db.get_agent_logs(agent_name="echo")
        assert len(logs[0].input_summary) == 200
