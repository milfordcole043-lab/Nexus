"""Base agent class for all Nexus agents."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nexus.db.database import DatabaseManager
from nexus.db.models import AgentLog
from nexus.llm.cascade import CascadeManager

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    INIT = "init"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentResult:
    """Result from an agent execution."""

    success: bool
    output: str = ""
    tokens_used: int = 0
    duration_ms: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for Nexus agents."""

    def __init__(
        self,
        name: str,
        description: str,
        cascade: CascadeManager,
        db: DatabaseManager,
    ):
        self.name = name
        self.description = description
        self.cascade = cascade
        self.db = db
        self.status = AgentStatus.INIT

    async def run(self, input_data: str = "") -> AgentResult:
        """Execute the agent with lifecycle management and logging."""
        self.status = AgentStatus.RUNNING
        start = time.perf_counter_ns()

        try:
            result = await self._execute(input_data)
            self.status = AgentStatus.DONE

            duration_ms = (time.perf_counter_ns() - start) // 1_000_000
            result.duration_ms = duration_ms

            await self.db.insert_agent_log(
                AgentLog(
                    agent_name=self.name,
                    action="run",
                    input_summary=input_data[:200] if input_data else None,
                    output_summary=result.output[:200] if result.output else None,
                    tokens_used=result.tokens_used,
                    duration_ms=duration_ms,
                    status="success" if result.success else "failure",
                    error_message=result.error,
                )
            )

            return result

        except Exception as e:
            self.status = AgentStatus.ERROR
            duration_ms = (time.perf_counter_ns() - start) // 1_000_000

            logger.error("Agent %s failed: %s", self.name, e)

            await self.db.insert_agent_log(
                AgentLog(
                    agent_name=self.name,
                    action="run",
                    input_summary=input_data[:200] if input_data else None,
                    tokens_used=0,
                    duration_ms=duration_ms,
                    status="error",
                    error_message=str(e),
                )
            )

            return AgentResult(
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

    @abstractmethod
    async def _execute(self, input_data: str) -> AgentResult:
        """Implement agent-specific logic. Subclasses override this."""
        ...
