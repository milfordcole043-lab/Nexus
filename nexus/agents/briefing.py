"""Daily Briefing Agent — compiles morning summaries from Nexus knowledge."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from nexus.agents.base import AgentResult, BaseAgent
from nexus.config import NexusConfig
from nexus.db.database import DatabaseManager
from nexus.db.models import Briefing
from nexus.llm.cascade import CascadeManager
from nexus.tools.notifications import send_notification

logger = logging.getLogger(__name__)


@dataclass
class BriefingResult:
    """Result from a briefing generation."""

    content: str
    summary: str
    sections: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class BriefingAgent(BaseAgent):
    """Generates daily briefing summaries from indexed knowledge."""

    def __init__(
        self,
        name: str,
        description: str,
        cascade: CascadeManager,
        db: DatabaseManager,
        config: NexusConfig,
        memory: Any | None = None,
    ):
        super().__init__(name, description, cascade, db)
        self.config = config
        self.memory = memory

    async def generate_briefing(self) -> BriefingResult:
        """Generate a full daily briefing."""
        now = datetime.now(UTC)
        lookback = self.config.briefing.lookback_hours
        since = (now - timedelta(hours=lookback)).isoformat()
        since_7d = (now - timedelta(days=7)).isoformat()

        # Gather all sections
        recent_files = await self._gather_recent_files(since)
        entity_highlights = await self._gather_entity_highlights(since_7d)
        git_activity = await self._gather_git_activity(lookback)
        system_status = await self._gather_system_status()

        sections = {
            "recent_files": recent_files,
            "entity_highlights": entity_highlights,
            "git_activity": git_activity,
            "system_status": system_status,
        }

        content, summary = self._compile_markdown(sections)

        # Store in DB
        briefing = Briefing(content=content, summary=summary)
        briefing_id = await self.db.insert_briefing(briefing)

        # Send notification
        try:
            sent = await send_notification(
                topic=self.config.briefing.ntfy_topic,
                title="Nexus Daily Briefing",
                message=summary,
            )
            if sent and briefing_id:
                await self.db.mark_briefing_delivered(briefing_id)
        except Exception as e:
            logger.warning("Notification failed: %s", e)

        return BriefingResult(
            content=content,
            summary=summary,
            sections=sections,
            generated_at=now.isoformat(),
        )

    async def _gather_recent_files(self, since: str) -> dict[str, Any]:
        """Gather recently indexed files grouped by category."""
        try:
            docs = await self.db.get_recent_documents(since)
            categories: dict[str, list[str]] = {}
            for doc in docs:
                cat = doc.category or "uncategorized"
                categories.setdefault(cat, []).append(doc.title)
            return {
                "total": len(docs),
                "categories": categories,
                "top_titles": [d.title for d in docs[:10]],
            }
        except Exception as e:
            logger.warning("Failed to gather recent files: %s", e)
            return {"total": 0, "categories": {}, "top_titles": []}

    async def _gather_entity_highlights(self, since: str) -> list[dict[str, Any]]:
        """Gather most active entities from recent period."""
        try:
            active = await self.db.get_most_active_entities(since)
            return [
                {"name": entity.name, "type": entity.type, "mention_count": count}
                for entity, count in active
            ]
        except Exception as e:
            logger.warning("Failed to gather entity highlights: %s", e)
            return []

    async def _gather_git_activity(self, lookback_hours: int) -> list[dict[str, Any]]:
        """Gather git log from watched directories."""
        if not self.config.briefing.include_git:
            return []

        results = []
        for watch_dir in self.config.resolved_watch_directories:
            if not watch_dir.is_dir():
                continue
            try:
                proc = await asyncio.create_subprocess_exec(
                    "git", "-C", str(watch_dir), "log",
                    "--oneline", f"--since={lookback_hours} hours ago",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                if proc.returncode == 0 and stdout:
                    lines = stdout.decode().strip().splitlines()
                    if lines:
                        results.append({
                            "directory": str(watch_dir),
                            "commits": lines,
                        })
            except FileNotFoundError:
                logger.debug("git not found, skipping git activity")
                return results
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug("Git activity failed for %s: %s", watch_dir, e)
                continue

        return results

    async def _gather_system_status(self) -> dict[str, Any]:
        """Gather system/DB stats."""
        try:
            stats = await self.db.get_stats()
            db_size = 0
            try:
                db_size = os.path.getsize(self.db.db_path)
            except OSError:
                pass
            return {"db_stats": stats, "db_size_bytes": db_size}
        except Exception as e:
            logger.warning("Failed to gather system status: %s", e)
            return {"db_stats": {}, "db_size_bytes": 0}

    def _compile_markdown(self, sections: dict[str, Any]) -> tuple[str, str]:
        """Compile sections into markdown briefing + short summary."""
        lines = ["# Daily Briefing", ""]

        # Recent files
        rf = sections["recent_files"]
        lines.append("## Recent Files")
        lines.append(f"**{rf['total']} files** indexed in the last {self.config.briefing.lookback_hours} hours.")
        lines.append("")
        if rf["categories"]:
            for cat, titles in rf["categories"].items():
                lines.append(f"- **{cat}**: {len(titles)} files")
            lines.append("")
        if rf["top_titles"]:
            lines.append("### Top Files")
            for title in rf["top_titles"][:5]:
                lines.append(f"- {title}")
            lines.append("")

        # Entity highlights
        eh = sections["entity_highlights"]
        lines.append("## Knowledge Highlights")
        if eh:
            for e in eh:
                lines.append(f"- **{e['name']}** ({e['type']}): {e['mention_count']} mentions")
        else:
            lines.append("No notable entity activity this week.")
        lines.append("")

        # Git activity
        ga = sections["git_activity"]
        if ga:
            lines.append("## Git Activity")
            for repo in ga:
                lines.append(f"### {repo['directory']}")
                for commit in repo["commits"][:10]:
                    lines.append(f"- {commit}")
                lines.append("")

        # System status
        ss = sections["system_status"]
        lines.append("## System Status")
        db_stats = ss.get("db_stats", {})
        db_size_mb = ss.get("db_size_bytes", 0) / (1024 * 1024)
        lines.append(f"- Documents: {db_stats.get('documents', 0)}")
        lines.append(f"- Entities: {db_stats.get('entities', 0)}")
        lines.append(f"- Embeddings: {db_stats.get('embeddings', 0)}")
        lines.append(f"- DB size: {db_size_mb:.1f} MB")

        content = "\n".join(lines)

        # Build summary
        summary_parts = [f"{rf['total']} files indexed"]
        if eh:
            summary_parts.append(f"{len(eh)} active entities")
        git_count = sum(len(r["commits"]) for r in ga)
        if git_count:
            summary_parts.append(f"{git_count} git commits")
        summary = "Daily Briefing: " + ", ".join(summary_parts) + "."

        return content, summary

    async def _execute(self, input_data: str) -> AgentResult:
        """BaseAgent lifecycle hook — delegates to generate_briefing()."""
        result = await self.generate_briefing()
        return AgentResult(
            success=True,
            output=result.summary,
            metadata={"sections": list(result.sections.keys())},
        )
