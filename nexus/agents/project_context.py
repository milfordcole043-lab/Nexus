"""Project Context Agent — generates context blocks for Claude Code session injection."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from nexus.agents.base import AgentResult, BaseAgent
from nexus.agents.memory.context import SourceDocument
from nexus.db.database import DatabaseManager
from nexus.llm.cascade import CascadeManager

logger = logging.getLogger(__name__)

KEY_FILES = ["CLAUDE.md", "README.md", "pyproject.toml", "package.json"]


@dataclass
class ProjectContext:
    """Full project context for Claude Code injection."""

    project_name: str
    branch: str | None = None
    recent_commits: list[str] = field(default_factory=list)
    uncommitted_changes: list[str] = field(default_factory=list)
    key_files: dict[str, str] = field(default_factory=dict)
    related_knowledge: list[SourceDocument] = field(default_factory=list)
    context_block: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class ProjectContextAgent(BaseAgent):
    """Assembles project context for Claude Code session injection."""

    def __init__(
        self,
        name: str,
        description: str,
        cascade: CascadeManager,
        db: DatabaseManager,
        memory: Any | None = None,
    ):
        super().__init__(name, description, cascade, db)
        self.memory = memory

    async def get_context(self, project_path: str) -> ProjectContext:
        """Generate full project context for a given path."""
        path = Path(project_path).resolve()
        project_name = path.name

        git_info = await self._gather_git_info(path)
        key_files = await self._scan_key_files(path)
        related = await self._search_memory(project_name)
        recent_activity = await self._get_recent_activity(path)

        context_block = self._format_context_block(
            project_name=project_name,
            branch=git_info.get("branch"),
            recent_commits=git_info.get("commits", []),
            uncommitted_changes=git_info.get("changes", []),
            key_files=key_files,
            related_knowledge=related,
            recent_activity=recent_activity,
        )

        return ProjectContext(
            project_name=project_name,
            branch=git_info.get("branch"),
            recent_commits=git_info.get("commits", []),
            uncommitted_changes=git_info.get("changes", []),
            key_files=key_files,
            related_knowledge=related,
            context_block=context_block,
        )

    async def _gather_git_info(self, path: Path) -> dict[str, Any]:
        """Gather git branch, recent commits, and uncommitted changes."""
        result: dict[str, Any] = {"branch": None, "commits": [], "changes": []}

        async def _run_git(*args: str) -> str | None:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "git", "-C", str(path), *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                if proc.returncode == 0:
                    return stdout.decode().strip()
            except FileNotFoundError:
                return None
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug("Git command failed: %s", e)
            return None

        # Branch
        branch = await _run_git("rev-parse", "--abbrev-ref", "HEAD")
        if branch is None:
            return result
        result["branch"] = branch

        # Recent commits
        commits_raw = await _run_git("log", "--oneline", "-5")
        if commits_raw:
            result["commits"] = commits_raw.splitlines()

        # Uncommitted changes
        changes_raw = await _run_git("status", "--porcelain")
        if changes_raw:
            result["changes"] = changes_raw.splitlines()

        return result

    async def _scan_key_files(self, path: Path) -> dict[str, str]:
        """Read first ~2000 chars of key project files."""
        found: dict[str, str] = {}

        for filename in KEY_FILES:
            filepath = path / filename
            if not filepath.is_file():
                continue
            try:
                content = await asyncio.to_thread(self._read_file_head, filepath)
                if filename == "pyproject.toml":
                    content = self._extract_pyproject_summary(content)
                elif filename == "package.json":
                    content = self._extract_package_json_summary(content)
                found[filename] = content
            except Exception as e:
                logger.debug("Failed to read %s: %s", filename, e)

        return found

    @staticmethod
    def _read_file_head(filepath: Path, max_chars: int = 2000) -> str:
        """Read first max_chars of a file."""
        with open(filepath, encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)

    @staticmethod
    def _extract_pyproject_summary(content: str) -> str:
        """Extract name, description, dependencies from pyproject.toml content."""
        lines = []
        in_deps = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("name =") or stripped.startswith("description ="):
                lines.append(stripped)
            elif stripped == "dependencies = [":
                in_deps = True
                lines.append("dependencies:")
            elif in_deps:
                if stripped == "]":
                    in_deps = False
                elif stripped.startswith('"'):
                    lines.append(f"  - {stripped.strip(',').strip('\"')}")
        return "\n".join(lines) if lines else content[:500]

    @staticmethod
    def _extract_package_json_summary(content: str) -> str:
        """Extract name, description from package.json content."""
        import json
        try:
            data = json.loads(content)
            parts = []
            if "name" in data:
                parts.append(f"name: {data['name']}")
            if "description" in data:
                parts.append(f"description: {data['description']}")
            return "\n".join(parts) if parts else content[:500]
        except json.JSONDecodeError:
            return content[:500]

    async def _search_memory(self, project_name: str) -> list[SourceDocument]:
        """Search memory agent for related knowledge."""
        if self.memory is None:
            return []
        try:
            from nexus.agents.memory.agent import QueryMode
            response = await self.memory.query(
                question=project_name, mode=QueryMode.SEARCH, top_k=5
            )
            return response.sources
        except Exception as e:
            logger.debug("Memory search failed: %s", e)
            return []

    async def _get_recent_activity(self, path: Path) -> list[dict[str, str]]:
        """Get recently indexed documents from this project directory."""
        try:
            since = (datetime.now(UTC) - timedelta(days=7)).isoformat()
            docs = await self.db.get_documents_by_path_prefix(str(path), since=since)
            return [
                {"title": d.title, "created_at": d.created_at}
                for d in docs[:20]
            ]
        except Exception as e:
            logger.debug("Recent activity lookup failed: %s", e)
            return []

    def _format_context_block(
        self,
        project_name: str,
        branch: str | None,
        recent_commits: list[str],
        uncommitted_changes: list[str],
        key_files: dict[str, str],
        related_knowledge: list[SourceDocument],
        recent_activity: list[dict[str, str]],
    ) -> str:
        """Format all gathered data as plain text context block."""
        lines = ["=== NEXUS PROJECT CONTEXT ==="]
        lines.append(f"Project: {project_name}")
        if branch:
            lines.append(f"Branch: {branch}")
        lines.append("")

        if recent_commits:
            lines.append("Recent Commits:")
            for commit in recent_commits:
                lines.append(f"  {commit}")
            lines.append("")

        if uncommitted_changes:
            lines.append("Uncommitted Changes:")
            for change in uncommitted_changes:
                lines.append(f"  {change}")
            lines.append("")

        if key_files:
            lines.append("Key Files:")
            for filename, content in key_files.items():
                excerpt = content[:500].replace("\n", "\n    ")
                lines.append(f"  [{filename}] {excerpt}")
            lines.append("")

        if related_knowledge:
            lines.append("Related Knowledge:")
            for src in related_knowledge:
                lines.append(f"  - {src.title} (score: {src.score:.2f})")
            lines.append("")

        if recent_activity:
            lines.append("Recent Activity (7d):")
            for item in recent_activity[:10]:
                lines.append(f"  - {item['title']} -- {item['created_at']}")
            lines.append("")

        lines.append("=== END NEXUS CONTEXT ===")
        return "\n".join(lines)

    async def _execute(self, input_data: str) -> AgentResult:
        """BaseAgent lifecycle hook — treats input_data as project path."""
        if not input_data.strip():
            return AgentResult(success=False, error="No project path provided")
        ctx = await self.get_context(input_data.strip())
        return AgentResult(
            success=True,
            output=ctx.context_block,
            metadata={"project_name": ctx.project_name, "branch": ctx.branch},
        )
