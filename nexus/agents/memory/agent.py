"""Memory agent — RAG query interface over the knowledge graph."""

from __future__ import annotations

import logging
import re
import time
from enum import Enum

from nexus.agents.base import AgentResult, BaseAgent
from nexus.agents.memory.context import RAGContextAssembler, SourceDocument
from nexus.agents.memory.entities import EntityExtractor
from nexus.agents.memory.prompts import MEMORY_SYSTEM_PROMPT, build_answer_prompt
from nexus.db.database import DatabaseManager
from nexus.db.models import AgentLog, Entity
from nexus.db.vectors import EmbeddingPipeline
from nexus.llm.cascade import CascadeManager

logger = logging.getLogger(__name__)


class QueryMode(str, Enum):
    SEARCH = "search"
    ANSWER = "answer"
    AUTO = "auto"


class MemoryResponse:
    """Response from a memory query."""

    def __init__(
        self,
        answer: str | None,
        sources: list[SourceDocument],
        mode: str,
        tokens_used: int,
        duration_ms: int,
    ):
        self.answer = answer
        self.sources = sources
        self.mode = mode
        self.tokens_used = tokens_used
        self.duration_ms = duration_ms

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [
                {
                    "doc_id": s.doc_id,
                    "title": s.title,
                    "file_path": s.file_path,
                    "score": round(s.score, 4),
                    "snippet": s.snippet[:500],
                }
                for s in self.sources
            ],
            "mode": self.mode,
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
        }


class MemoryAgent(BaseAgent):
    """RAG-based knowledge retrieval agent."""

    def __init__(
        self,
        name: str,
        description: str,
        cascade: CascadeManager,
        db: DatabaseManager,
        pipeline: EmbeddingPipeline,
    ):
        super().__init__(name, description, cascade, db)
        self.pipeline = pipeline
        self._assembler = RAGContextAssembler(pipeline, db)
        self._extractor = EntityExtractor()

    async def query(
        self,
        question: str,
        mode: QueryMode = QueryMode.AUTO,
        top_k: int = 10,
    ) -> MemoryResponse:
        """Execute a knowledge query with semantic + entity search."""
        start = time.perf_counter_ns()
        resolved_mode = self._resolve_mode(question, mode)
        tokens_used = 0

        # Step 1: Semantic search
        vector_results = await self.pipeline.search_similar(question, top_k=top_k)

        # Step 2: Entity graph search — detect entity names in question, look up linked docs
        entity_docs = await self._entity_search(question)

        # Step 3: Merge + deduplicate by doc_id (keep best score)
        merged: dict[int, tuple] = {}
        for doc, score in vector_results:
            if doc.id is not None:
                merged[doc.id] = (doc, score)

        for doc, score in entity_docs:
            if doc.id is not None:
                if doc.id not in merged or score > merged[doc.id][1]:
                    merged[doc.id] = (doc, score)

        # Sort by score descending, limit to top_k
        sorted_results = sorted(merged.values(), key=lambda x: x[1], reverse=True)[:top_k]

        if not sorted_results:
            duration_ms = (time.perf_counter_ns() - start) // 1_000_000
            await self._log_query(question, resolved_mode.value, 0, duration_ms, 0)
            return MemoryResponse(
                answer="No relevant documents found." if resolved_mode == QueryMode.ANSWER else None,
                sources=[],
                mode=resolved_mode.value,
                tokens_used=0,
                duration_ms=duration_ms,
            )

        # Step 4: Build source snippets
        query_vec = await self.pipeline.embed_text(question)
        sources = await self._assembler.build_sources(query_vec, sorted_results)

        # Step 5: If search mode, return sources only
        if resolved_mode == QueryMode.SEARCH:
            duration_ms = (time.perf_counter_ns() - start) // 1_000_000
            await self._log_query(question, resolved_mode.value, len(sources), duration_ms, 0)
            return MemoryResponse(
                answer=None,
                sources=sources,
                mode=resolved_mode.value,
                tokens_used=0,
                duration_ms=duration_ms,
            )

        # Step 6: Format context and call LLM
        context_block = self._assembler.format_context(sources)
        prompt = build_answer_prompt(question, context_block)

        response = await self.cascade.generate(
            prompt=prompt,
            system_prompt=MEMORY_SYSTEM_PROMPT,
            max_tokens=1024,
            temperature=0.3,
        )
        tokens_used = response.tokens_used

        duration_ms = (time.perf_counter_ns() - start) // 1_000_000
        await self._log_query(question, resolved_mode.value, len(sources), duration_ms, tokens_used)

        return MemoryResponse(
            answer=response.text,
            sources=sources,
            mode=resolved_mode.value,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
        )

    async def extract_entities(self, doc_id: int, content: str) -> list[int]:
        """Extract entities from content and link to document. Called by file watcher hook."""
        extracted = self._extractor.extract(content)
        entity_ids: list[int] = []

        for ext in extracted:
            entity = Entity(name=ext.name, type=ext.type)
            entity_id = await self.db.insert_entity(entity)
            await self.db.link_entity_to_document(
                doc_id=doc_id,
                entity_id=entity_id,
                confidence=ext.confidence,
                context_snippet=ext.context_snippet,
            )
            entity_ids.append(entity_id)

        return entity_ids

    async def _execute(self, input_data: str) -> AgentResult:
        """BaseAgent lifecycle hook — delegates to query() in auto mode."""
        if not input_data.strip():
            return AgentResult(success=True, output="Memory agent ready")
        result = await self.query(input_data)
        return AgentResult(
            success=True,
            output=result.answer or f"Found {len(result.sources)} sources",
            tokens_used=result.tokens_used,
        )

    async def _entity_search(self, question: str) -> list[tuple]:
        """Search for documents linked to entities mentioned in the question."""
        # Extract entity names from the question
        extracted = self._extractor.extract(question)
        results: list[tuple] = []

        for ext in extracted:
            # Look up entity by exact name
            entities = await self.db.get_entity_by_name(ext.name)
            for entity in entities:
                if entity.id is not None:
                    docs = await self.db.get_documents_for_entity(entity.id)
                    for doc in docs:
                        # Entity matches get a fixed score boost
                        results.append((doc, 0.85))

        return results

    def _resolve_mode(self, question: str, mode: QueryMode) -> QueryMode:
        """Resolve AUTO mode using heuristics."""
        if mode != QueryMode.AUTO:
            return mode

        q = question.strip().lower()

        # Search heuristics
        search_re = re.compile(r"^(find|list|show|search|which files)\b")
        if search_re.match(q):
            return QueryMode.SEARCH

        # Answer heuristics
        answer_re = re.compile(
            r"^(what|why|how|when|where|who|explain|describe|summarize|compare)\b"
        )
        if answer_re.match(q):
            return QueryMode.ANSWER

        if q.endswith("?"):
            return QueryMode.ANSWER

        # Short queries without ? → search
        if len(q.split()) <= 3 and not q.endswith("?"):
            return QueryMode.SEARCH

        return QueryMode.ANSWER

    async def _log_query(
        self,
        question: str,
        mode: str,
        num_sources: int,
        duration_ms: int,
        tokens_used: int,
    ) -> None:
        """Log a query to agent_logs."""
        await self.db.insert_agent_log(
            AgentLog(
                agent_name=self.name,
                action="query",
                input_summary=question[:200],
                output_summary=f"mode={mode} sources={num_sources}",
                tokens_used=tokens_used,
                duration_ms=duration_ms,
                status="success",
            )
        )
