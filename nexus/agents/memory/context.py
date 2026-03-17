"""RAG context assembly — builds source snippets and formats LLM context."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nexus.db.database import DatabaseManager
from nexus.db.models import Document
from nexus.db.vectors import EmbeddingPipeline


@dataclass
class SourceDocument:
    """A source document with its best snippet for attribution."""

    doc_id: int
    title: str
    file_path: str | None
    score: float
    snippet: str


class RAGContextAssembler:
    """Builds source snippets and formats context for LLM prompts."""

    def __init__(self, pipeline: EmbeddingPipeline, db: DatabaseManager):
        self.pipeline = pipeline
        self.db = db

    async def build_sources(
        self, query_vec: np.ndarray, results: list[tuple[Document, float]]
    ) -> list[SourceDocument]:
        """For each doc, extract best snippet.

        Short docs (single chunk): full content.
        Long docs (multi-chunk): best chunk selected by re-scoring stored vectors.
        """
        sources: list[SourceDocument] = []
        for doc, score in results:
            if doc.id is None:
                continue
            content = doc.content
            chunks = self.pipeline._chunk_text(content)

            if len(chunks) <= 1:
                # Short doc — use full content
                snippet = content
            else:
                # Long doc — find best chunk via stored embeddings
                embeddings = await self.db.get_embeddings_for_document(doc.id)
                if embeddings:
                    best_idx = 0
                    best_sim = -1.0
                    for emb in embeddings:
                        vec = np.frombuffer(emb.vector, dtype=np.float32)
                        sim = EmbeddingPipeline._cosine_similarity(query_vec, vec)
                        if sim > best_sim:
                            best_sim = sim
                            best_idx = emb.chunk_index
                    if best_idx < len(chunks):
                        snippet = chunks[best_idx]
                    else:
                        snippet = chunks[0]
                else:
                    snippet = chunks[0]

            sources.append(
                SourceDocument(
                    doc_id=doc.id,
                    title=doc.title,
                    file_path=doc.file_path,
                    score=score,
                    snippet=snippet,
                )
            )
        return sources

    def format_context(
        self, sources: list[SourceDocument], token_budget: int = 3500
    ) -> str:
        """Assemble numbered source blocks within token budget.

        Estimates tokens as words * 1.3.
        """
        blocks: list[str] = []
        tokens_used = 0

        for i, src in enumerate(sources, 1):
            header = f"[Source {i}] {src.title}"
            if src.file_path:
                header += f" ({src.file_path})"
            block = f"{header}\n{src.snippet}"

            # Estimate tokens
            word_count = len(block.split())
            estimated_tokens = int(word_count * 1.3)

            if tokens_used + estimated_tokens > token_budget:
                # Try to fit a truncated version
                remaining_tokens = token_budget - tokens_used
                remaining_words = int(remaining_tokens / 1.3)
                if remaining_words > 20:
                    words = block.split()[:remaining_words]
                    blocks.append(" ".join(words) + "...")
                break

            blocks.append(block)
            tokens_used += estimated_tokens

        return "\n\n".join(blocks)
