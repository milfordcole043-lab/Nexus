"""Embedding pipeline with cosine similarity search."""

from __future__ import annotations

import numpy as np

from nexus.db.database import DatabaseManager
from nexus.db.models import Document, Embedding
from nexus.llm.cascade import CascadeManager


class EmbeddingPipeline:
    """Handles embedding generation, storage, and similarity search."""

    def __init__(
        self,
        cascade: CascadeManager,
        db: DatabaseManager,
        model_name: str = "nomic-embed-text",
        dimensions: int = 768,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.cascade = cascade
        self.db = db
        self.model_name = model_name
        self.dimensions = dimensions
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding vector for text."""
        floats = await self.cascade.embed(text)
        return np.array(floats, dtype=np.float32)

    async def embed_and_store(self, doc_id: int, text: str) -> list[int]:
        """Chunk text, embed each chunk, store in DB. Returns embedding IDs."""
        chunks = self._chunk_text(text)
        ids = []
        for i, chunk in enumerate(chunks):
            vector = await self.embed_text(chunk)
            emb = Embedding(
                document_id=doc_id,
                vector=vector.tobytes(),
                model_name=self.model_name,
                dimensions=self.dimensions,
                chunk_index=i,
            )
            emb_id = await self.db.insert_embedding(emb)
            ids.append(emb_id)
        return ids

    async def search_similar(
        self, query: str, top_k: int = 5
    ) -> list[tuple[Document, float]]:
        """Search for documents similar to query text.

        Returns list of (document, similarity_score) tuples, deduplicated by document_id.
        """
        query_vec = await self.embed_text(query)
        all_embeddings = await self.db.get_all_embeddings()

        if not all_embeddings:
            return []

        # Score each embedding
        scored: dict[int, float] = {}
        for emb in all_embeddings:
            vec = np.frombuffer(emb.vector, dtype=np.float32)
            sim = self._cosine_similarity(query_vec, vec)
            # Keep best score per document
            if emb.document_id not in scored or sim > scored[emb.document_id]:
                scored[emb.document_id] = sim

        # Sort by similarity descending
        top_doc_ids = sorted(scored, key=scored.get, reverse=True)[:top_k]  # type: ignore[arg-type]

        results = []
        for doc_id in top_doc_ids:
            doc = await self.db.get_document(doc_id)
            if doc:
                results.append((doc, scored[doc_id]))
        return results

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping word-based chunks."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
