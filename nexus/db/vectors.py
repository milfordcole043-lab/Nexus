"""Embedding pipeline with hybrid search (LanceDB + FTS5 + RRF)."""

from __future__ import annotations

import logging

import numpy as np

from nexus.db.database import DatabaseManager
from nexus.db.models import Document, Embedding
from nexus.llm.cascade import CascadeManager

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """Handles embedding generation, storage, and similarity search.

    When lancedb_store is provided, uses LanceDB for vector search and
    SQLite FTS5 for keyword search, combined via Reciprocal Rank Fusion.
    When lancedb_store is None, falls back to numpy brute-force (backward compat).
    """

    def __init__(
        self,
        cascade: CascadeManager,
        db: DatabaseManager,
        model_name: str = "nomic-embed-text",
        dimensions: int = 768,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        lancedb_store: object | None = None,
        search_mode: str = "hybrid",
    ):
        self.cascade = cascade
        self.db = db
        self.model_name = model_name
        self.dimensions = dimensions
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.lancedb_store = lancedb_store
        self.search_mode = search_mode

    async def initialize(self) -> None:
        """Initialize LanceDB store and migrate existing SQLite embeddings."""
        if self.lancedb_store is None:
            return

        await self.lancedb_store.initialize()

        # Migrate existing SQLite embeddings to LanceDB if LanceDB is empty
        if await self.lancedb_store.is_empty():
            all_embs = await self.db.get_all_embeddings()
            if all_embs:
                records = []
                for emb in all_embs:
                    vec = np.frombuffer(emb.vector, dtype=np.float32).tolist()
                    records.append({
                        "document_id": emb.document_id,
                        "chunk_index": emb.chunk_index,
                        "vector": vec,
                    })
                await self.lancedb_store.add_embeddings(records)
                logger.info("Migrated %d embeddings from SQLite to LanceDB", len(records))

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding vector for text."""
        floats = await self.cascade.embed(text)
        return np.array(floats, dtype=np.float32)

    async def embed_and_store(self, doc_id: int, text: str) -> list[int]:
        """Chunk text, embed each chunk, store in active backend. Returns embedding IDs."""
        chunks = self._chunk_text(text)
        ids = []
        lance_records = []

        for i, chunk in enumerate(chunks):
            vector = await self.embed_text(chunk)

            if self.lancedb_store is not None:
                lance_records.append({
                    "document_id": doc_id,
                    "chunk_index": i,
                    "vector": vector.tolist(),
                })
                ids.append(i)
            else:
                emb = Embedding(
                    document_id=doc_id,
                    vector=vector.tobytes(),
                    model_name=self.model_name,
                    dimensions=self.dimensions,
                    chunk_index=i,
                )
                emb_id = await self.db.insert_embedding(emb)
                ids.append(emb_id)

        if lance_records:
            await self.lancedb_store.add_embeddings(lance_records)

        return ids

    async def search_similar(
        self, query: str, top_k: int = 5
    ) -> list[tuple[Document, float]]:
        """Search for documents similar to query text.

        Returns list of (document, similarity_score) tuples, deduplicated by document_id.
        """
        if self.lancedb_store is None:
            return await self._search_bruteforce(query, top_k)

        if self.search_mode == "vector":
            ranked = await self._search_vector(query, top_k)
        elif self.search_mode == "keyword":
            ranked = await self._search_keyword(query, top_k)
        else:  # hybrid
            vector_results = await self._search_vector(query, top_k)
            keyword_results = await self._search_keyword(query, top_k)
            ranked = self._fuse_rrf(vector_results, keyword_results, top_k)

        results = []
        for doc_id, score in ranked:
            doc = await self.db.get_document(doc_id)
            if doc:
                results.append((doc, score))
        return results

    async def delete_embeddings(self, doc_id: int) -> None:
        """Delete embeddings for a document from the active store."""
        if self.lancedb_store:
            await self.lancedb_store.delete_by_document(doc_id)
        else:
            await self.db.delete_embeddings_for_document(doc_id)

    # --- Private search methods ---

    async def _search_bruteforce(
        self, query: str, top_k: int
    ) -> list[tuple[Document, float]]:
        """Original numpy brute-force search (backward compat)."""
        query_vec = await self.embed_text(query)
        all_embeddings = await self.db.get_all_embeddings()

        if not all_embeddings:
            return []

        scored: dict[int, float] = {}
        for emb in all_embeddings:
            vec = np.frombuffer(emb.vector, dtype=np.float32)
            sim = self._cosine_similarity(query_vec, vec)
            if emb.document_id not in scored or sim > scored[emb.document_id]:
                scored[emb.document_id] = sim

        top_doc_ids = sorted(scored, key=scored.get, reverse=True)[:top_k]  # type: ignore[arg-type]

        results = []
        for doc_id in top_doc_ids:
            doc = await self.db.get_document(doc_id)
            if doc:
                results.append((doc, scored[doc_id]))
        return results

    async def _search_vector(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """LanceDB vector search returning (doc_id, similarity_score) sorted desc."""
        query_vec = await self.embed_text(query)
        return await self.lancedb_store.search(query_vec.tolist(), top_k)

    async def _search_keyword(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """FTS5 keyword search returning (doc_id, positive_score) sorted desc."""
        fts_results = await self.db.search_fts(query, limit=top_k)
        # BM25 rank is negative (closer to 0 = better). Convert: score = -rank
        return [(doc_id, -rank) for doc_id, rank in fts_results]

    @staticmethod
    def _fuse_rrf(
        vector_results: list[tuple[int, float]],
        keyword_results: list[tuple[int, float]],
        top_k: int,
        k: int = 60,
    ) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion. Returns (doc_id, rrf_score) sorted desc."""
        scores: dict[int, float] = {}
        for rank, (doc_id, _) in enumerate(vector_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        for rank, (doc_id, _) in enumerate(keyword_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [(doc_id, scores[doc_id]) for doc_id in sorted_ids]

    # --- Unchanged helpers ---

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
