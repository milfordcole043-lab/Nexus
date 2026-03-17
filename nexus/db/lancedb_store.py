"""Async wrapper around LanceDB for vector storage and search."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import lancedb
import pyarrow as pa

logger = logging.getLogger(__name__)


class LanceDBStore:
    """Thin async wrapper around synchronous LanceDB operations."""

    TABLE_NAME = "embeddings"

    def __init__(self, path: str | Path, dimensions: int = 768):
        self.path = Path(path)
        self.dimensions = dimensions
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None

    def _get_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("document_id", pa.int64()),
            pa.field("chunk_index", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), self.dimensions)),
        ])

    async def initialize(self) -> None:
        """Connect and create/open the embeddings table."""
        loop = asyncio.get_event_loop()
        self._db = await loop.run_in_executor(None, lancedb.connect, str(self.path))

        table_names = await loop.run_in_executor(None, self._db.list_tables)
        if self.TABLE_NAME in table_names:
            self._table = await loop.run_in_executor(
                None, self._db.open_table, self.TABLE_NAME
            )
        else:
            schema = self._get_schema()
            self._table = await loop.run_in_executor(
                None, lambda: self._db.create_table(self.TABLE_NAME, schema=schema)
            )

        logger.info("LanceDB initialized at %s", self.path)

    async def add_embeddings(self, records: list[dict]) -> None:
        """Batch-insert embeddings. Each dict: {document_id, chunk_index, vector}."""
        if not records or self._table is None:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._table.add, records)

    async def search(self, query_vec: list[float], top_k: int = 20) -> list[tuple[int, float]]:
        """Vector search returning (doc_id, cosine_similarity) deduplicated by doc_id."""
        if self._table is None:
            return []

        loop = asyncio.get_event_loop()

        def _search():
            results = (
                self._table.search(query_vec)
                .metric("cosine")
                .limit(top_k * 3)  # over-fetch to allow dedup
                .to_list()
            )
            # LanceDB cosine returns distance (0=identical, 2=opposite)
            # Convert to similarity: 1 - distance
            scored: dict[int, float] = {}
            for row in results:
                doc_id = int(row["document_id"])
                similarity = 1.0 - float(row["_distance"])
                if doc_id not in scored or similarity > scored[doc_id]:
                    scored[doc_id] = similarity

            sorted_ids = sorted(scored, key=scored.get, reverse=True)[:top_k]
            return [(doc_id, scored[doc_id]) for doc_id in sorted_ids]

        return await loop.run_in_executor(None, _search)

    async def delete_by_document(self, doc_id: int) -> None:
        """Delete all embeddings for a document."""
        if self._table is None:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: self._table.delete(f"document_id = {doc_id}")
        )

    async def count(self) -> int:
        """Count total embeddings."""
        if self._table is None:
            return 0
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._table.count_rows())

    async def is_empty(self) -> bool:
        """Check if the table has any data."""
        return await self.count() == 0
