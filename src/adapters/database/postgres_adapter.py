"""
PostgreSQL Database Adapter with pgvector support.

Implements the AsyncDatabasePort interface using asyncpg and SQLAlchemy async.
Provides embedding storage, read-only query execution, and vector similarity search
for the multi-agent NL2SQL workflow.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Any, Dict

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from src.ports.database_port import AsyncDatabasePort
from src.ports.embedding_port import EmbeddingPort
from src.domain.models import Receipt, Item
from src.settings import settings

logger = logging.getLogger(__name__)


class PostgresAdapter(AsyncDatabasePort):
    """PostgreSQL database adapter with pgvector support."""

    def __init__(self, database_url: str = None, embedding_provider: EmbeddingPort = None):
        """
        Initialize PostgreSQL adapter.

        Args:
            database_url: PostgreSQL connection string.
                          Defaults to settings.DATABASE_URL
            embedding_provider: Embedding provider for generating embeddings.
                               Defaults to None, must be set via set_embedding_provider()
        """
        self.database_url = database_url or settings.DATABASE_URL
        self.pool: Optional[asyncpg.Pool] = None
        self._embedding_provider: Optional[EmbeddingPort] = embedding_provider

    def set_embedding_provider(self, embedding_provider: EmbeddingPort) -> None:
        """Set the embedding provider for this adapter."""
        self._embedding_provider = embedding_provider

    async def connect(self) -> None:
        """Initialize connection pool and create tables/extensions."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                init=self._init_connection,
                statement_cache_size=0,
            )
            async with self.pool.acquire() as conn:
                # Enable pgvector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                # Create items table with embedding column
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS items (
                        id SERIAL PRIMARY KEY,
                        receipt_id INTEGER NOT NULL,
                        item_name TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        unit_price REAL NOT NULL,
                        total_price REAL NOT NULL,
                        purchase_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        item_name_embedding vector(768)
                    )
                """)
                # Create index for vector similarity search
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_items_embedding
                    ON items USING ivfflat (item_name_embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)

            logger.info("PostgreSQL connection pool created and schema initialized.")

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        """Initialize each connection in the pool with pgvector type."""
        await register_vector(conn)

    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed.")

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate a 768-dim embedding for the given text using the configured embedding provider.

        Falls back to a zero vector if embedding provider is not set or generation fails.
        """
        if not self._embedding_provider:
            logger.warning("No embedding provider set. Using zero vector fallback.")
            return [0.0] * 768

        return await self._embedding_provider.generate_embedding(text)

    async def save_receipt(self, receipt: Receipt) -> bool:
        """
        Save receipt items to PostgreSQL including embeddings.

        Each item gets an auto generated embedding for its item_name.
        """
        if not self.pool:
            logger.error("Database pool not initialized. Call connect() first.")
            return False

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    for item in receipt.items:
                        # Generate embedding for item name
                        embedding = await self._generate_embedding(item.item_name)
                        embedding_np = np.array(embedding, dtype=np.float32)

                        purchase_date = item.purchase_date or datetime.now(timezone.utc)

                        await conn.execute(
                            """
                            INSERT INTO items 
                                (receipt_id, item_name, quantity, unit_price, 
                                 total_price, purchase_date, item_name_embedding)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            """,
                            receipt.receipt_id,
                            item.item_name,
                            item.quantity,
                            item.unit_price,
                            item.total_price,
                            purchase_date,
                            embedding_np,
                        )

            logger.info(
                f"Saved receipt {receipt.receipt_id} with {len(receipt.items)} items."
            )
            return True

        except Exception as e:
            logger.error(f"Error saving receipt: {e}")
            return False

    async def query_spending(self, item_name: str, days: int = 7) -> float:
        """Query total spending for an item within specified days."""
        if not self.pool:
            return 0.0

        try:
            async with self.pool.acquire() as conn:
                if days > 0:
                    result = await conn.fetchval(
                        """
                        SELECT COALESCE(SUM(total_price), 0)
                        FROM items
                        WHERE item_name ILIKE $1
                          AND purchase_date >= NOW() - INTERVAL '1 day' * $2
                        """,
                        f"%{item_name}%",
                        days,
                    )
                else:
                    # All-time spending
                    result = await conn.fetchval(
                        """
                        SELECT COALESCE(SUM(total_price), 0)
                        FROM items
                        WHERE item_name ILIKE $1
                        """,
                        f"%{item_name}%",
                    )
                return float(result) if result else 0.0

        except Exception as e:
            logger.error(f"Error querying spending: {e}")
            return 0.0

    async def get_items(self, receipt_id: Optional[int] = None) -> List[Item]:
        """Get items, optionally filtered by receipt_id."""
        if not self.pool:
            return []

        try:
            async with self.pool.acquire() as conn:
                if receipt_id is not None:
                    rows = await conn.fetch(
                        """
                        SELECT item_name, quantity, unit_price, total_price, purchase_date
                        FROM items WHERE receipt_id = $1
                        """,
                        receipt_id,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT item_name, quantity, unit_price, total_price, purchase_date
                        FROM items
                        """
                    )

                return [
                    Item(
                        item_name=row["item_name"],
                        quantity=row["quantity"],
                        unit_price=row["unit_price"],
                        total_price=row["total_price"],
                        purchase_date=row["purchase_date"],
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Error getting items: {e}")
            return []

    async def execute_read_query(
        self,
        sql: str,
        params: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a read-only SQL query. Only SELECT statements are allowed.

        Used by the Database Analyst Agent for NL2SQL queries.
        """
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")

        # Safety: only allow SELECT statements
        cleaned = sql.strip().upper()
        if not cleaned.startswith("SELECT"):
            raise ValueError(
                "Only SELECT queries are allowed. "
                f"Received query starting with: {cleaned[:20]}"
            )

        # Block dangerous keywords
        dangerous_keywords = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE", "GRANT", "REVOKE"]
        for keyword in dangerous_keywords:
            # Check for keyword as a standalone word
            if f" {keyword} " in f" {cleaned} ":
                raise ValueError(
                    f"Query contains forbidden keyword: {keyword}"
                )

        try:
            async with self.pool.acquire() as conn:
                # Use a read-only transaction for extra safety
                async with conn.transaction(readonly=True):
                    if params:
                        rows = await conn.fetch(sql, *params)
                    else:
                        rows = await conn.fetch(sql)

                    return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Error executing read query: {e}")
            raise

    async def search_similar_items(
        self,
        query_embedding: List[float],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for items with similar name embeddings using pgvector cosine distance.

        Returns distinct item names with their similarity scores.
        """
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")

        try:
            embedding_np = np.array(query_embedding, dtype=np.float32)

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT DISTINCT ON (item_name)
                        item_name,
                        1 - (item_name_embedding <=> $1) AS similarity_score,
                        total_price,
                        purchase_date
                    FROM items
                    WHERE item_name_embedding IS NOT NULL
                    ORDER BY item_name, item_name_embedding <=> $1
                    LIMIT $2
                    """,
                    embedding_np,
                    limit,
                )

                return [
                    {
                        "item_name": row["item_name"],
                        "similarity_score": float(row["similarity_score"]),
                        "total_price": float(row["total_price"]),
                        "purchase_date": (
                            row["purchase_date"].isoformat()
                            if row["purchase_date"]
                            else None
                        ),
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Error searching similar items: {e}")
            return []
        
    ##### have to check this later this is a better approch
    # async def search_similar_items(
    #     self,
    #     query_embedding: List[float],
    #     limit: int = 5,
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Search for items with similar name embeddings using pgvector cosine distance.

    #     Returns distinct item names with their similarity scores,
    #     globally sorted by similarity (highest first).
    #     """
    #     if not self.pool:
    #         raise RuntimeError("Database pool not initialized. Call connect() first.")

    #     try:
    #         # Convert embedding to numpy float32 (required for pgvector)
    #         embedding_np = np.array(query_embedding, dtype=np.float32)

    #         async with self.pool.acquire() as conn:
    #             rows = await conn.fetch(
    #                 """
    #                 SELECT *
    #                 FROM (
    #                     SELECT DISTINCT ON (item_name)
    #                         item_name,
    #                         1 - (item_name_embedding <=> $1) AS similarity_score,
    #                         total_price,
    #                         purchase_date
    #                     FROM items
    #                     WHERE item_name_embedding IS NOT NULL
    #                     ORDER BY item_name, item_name_embedding <=> $1
    #                 ) sub
    #                 ORDER BY similarity_score DESC
    #                 LIMIT $2
    #                 """,
    #                 embedding_np,
    #                 limit,
    #             )

    #             return [
    #                 {
    #                     "item_name": row["item_name"],
    #                     "similarity_score": float(row["similarity_score"]),
    #                     "total_price": float(row["total_price"]),
    #                     "purchase_date": (
    #                         row["purchase_date"].isoformat()
    #                         if row["purchase_date"]
    #                         else None
    #                     ),
    #                 }
    #                 for row in rows
    #             ]

    #     except Exception as e:
    #         logger.error(f"Error searching similar items: {e}")
    #         return []
