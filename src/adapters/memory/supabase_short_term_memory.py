"""
Supabase Short-Term Memory Adapter

Stores recent conversation turns in a Supabase PostgreSQL table.
"""

import json

import asyncpg

from src.domain.models import MemoryRecord
from src.ports.memory_port import ShortTermMemoryPort
from src.settings import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class SupabaseShortTermMemory(ShortTermMemoryPort):
    """Recent conversation storage backed by Supabase Postgres."""

    def __init__(
        self,
        database_url: str | None = None,
        table_name: str | None = None,
        ttl_hours: int | None = None,
    ) -> None:
        self._database_url = database_url or settings.DATABASE_URL
        self._table_name = table_name or settings.SUPABASE_MEMORY_TABLE
        self._ttl_hours = ttl_hours or settings.SHORT_TERM_MEMORY_TTL_HOURS
        self._pool: asyncpg.Pool | None = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        if not self._database_url:
            logger.warning("DATABASE_URL is not configured. Short-term memory is disabled.")
            self._initialized = True
            return

        self._pool = await asyncpg.create_pool(self._database_url, min_size=1, max_size=5)
        await self._ensure_schema()
        self._initialized = True
        logger.info("Supabase short-term memory initialized")

    async def _ensure_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT,
                    source TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_session_created_at
                ON {self._table_name} (session_id, created_at DESC);
                """
            )

    async def add_message(self, record: MemoryRecord) -> None:
        await self.initialize()
        if self._pool is None:
            return
        assert self._pool is not None

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._table_name}
                    (session_id, user_id, source, role, content, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7);
                """,
                record.session_id,
                record.user_id,
                record.source,
                record.role,
                record.content,
                json.dumps(record.metadata),
                record.created_at,
            )
            await conn.execute(
                f"""
                DELETE FROM {self._table_name}
                WHERE session_id = $1
                  AND created_at < NOW() - ($2::text || ' hours')::interval;
                """,
                record.session_id,
                str(self._ttl_hours),
            )

    async def get_recent_messages(
        self,
        session_id: str,
        *,
        limit: int,
    ) -> list[MemoryRecord]:
        await self.initialize()
        if self._pool is None:
            return []
        assert self._pool is not None

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT session_id, user_id, source, role, content, metadata, created_at
                FROM {self._table_name}
                WHERE session_id = $1
                  AND created_at >= NOW() - ($2::text || ' hours')::interval
                ORDER BY created_at DESC
                LIMIT $3;
                """,
                session_id,
                str(self._ttl_hours),
                limit,
            )

        records = [
            MemoryRecord(
                session_id=row["session_id"],
                user_id=row["user_id"],
                source=row["source"],
                role=row["role"],
                content=row["content"],
                metadata=row["metadata"] or {},
                created_at=row["created_at"],
            )
            for row in reversed(rows)
        ]
        return records

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._initialized = False
