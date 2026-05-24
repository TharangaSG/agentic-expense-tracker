"""
Qdrant Long-Term Memory Adapter

Stores semantic memory snippets for cross-session recall.
"""

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from src.domain.models import MemorySearchResult
from src.ports.memory_port import LongTermMemoryPort
from src.settings import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class QdrantLongTermMemory(LongTermMemoryPort):
    """Semantic memory storage backed by Qdrant."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        vector_size: int | None = None,
    ) -> None:
        self._collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self._vector_size = vector_size or settings.MEMORY_EMBEDDING_DIMENSION
        resolved_url = url or settings.QDRANT_URL
        self._client = (
            QdrantClient(
                url=resolved_url,
                api_key=api_key or settings.QDRANT_API_KEY or None,
            )
            if resolved_url
            else None
        )
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        if self._client is None:
            logger.warning("Qdrant URL is not configured. Long-term memory is disabled.")
            self._initialized = True
            return

        collections = self._client.collection_exists(collection_name=self._collection_name)
        if not collections:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=rest.VectorParams(
                    size=self._vector_size,
                    distance=rest.Distance.COSINE,
                ),
            )
        self._initialized = True
        logger.info("Qdrant long-term memory initialized")

    async def store_memory(
        self,
        *,
        memory_id: str,
        embedding: list[float],
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        await self.initialize()
        if self._client is None:
            return
        payload = {"content": content, **metadata}
        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                rest.PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

    async def search(
        self,
        *,
        embedding: list[float],
        session_id: str | None,
        user_id: str | None,
        limit: int,
    ) -> list[MemorySearchResult]:
        await self.initialize()
        if self._client is None:
            return []

        filters: list[rest.FieldCondition] = []
        if user_id:
            filters.append(rest.FieldCondition(key="user_id", match=rest.MatchValue(value=user_id)))
        elif session_id:
            filters.append(rest.FieldCondition(key="session_id", match=rest.MatchValue(value=session_id)))

        query_filter = rest.Filter(must=filters) if filters else None
        hits = self._client.search(
            collection_name=self._collection_name,
            query_vector=embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [
            MemorySearchResult(
                content=(hit.payload or {}).get("content", ""),
                score=hit.score,
                session_id=(hit.payload or {}).get("session_id"),
                user_id=(hit.payload or {}).get("user_id"),
                source=(hit.payload or {}).get("source"),
                metadata={k: v for k, v in (hit.payload or {}).items() if k != "content"},
            )
            for hit in hits
            if (hit.payload or {}).get("content")
        ]

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
        self._initialized = False
