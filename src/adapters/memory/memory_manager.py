"""
Memory Manager

Coordinates short-term conversational state and long-term semantic memory.
"""

from __future__ import annotations

from uuid import uuid4

from src.domain.models import MemoryRecord, Message
from src.ports.embedding_port import EmbeddingPort
from src.ports.memory_port import LongTermMemoryPort, ShortTermMemoryPort
from src.settings import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """Combines Supabase and Qdrant memory for the main agent."""

    def __init__(
        self,
        short_term_memory: ShortTermMemoryPort,
        long_term_memory: LongTermMemoryPort,
        embedding_provider: EmbeddingPort,
        *,
        short_term_limit: int | None = None,
        long_term_top_k: int | None = None,
        min_content_length: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._short_term_memory = short_term_memory
        self._long_term_memory = long_term_memory
        self._embedding_provider = embedding_provider
        self._short_term_limit = short_term_limit or settings.SHORT_TERM_MEMORY_LIMIT
        self._long_term_top_k = long_term_top_k or settings.MEMORY_TOP_K
        self._min_content_length = min_content_length or settings.MEMORY_MIN_CONTENT_LENGTH
        self._enabled = settings.MEMORY_ENABLED if enabled is None else enabled
        self._initialized = False

    async def initialize(self) -> None:
        if not self._enabled or self._initialized:
            return

        await self._short_term_memory.initialize()
        await self._long_term_memory.initialize()
        self._initialized = True

    async def build_context(
        self,
        *,
        session_id: str | None,
        user_id: str | None,
        source: str,
        user_input: str,
    ) -> tuple[list[Message], str | None]:
        if not self._enabled or not session_id:
            return [], None

        await self.initialize()
        recent_records = await self._short_term_memory.get_recent_messages(
            session_id,
            limit=self._short_term_limit,
        )
        context_messages = [
            Message(role=record.role, content=record.content)
            for record in recent_records
            if record.content
        ]

        long_term_context = None
        if self._is_memorable(user_input):
            embedding = await self._embedding_provider.generate_embedding(user_input)
            related_memories = await self._long_term_memory.search(
                embedding=embedding,
                session_id=session_id,
                user_id=user_id,
                limit=self._long_term_top_k,
            )
            if related_memories:
                memory_lines = [f"- {memory.content}" for memory in related_memories]
                long_term_context = (
                    "Relevant past memories that may help with this reply:\n"
                    + "\n".join(memory_lines)
                )

        return context_messages, long_term_context

    async def store_turn(
        self,
        *,
        session_id: str | None,
        user_id: str | None,
        source: str,
        user_input: str,
        assistant_response: str,
    ) -> None:
        if not self._enabled or not session_id:
            return

        await self.initialize()

        await self._store_short_term(
            session_id=session_id,
            user_id=user_id,
            source=source,
            user_input=user_input,
            assistant_response=assistant_response,
        )
        await self._store_long_term(
            session_id=session_id,
            user_id=user_id,
            source=source,
            user_input=user_input,
            assistant_response=assistant_response,
        )

    async def close(self) -> None:
        await self._short_term_memory.close()
        await self._long_term_memory.close()
        self._initialized = False

    async def _store_short_term(
        self,
        *,
        session_id: str,
        user_id: str | None,
        source: str,
        user_input: str,
        assistant_response: str,
    ) -> None:
        records = [
            MemoryRecord(
                session_id=session_id,
                user_id=user_id,
                source=source,
                role="user",
                content=user_input,
                metadata={"channel": source},
            ),
            MemoryRecord(
                session_id=session_id,
                user_id=user_id,
                source=source,
                role="assistant",
                content=assistant_response,
                metadata={"channel": source},
            ),
        ]

        for record in records:
            if self._is_memorable(record.content):
                await self._short_term_memory.add_message(record)

    async def _store_long_term(
        self,
        *,
        session_id: str,
        user_id: str | None,
        source: str,
        user_input: str,
        assistant_response: str,
    ) -> None:
        content_parts = [part.strip() for part in [user_input, assistant_response] if self._is_memorable(part)]
        if not content_parts:
            return

        content = "\n".join(
            [
                f"User: {user_input.strip()}",
                f"Assistant: {assistant_response.strip()}",
            ]
        )
        embedding = await self._embedding_provider.generate_embedding(user_input)
        await self._long_term_memory.store_memory(
            memory_id=str(uuid4()),
            embedding=embedding,
            content=content,
            metadata={
                "session_id": session_id,
                "user_id": user_id,
                "source": source,
            },
        )

    def _is_memorable(self, content: str | None) -> bool:
        return bool(content and len(content.strip()) >= self._min_content_length)


_memory_manager: MemoryManager | None = None


def configure_memory_manager(memory_manager: MemoryManager) -> None:
    global _memory_manager
    _memory_manager = memory_manager


def get_memory_manager() -> MemoryManager:
    if _memory_manager is None:
        raise RuntimeError("Memory manager has not been configured.")
    return _memory_manager


async def close_memory_manager() -> None:
    if _memory_manager is not None:
        await _memory_manager.close()
