"""
Memory Port Interfaces

Contracts for short-term and long-term conversational memory adapters.
"""

from abc import ABC, abstractmethod
from typing import Any

from src.domain.models import MemoryRecord, MemorySearchResult


class ShortTermMemoryPort(ABC):
    """Port for recent conversational state."""

    @abstractmethod
    async def initialize(self) -> None:
        """Prepare the backing store."""
        pass

    @abstractmethod
    async def add_message(self, record: MemoryRecord) -> None:
        """Persist a single chat message."""
        pass

    @abstractmethod
    async def get_recent_messages(
        self,
        session_id: str,
        *,
        limit: int,
    ) -> list[MemoryRecord]:
        """Fetch recent messages for a session."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Release any resources."""
        pass


class LongTermMemoryPort(ABC):
    """Port for semantic memory recall."""

    @abstractmethod
    async def initialize(self) -> None:
        """Prepare the vector store."""
        pass

    @abstractmethod
    async def store_memory(
        self,
        *,
        memory_id: str,
        embedding: list[float],
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        """Persist a semantic memory document."""
        pass

    @abstractmethod
    async def search(
        self,
        *,
        embedding: list[float],
        session_id: str | None,
        user_id: str | None,
        limit: int,
    ) -> list[MemorySearchResult]:
        """Search semantically related memories."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Release any resources."""
        pass
