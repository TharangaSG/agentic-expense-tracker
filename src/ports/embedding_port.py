"""
Embedding Port Interface

Defines the contract for embedding providers using Pydantic models.
"""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingPort(ABC):
    """Port interface for embedding providers"""

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for the given text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the current embedding model name"""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vector"""
        pass
