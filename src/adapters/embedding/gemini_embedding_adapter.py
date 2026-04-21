"""
Gemini Embedding Adapter

Implements EmbeddingPort using Google Gemini embedding models.
Supports both gemini-embedding-001 and text-embedding-004 models.
"""

import logging
from typing import List

from google import genai
from google.genai import types

from src.ports.embedding_port import EmbeddingPort
from src.settings import settings

logger = logging.getLogger(__name__)


class GeminiEmbeddingAdapter(EmbeddingPort):
    """Gemini embedding adapter implementing EmbeddingPort"""

    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        output_dimensionality: int = 768,
    ):
        """
        Initialize Gemini embedding adapter.

        Args:
            model_name: Gemini embedding model to use
            output_dimensionality: Dimension of output embedding vector
        """
        self._model_name = model_name
        self._output_dimensionality = output_dimensionality
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a 768-dim embedding for the given text using Gemini.

        Uses gemini-embedding-001 with output_dimensionality=768 by default.
        Falls back to a zero vector if embedding generation fails.

        Args:
            text: Input text to embed

        Returns:
            List of 768 floats representing the embedding vector
        """
        try:
            result = await self._client.aio.models.embed_content(
                model=self._model_name,
                contents=text,
                config=types.EmbedContentConfig(
                    output_dimensionality=self._output_dimensionality,
                ),
            )
            return result.embeddings[0].values

        except Exception as e:
            logger.error(
                f"Embedding generation failed for model '{self._model_name}': {e}"
            )
            return [0.0] * self._output_dimensionality

    def get_model_name(self) -> str:
        """Get the current embedding model name"""
        return self._model_name

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vector"""
        return self._output_dimensionality
