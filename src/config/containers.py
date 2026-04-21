"""
Dependency Injection Container

Uses dependency-injector for professional DI management.
Provides singleton instances of all providers based on configuration.
"""

import logging

from dependency_injector import containers, providers

from src.utils.logging_config import get_logger
from src.ports.llm_port import LLMPort
from src.ports.stt_port import STTPort
from src.ports.tts_port import TTSPort
from src.ports.vision_port import VisionPort
from src.ports.database_port import DatabasePort, AsyncDatabasePort
from src.ports.embedding_port import EmbeddingPort

from src.adapters.llm.gemini_adapter import GeminiLLMAdapter
from src.adapters.llm.groq_adapter import GroqLLMAdapter
from src.adapters.stt.gemini_stt_adapter import GeminiSTTAdapter
from src.adapters.tts.elevenlabs_adapter import ElevenLabsTTSAdapter
from src.adapters.vision.groq_vision_adapter import GroqVisionAdapter
from src.adapters.database.sqlite_adapter import SQLiteDatabaseAdapter
from src.adapters.database.postgres_adapter import PostgresAdapter
from src.adapters.embedding.gemini_embedding_adapter import GeminiEmbeddingAdapter

from src.settings import settings

logger = get_logger(__name__)


class Container(containers.DeclarativeContainer):
    """Application DI Container"""
    
    # Configuration
    config = providers.Configuration()
    
    # LLM Provider Factory
    llm_provider = providers.Selector(
        config.llm_provider,
        gemini=providers.Singleton(GeminiLLMAdapter),
        groq=providers.Singleton(GroqLLMAdapter),
    )
    
    # STT Provider Factory
    stt_provider = providers.Selector(
        config.stt_provider,
        gemini=providers.Singleton(GeminiSTTAdapter),
    )
    
    # TTS Provider Factory
    tts_provider = providers.Selector(
        config.tts_provider,
        elevenlabs=providers.Singleton(ElevenLabsTTSAdapter),
    )
    
    # Vision Provider Factory
    vision_provider = providers.Selector(
        config.vision_provider,
        groq=providers.Singleton(GroqVisionAdapter),
    )
    
    # Database Provider (Sync - legacy SQLite)
    database = providers.Selector(
        config.database_provider,
        sqlite=providers.Singleton(SQLiteDatabaseAdapter),
        postgres=providers.Singleton(SQLiteDatabaseAdapter),  # Fallback for sync interface
    )
    
    # Embedding Provider Factory
    embedding_provider = providers.Selector(
        config.embedding_provider,
        gemini=providers.Singleton(
            GeminiEmbeddingAdapter,
            model_name=settings.EMBEDDING_MODEL_NAME,
            output_dimensionality=settings.EMBEDDING_DIMENSION,
        ),
    )

    # Async Database Provider (PostgreSQL with pgvector)
    # Note: embedding_provider is injected via factory below
    async_database = providers.Singleton(
        PostgresAdapter,
        database_url=settings.DATABASE_URL,
    )


# Create and configure the container
container = Container()
container.config.llm_provider.from_value(settings.LLM_PROVIDER.lower())
container.config.stt_provider.from_value(settings.STT_PROVIDER.lower())
container.config.tts_provider.from_value(settings.TTS_PROVIDER.lower())
container.config.vision_provider.from_value(settings.VISION_PROVIDER.lower())
container.config.database_provider.from_value(settings.DATABASE_PROVIDER.lower())
container.config.embedding_provider.from_value(settings.EMBEDDING_PROVIDER.lower())

# Wire embedding provider into PostgresAdapter
_db_instance = container.async_database()
_db_instance.set_embedding_provider(container.embedding_provider())

logger.info(
    f"Dependency injection container configured | "
    f"LLM: {settings.LLM_PROVIDER} | STT: {settings.STT_PROVIDER} | "
    f"TTS: {settings.TTS_PROVIDER} | Vision: {settings.VISION_PROVIDER} | "
    f"Embedding: {settings.EMBEDDING_PROVIDER} | Database: {settings.DATABASE_PROVIDER}"
)


# Convenience functions for backward compatibility
def get_llm_provider() -> LLMPort:
    """Get configured LLM provider instance."""
    return container.llm_provider()


def get_stt_provider() -> STTPort:
    """Get configured Speech-to-Text provider instance."""
    return container.stt_provider()


def get_tts_provider() -> TTSPort:
    """Get configured Text-to-Speech provider instance."""
    return container.tts_provider()


def get_vision_provider() -> VisionPort:
    """Get configured Vision provider instance."""
    return container.vision_provider()


def get_database() -> DatabasePort:
    """Get sync database instance (legacy SQLite)."""
    return container.database()


def get_async_database() -> AsyncDatabasePort:
    """Get async database instance (PostgreSQL with pgvector)."""
    return container.async_database()


def get_embedding_provider() -> EmbeddingPort:
    """Get configured embedding provider instance."""
    return container.embedding_provider()


def reset_providers():
    """Reset all provider instances (useful for testing)."""
    container.reset_singletons()
