"""
Dependency Injection Container

Uses dependency-injector for professional DI management.
Provides singleton instances of all providers based on configuration.
"""

from dependency_injector import containers, providers

from src.ports.llm_port import LLMPort
from src.ports.stt_port import STTPort
from src.ports.tts_port import TTSPort
from src.ports.vision_port import VisionPort
from src.ports.database_port import DatabasePort

from src.adapters.llm.gemini_adapter import GeminiLLMAdapter
from src.adapters.llm.groq_adapter import GroqLLMAdapter
from src.adapters.stt.gemini_stt_adapter import GeminiSTTAdapter
from src.adapters.tts.elevenlabs_adapter import ElevenLabsTTSAdapter
from src.adapters.vision.groq_vision_adapter import GroqVisionAdapter
from src.adapters.database.sqlite_adapter import SQLiteDatabaseAdapter

from src.settings import settings


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
    
    # Database Provider (Singleton)
    database = providers.Singleton(SQLiteDatabaseAdapter)


# Create and configure the container
container = Container()
container.config.llm_provider.from_value(settings.LLM_PROVIDER.lower())
container.config.stt_provider.from_value(settings.STT_PROVIDER.lower())
container.config.tts_provider.from_value(settings.TTS_PROVIDER.lower())
container.config.vision_provider.from_value(settings.VISION_PROVIDER.lower())


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
    """Get database instance."""
    return container.database()


def reset_providers():
    """Reset all provider instances (useful for testing)."""
    container.reset_singletons()
