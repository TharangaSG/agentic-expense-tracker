from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    GROQ_API_KEY: str
    GEMINI_API_KEY: str
    ELEVENLABS_API_KEY: str

    # Provider Selection - Change these to switch providers
    LLM_PROVIDER: str = "groq"  # Options: "gemini", "groq"
    STT_PROVIDER: str = "gemini"  # Options: "gemini"
    TTS_PROVIDER: str = "elevenlabs"  # Options: "elevenlabs"
    VISION_PROVIDER: str = "groq"  # Options: "groq"
    EMBEDDING_PROVIDER: str = "gemini"  # Options: "gemini"

    # Database Configuration
    DATABASE_PROVIDER: str = "postgres"  # Options: "sqlite", "postgres"
    DATABASE_URL: str

    # Model Configuration
    MAIN_MODEL_NAME: str = "llama-3.3-70b-versatile"
    VISION_MODEL_NAME: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    ELEVENLABS_VOICE_ID: str = "Xb7hH8MSUJpSbSDYk0k2"
    STT_MODEL_NAME: str = "gemini-2.5-flash"
    TTS_MODEL_NAME: str = "eleven_multilingual_v2"
    EMBEDDING_MODEL_NAME: str = "gemini-embedding-001"
    EMBEDDING_DIMENSION: int = 768

    # Conversation Memory Configuration
    MEMORY_ENABLED: bool = True
    MEMORY_TOP_K: int = 3
    SHORT_TERM_MEMORY_LIMIT: int = 8
    SHORT_TERM_MEMORY_TTL_HOURS: int = 24
    MEMORY_MIN_CONTENT_LENGTH: int = 12

    # Supabase short-term memory
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    SUPABASE_SCHEMA: str = "public"
    SUPABASE_MEMORY_TABLE: str = "conversation_messages"

    # Qdrant long-term memory
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "conversation_memory"
    MEMORY_EMBEDDING_DIMENSION: int = 768

    # Define a threshold for detecting silence and a timeout for ending a turn
    SILENCE_THRESHOLD: int = 3500
    SILENCE_TIMEOUT: float = 1300.  # ms

    # WhatsApp API credentials
    WHATSAPP_TOKEN: str
    WHATSAPP_PHONE_NUMBER_ID: str
    WHATSAPP_VERIFY_TOKEN: str

    # Telegram Bot API credentials
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_WEBHOOK_SECRET_TOKEN: str = ""

    # Langfuse observability
    LANGFUSE_ENABLED: bool = True
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_BASE_URL: str = "https://cloud.langfuse.com"
    LANGFUSE_TRACING_ENVIRONMENT: str = "development"


# Load settings
settings = Settings()

# Log configuration on module load (masking sensitive values)
logger.info(
    f"Settings loaded | "
    f"LLM Provider: {settings.LLM_PROVIDER} | Model: {settings.MAIN_MODEL_NAME} | "
    f"STT Provider: {settings.STT_PROVIDER} | Model: {settings.STT_MODEL_NAME} | "
    f"TTS Provider: {settings.TTS_PROVIDER} | Model: {settings.TTS_MODEL_NAME} | "
    f"Vision Provider: {settings.VISION_PROVIDER} | Model: {settings.VISION_MODEL_NAME} | "
    f"Embedding Provider: {settings.EMBEDDING_PROVIDER} | Model: {settings.EMBEDDING_MODEL_NAME} | "
    f"Database Provider: {settings.DATABASE_PROVIDER} | "
    f"Memory Enabled: {settings.MEMORY_ENABLED} | "
    f"Langfuse Enabled: {settings.LANGFUSE_ENABLED and bool(settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY)} | "
)
