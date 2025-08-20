from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    GROQ_API_KEY: str
    GEMINI_API_KEY: str
    ELEVENLABS_API_KEY: str 

    MAIN_MODEL_NAME: str = "gemini-2.5-flash"
    VISION_MODEL_NAME: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    ELEVENLABS_VOICE_ID: str = "Xb7hH8MSUJpSbSDYk0k2"

    # Define a threshold for detecting silence and a timeout for ending a turn
    SILENCE_THRESHOLD: int = 3500
    SILENCE_TIMEOUT: float = 1300.  # ms

settings = Settings()
