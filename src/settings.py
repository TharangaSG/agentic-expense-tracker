from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    GROQ_API_KEY: str
    GEMINI_API_KEY: str

    MAIN_MODEL_NAME: str = "gemini-2.5-flash"
    VISION_MODEL_NAME: str = "meta-llama/llama-4-scout-17b-16e-instruct"

settings = Settings()
