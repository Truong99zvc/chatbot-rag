from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # App
    APP_NAME: str = "Chatbot PDF RAG"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Google AI
    GOOGLE_API_KEY: str = ""
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    LLM_MODEL: str = "gemini-2.0-flash"
    LLM_TEMPERATURE: float = 0.2

    # Vector Store
    FAISS_INDEX_DIR: Path = Path("vectorstores/faiss/current_index")

    # Chunking
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200

    # RAG
    TOP_K_RESULTS: int = 4
    MAX_HISTORY_TURNS: int = 6

    # PostgreSQL
    DATABASE_URL: str = "postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb"

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW: int = 60  # seconds


settings = Settings()
