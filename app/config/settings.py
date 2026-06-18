from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # App
    APP_NAME: str = "UIT Quy Chế Đào Tạo Chatbot"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Google AI
    GOOGLE_API_KEY: str = ""
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    LLM_MODEL: str = "gemini-2.0-flash"
    LLM_TEMPERATURE: float = 0.2

    # Vector Store
    FAISS_INDEX_DIR: Path = Path("vectorstores/faiss/current_index")

    # Data
    DATA_DIR: Path = Path("data")

    # Session history (JSON file on disk, per session_id)
    SESSION_STORE_FILE: Path = Path("vectorstores/sessions.json")

    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # RAG
    TOP_K_RESULTS: int = 5
    MAX_HISTORY_TURNS: int = 6


settings = Settings()
