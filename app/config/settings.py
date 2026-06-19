import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Fix for Windows OpenMP crash with PyTorch/HuggingFace
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # App
    APP_NAME: str = "UIT Academic Policies Chatbot"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # HuggingFace
    HF_TOKEN: str = ""
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
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

    # Rate Limiting
    RATE_LIMIT_WINDOW: int = 60
    RATE_LIMIT_MAX_REQUESTS: int = 20


settings = Settings()
