from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config.settings import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFaceEmbeddings instance.

    Model: intfloat/multilingual-e5-large
    - Runs fully locally (downloaded once to ~/.cache/huggingface)
    - Excellent multilingual support including Vietnamese
    - No API token required after the initial download

    The lru_cache ensures the model is loaded into memory only once
    and reused across all requests.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
