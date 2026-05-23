from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config.settings import settings


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Return a cached GoogleGenerativeAIEmbeddings instance.
    The cache ensures the object (and any underlying connection) is reused
    across requests rather than recreated on every call.
    """
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
    )
