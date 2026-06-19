from functools import lru_cache
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from app.config.settings import settings

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEndpointEmbeddings:
    """
    Returns the HuggingFace Embeddings model via Inference API.
    This avoids local PyTorch Windows crash issues (exit code 1).
    """
    return HuggingFaceEndpointEmbeddings(
        model=settings.EMBEDDING_MODEL,
        task="feature-extraction",
        huggingfacehub_api_token=settings.HF_TOKEN,
    )
