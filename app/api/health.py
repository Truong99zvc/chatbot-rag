from fastapi import APIRouter
from app.config.settings import settings

router = APIRouter()


@router.get("", summary="Health check")
async def health_check() -> dict:
    """Return service liveness and knowledge-base index status."""
    index_ready = settings.FAISS_INDEX_DIR.exists()
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "index_ready": index_ready,
        "index_path": str(settings.FAISS_INDEX_DIR),
    }
