from fastapi import APIRouter
from app.config.settings import settings

router = APIRouter()


@router.get("", summary="Health check")
async def health_check() -> dict:
    """Return service liveness status."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }
