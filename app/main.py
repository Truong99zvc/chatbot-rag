"""
UIT Quy Chế Đào Tạo Chatbot — FastAPI Application Entry Point

Knowledge base: Quy chế, quy định, quy trình đào tạo đại học chính quy UIT
                (pre-indexed via `make build-index`)
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config.settings import settings
from app.api import health, rag
from app.middleware.error_handler import register_error_handlers
from app.middleware.logger import LoggerMiddleware
from app.middleware.rate_limiter import RateLimiterMiddleware
from app.database.connection import init_db

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQL database tables
    init_db()

    # Startup: validate Qdrant connection or local path
    import logging
    logger = logging.getLogger(__name__)
    try:
        from qdrant_client import QdrantClient
        if settings.QDRANT_URL:
            client = QdrantClient(url=settings.QDRANT_URL)
            client.collection_exists(settings.QDRANT_COLLECTION)
        else:
            if not settings.QDRANT_PATH.exists():
                logger.warning(
                    "Qdrant local storage not found at '%s'. "
                    "Run `make build-index` before sending queries.",
                    settings.QDRANT_PATH,
                )
    except Exception as e:
        logger.warning("Could not validate Qdrant connection/storage on startup: %s", e)
    yield
    # Shutdown: nothing to clean up (FAISS is disk-backed)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "Chatbot hỏi đáp về Quy chế, Quy định và Quy trình Đào tạo "
            "Đại học Chính quy của Trường ĐH Công nghệ Thông tin – ĐHQG TP.HCM (UIT)."
        ),
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow all origins for development; restrict in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware (last added = outermost)
    app.add_middleware(LoggerMiddleware)
    app.add_middleware(RateLimiterMiddleware)

    # Global error handlers
    register_error_handlers(app)

    # Routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(rag.router, prefix="/api/v1/rag", tags=["RAG – Quy chế UIT"])

    # Serve the chat UI at the root
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False, tags=["UI"])
        async def serve_ui() -> FileResponse:
            """Serve the chat web interface."""
            return FileResponse(str(STATIC_DIR / "index.html"))

    return app


app = create_app()
