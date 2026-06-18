"""
UIT Quy Chế Đào Tạo Chatbot — FastAPI Application Entry Point

Knowledge base: Quy chế, quy định, quy trình đào tạo đại học chính quy UIT
                (pre-indexed via `make build-index`)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.api import health, rag
from app.middleware.error_handler import register_error_handlers
from app.middleware.logger import LoggerMiddleware
from app.middleware.rate_limiter import RateLimiterMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: validate that the FAISS index exists
    if not settings.FAISS_INDEX_DIR.exists():
        import logging
        logging.getLogger(__name__).warning(
            "FAISS index not found at '%s'. "
            "Run `make build-index` before sending queries.",
            settings.FAISS_INDEX_DIR,
        )
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

    return app


app = create_app()
