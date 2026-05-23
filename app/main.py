from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.api import documents, health, rag
from app.middleware.error_handler import register_error_handlers
from app.middleware.logger import LoggerMiddleware
from app.middleware.rate_limiter import RateLimiterMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pre-load vector store, warm up connections, etc.
    yield
    # Shutdown: clean up resources


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware (added in reverse order — last added = outermost)
    app.add_middleware(LoggerMiddleware)
    app.add_middleware(RateLimiterMiddleware)

    # Global error handlers
    register_error_handlers(app)

    # Routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
    app.include_router(rag.router, prefix="/api/v1/rag", tags=["RAG"])

    return app


app = create_app()
