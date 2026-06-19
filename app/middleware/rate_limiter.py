import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config.settings import settings


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Simple in-memory sliding-window rate limiter per client IP."""

    def __init__(self, app) -> None:
        super().__init__(app)
        self._timestamps: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = settings.RATE_LIMIT_WINDOW
        limit = settings.RATE_LIMIT_MAX_REQUESTS

        # Remove timestamps outside the current window
        self._timestamps[client_ip] = [
            ts for ts in self._timestamps[client_ip] if now - ts < window
        ]

        if len(self._timestamps[client_ip]) >= limit:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "detail": f"Max {limit} requests per {window}s. Try again later.",
                },
            )

        self._timestamps[client_ip].append(now)
        return await call_next(request)
