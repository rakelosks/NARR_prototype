"""
In-memory sliding-window rate limiter middleware.

Limits requests per client (identified by IP address) within a
configurable time window.  Returns HTTP 429 when the limit is hit.

Configuration (via environment / config.py):
    RATE_LIMIT_RPM   – max requests per minute per client  (default 60)

Exempt paths:
    /health, /docs, /redoc, /openapi.json
"""

import logging
import time
import hashlib
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config import settings

logger = logging.getLogger(__name__)

# Paths that are never rate-limited
_EXEMPT_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter.

    Stores per-client request timestamps in memory and rejects
    requests that exceed the configured limit within the window.
    """

    def __init__(self, app, requests_per_minute: int = 0):
        super().__init__(app)
        # 0 means "use config value"; fall back to 60
        self.rpm = requests_per_minute or settings.rate_limit_rpm
        self.window = 60.0  # seconds
        # client_id → list of timestamps
        self._hits: dict[str, list[float]] = defaultdict(list)

    def _client_id(self, request: Request) -> str:
        """
        Identify the client.  Prefers the API key (if present) so
        that all requests from the same key share a bucket, regardless
        of proxy / IP.  Falls back to the client IP.
        """
        api_key = request.headers.get("x-api-key")
        if api_key:
            digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
            return f"key:{digest}"
        # Only trust X-Forwarded-For when explicitly enabled behind a trusted proxy.
        if settings.trust_proxy_headers:
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                return f"ip:{forwarded.split(',')[0].strip()}"
        return f"ip:{request.client.host}" if request.client else "ip:unknown"

    def _prune(self, timestamps: list[float], now: float) -> list[float]:
        """Remove timestamps outside the current window."""
        cutoff = now - self.window
        return [t for t in timestamps if t > cutoff]

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting if disabled (rpm <= 0)
        if self.rpm <= 0:
            return await call_next(request)

        # Exempt paths
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        client = self._client_id(request)
        now = time.monotonic()

        # Prune old entries
        self._hits[client] = self._prune(self._hits[client], now)

        if len(self._hits[client]) >= self.rpm:
            retry_after = int(self.window - (now - self._hits[client][0])) + 1
            logger.warning(
                f"Rate limit exceeded for {client} "
                f"({len(self._hits[client])}/{self.rpm} rpm)"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded ({self.rpm} requests/minute). "
                              f"Retry after {retry_after}s.",
                },
                headers={"Retry-After": str(retry_after)},
            )

        self._hits[client].append(now)

        response = await call_next(request)

        # Add rate-limit headers so the client can self-throttle
        remaining = max(0, self.rpm - len(self._hits[client]))
        response.headers["X-RateLimit-Limit"] = str(self.rpm)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response
