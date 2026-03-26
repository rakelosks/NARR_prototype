"""
API key authentication.

Validates requests via the ``X-API-Key`` header against
the key configured in ``NARR_API_KEY``.

If no key is configured (empty string), authentication is
disabled so local development works without extra setup.

Usage — protect a single router:

    from app.middleware.auth import require_api_key

    router = APIRouter(dependencies=[Depends(require_api_key)])

Usage — protect the entire app (see main.py).
"""

import logging
from typing import Optional

from fastapi import HTTPException, Security, Request
from fastapi.security import APIKeyHeader

from config import settings

logger = logging.getLogger(__name__)

_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    request: Request,
    api_key: Optional[str] = Security(_header_scheme),
):
    """
    FastAPI dependency that enforces API key authentication.

    - If ``NARR_API_KEY`` is not set, all requests are allowed (dev mode).
    - If set, the request must include a matching ``X-API-Key`` header.
    - ``/health`` and ``/docs`` are always exempt.
    """
    configured_key = settings.narr_api_key

    # No key configured → auth disabled (local development)
    if not configured_key:
        return None

    # Exempt paths (health checks, OpenAPI docs)
    exempt = {"/health", "/docs", "/redoc", "/openapi.json"}
    if request.url.path in exempt:
        return None

    if not api_key:
        logger.warning(f"Unauthenticated request to {request.url.path}")
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include an X-API-Key header.",
        )

    if api_key != configured_key:
        logger.warning(f"Invalid API key for {request.url.path}")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return api_key
