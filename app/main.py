"""
Main FastAPI application entry point.
Run with: uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from data.cache.parquet_cache import cleanup_expired
from app.api.datasets import router as datasets_router
from app.api.narratives import router as narratives_router
from app.api.visualizations import router as visualizations_router
from app.api.jobs import router as jobs_router
from app.middleware.auth import require_api_key
from app.middleware.rate_limit import RateLimitMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Log security configuration on startup
    if settings.narr_api_key:
        logger.info("API key authentication enabled")
    else:
        logger.info("API key authentication disabled (no NARR_API_KEY set)")
    if settings.rate_limit_rpm > 0:
        logger.info(f"Rate limiting enabled: {settings.rate_limit_rpm} rpm per client")
    else:
        logger.info("Rate limiting disabled")
    cleanup_expired()
    yield


app = FastAPI(
    title="Smart City Narrative Visualization Platform",
    description="AI-powered open data narrative visualization system for smart cities",
    version="0.1.0",
    lifespan=lifespan,
    # Apply API key auth globally; /health and /docs are exempt inside the dependency
    dependencies=[Depends(require_api_key)],
)

# --- Middleware stack (applied bottom-to-top) ---

# CORS for Streamlit and other frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (runs before auth so abusive unauthenticated requests are rejected early)
app.add_middleware(RateLimitMiddleware)

# Register routers
app.include_router(datasets_router)
app.include_router(narratives_router)
app.include_router(visualizations_router)
app.include_router(jobs_router)


@app.get("/health")
async def health_check():
    """Health check endpoint (no auth required)."""
    return {
        "status": "ok",
        "portal_url": settings.ckan_portal_url,
        "portal_language": settings.ckan_portal_language,
    }