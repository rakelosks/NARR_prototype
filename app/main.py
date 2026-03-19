"""
Main FastAPI application entry point.
Run with: uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from data.cache.parquet_cache import cleanup_expired
from app.api.datasets import router as datasets_router
from app.api.narratives import router as narratives_router
from app.api.visualizations import router as visualizations_router
from app.api.jobs import router as jobs_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_expired()
    yield


app = FastAPI(
    title="Smart City Narrative Visualization Platform",
    description="AI-powered open data narrative visualization system for smart cities",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for Streamlit and other frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(datasets_router)
app.include_router(narratives_router)
app.include_router(visualizations_router)
app.include_router(jobs_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "portal_url": settings.ckan_portal_url,
        "portal_language": settings.ckan_portal_language,
    }