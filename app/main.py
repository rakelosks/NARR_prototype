"""
Main FastAPI application entry point.
Run with: uvicorn app.main:app --reload
"""

from fastapi import FastAPI

app = FastAPI(
    title="Smart City Narrative Visualization Platform",
    description="AI-powered open data narrative visualization system for smart cities",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
