"""
API routes for dataset operations.
Handles dataset ingestion, listing, and querying.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/")
async def list_datasets():
    """List all registered datasets."""
    # TODO: Implement dataset listing from SQLite registry
    return {"datasets": []}


@router.post("/ingest")
async def ingest_dataset():
    """Ingest a new dataset from a URL or file."""
    # TODO: Implement dataset ingestion pipeline
    return {"status": "not implemented"}
