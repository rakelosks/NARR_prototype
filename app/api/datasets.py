"""
API routes for dataset operations.
Handles CKAN catalog browsing and template configuration listing.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from config import settings
from data.ingestion.ckan_client import CKANClient, CKANError
from data.storage.catalog_index import CatalogIndex
from data.storage.metadata import MetadataStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/datasets", tags=["datasets"])

# Shared instances
catalog_index = CatalogIndex(db_path=settings.metadata_db_path)
metadata_store = MetadataStore(db_path=settings.metadata_db_path)


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class CatalogRefreshRequest(BaseModel):
    portal_url: str
    full: bool = False


# ---------------------------------------------------------------------------
# Catalog endpoints
# ---------------------------------------------------------------------------

@router.post("/catalog/refresh")
async def refresh_catalog(request: CatalogRefreshRequest):
    """Refresh the catalog index from a CKAN portal."""
    try:
        client = CKANClient(request.portal_url)
        await catalog_index.refresh(client, portal_url=request.portal_url, full=request.full)
        count = catalog_index.count(portal_url=request.portal_url)
        return {"status": "success", "datasets_indexed": count}
    except CKANError as e:
        logger.exception("Catalog refresh failed due to CKAN API error")
        raise HTTPException(status_code=502, detail="Failed to refresh catalog from CKAN portal")
    except Exception as e:
        logger.exception("Catalog refresh failed unexpectedly")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/catalog/search")
async def search_catalog(
    q: Optional[str] = None,
    portal_url: Optional[str] = None,
    format: Optional[str] = None,
    organization: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    offset: int = 0,
):
    """Search the local catalog index."""
    results = catalog_index.search(
        query=q,
        portal_url=portal_url,
        format_filter=format,
        organization=organization,
        limit=limit,
        offset=offset,
    )
    return {"results": results, "count": len(results)}


@router.get("/catalog/{dataset_id}")
async def get_catalog_entry(dataset_id: str):
    """Get a single catalog entry."""
    entry = catalog_index.get_entry(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found in catalog")
    return entry


# ---------------------------------------------------------------------------
# Template configuration endpoints
# ---------------------------------------------------------------------------

@router.get("/")
async def list_configs():
    """List all datasets with saved template configurations."""
    configs = metadata_store.list_configs()
    return {"datasets": configs}


@router.get("/{dataset_id}")
async def get_config(dataset_id: str):
    """Get template configuration for a dataset."""
    config = metadata_store.get_config(dataset_id)
    if not config:
        raise HTTPException(status_code=404, detail="No configuration found for this dataset")
    return config
