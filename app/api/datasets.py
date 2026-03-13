"""
API routes for dataset operations.
Handles CKAN catalog browsing, dataset ingestion, and listing.
"""

import uuid
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from data.ingestion.ckan_client import CKANClient, CKANError
from data.ingestion.loader import load_from_url, load_from_file
from data.storage.catalog_index import CatalogIndex
from data.storage.metadata import MetadataStore
from data.cache.parquet_cache import save_snapshot

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/datasets", tags=["datasets"])

# Shared instances (in production these would be injected via dependency injection)
catalog_index = CatalogIndex()
metadata_store = MetadataStore()


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class IngestFromURLRequest(BaseModel):
    url: str
    name: str
    description: Optional[str] = None
    format: Optional[str] = None


class IngestFromCKANRequest(BaseModel):
    portal_url: str
    dataset_id: str
    resource_index: int = 0


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
        raise HTTPException(status_code=502, detail=f"CKAN API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
# Ingestion endpoints
# ---------------------------------------------------------------------------

@router.post("/ingest/url")
async def ingest_from_url(request: IngestFromURLRequest):
    """Ingest a dataset from a direct URL."""
    try:
        df, result = await load_from_url(request.url, format=request.format)

        dataset_id = str(uuid.uuid4())[:8]
        snapshot_path = save_snapshot(df, dataset_id)

        metadata_store.register_dataset(
            dataset_id=dataset_id,
            name=request.name,
            source_url=request.url,
            description=request.description,
            row_count=result.row_count,
        )

        return {
            "dataset_id": dataset_id,
            "name": request.name,
            "row_count": result.row_count,
            "column_count": result.column_count,
            "format": result.format,
            "columns": [c.model_dump() for c in result.columns],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ingest/ckan")
async def ingest_from_ckan(request: IngestFromCKANRequest):
    """Ingest a dataset from a CKAN portal."""
    try:
        client = CKANClient(request.portal_url)
        dataset = await client.get_dataset(request.dataset_id)

        supported = dataset.supported_resources
        if not supported:
            raise HTTPException(
                status_code=400,
                detail=f"No supported resources found in dataset '{request.dataset_id}'",
            )

        idx = min(request.resource_index, len(supported) - 1)
        resource = supported[idx]

        df = await client.download_resource_as_dataframe(resource)

        dataset_id = str(uuid.uuid4())[:8]
        snapshot_path = save_snapshot(df, dataset_id)

        metadata_store.register_dataset(
            dataset_id=dataset_id,
            name=dataset.title or dataset.name,
            source_url=resource.url,
            description=dataset.notes,
            row_count=len(df),
        )

        return {
            "dataset_id": dataset_id,
            "name": dataset.title or dataset.name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "format": resource.normalized_format,
            "columns": list(df.columns),
            "resource_name": resource.name,
        }
    except CKANError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Listing endpoints
# ---------------------------------------------------------------------------

@router.get("/")
async def list_datasets():
    """List all ingested datasets."""
    datasets = metadata_store.list_datasets()
    return {"datasets": datasets}


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get metadata for a single ingested dataset."""
    datasets = metadata_store.list_datasets()
    for ds in datasets:
        if ds["id"] == dataset_id:
            return ds
    raise HTTPException(status_code=404, detail="Dataset not found")