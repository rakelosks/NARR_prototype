"""
CKAN connector for catalog discovery and resource download.
Supports searching, listing, and fetching datasets from any CKAN instance.

Includes request throttling to avoid overwhelming CKAN portals,
especially during full catalog builds.
"""

import httpx
import logging
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum

from data.ingestion.throttle import Throttle, retry_with_backoff

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ResourceFormat(str, Enum):
    """Supported resource file formats."""
    CSV = "csv"
    JSON = "json"
    GEOJSON = "geojson"
    XLS = "xls"
    XLSX = "xlsx"


class CKANResource(BaseModel):
    """A single downloadable resource within a CKAN dataset."""
    id: str
    name: str = ""
    description: str = ""
    format: str = ""
    url: str = ""
    size: Optional[int] = None
    last_modified: Optional[str] = None

    @property
    def normalized_format(self) -> str:
        """Normalize format string for comparison."""
        return self.format.strip().lower()

    @property
    def is_supported(self) -> bool:
        """Check if this resource is in a supported format."""
        supported = {f.value for f in ResourceFormat}
        return self.normalized_format in supported


class CKANDataset(BaseModel):
    """Metadata for a CKAN dataset (package)."""
    id: str
    name: str
    title: str = ""
    notes: str = Field(default="", description="Dataset description")
    tags: list[str] = []
    organization: Optional[str] = None
    metadata_created: Optional[str] = None
    metadata_modified: Optional[str] = None
    num_resources: int = 0
    resources: list[CKANResource] = []

    @property
    def supported_resources(self) -> list[CKANResource]:
        """Return only resources in supported formats."""
        return [r for r in self.resources if r.is_supported]


class CKANCatalogEntry(BaseModel):
    """Lightweight catalog entry for index storage."""
    dataset_id: str
    name: str
    title: str = ""
    description: str = ""
    tags: list[str] = []
    organization: Optional[str] = None
    resource_formats: list[str] = []
    num_resources: int = 0
    metadata_modified: Optional[str] = None


# ---------------------------------------------------------------------------
# CKAN Client
# ---------------------------------------------------------------------------

class CKANClient:
    """
    Client for interacting with a CKAN instance API.

    Usage:
        client = CKANClient("https://gagnagatt.reykjavik.is/en/api/3")
        datasets = await client.search_datasets("transport")
        dataset = await client.get_dataset("some-dataset-id")
        data = await client.download_resource(dataset.resources[0])
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_concurrent: int = 3,
        min_interval: float = 0.2,
        max_retries: int = 3,
    ):
        """
        Initialize the CKAN client.

        Args:
            base_url: The CKAN API base URL (e.g. https://gagnagatt.reykjavik.is/en/api/3)
            timeout: Request timeout in seconds.
            max_concurrent: Max concurrent requests to the portal.
            min_interval: Minimum seconds between requests.
            max_retries: Number of retries on transient failures.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._throttle = Throttle(
            max_concurrent=max_concurrent,
            min_interval=min_interval,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_url(self, action: str) -> str:
        """Build the full URL for a CKAN API action."""
        return f"{self.base_url}/action/{action}"

    async def _request(self, action: str, params: Optional[dict] = None) -> dict:
        """Make a throttled GET request to the CKAN API with retry."""
        url = self._build_url(action)

        async def _do_request():
            async with self._throttle:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    return response.json()

        try:
            body = await retry_with_backoff(
                _do_request,
                max_retries=self.max_retries,
            )
        except httpx.ConnectError:
            raise CKANError(f"Cannot connect to CKAN portal at {self.base_url}")
        except httpx.TimeoutException:
            raise CKANError(f"CKAN request timed out after {self.timeout}s: {action}")
        except httpx.HTTPStatusError as e:
            raise CKANError(f"CKAN HTTP {e.response.status_code} on {action}: {e.response.text[:200]}")

        if not body.get("success"):
            error = body.get("error", {})
            raise CKANError(
                f"CKAN API error on {action}: {error.get('message', 'unknown error')}"
            )

        return body["result"]

    def _parse_dataset(self, raw: dict) -> CKANDataset:
        """Parse a raw CKAN package dict into a CKANDataset model."""
        resources = [
            CKANResource(
                id=r.get("id") or "",
                name=r.get("name") or "",
                description=r.get("description") or "",
                format=r.get("format") or "",
                url=r.get("url") or "",
                size=r.get("size"),
                last_modified=r.get("last_modified"),
            )
            for r in raw.get("resources", [])
        ]

        tags = [t["display_name"] for t in raw.get("tags", []) if "display_name" in t]

        org = None
        if raw.get("organization"):
            org = raw["organization"].get("title") or raw["organization"].get("name")

        return CKANDataset(
            id=raw["id"],
            name=raw.get("name", ""),
            title=raw.get("title", ""),
            notes=raw.get("notes", ""),
            tags=tags,
            organization=org,
            metadata_created=raw.get("metadata_created"),
            metadata_modified=raw.get("metadata_modified"),
            num_resources=raw.get("num_resources", len(resources)),
            resources=resources,
        )

    def _to_catalog_entry(self, dataset: CKANDataset) -> CKANCatalogEntry:
        """Convert a full dataset to a lightweight catalog entry."""
        return CKANCatalogEntry(
            dataset_id=dataset.id,
            name=dataset.name,
            title=dataset.title,
            description=dataset.notes,
            tags=dataset.tags,
            organization=dataset.organization,
            resource_formats=list({r.normalized_format for r in dataset.resources}),
            num_resources=dataset.num_resources,
            metadata_modified=dataset.metadata_modified,
        )

    # ------------------------------------------------------------------
    # Catalog discovery
    # ------------------------------------------------------------------

    async def search_datasets(
        self,
        query: str,
        rows: int = 20,
        start: int = 0,
    ) -> list[CKANDataset]:
        """
        Search datasets by keyword.

        Args:
            query: Search query string.
            rows: Maximum number of results to return.
            start: Offset for pagination.

        Returns:
            List of matching CKANDataset objects.
        """
        result = await self._request(
            "package_search",
            params={"q": query, "rows": rows, "start": start},
        )
        return [self._parse_dataset(pkg) for pkg in result.get("results", [])]

    async def list_datasets(self, limit: int = 100, offset: int = 0) -> list[str]:
        """
        List all dataset names/IDs from the portal.

        Args:
            limit: Maximum number of dataset names to return.
            offset: Offset for pagination.

        Returns:
            List of dataset name strings.
        """
        result = await self._request(
            "package_list",
            params={"limit": limit, "offset": offset},
        )
        return result  # Returns a list of dataset name strings

    async def get_dataset(self, dataset_id: str) -> CKANDataset:
        """
        Fetch full metadata for a single dataset.

        Args:
            dataset_id: The dataset ID or name.

        Returns:
            CKANDataset with full metadata and resources.
        """
        result = await self._request(
            "package_show",
            params={"id": dataset_id},
        )
        return self._parse_dataset(result)

    async def get_all_datasets_metadata(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[CKANDataset]:
        """
        Fetch metadata for all datasets (paginated).
        Useful for building the catalog index.

        Args:
            limit: Number of datasets per page.
            offset: Offset for pagination.

        Returns:
            List of CKANDataset objects with full metadata.
        """
        result = await self._request(
            "package_search",
            params={"q": "*:*", "rows": limit, "start": offset},
        )
        return [self._parse_dataset(pkg) for pkg in result.get("results", [])]

    async def build_catalog(self) -> list[CKANCatalogEntry]:
        """
        Build a complete catalog index by fetching all datasets.
        Handles pagination automatically.

        Returns:
            List of CKANCatalogEntry objects for index storage.
        """
        catalog = []
        offset = 0
        page_size = 100

        while True:
            datasets = await self.get_all_datasets_metadata(
                limit=page_size, offset=offset
            )
            if not datasets:
                break

            catalog.extend(self._to_catalog_entry(ds) for ds in datasets)
            logger.info(f"Fetched {len(catalog)} catalog entries so far...")

            if len(datasets) < page_size:
                break
            offset += page_size

        logger.info(f"Catalog complete: {len(catalog)} datasets indexed.")
        return catalog

    # ------------------------------------------------------------------
    # Resource download
    # ------------------------------------------------------------------

    async def download_resource(self, resource: CKANResource) -> bytes:
        """
        Download a resource file as raw bytes.

        Args:
            resource: The CKANResource to download.

        Returns:
            Raw file bytes.
        """
        if not resource.url:
            raise CKANError(f"Resource {resource.id} has no download URL")

        resource_url = resource.url

        async def _do_download():
            async with self._throttle:
                async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                    response = await client.get(resource_url)
                    response.raise_for_status()
                    return response.content

        try:
            return await retry_with_backoff(
                _do_download,
                max_retries=self.max_retries,
            )
        except httpx.ConnectError:
            raise CKANError(f"Cannot connect to download resource: {resource_url}")
        except httpx.TimeoutException:
            raise CKANError(f"Download timed out after {self.timeout}s: {resource_url}")
        except httpx.HTTPStatusError as e:
            raise CKANError(f"Download failed (HTTP {e.response.status_code}): {resource_url}")

    async def download_resource_as_dataframe(self, resource: CKANResource):
        """
        Download a resource and parse it into a pandas DataFrame.

        Args:
            resource: The CKANResource to download.

        Returns:
            pandas DataFrame with the resource data.

        Raises:
            CKANError: If the format is not supported.
        """
        import pandas as pd
        from io import BytesIO, StringIO

        raw = await self.download_resource(resource)
        fmt = resource.normalized_format

        try:
            if fmt == "csv":
                import csv as _csv
                try:
                    sample = raw[:8192].decode("utf-8", errors="replace")
                    dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t|")
                    sep = dialect.delimiter
                except _csv.Error:
                    sep = ","
                from data.ingestion.loader import _fix_european_decimals
                df = pd.read_csv(BytesIO(raw), sep=sep)
                return _fix_european_decimals(df)
            elif fmt == "json":
                return pd.read_json(BytesIO(raw))
            elif fmt == "geojson":
                import json
                geojson = json.loads(raw)
                features = geojson.get("features", [])
                if not features:
                    raise CKANError("GeoJSON contains no features")
                rows = []
                for f in features:
                    row = f.get("properties", {})
                    row["geometry"] = f.get("geometry")
                    rows.append(row)
                return pd.DataFrame(rows)
            elif fmt in ("xls", "xlsx"):
                return pd.read_excel(BytesIO(raw))
            else:
                raise CKANError(f"Unsupported resource format: {resource.format}")
        except CKANError:
            raise
        except Exception as e:
            raise CKANError(
                f"Failed to parse {fmt} resource '{resource.name}': {e}"
            )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CKANError(Exception):
    """Raised when a CKAN API call fails."""
    pass
