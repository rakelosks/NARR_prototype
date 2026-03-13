"""
Fallback URL and file connector.
Handles direct dataset loading from URLs and local files
when data is not sourced through a CKAN portal.

Supports: CSV, JSON, GeoJSON, XLS, XLSX
"""

import os
import json
import logging
from typing import Optional
from enum import Enum

import httpx
import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class FileFormat(str, Enum):
    """Supported file formats for ingestion."""
    CSV = "csv"
    JSON = "json"
    GEOJSON = "geojson"
    XLS = "xls"
    XLSX = "xlsx"


class ColumnTypeInfo(BaseModel):
    """Inferred type information for a single column."""
    name: str
    dtype: str  # numerical, categorical, temporal, geospatial
    pandas_dtype: str  # original pandas dtype string
    nullable: bool = True
    sample_values: list[str] = []


class IngestResult(BaseModel):
    """Result of a dataset ingestion operation."""
    source: str  # URL or file path
    format: str
    row_count: int
    column_count: int
    columns: list[ColumnTypeInfo]


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

# Map of file extensions to FileFormat
EXTENSION_MAP = {
    ".csv": FileFormat.CSV,
    ".json": FileFormat.JSON,
    ".geojson": FileFormat.GEOJSON,
    ".xls": FileFormat.XLS,
    ".xlsx": FileFormat.XLSX,
}

# Common MIME types to FileFormat
MIME_MAP = {
    "text/csv": FileFormat.CSV,
    "application/csv": FileFormat.CSV,
    "application/json": FileFormat.JSON,
    "application/geo+json": FileFormat.GEOJSON,
    "application/vnd.ms-excel": FileFormat.XLS,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": FileFormat.XLSX,
}


def detect_format_from_path(path: str) -> Optional[FileFormat]:
    """Detect file format from the file extension."""
    ext = os.path.splitext(path.lower().split("?")[0])[1]  # strip query params
    return EXTENSION_MAP.get(ext)


def detect_format_from_content_type(content_type: str) -> Optional[FileFormat]:
    """Detect file format from an HTTP Content-Type header."""
    # Take just the MIME type, ignore charset etc.
    mime = content_type.split(";")[0].strip().lower()
    return MIME_MAP.get(mime)


def detect_format(
    path: str,
    content_type: Optional[str] = None,
    explicit_format: Optional[str] = None,
) -> FileFormat:
    """
    Detect file format using multiple signals.
    Priority: explicit > content-type > extension.

    Raises:
        ValueError: If format cannot be determined.
    """
    if explicit_format:
        try:
            return FileFormat(explicit_format.lower())
        except ValueError:
            pass

    if content_type:
        fmt = detect_format_from_content_type(content_type)
        if fmt:
            return fmt

    fmt = detect_format_from_path(path)
    if fmt:
        return fmt

    raise ValueError(
        f"Cannot determine format for '{path}'. "
        f"Provide an explicit format: {[f.value for f in FileFormat]}"
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_bytes(raw: bytes, fmt: FileFormat) -> pd.DataFrame:
    """
    Parse raw bytes into a pandas DataFrame based on format.

    Args:
        raw: Raw file bytes.
        fmt: The file format.

    Returns:
        Parsed DataFrame.
    """
    from io import BytesIO

    if fmt == FileFormat.CSV:
        return pd.read_csv(BytesIO(raw))

    elif fmt == FileFormat.JSON:
        # Try tabular JSON first, fall back to records
        try:
            return pd.read_json(BytesIO(raw))
        except ValueError:
            data = json.loads(raw)
            if isinstance(data, list):
                return pd.DataFrame(data)
            raise ValueError("JSON must be an array of objects or tabular format")

    elif fmt == FileFormat.GEOJSON:
        geojson = json.loads(raw)
        features = geojson.get("features", [])
        if not features:
            raise ValueError("GeoJSON contains no features")
        rows = []
        for f in features:
            row = f.get("properties", {})
            row["geometry"] = f.get("geometry")
            rows.append(row)
        return pd.DataFrame(rows)

    elif fmt in (FileFormat.XLS, FileFormat.XLSX):
        return pd.read_excel(BytesIO(raw))

    else:
        raise ValueError(f"Unsupported format: {fmt}")


# ---------------------------------------------------------------------------
# Column type inference
# ---------------------------------------------------------------------------

# Keywords that suggest a column contains geospatial data
GEO_KEYWORDS = {"lat", "lon", "lng", "latitude", "longitude", "geom", "geometry", "wkt", "coord"}


def infer_column_types(df: pd.DataFrame) -> list[ColumnTypeInfo]:
    """
    Infer semantic column types from a DataFrame.

    Types:
        - numerical: int or float columns
        - categorical: string/object columns with limited unique values
        - temporal: datetime columns or string columns that parse as dates
        - geospatial: columns with geo-related names or geometry data

    Returns:
        List of ColumnTypeInfo for each column.
    """
    results = []

    for col in df.columns:
        series = df[col]
        pandas_dtype = str(series.dtype)
        nullable = bool(series.isna().any())
        sample = [str(v) for v in series.dropna().head(3).tolist()]
        col_lower = col.lower().strip()

        # Check geospatial by column name
        if col_lower in GEO_KEYWORDS or any(kw in col_lower for kw in GEO_KEYWORDS):
            dtype = "geospatial"

        # Check if it's already datetime
        elif pd.api.types.is_datetime64_any_dtype(series):
            dtype = "temporal"

        # Check numeric
        elif pd.api.types.is_numeric_dtype(series):
            dtype = "numerical"

        # For object/string columns, try to detect temporal
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            dtype = _infer_string_column(series, col_lower)

        else:
            dtype = "categorical"

        results.append(ColumnTypeInfo(
            name=col,
            dtype=dtype,
            pandas_dtype=pandas_dtype,
            nullable=nullable,
            sample_values=sample,
        ))

    return results


def _infer_string_column(series: pd.Series, col_name: str) -> str:
    """Infer whether a string column is temporal or categorical."""
    # Try parsing as dates on a sample
    sample = series.dropna().head(50)
    if len(sample) == 0:
        return "categorical"

    try:
        parsed = pd.to_datetime(sample, infer_datetime_format=True)
        # If most values parsed successfully, it's temporal
        if parsed.notna().sum() / len(sample) > 0.8:
            return "temporal"
    except (ValueError, TypeError):
        pass

    # Date-like column names
    date_keywords = {"date", "time", "timestamp", "created", "modified", "year", "month", "day"}
    if any(kw in col_name for kw in date_keywords):
        return "temporal"

    return "categorical"


# ---------------------------------------------------------------------------
# Loader functions
# ---------------------------------------------------------------------------

async def load_from_url(
    url: str,
    format: Optional[str] = None,
    timeout: float = 60.0,
) -> tuple[pd.DataFrame, IngestResult]:
    """
    Load a dataset from a URL.

    Args:
        url: Direct URL to the data file.
        format: Explicit format override (csv, json, geojson, xls, xlsx).
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (DataFrame, IngestResult metadata).
    """
    logger.info(f"Downloading from URL: {url}")

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        raw = response.content

    fmt = detect_format(url, content_type=content_type, explicit_format=format)
    logger.info(f"Detected format: {fmt.value}")

    df = parse_bytes(raw, fmt)
    columns = infer_column_types(df)

    result = IngestResult(
        source=url,
        format=fmt.value,
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
    )

    logger.info(f"Loaded {result.row_count} rows x {result.column_count} columns from {url}")
    return df, result


def load_from_file(
    filepath: str,
    format: Optional[str] = None,
) -> tuple[pd.DataFrame, IngestResult]:
    """
    Load a dataset from a local file.

    Args:
        filepath: Path to the local file.
        format: Explicit format override.

    Returns:
        Tuple of (DataFrame, IngestResult metadata).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info(f"Loading from file: {filepath}")

    fmt = detect_format(filepath, explicit_format=format)

    with open(filepath, "rb") as f:
        raw = f.read()

    df = parse_bytes(raw, fmt)
    columns = infer_column_types(df)

    result = IngestResult(
        source=filepath,
        format=fmt.value,
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
    )

    logger.info(f"Loaded {result.row_count} rows x {result.column_count} columns from {filepath}")
    return df, result