"""
Pydantic models for dataset-related request/response schemas.
"""

from pydantic import BaseModel
from typing import Optional
from enum import Enum


class ColumnType(str, Enum):
    """Supported column data types."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    GEOSPATIAL = "geospatial"


class ColumnSchema(BaseModel):
    """Schema for a single dataset column."""
    name: str
    dtype: ColumnType
    nullable: bool = True


class DatasetMetadata(BaseModel):
    """Metadata for a registered dataset."""
    id: str
    name: str
    source_url: Optional[str] = None
    description: Optional[str] = None
    row_count: Optional[int] = None
    columns: list[ColumnSchema] = []


class IngestRequest(BaseModel):
    """Request schema for dataset ingestion."""
    url: Optional[str] = None
    name: str
    description: Optional[str] = None
