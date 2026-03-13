"""
Schema validation using the frictionless framework.
Validates tabular dataset structure, detects schema inconsistencies,
and enforces data quality rules before further processing.
"""

import logging
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field
from frictionless import Resource, Schema, Field as FrictionlessField, Detector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ValidationError(BaseModel):
    """A single validation error found in the dataset."""
    row: Optional[int] = None
    column: Optional[str] = None
    error_type: str
    message: str


class ValidationReport(BaseModel):
    """Result of validating a dataset."""
    is_valid: bool
    total_errors: int
    error_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Count of errors by type",
    )
    errors: list[ValidationError] = Field(
        default_factory=list,
        description="First N errors (capped to avoid huge reports)",
    )
    schema_fields: list[dict] = Field(
        default_factory=list,
        description="Detected schema fields with name and type",
    )
    row_count: int = 0
    stats: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def validate_file(filepath: str, max_errors: int = 50) -> ValidationReport:
    """
    Validate a tabular file using the frictionless framework.

    Detects:
        - Type errors (e.g. string in a numeric column)
        - Missing required values
        - Schema inconsistencies
        - Duplicate rows
        - Formatting issues

    Args:
        filepath: Path to the file (CSV, Excel, JSON).
        max_errors: Maximum number of individual errors to include.

    Returns:
        ValidationReport with errors and detected schema.
    """
    logger.info(f"Validating file: {filepath}")

    resource = Resource(filepath)
    report = resource.validate()

    # Extract schema
    schema_fields = []
    if resource.schema and resource.schema.fields:
        schema_fields = [
            {"name": f.name, "type": f.type}
            for f in resource.schema.fields
        ]

    # Parse errors
    errors = []
    error_summary: dict[str, int] = {}

    for task_report in report.tasks:
        for error in task_report.errors:
            error_type = error.type if hasattr(error, "type") else "unknown"
            error_summary[error_type] = error_summary.get(error_type, 0) + 1

            if len(errors) < max_errors:
                row_num = None
                col_name = None

                if hasattr(error, "row_number"):
                    row_num = error.row_number
                if hasattr(error, "field_name"):
                    col_name = error.field_name

                errors.append(ValidationError(
                    row=row_num,
                    column=col_name,
                    error_type=error_type,
                    message=str(error.message) if hasattr(error, "message") else str(error),
                ))

    total_errors = sum(error_summary.values())

    result = ValidationReport(
        is_valid=report.valid,
        total_errors=total_errors,
        error_summary=error_summary,
        errors=errors,
        schema_fields=schema_fields,
        row_count=report.stats.get("tasks", [{}])[0].get("rows", 0) if report.stats.get("tasks") else 0,
        stats=report.stats if isinstance(report.stats, dict) else {},
    )

    logger.info(
        f"Validation complete: {'VALID' if result.is_valid else 'INVALID'}, "
        f"{total_errors} errors"
    )
    return result


def validate_dataframe(
    df: pd.DataFrame,
    expected_schema: Optional[dict] = None,
    max_errors: int = 50,
) -> ValidationReport:
    """
    Validate a pandas DataFrame using frictionless.

    Args:
        df: DataFrame to validate.
        expected_schema: Optional schema dict to validate against.
            Format: {"fields": [{"name": "col", "type": "integer"}, ...]}
        max_errors: Maximum errors to report.

    Returns:
        ValidationReport.
    """
    logger.info(f"Validating DataFrame: {len(df)} rows x {len(df.columns)} columns")

    # Build resource from DataFrame
    resource = Resource(df)

    # Apply expected schema if provided
    if expected_schema:
        fields = [
            FrictionlessField.from_descriptor(f)
            for f in expected_schema.get("fields", [])
        ]
        resource.schema = Schema(fields=fields)

    report = resource.validate()

    # Extract detected schema
    schema_fields = []
    if resource.schema and resource.schema.fields:
        schema_fields = [
            {"name": f.name, "type": f.type}
            for f in resource.schema.fields
        ]

    # Parse errors
    errors = []
    error_summary: dict[str, int] = {}

    for task_report in report.tasks:
        for error in task_report.errors:
            error_type = error.type if hasattr(error, "type") else "unknown"
            error_summary[error_type] = error_summary.get(error_type, 0) + 1

            if len(errors) < max_errors:
                row_num = getattr(error, "row_number", None)
                col_name = getattr(error, "field_name", None)

                errors.append(ValidationError(
                    row=row_num,
                    column=col_name,
                    error_type=error_type,
                    message=str(error.message) if hasattr(error, "message") else str(error),
                ))

    total_errors = sum(error_summary.values())

    return ValidationReport(
        is_valid=report.valid,
        total_errors=total_errors,
        error_summary=error_summary,
        errors=errors,
        schema_fields=schema_fields,
        row_count=len(df),
    )


def detect_schema(filepath: str) -> list[dict]:
    """
    Detect the schema of a tabular file without full validation.
    Faster than validate_file when you only need the schema.

    Args:
        filepath: Path to the file.

    Returns:
        List of field descriptors [{"name": "...", "type": "..."}, ...]
    """
    resource = Resource(filepath)
    resource.infer()

    if resource.schema and resource.schema.fields:
        return [
            {"name": f.name, "type": f.type}
            for f in resource.schema.fields
        ]
    return []


def detect_schema_from_dataframe(df: pd.DataFrame) -> list[dict]:
    """
    Detect the schema of a DataFrame.

    Args:
        df: DataFrame to analyze.

    Returns:
        List of field descriptors.
    """
    resource = Resource(df)
    resource.infer()

    if resource.schema and resource.schema.fields:
        return [
            {"name": f.name, "type": f.type}
            for f in resource.schema.fields
        ]
    return []
