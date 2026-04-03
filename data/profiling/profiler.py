"""
Data profiling engine.
Computes column-level statistics, type inference, null rates,
and dataset-level summaries for ingested datasets.

Uses pandas for initial profiling and DuckDB for aggregate statistics
on larger datasets.
"""

import re
import logging
from typing import Optional
from datetime import datetime

import pandas as pd
import duckdb
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class NumericStats(BaseModel):
    """Statistics for a numerical column."""
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None


class CategoricalStats(BaseModel):
    """Statistics for a categorical column."""
    unique_count: int = 0
    top_values: list[dict] = Field(
        default_factory=list,
        description="Top values with counts, e.g. [{'value': 'A', 'count': 10}]",
    )
    is_unique: bool = False


class TemporalStats(BaseModel):
    """Statistics for a temporal column."""
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    date_range_days: Optional[int] = None
    inferred_frequency: Optional[str] = None  # daily, weekly, monthly, yearly


class GeospatialStats(BaseModel):
    """Statistics for a geospatial column."""
    min_lat: Optional[float] = None
    max_lat: Optional[float] = None
    min_lon: Optional[float] = None
    max_lon: Optional[float] = None
    has_geometry: bool = False


class ColumnProfile(BaseModel):
    """Complete profile for a single column."""
    name: str
    semantic_type: str  # numerical, categorical, temporal, geospatial
    pandas_dtype: str
    total_count: int
    null_count: int
    null_rate: float
    sample_values: list[str] = []

    # Type-specific stats (only one will be populated)
    numeric_stats: Optional[NumericStats] = None
    categorical_stats: Optional[CategoricalStats] = None
    temporal_stats: Optional[TemporalStats] = None
    geospatial_stats: Optional[GeospatialStats] = None

    # Keyword dictionary signal (populated during profiling)
    keyword_signal: Optional["ColumnSignalRef"] = None


class ColumnSignalRef(BaseModel):
    """Lightweight reference to keyword dictionary results stored on a column profile."""
    matched_canonicals: list[str] = []
    domain_signals: list[str] = []
    role_hints: list[str] = []


class DatasetProfile(BaseModel):
    """Complete profile for a dataset."""
    dataset_id: str
    source: str = ""
    row_count: int
    column_count: int
    total_null_rate: float
    columns: list[ColumnProfile]
    profiled_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def column_types_summary(self) -> dict[str, int]:
        """Count of columns by semantic type."""
        summary: dict[str, int] = {}
        for col in self.columns:
            summary[col.semantic_type] = summary.get(col.semantic_type, 0) + 1
        return summary

    @property
    def has_temporal(self) -> bool:
        return any(c.semantic_type == "temporal" for c in self.columns)

    @property
    def has_geospatial(self) -> bool:
        return any(c.semantic_type == "geospatial" for c in self.columns)

    @property
    def has_numerical(self) -> bool:
        return any(c.semantic_type == "numerical" for c in self.columns)

    @property
    def has_categorical(self) -> bool:
        return any(c.semantic_type == "categorical" for c in self.columns)


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

GEO_KEYWORDS = {
    "lat", "lon", "lng", "latitude", "longitude",
    "geom", "geometry", "wkt", "coord", "x_coord", "y_coord",
    "x", "y",
}

_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])")


def _column_name_words(col_name: str) -> set[str]:
    """
    Split a column name into a set of lowercase word tokens.

    Handles common separators (_, -, ., whitespace) and camelCase.
    E.g. "myLatitude_value" → {"my", "latitude", "value"}
         "flat_rate" → {"flat", "rate"}
    """
    # Split camelCase first
    expanded = _CAMEL_SPLIT.sub("_", col_name)
    # Split on _, -, ., whitespace
    tokens = re.split(r"[_\-.\s]+", expanded.lower())
    return {t for t in tokens if t}


# Keywords that suggest a column is an ID, not a measure
_ID_KEYWORDS = {"id", "index", "idx", "key", "code", "nr", "num", "number", "pk"}

# Keywords that override ID detection (these are real measures)
_MEASURE_KEYWORDS = {
    "amount", "count", "total", "value", "budget", "spending", "revenue",
    "rate", "score", "price", "cost", "salary", "income", "population",
    "quantity", "sum", "avg", "average", "median", "percent", "percentage",
}

# Keywords for year-like column names
_YEAR_KEYWORDS = {"year", "yr", "anno", "aar"}


def is_viable_measure(series: pd.Series, col_name: str, total_rows: int) -> bool:
    """
    Check whether a numerical column is a viable measure for charting.

    Returns False for ID columns, boolean/binary columns, constant columns,
    and columns with very high null rates.
    """
    clean = series.dropna()
    n_clean = len(clean)

    # High null rate (>80%)
    if total_rows > 0 and (total_rows - n_clean) / total_rows > 0.8:
        return False

    if n_clean == 0:
        return False

    nunique = clean.nunique()

    # Constant column (no variance)
    if nunique <= 1:
        return False

    # Boolean/binary (only 2 unique values that are 0/1 or True/False)
    if nunique == 2:
        unique_vals = set(clean.unique())
        if unique_vals <= {0, 1, True, False, 0.0, 1.0}:
            return False

    # ID detection
    words = _column_name_words(col_name)

    # If name contains measure-like words, it's a real measure
    if words & _MEASURE_KEYWORDS:
        return True

    # If name contains ID-like words, it's not a measure
    if words & _ID_KEYWORDS:
        return False

    # High cardinality with no repeats → likely an ID/index
    if total_rows > 10 and nunique / total_rows > 0.95:
        return False

    return True


def infer_semantic_type(series: pd.Series, col_name: str) -> str:
    """
    Infer the semantic type of a column.

    Returns one of: numerical, categorical, temporal, geospatial
    """
    col_lower = col_name.lower().strip()
    words = _column_name_words(col_lower)

    # Geospatial by name (word-boundary matching, not substring)
    if words & GEO_KEYWORDS:
        return "geospatial"

    # Check for geometry objects (dicts with 'type' and 'coordinates')
    if pd.api.types.is_object_dtype(series):
        sample = series.dropna().head(5)
        if len(sample) > 0:
            first = sample.iloc[0]
            if isinstance(first, dict) and "type" in first and "coordinates" in first:
                return "geospatial"

    # Already datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "temporal"

    # Numeric
    if pd.api.types.is_numeric_dtype(series):
        # Check if this looks like a year column (integers in 1900-2100)
        if pd.api.types.is_integer_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                vals = clean.astype(int)
                name_hints = bool(words & _YEAR_KEYWORDS)
                in_year_range = int(vals.min()) >= 1900 and int(vals.max()) <= 2100
                few_unique = clean.nunique() < 200
                if in_year_range and (name_hints or few_unique):
                    return "temporal"
        return "numerical"

    # String columns — check if temporal or actually numeric
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        if _looks_temporal(series, col_lower):
            return "temporal"
        if _looks_numeric(series):
            return "numerical"

    return "categorical"


def _looks_numeric(series: pd.Series, threshold: float = 0.8) -> bool:
    """Check if a string/object column actually contains numeric values.

    Many open-data portals store all columns as text even when the values
    are numbers. This function samples up to 50 non-null values and
    attempts ``pd.to_numeric`` coercion.  If at least *threshold* of the
    sample converts successfully, the column is treated as numeric.
    """
    sample = series.dropna().head(50)
    if len(sample) == 0:
        return False
    coerced = pd.to_numeric(sample, errors="coerce")
    success_rate = coerced.notna().sum() / len(sample)
    return success_rate >= threshold


def _looks_temporal(series: pd.Series, col_name: str) -> bool:
    """Check if a string column likely contains date/time values."""
    date_keywords = {
        "date", "time", "timestamp", "created", "modified",
        "updated", "year", "month", "day", "period",
    }
    if _column_name_words(col_name) & date_keywords:
        return True

    sample = series.dropna().head(30)
    if len(sample) == 0:
        return False

    try:
        parsed = pd.to_datetime(sample, format="mixed", dayfirst=True)
        success_rate = parsed.notna().sum() / len(sample)
        return success_rate > 0.8
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Column profiling
# ---------------------------------------------------------------------------

def _profile_numeric(series: pd.Series) -> NumericStats:
    """Compute statistics for a numerical column."""
    clean = series.dropna()
    if len(clean) == 0:
        return NumericStats()

    return NumericStats(
        min=float(clean.min()),
        max=float(clean.max()),
        mean=round(float(clean.mean()), 4),
        median=round(float(clean.median()), 4),
        std=round(float(clean.std()), 4) if len(clean) > 1 else None,
        q25=round(float(clean.quantile(0.25)), 4),
        q75=round(float(clean.quantile(0.75)), 4),
    )


def _profile_categorical(series: pd.Series) -> CategoricalStats:
    """Compute statistics for a categorical column."""
    clean = series.dropna()
    unique_count = clean.nunique()
    is_unique = unique_count == len(clean) and len(clean) > 0

    value_counts = clean.value_counts().head(10)
    top_values = [
        {"value": str(val), "count": int(count)}
        for val, count in value_counts.items()
    ]

    return CategoricalStats(
        unique_count=unique_count,
        top_values=top_values,
        is_unique=is_unique,
    )


def _profile_temporal(series: pd.Series) -> TemporalStats:
    """Compute statistics for a temporal column."""
    # Try to parse if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(series):
        try:
            series = pd.to_datetime(series, format="mixed", dayfirst=True)
        except (ValueError, TypeError):
            return TemporalStats()

    clean = series.dropna()
    if len(clean) == 0:
        return TemporalStats()

    min_date = clean.min()
    max_date = clean.max()
    range_days = (max_date - min_date).days

    # Infer frequency
    frequency = None
    if len(clean) > 2:
        try:
            freq = pd.infer_freq(clean.sort_values())
            if freq:
                freq_map = {"D": "daily", "W": "weekly", "MS": "monthly", "YS": "yearly", "M": "monthly", "Y": "yearly"}
                frequency = freq_map.get(freq, freq)
        except (ValueError, TypeError):
            # Estimate from median gap
            gaps = clean.sort_values().diff().dropna()
            if len(gaps) > 0:
                median_gap = gaps.median().days
                if median_gap <= 1:
                    frequency = "daily"
                elif median_gap <= 8:
                    frequency = "weekly"
                elif median_gap <= 35:
                    frequency = "monthly"
                elif median_gap <= 370:
                    frequency = "yearly"

    return TemporalStats(
        min_date=str(min_date.date()) if pd.notna(min_date) else None,
        max_date=str(max_date.date()) if pd.notna(max_date) else None,
        date_range_days=range_days,
        inferred_frequency=frequency,
    )


def _profile_geospatial(series: pd.Series, col_name: str) -> GeospatialStats:
    """Compute statistics for a geospatial column."""
    col_lower = col_name.lower()
    clean = series.dropna()

    # Check for geometry objects
    if len(clean) > 0 and isinstance(clean.iloc[0], dict):
        return GeospatialStats(has_geometry=True)

    # Numeric lat/lon
    if pd.api.types.is_numeric_dtype(series):
        words = _column_name_words(col_lower)
        is_lat = bool(words & {"lat", "latitude", "y_coord", "y"})
        is_lon = bool(words & {"lon", "lng", "longitude", "x_coord", "x"})

        vals = clean.astype(float)
        if is_lat:
            return GeospatialStats(min_lat=float(vals.min()), max_lat=float(vals.max()))
        elif is_lon:
            return GeospatialStats(min_lon=float(vals.min()), max_lon=float(vals.max()))

    return GeospatialStats()


def profile_column(series: pd.Series, col_name: str) -> ColumnProfile:
    """Profile a single column."""
    semantic_type = infer_semantic_type(series, col_name)
    total = len(series)
    nulls = int(series.isna().sum())
    null_rate = round(nulls / total, 4) if total > 0 else 0.0
    sample = [str(v) for v in series.dropna().head(5).tolist()]

    # Resolve keyword dictionary signal
    keyword_signal = None
    try:
        from data.profiling.keyword_dictionary import resolve_column
        signal = resolve_column(col_name)
        if signal.matched_canonicals:
            keyword_signal = ColumnSignalRef(
                matched_canonicals=signal.matched_canonicals,
                domain_signals=signal.domain_signals,
                role_hints=signal.role_hints,
            )
    except ImportError:
        pass

    profile = ColumnProfile(
        name=col_name,
        semantic_type=semantic_type,
        pandas_dtype=str(series.dtype),
        total_count=total,
        null_count=nulls,
        null_rate=null_rate,
        sample_values=sample,
        keyword_signal=keyword_signal,
    )

    if semantic_type == "numerical":
        profile.numeric_stats = _profile_numeric(series)
    elif semantic_type == "categorical":
        profile.categorical_stats = _profile_categorical(series)
    elif semantic_type == "temporal":
        profile.temporal_stats = _profile_temporal(series)
    elif semantic_type == "geospatial":
        profile.geospatial_stats = _profile_geospatial(series, col_name)

    return profile


# ---------------------------------------------------------------------------
# Dataset profiling
# ---------------------------------------------------------------------------

def coerce_numeric_text_columns(df: pd.DataFrame) -> list[str]:
    """Convert text columns that actually contain numbers to numeric dtype.

    Many open-data portals publish all columns as text.  This function
    inspects every object/string column and, if >=80 % of its non-null
    values are parseable as numbers, converts the column in-place using
    ``pd.to_numeric``.

    Returns the list of column names that were coerced.
    """
    coerced: list[str] = []
    for col in df.columns:
        if not (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ):
            continue
        if _looks_numeric(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            coerced.append(col)
            logger.info(f"Coerced text column '{col}' to numeric")
    return coerced


def profile_dataset(
    df: pd.DataFrame,
    dataset_id: str,
    source: str = "",
) -> DatasetProfile:
    """
    Profile an entire dataset.

    Args:
        df: The DataFrame to profile.
        dataset_id: Unique identifier for the dataset.
        source: Source URL or file path.

    Returns:
        DatasetProfile with column-level stats and dataset summary.
    """
    logger.info(f"Profiling dataset '{dataset_id}': {len(df)} rows x {len(df.columns)} columns")

    columns = [profile_column(df[col], col) for col in df.columns]

    total_cells = len(df) * len(df.columns)
    total_nulls = sum(c.null_count for c in columns)
    total_null_rate = round(total_nulls / total_cells, 4) if total_cells > 0 else 0.0

    profile = DatasetProfile(
        dataset_id=dataset_id,
        source=source,
        row_count=len(df),
        column_count=len(df.columns),
        total_null_rate=total_null_rate,
        columns=columns,
    )

    logger.info(
        f"Profile complete: {profile.column_types_summary}, "
        f"null rate: {profile.total_null_rate:.1%}"
    )
    return profile


def profile_dataset_duckdb(
    df: pd.DataFrame,
    dataset_id: str,
    source: str = "",
) -> DatasetProfile:
    """
    Profile a dataset using DuckDB for aggregate statistics.
    More efficient for large datasets.

    Args:
        df: The DataFrame to profile.
        dataset_id: Unique identifier.
        source: Source URL or file path.

    Returns:
        DatasetProfile.
    """
    logger.info(f"Profiling dataset '{dataset_id}' with DuckDB: {len(df)} rows")

    conn = duckdb.connect()
    conn.register("dataset", df)

    columns = []
    for col in df.columns:
        semantic_type = infer_semantic_type(df[col], col)
        safe_col = f'"{col}"'

        # Get null stats from DuckDB
        stats = conn.execute(f"""
            SELECT
                COUNT(*) as total,
                COUNT({safe_col}) as non_null,
                COUNT(*) - COUNT({safe_col}) as null_count
            FROM dataset
        """).fetchone()

        total, non_null, null_count = stats
        null_rate = round(null_count / total, 4) if total > 0 else 0.0
        sample = [str(v) for v in df[col].dropna().head(5).tolist()]

        profile = ColumnProfile(
            name=col,
            semantic_type=semantic_type,
            pandas_dtype=str(df[col].dtype),
            total_count=total,
            null_count=null_count,
            null_rate=null_rate,
            sample_values=sample,
        )

        # Numeric stats via DuckDB
        if semantic_type == "numerical" and pd.api.types.is_numeric_dtype(df[col]):
            try:
                num_stats = conn.execute(f"""
                    SELECT
                        MIN({safe_col}) as min_val,
                        MAX({safe_col}) as max_val,
                        AVG({safe_col}) as mean_val,
                        MEDIAN({safe_col}) as median_val,
                        STDDEV({safe_col}) as std_val,
                        QUANTILE_CONT({safe_col}, 0.25) as q25,
                        QUANTILE_CONT({safe_col}, 0.75) as q75
                    FROM dataset
                    WHERE {safe_col} IS NOT NULL
                """).fetchone()

                profile.numeric_stats = NumericStats(
                    min=float(num_stats[0]) if num_stats[0] is not None else None,
                    max=float(num_stats[1]) if num_stats[1] is not None else None,
                    mean=round(float(num_stats[2]), 4) if num_stats[2] is not None else None,
                    median=round(float(num_stats[3]), 4) if num_stats[3] is not None else None,
                    std=round(float(num_stats[4]), 4) if num_stats[4] is not None else None,
                    q25=round(float(num_stats[5]), 4) if num_stats[5] is not None else None,
                    q75=round(float(num_stats[6]), 4) if num_stats[6] is not None else None,
                )
            except Exception as e:
                logger.warning(f"DuckDB numeric profiling failed for '{col}': {e}")
                profile.numeric_stats = _profile_numeric(df[col])
        elif semantic_type == "categorical":
            profile.categorical_stats = _profile_categorical(df[col])
        elif semantic_type == "temporal":
            profile.temporal_stats = _profile_temporal(df[col])
        elif semantic_type == "geospatial":
            profile.geospatial_stats = _profile_geospatial(df[col], col)

        columns.append(profile)

    conn.close()

    total_cells = len(df) * len(df.columns)
    total_nulls = sum(c.null_count for c in columns)
    total_null_rate = round(total_nulls / total_cells, 4) if total_cells > 0 else 0.0

    return DatasetProfile(
        dataset_id=dataset_id,
        source=source,
        row_count=len(df),
        column_count=len(df.columns),
        total_null_rate=total_null_rate,
        columns=columns,
    )
