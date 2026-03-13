"""
Template-driven analytics engine.
Runs aggregation queries tailored to each template type using DuckDB.
Produces summary metrics that feed into narrative generation and visualization.
"""

import logging
from typing import Optional

import duckdb
import pandas as pd
from pydantic import BaseModel, Field

from data.profiling.template_definitions import TemplateType
from data.profiling.matcher import MatchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TimeSeriesMetrics(BaseModel):
    """Aggregated metrics for a time-series dataset."""
    time_column: str
    measure_columns: list[str]
    group_column: Optional[str] = None
    total_periods: int = 0
    date_range: dict = Field(default_factory=dict)  # {min, max, range_days}
    trend: dict = Field(default_factory=dict)  # per measure: {direction, pct_change, first, last}
    period_aggregations: list[dict] = Field(default_factory=list)  # aggregated rows
    summary_stats: dict = Field(default_factory=dict)  # per measure: {mean, min, max, total}


class CategoricalMetrics(BaseModel):
    """Aggregated metrics for a categorical comparison dataset."""
    category_column: str
    measure_columns: list[str]
    total_categories: int = 0
    rankings: dict = Field(default_factory=dict)  # per measure: [{category, value}, ...]
    comparisons: list[dict] = Field(default_factory=list)  # aggregated rows
    summary_stats: dict = Field(default_factory=dict)  # per measure: {mean, min, max, total, std}


class GeospatialMetrics(BaseModel):
    """Aggregated metrics for a geospatial dataset."""
    lat_column: Optional[str] = None
    lon_column: Optional[str] = None
    geometry_column: Optional[str] = None
    measure_column: Optional[str] = None
    category_column: Optional[str] = None
    total_points: int = 0
    bounding_box: dict = Field(default_factory=dict)  # {min_lat, max_lat, min_lon, max_lon}
    category_distribution: list[dict] = Field(default_factory=list)
    points: list[dict] = Field(default_factory=list)  # [{lat, lon, ...metadata}]


class AnalyticsResult(BaseModel):
    """Complete analytics result for a dataset."""
    dataset_id: str
    template_type: TemplateType
    matched_columns: dict[str, str]
    metrics: dict = Field(default_factory=dict)  # The actual metrics (type depends on template)
    aggregation_table: list[dict] = Field(
        default_factory=list,
        description="Flat table of aggregated data for visualization",
    )


# ---------------------------------------------------------------------------
# Analytics engine
# ---------------------------------------------------------------------------

class AnalyticsEngine:
    """
    Runs template-driven aggregations using DuckDB.
    Each template type has its own aggregation strategy.
    """

    def analyze(
        self,
        df: pd.DataFrame,
        match: MatchResult,
    ) -> AnalyticsResult:
        """
        Run analytics based on the matched template.

        Args:
            df: The source DataFrame.
            match: The template match result containing column mappings.

        Returns:
            AnalyticsResult with template-specific metrics.
        """
        if not match.best_match or not match.best_match.is_viable:
            raise ValueError(f"No viable template match for dataset '{match.dataset_id}'")

        template_type = match.best_match.template_id
        columns = match.best_match.matched_columns

        logger.info(
            f"Running {template_type.value} analytics on '{match.dataset_id}' "
            f"with columns: {columns}"
        )

        conn = duckdb.connect()
        conn.register("dataset", df)

        try:
            if template_type == TemplateType.TIME_SERIES:
                return self._analyze_time_series(conn, df, match.dataset_id, columns)
            elif template_type == TemplateType.CATEGORICAL:
                return self._analyze_categorical(conn, df, match.dataset_id, columns)
            elif template_type == TemplateType.GEOSPATIAL:
                return self._analyze_geospatial(conn, df, match.dataset_id, columns)
            else:
                raise ValueError(f"Unknown template type: {template_type}")
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Time series analytics
    # ------------------------------------------------------------------

    def _analyze_time_series(
        self,
        conn: duckdb.DuckDBPyConnection,
        df: pd.DataFrame,
        dataset_id: str,
        columns: dict[str, str],
    ) -> AnalyticsResult:
        time_col = columns["time_axis"]
        safe_time = f'"{time_col}"'

        # Find all numerical columns as measures
        measure_cols = self._find_measure_columns(df, columns)
        group_col = columns.get("series_group")

        # Date range
        date_range_row = conn.execute(f"""
            SELECT
                MIN({safe_time}) as min_date,
                MAX({safe_time}) as max_date,
                COUNT(DISTINCT {safe_time}) as total_periods
            FROM dataset
        """).fetchone()

        date_range = {
            "min": str(date_range_row[0]),
            "max": str(date_range_row[1]),
        }
        total_periods = date_range_row[2]

        # Summary stats per measure
        summary_stats = {}
        for mcol in measure_cols:
            safe_m = f'"{mcol}"'
            stats = conn.execute(f"""
                SELECT
                    AVG({safe_m}) as mean_val,
                    MIN({safe_m}) as min_val,
                    MAX({safe_m}) as max_val,
                    SUM({safe_m}) as total_val,
                    STDDEV({safe_m}) as std_val
                FROM dataset
                WHERE {safe_m} IS NOT NULL
            """).fetchone()
            summary_stats[mcol] = {
                "mean": round(float(stats[0]), 2) if stats[0] is not None else None,
                "min": float(stats[1]) if stats[1] is not None else None,
                "max": float(stats[2]) if stats[2] is not None else None,
                "total": float(stats[3]) if stats[3] is not None else None,
                "std": round(float(stats[4]), 2) if stats[4] is not None else None,
            }

        # Trend: compare first vs last period
        trend = {}
        for mcol in measure_cols:
            safe_m = f'"{mcol}"'
            first_last = conn.execute(f"""
                WITH ordered AS (
                    SELECT {safe_m},
                        ROW_NUMBER() OVER (ORDER BY {safe_time} ASC) as rn_asc,
                        ROW_NUMBER() OVER (ORDER BY {safe_time} DESC) as rn_desc
                    FROM dataset
                    WHERE {safe_m} IS NOT NULL
                )
                SELECT
                    (SELECT {safe_m} FROM ordered WHERE rn_asc = 1) as first_val,
                    (SELECT {safe_m} FROM ordered WHERE rn_desc = 1) as last_val
            """).fetchone()

            first_val = float(first_last[0]) if first_last[0] is not None else 0
            last_val = float(first_last[1]) if first_last[1] is not None else 0
            pct_change = round(((last_val - first_val) / first_val) * 100, 2) if first_val != 0 else 0

            trend[mcol] = {
                "first": first_val,
                "last": last_val,
                "pct_change": pct_change,
                "direction": "increasing" if last_val > first_val else "decreasing" if last_val < first_val else "stable",
            }

        # Build aggregation table
        select_cols = [safe_time] + [f'"{m}"' for m in measure_cols]
        if group_col:
            select_cols.append(f'"{group_col}"')

        agg_rows = conn.execute(f"""
            SELECT {', '.join(select_cols)}
            FROM dataset
            ORDER BY {safe_time}
        """).df().to_dict(orient="records")

        metrics = TimeSeriesMetrics(
            time_column=time_col,
            measure_columns=measure_cols,
            group_column=group_col,
            total_periods=total_periods,
            date_range=date_range,
            trend=trend,
            summary_stats=summary_stats,
        )

        return AnalyticsResult(
            dataset_id=dataset_id,
            template_type=TemplateType.TIME_SERIES,
            matched_columns=columns,
            metrics=metrics.model_dump(),
            aggregation_table=agg_rows,
        )

    # ------------------------------------------------------------------
    # Categorical analytics
    # ------------------------------------------------------------------

    def _analyze_categorical(
        self,
        conn: duckdb.DuckDBPyConnection,
        df: pd.DataFrame,
        dataset_id: str,
        columns: dict[str, str],
    ) -> AnalyticsResult:
        cat_col = columns["category"]
        safe_cat = f'"{cat_col}"'
        measure_cols = self._find_measure_columns(df, columns)

        # Total categories
        total_cats = conn.execute(f"""
            SELECT COUNT(DISTINCT {safe_cat}) FROM dataset
        """).fetchone()[0]

        # Summary stats and rankings per measure
        summary_stats = {}
        rankings = {}
        for mcol in measure_cols:
            safe_m = f'"{mcol}"'

            stats = conn.execute(f"""
                SELECT
                    AVG({safe_m}) as mean_val,
                    MIN({safe_m}) as min_val,
                    MAX({safe_m}) as max_val,
                    SUM({safe_m}) as total_val,
                    STDDEV({safe_m}) as std_val
                FROM dataset
                WHERE {safe_m} IS NOT NULL
            """).fetchone()
            summary_stats[mcol] = {
                "mean": round(float(stats[0]), 2) if stats[0] is not None else None,
                "min": float(stats[1]) if stats[1] is not None else None,
                "max": float(stats[2]) if stats[2] is not None else None,
                "total": float(stats[3]) if stats[3] is not None else None,
                "std": round(float(stats[4]), 2) if stats[4] is not None else None,
            }

            ranking = conn.execute(f"""
                SELECT {safe_cat} as category, {safe_m} as value
                FROM dataset
                WHERE {safe_m} IS NOT NULL
                ORDER BY {safe_m} DESC
            """).df().to_dict(orient="records")
            rankings[mcol] = ranking

        # Build aggregation table
        select_cols = [safe_cat] + [f'"{m}"' for m in measure_cols]
        agg_rows = conn.execute(f"""
            SELECT {', '.join(select_cols)}
            FROM dataset
            ORDER BY {safe_cat}
        """).df().to_dict(orient="records")

        metrics = CategoricalMetrics(
            category_column=cat_col,
            measure_columns=measure_cols,
            total_categories=total_cats,
            rankings=rankings,
            summary_stats=summary_stats,
        )

        return AnalyticsResult(
            dataset_id=dataset_id,
            template_type=TemplateType.CATEGORICAL,
            matched_columns=columns,
            metrics=metrics.model_dump(),
            aggregation_table=agg_rows,
        )

    # ------------------------------------------------------------------
    # Geospatial analytics
    # ------------------------------------------------------------------

    def _analyze_geospatial(
        self,
        conn: duckdb.DuckDBPyConnection,
        df: pd.DataFrame,
        dataset_id: str,
        columns: dict[str, str],
    ) -> AnalyticsResult:
        # Find lat/lon columns
        lat_col, lon_col, geom_col = self._find_geo_columns(df)
        measure_col = columns.get("measure")
        category_col = columns.get("category")

        bounding_box = {}
        if lat_col and lon_col:
            safe_lat = f'"{lat_col}"'
            safe_lon = f'"{lon_col}"'
            bb = conn.execute(f"""
                SELECT
                    MIN({safe_lat}) as min_lat, MAX({safe_lat}) as max_lat,
                    MIN({safe_lon}) as min_lon, MAX({safe_lon}) as max_lon
                FROM dataset
                WHERE {safe_lat} IS NOT NULL AND {safe_lon} IS NOT NULL
            """).fetchone()
            bounding_box = {
                "min_lat": float(bb[0]), "max_lat": float(bb[1]),
                "min_lon": float(bb[2]), "max_lon": float(bb[3]),
            }

        total_points = conn.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]

        # Category distribution
        category_distribution = []
        if category_col:
            safe_cc = f'"{category_col}"'
            dist = conn.execute(f"""
                SELECT {safe_cc} as category, COUNT(*) as count
                FROM dataset
                GROUP BY {safe_cc}
                ORDER BY count DESC
            """).df().to_dict(orient="records")
            category_distribution = dist

        # Build points table
        select_parts = []
        for col in df.columns:
            select_parts.append(f'"{col}"')

        points = conn.execute(f"""
            SELECT {', '.join(select_parts)}
            FROM dataset
        """).df().to_dict(orient="records")

        metrics = GeospatialMetrics(
            lat_column=lat_col,
            lon_column=lon_col,
            geometry_column=geom_col,
            measure_column=measure_col,
            category_column=category_col,
            total_points=total_points,
            bounding_box=bounding_box,
            category_distribution=category_distribution,
        )

        return AnalyticsResult(
            dataset_id=dataset_id,
            template_type=TemplateType.GEOSPATIAL,
            matched_columns=columns,
            metrics=metrics.model_dump(),
            aggregation_table=points,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_measure_columns(self, df: pd.DataFrame, columns: dict) -> list[str]:
        """Find all numerical columns suitable as measures."""
        from data.profiling.profiler import infer_semantic_type

        measures = []
        for col in df.columns:
            if infer_semantic_type(df[col], col) == "numerical":
                measures.append(col)
        return measures

    def _find_geo_columns(self, df: pd.DataFrame) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Find latitude, longitude, and geometry columns."""
        lat_col = None
        lon_col = None
        geom_col = None

        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in {"lat", "latitude", "y_coord"}):
                lat_col = col
            elif any(kw in col_lower for kw in {"lon", "lng", "longitude", "x_coord"}):
                lon_col = col
            elif any(kw in col_lower for kw in {"geom", "geometry", "wkt"}):
                geom_col = col

        return lat_col, lon_col, geom_col
