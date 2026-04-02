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

from data.profiling.template_definitions import TemplateType, get_parent_archetype
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
            # Determine the archetype to use for base analytics
            parent = get_parent_archetype(template_type)
            base_type = parent if parent else template_type

            # Map domain-specific roles to archetype roles for base analytics
            archetype_columns = self._map_to_archetype_roles(columns, base_type)

            # Run parent/archetype analytics
            if base_type == TemplateType.TIME_SERIES:
                result = self._analyze_time_series(conn, df, match.dataset_id, archetype_columns)
            elif base_type == TemplateType.CATEGORICAL:
                result = self._analyze_categorical(conn, df, match.dataset_id, archetype_columns)
            elif base_type == TemplateType.GEOSPATIAL:
                result = self._analyze_geospatial(conn, df, match.dataset_id, archetype_columns)
            else:
                raise ValueError(f"Unknown base template type: {base_type}")

            # Override template_type to the actual matched type (not the parent)
            result.template_type = template_type

            # Apply domain-specific extensions
            domain_extensions = {
                TemplateType.BUDGET: self._extend_budget,
                TemplateType.ENVIRONMENTAL: self._extend_environmental,
                TemplateType.TRANSPORT: self._extend_transport,
                TemplateType.DEMOGRAPHIC: self._extend_demographic,
                TemplateType.FACILITY: self._extend_facility,
                TemplateType.INCIDENT: self._extend_incident,
                TemplateType.HOUSING: self._extend_housing,
            }

            extend_fn = domain_extensions.get(template_type)
            if extend_fn:
                result = extend_fn(conn, df, columns, result)
                logger.info(f"Applied {template_type.value} domain extensions")

            return result
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Role mapping for domain → archetype
    # ------------------------------------------------------------------

    # Domain roles that map to archetype roles for base analytics
    _ROLE_ALIASES: dict[str, str] = {
        # Time-series archetype roles
        "financial_measure": "measure",
        "env_measure": "measure",
        "traffic_measure": "measure",
        "date": "time_axis",
        "event_time": "time_axis",
        # Categorical archetype roles
        "area": "category",
        "population_measure": "measure",
        "facility_type": "category",
        "event_type": "category",
        "type": "category",
        "value": "measure",
        # Geospatial archetype (location is already standard)
        # Grouping roles
        "department": "series_group",
        "budget_category": "series_group",
        "station": "series_group",
        "route": "series_group",
        "direction": "series_group",
        "mode": "series_group",
        "demographic_group": "series_group",
    }

    def _map_to_archetype_roles(
        self, columns: dict[str, str], archetype: TemplateType
    ) -> dict[str, str]:
        """
        Map domain-specific column roles to the archetype's expected roles.
        Preserves original roles alongside the mapped ones so domain extensions
        can still find them.
        """
        mapped = dict(columns)
        for domain_role, col_name in columns.items():
            archetype_role = self._ROLE_ALIASES.get(domain_role)
            if archetype_role and archetype_role not in mapped:
                mapped[archetype_role] = col_name
        return mapped

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

        # Trend: first-vs-last, peak detection, and shape analysis
        trend = {}
        for mcol in measure_cols:
            safe_m = f'"{mcol}"'

            # When data is grouped, aggregate per period so we compare
            # like-for-like totals instead of random group rows.
            if group_col:
                safe_group = f'"{group_col}"'
                agg_labels = ", ".join(
                    f"'{lbl}'" for lbl in self._AGGREGATE_LABELS
                )
                ts_rows = conn.execute(f"""
                    SELECT {safe_time} as period, SUM({safe_m}) as val
                    FROM dataset
                    WHERE {safe_m} IS NOT NULL
                      AND LOWER(TRIM({safe_group})) NOT IN ({agg_labels})
                    GROUP BY {safe_time}
                    ORDER BY {safe_time}
                """).fetchall()
            else:
                ts_rows = conn.execute(f"""
                    SELECT {safe_time} as period, {safe_m} as val
                    FROM dataset
                    WHERE {safe_m} IS NOT NULL
                    ORDER BY {safe_time}
                """).fetchall()

            if not ts_rows:
                continue

            first_val = float(ts_rows[0][1])
            last_val = float(ts_rows[-1][1])
            pct_change = round(((last_val - first_val) / first_val) * 100, 2) if first_val != 0 else 0

            # Peak and trough detection
            values = [float(r[1]) for r in ts_rows]
            periods = [str(r[0]) for r in ts_rows]
            peak_val = max(values)
            peak_idx = values.index(peak_val)
            trough_val = min(values)
            trough_idx = values.index(trough_val)

            trend_info = {
                "first": first_val,
                "last": last_val,
                "pct_change": pct_change,
                "direction": "increasing" if last_val > first_val else "decreasing" if last_val < first_val else "stable",
                "peak": peak_val,
                "peak_period": periods[peak_idx],
                "trough": trough_val,
                "trough_period": periods[trough_idx],
            }

            # Detect shape: is it rise-then-fall, steady, etc.?
            # If peak is not at the end and the drop from peak is significant
            if peak_idx < len(values) - 1 and peak_val > 0:
                drop_from_peak = peak_val - last_val
                drop_pct = round((drop_from_peak / peak_val) * 100, 1)
                if drop_pct > 15:
                    trend_info["shape"] = "rise_then_fall"
                    trend_info["drop_from_peak_pct"] = drop_pct
                    trend_info["drop_from_peak"] = round(drop_from_peak, 2)

            # If trough is not at the start and the recovery is significant
            if trough_idx > 0 and trough_idx < len(values) - 1:
                recovery = last_val - trough_val
                if trough_val > 0:
                    recovery_pct = round((recovery / trough_val) * 100, 1)
                    if recovery_pct > 15:
                        trend_info["recovery_from_trough_pct"] = recovery_pct

            trend[mcol] = trend_info

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
        # Use matcher-provided columns first, fall back to name scan
        lat_col = columns.get("latitude")
        lon_col = columns.get("longitude")
        geom_col = columns.get("geometry")
        if not lat_col and not lon_col and not geom_col:
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
    # Domain extension methods
    # ------------------------------------------------------------------

    def _extend_budget(
        self,
        conn: duckdb.DuckDBPyConnection,
        df: pd.DataFrame,
        columns: dict[str, str],
        result: AnalyticsResult,
    ) -> AnalyticsResult:
        """Add budget-specific metrics: YoY change, department share, cumulative trajectory."""
        time_col = columns.get("time_axis", columns.get("date", ""))
        measure_cols = result.metrics.get("measure_columns", [])
        dept_col = columns.get("department") or columns.get("budget_category")
        extensions: dict = {}

        if measure_cols and time_col:
            mcol = measure_cols[0]
            safe_m = f'"{mcol}"'
            safe_t = f'"{time_col}"'

            # Year-over-year change via LAG window
            try:
                yoy = conn.execute(f"""
                    SELECT {safe_t} as period,
                           {safe_m} as current_val,
                           LAG({safe_m}) OVER (ORDER BY {safe_t}) as prev_val,
                           CASE WHEN LAG({safe_m}) OVER (ORDER BY {safe_t}) != 0
                                THEN ROUND(({safe_m} - LAG({safe_m}) OVER (ORDER BY {safe_t}))
                                     / ABS(LAG({safe_m}) OVER (ORDER BY {safe_t})) * 100, 2)
                                ELSE NULL END as yoy_pct
                    FROM dataset
                    ORDER BY {safe_t}
                """).df().to_dict(orient="records")
                extensions["year_over_year_change"] = yoy
            except Exception as e:
                logger.warning(f"Budget YoY extension failed: {e}")

            # Cumulative trajectory (running sum)
            try:
                cumul = conn.execute(f"""
                    SELECT {safe_t} as period,
                           {safe_m} as value,
                           SUM({safe_m}) OVER (ORDER BY {safe_t}) as cumulative
                    FROM dataset
                    ORDER BY {safe_t}
                """).df().to_dict(orient="records")
                extensions["cumulative_trajectory"] = cumul
            except Exception as e:
                logger.warning(f"Budget cumulative extension failed: {e}")

        # Department share (% of total)
        if dept_col and measure_cols:
            mcol = measure_cols[0]
            safe_m = f'"{mcol}"'
            safe_d = f'"{dept_col}"'
            try:
                shares = conn.execute(f"""
                    SELECT {safe_d} as department,
                           SUM({safe_m}) as total,
                           ROUND(SUM({safe_m}) * 100.0 / SUM(SUM({safe_m})) OVER (), 2) as pct_share
                    FROM dataset
                    WHERE {safe_m} IS NOT NULL
                    GROUP BY {safe_d}
                    ORDER BY total DESC
                """).df().to_dict(orient="records")
                extensions["department_share"] = shares
            except Exception as e:
                logger.warning(f"Budget department share extension failed: {e}")

        # Budget vs actual (if secondary measure exists)
        secondary = columns.get("secondary_measure")
        if secondary and measure_cols:
            primary = measure_cols[0]
            safe_p = f'"{primary}"'
            safe_s = f'"{secondary}"'
            safe_t = f'"{time_col}"'
            try:
                variance = conn.execute(f"""
                    SELECT {safe_t} as period,
                           {safe_p} as budget,
                           {safe_s} as actual,
                           {safe_s} - {safe_p} as variance,
                           CASE WHEN {safe_p} != 0
                                THEN ROUND(({safe_s} - {safe_p}) / ABS({safe_p}) * 100, 2)
                                ELSE NULL END as variance_pct
                    FROM dataset
                    ORDER BY {safe_t}
                """).df().to_dict(orient="records")
                extensions["budget_vs_actual"] = variance
            except Exception as e:
                logger.warning(f"Budget vs actual extension failed: {e}")

        result.metrics["domain_extensions"] = extensions
        return result

    def _extend_environmental(
        self,
        conn: duckdb.DuckDBPyConnection,
        df: pd.DataFrame,
        columns: dict[str, str],
        result: AnalyticsResult,
    ) -> AnalyticsResult:
        """Add environmental metrics: exceedance analysis, daily/seasonal patterns."""
        measure_cols = result.metrics.get("measure_columns", [])
        time_col = columns.get("time_axis", "")
        station_col = columns.get("station")
        extensions: dict = {}

        # Exceedance analysis against WHO thresholds
        try:
            from data.profiling.keyword_dictionary import resolve_column, get_who_threshold
            for mcol in measure_cols:
                signal = resolve_column(mcol)
                for canonical in signal.matched_canonicals:
                    threshold = get_who_threshold(canonical)
                    if threshold:
                        # Use annual threshold if available, else 24h
                        limit = threshold.get("annual") or threshold.get("24h")
                        if limit is not None:
                            safe_m = f'"{mcol}"'
                            exc = conn.execute(f"""
                                SELECT
                                    COUNT(*) as total_readings,
                                    SUM(CASE WHEN {safe_m} > {limit} THEN 1 ELSE 0 END) as exceedances,
                                    ROUND(SUM(CASE WHEN {safe_m} > {limit} THEN 1 ELSE 0 END) * 100.0
                                          / COUNT(*), 2) as exceedance_pct
                                FROM dataset
                                WHERE {safe_m} IS NOT NULL
                            """).fetchone()
                            extensions[f"exceedance_{mcol}"] = {
                                "pollutant": canonical,
                                "who_threshold": limit,
                                "unit": threshold.get("unit", ""),
                                "total_readings": exc[0],
                                "exceedances": exc[1],
                                "exceedance_pct": float(exc[2]) if exc[2] else 0,
                            }
                        break  # One threshold per measure column
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Environmental exceedance extension failed: {e}")

        # Station comparison (if station column exists)
        if station_col and measure_cols:
            mcol = measure_cols[0]
            safe_m = f'"{mcol}"'
            safe_s = f'"{station_col}"'
            try:
                comparison = conn.execute(f"""
                    SELECT {safe_s} as station,
                           ROUND(AVG({safe_m}), 2) as avg_reading,
                           MIN({safe_m}) as min_reading,
                           MAX({safe_m}) as max_reading,
                           COUNT(*) as reading_count
                    FROM dataset
                    WHERE {safe_m} IS NOT NULL
                    GROUP BY {safe_s}
                    ORDER BY avg_reading DESC
                """).df().to_dict(orient="records")
                extensions["station_comparison"] = comparison
            except Exception as e:
                logger.warning(f"Environmental station comparison failed: {e}")

        result.metrics["domain_extensions"] = extensions
        return result

    def _extend_transport(
        self,
        conn: duckdb.DuckDBPyConnection,
        df: pd.DataFrame,
        columns: dict[str, str],
        result: AnalyticsResult,
    ) -> AnalyticsResult:
        """Add transport metrics: peak hours, weekday vs weekend, route ranking."""
        measure_cols = result.metrics.get("measure_columns", [])
        time_col = columns.get("time_axis", "")
        route_col = columns.get("route")
        direction_col = columns.get("direction")
        extensions: dict = {}

        if measure_cols and time_col:
            mcol = measure_cols[0]
            safe_m = f'"{mcol}"'
            safe_t = f'"{time_col}"'

            # Peak hour pattern (hourly average)
            try:
                hourly = conn.execute(f"""
                    SELECT EXTRACT(HOUR FROM CAST({safe_t} AS TIMESTAMP)) as hour,
                           ROUND(AVG({safe_m}), 2) as avg_value
                    FROM dataset
                    WHERE {safe_m} IS NOT NULL
                    GROUP BY hour
                    ORDER BY hour
                """).df().to_dict(orient="records")
                if hourly and any(r.get("hour") is not None for r in hourly):
                    extensions["peak_hour_pattern"] = hourly
            except Exception as e:
                logger.debug(f"Transport hourly pattern failed (may not be hourly data): {e}")

            # Weekday vs weekend comparison
            try:
                dow = conn.execute(f"""
                    SELECT
                        CASE WHEN EXTRACT(DOW FROM CAST({safe_t} AS TIMESTAMP)) IN (0, 6)
                             THEN 'weekend' ELSE 'weekday' END as day_type,
                        ROUND(AVG({safe_m}), 2) as avg_value,
                        COUNT(*) as observations
                    FROM dataset
                    WHERE {safe_m} IS NOT NULL
                    GROUP BY day_type
                """).df().to_dict(orient="records")
                if len(dow) == 2:
                    extensions["weekday_vs_weekend"] = dow
            except Exception as e:
                logger.debug(f"Transport weekday/weekend comparison failed: {e}")

        # Route ranking
        if route_col and measure_cols:
            mcol = measure_cols[0]
            safe_m = f'"{mcol}"'
            safe_r = f'"{route_col}"'
            try:
                ranking = conn.execute(f"""
                    SELECT {safe_r} as route,
                           ROUND(AVG({safe_m}), 2) as avg_value,
                           SUM({safe_m}) as total_value
                    FROM dataset
                    WHERE {safe_m} IS NOT NULL
                    GROUP BY {safe_r}
                    ORDER BY total_value DESC
                """).df().to_dict(orient="records")
                extensions["route_ranking"] = ranking
            except Exception as e:
                logger.warning(f"Transport route ranking failed: {e}")

        # Directional flow
        if direction_col and measure_cols:
            mcol = measure_cols[0]
            safe_m = f'"{mcol}"'
            safe_d = f'"{direction_col}"'
            try:
                directional = conn.execute(f"""
                    SELECT {safe_d} as direction,
                           ROUND(AVG({safe_m}), 2) as avg_value,
                           SUM({safe_m}) as total_value
                    FROM dataset
                    WHERE {safe_m} IS NOT NULL
                    GROUP BY {safe_d}
                    ORDER BY total_value DESC
                """).df().to_dict(orient="records")
                extensions["directional_flow"] = directional
            except Exception as e:
                logger.warning(f"Transport directional flow failed: {e}")

        result.metrics["domain_extensions"] = extensions
        return result

    def _extend_demographic(
        self,
        conn: duckdb.DuckDBPyConnection,
        df: pd.DataFrame,
        columns: dict[str, str],
        result: AnalyticsResult,
    ) -> AnalyticsResult:
        """Add demographic metrics: population share, group composition, growth rate."""
        measure_cols = result.metrics.get("measure_columns", [])
        area_col = columns.get("area", columns.get("category", ""))
        group_col = columns.get("demographic_group")
        year_col = columns.get("year")
        extensions: dict = {}

        if measure_cols and area_col:
            mcol = measure_cols[0]
            safe_m = f'"{mcol}"'
            safe_a = f'"{area_col}"'

            # Population share (% of total per area)
            try:
                shares = conn.execute(f"""
                    SELECT {safe_a} as area,
                           SUM({safe_m}) as total,
                           ROUND(SUM({safe_m}) * 100.0 / SUM(SUM({safe_m})) OVER (), 2) as pct_share
                    FROM dataset
                    WHERE {safe_m} IS NOT NULL
                    GROUP BY {safe_a}
                    ORDER BY total DESC
                """).df().to_dict(orient="records")
                extensions["population_share"] = shares
            except Exception as e:
                logger.warning(f"Demographic population share failed: {e}")

        # Group composition (pivot by demographic_group)
        if group_col and measure_cols and area_col:
            mcol = measure_cols[0]
            safe_m = f'"{mcol}"'
            safe_a = f'"{area_col}"'
            safe_g = f'"{group_col}"'
            try:
                composition = conn.execute(f"""
                    SELECT {safe_a} as area, {safe_g} as group_name,
                           SUM({safe_m}) as value
                    FROM dataset
                    WHERE {safe_m} IS NOT NULL
                    GROUP BY {safe_a}, {safe_g}
                    ORDER BY {safe_a}, value DESC
                """).df().to_dict(orient="records")
                extensions["group_composition"] = composition
            except Exception as e:
                logger.warning(f"Demographic group composition failed: {e}")

        # Growth rate (if year column exists)
        if year_col and measure_cols and area_col:
            mcol = measure_cols[0]
            safe_m = f'"{mcol}"'
            safe_a = f'"{area_col}"'
            safe_y = f'"{year_col}"'
            try:
                growth = conn.execute(f"""
                    WITH yearly AS (
                        SELECT {safe_a} as area, {safe_y} as year, SUM({safe_m}) as pop
                        FROM dataset
                        WHERE {safe_m} IS NOT NULL
                        GROUP BY {safe_a}, {safe_y}
                    )
                    SELECT area, year, pop,
                           LAG(pop) OVER (PARTITION BY area ORDER BY year) as prev_pop,
                           CASE WHEN LAG(pop) OVER (PARTITION BY area ORDER BY year) != 0
                                THEN ROUND((pop - LAG(pop) OVER (PARTITION BY area ORDER BY year))
                                     / ABS(LAG(pop) OVER (PARTITION BY area ORDER BY year)) * 100, 2)
                                ELSE NULL END as growth_pct
                    FROM yearly
                    ORDER BY area, year
                """).df().to_dict(orient="records")
                extensions["growth_rate"] = growth
            except Exception as e:
                logger.warning(f"Demographic growth rate failed: {e}")

        result.metrics["domain_extensions"] = extensions
        return result

    def _extend_facility(
        self,
        conn: duckdb.DuckDBPyConnection,
        df: pd.DataFrame,
        columns: dict[str, str],
        result: AnalyticsResult,
    ) -> AnalyticsResult:
        """Add facility metrics: type counts, district distribution, coverage ratio."""
        type_col = columns.get("facility_type", columns.get("category", ""))
        district_col = columns.get("district")
        capacity_col = columns.get("capacity")
        extensions: dict = {}

        if type_col:
            safe_t = f'"{type_col}"'
            # Type counts
            try:
                counts = conn.execute(f"""
                    SELECT {safe_t} as facility_type, COUNT(*) as count
                    FROM dataset
                    GROUP BY {safe_t}
                    ORDER BY count DESC
                """).df().to_dict(orient="records")
                extensions["type_counts"] = counts
            except Exception as e:
                logger.warning(f"Facility type counts failed: {e}")

        if district_col:
            safe_d = f'"{district_col}"'
            # District distribution
            try:
                dist = conn.execute(f"""
                    SELECT {safe_d} as district, COUNT(*) as count
                    FROM dataset
                    GROUP BY {safe_d}
                    ORDER BY count DESC
                """).df().to_dict(orient="records")
                extensions["district_distribution"] = dist
            except Exception as e:
                logger.warning(f"Facility district distribution failed: {e}")

            # Coverage ratio (distinct types per district)
            if type_col:
                safe_t = f'"{type_col}"'
                try:
                    coverage = conn.execute(f"""
                        SELECT {safe_d} as district,
                               COUNT(DISTINCT {safe_t}) as distinct_types,
                               COUNT(*) as total_facilities
                        FROM dataset
                        GROUP BY {safe_d}
                        ORDER BY distinct_types DESC
                    """).df().to_dict(orient="records")
                    extensions["coverage_ratio"] = coverage
                except Exception as e:
                    logger.warning(f"Facility coverage ratio failed: {e}")

        # Capacity by type
        if capacity_col and type_col:
            safe_c = f'"{capacity_col}"'
            safe_t = f'"{type_col}"'
            try:
                cap = conn.execute(f"""
                    SELECT {safe_t} as facility_type,
                           ROUND(AVG({safe_c}), 2) as avg_capacity,
                           SUM({safe_c}) as total_capacity
                    FROM dataset
                    WHERE {safe_c} IS NOT NULL
                    GROUP BY {safe_t}
                    ORDER BY total_capacity DESC
                """).df().to_dict(orient="records")
                extensions["capacity_by_type"] = cap
            except Exception as e:
                logger.warning(f"Facility capacity by type failed: {e}")

        result.metrics["domain_extensions"] = extensions
        return result

    def _extend_incident(
        self,
        conn: duckdb.DuckDBPyConnection,
        df: pd.DataFrame,
        columns: dict[str, str],
        result: AnalyticsResult,
    ) -> AnalyticsResult:
        """Add incident metrics: volume trend, type breakdown, temporal patterns."""
        time_col = columns.get("event_time", "")
        type_col = columns.get("event_type", "")
        status_col = columns.get("status")
        district_col = columns.get("district")
        extensions: dict = {}

        # Volume trend (monthly count)
        if time_col:
            safe_t = f'"{time_col}"'
            try:
                volume = conn.execute(f"""
                    SELECT DATE_TRUNC('month', CAST({safe_t} AS TIMESTAMP)) as month,
                           COUNT(*) as incident_count
                    FROM dataset
                    GROUP BY month
                    ORDER BY month
                """).df().to_dict(orient="records")
                extensions["volume_trend"] = volume
            except Exception as e:
                logger.debug(f"Incident volume trend failed: {e}")

            # Day-of-week pattern
            try:
                dow_pattern = conn.execute(f"""
                    SELECT EXTRACT(DOW FROM CAST({safe_t} AS TIMESTAMP)) as day_of_week,
                           COUNT(*) as count
                    FROM dataset
                    GROUP BY day_of_week
                    ORDER BY day_of_week
                """).df().to_dict(orient="records")
                extensions["temporal_patterns"] = dow_pattern
            except Exception as e:
                logger.debug(f"Incident temporal patterns failed: {e}")

        # Type breakdown
        if type_col:
            safe_ty = f'"{type_col}"'
            try:
                breakdown = conn.execute(f"""
                    SELECT {safe_ty} as event_type, COUNT(*) as count
                    FROM dataset
                    GROUP BY {safe_ty}
                    ORDER BY count DESC
                """).df().to_dict(orient="records")
                extensions["type_breakdown"] = breakdown
            except Exception as e:
                logger.warning(f"Incident type breakdown failed: {e}")

        # Hotspot areas (by district if available)
        if district_col:
            safe_d = f'"{district_col}"'
            try:
                hotspots = conn.execute(f"""
                    SELECT {safe_d} as area, COUNT(*) as count
                    FROM dataset
                    GROUP BY {safe_d}
                    ORDER BY count DESC
                """).df().to_dict(orient="records")
                extensions["hotspot_areas"] = hotspots
            except Exception as e:
                logger.warning(f"Incident hotspot areas failed: {e}")

        # Resolution stats (if status column exists)
        if status_col:
            safe_s = f'"{status_col}"'
            try:
                resolution = conn.execute(f"""
                    SELECT {safe_s} as status, COUNT(*) as count
                    FROM dataset
                    GROUP BY {safe_s}
                    ORDER BY count DESC
                """).df().to_dict(orient="records")
                extensions["resolution_stats"] = resolution
            except Exception as e:
                logger.warning(f"Incident resolution stats failed: {e}")

        result.metrics["domain_extensions"] = extensions
        return result

    def _extend_housing(
        self,
        conn: duckdb.DuckDBPyConnection,
        df: pd.DataFrame,
        columns: dict[str, str],
        result: AnalyticsResult,
    ) -> AnalyticsResult:
        """Add housing metrics: volume trend, type breakdown, avg value, district activity."""
        time_col = columns.get("date", columns.get("time_axis", ""))
        type_col = columns.get("type", "")
        value_col = columns.get("value", "")
        district_col = columns.get("district")
        extensions: dict = {}

        # Volume trend (monthly count)
        if time_col:
            safe_t = f'"{time_col}"'
            try:
                volume = conn.execute(f"""
                    SELECT DATE_TRUNC('month', CAST({safe_t} AS TIMESTAMP)) as month,
                           COUNT(*) as permit_count
                    FROM dataset
                    GROUP BY month
                    ORDER BY month
                """).df().to_dict(orient="records")
                extensions["volume_trend"] = volume
            except Exception as e:
                logger.debug(f"Housing volume trend failed: {e}")

        # Type breakdown
        if type_col:
            safe_ty = f'"{type_col}"'
            try:
                breakdown = conn.execute(f"""
                    SELECT {safe_ty} as permit_type, COUNT(*) as count
                    FROM dataset
                    GROUP BY {safe_ty}
                    ORDER BY count DESC
                """).df().to_dict(orient="records")
                extensions["type_breakdown"] = breakdown
            except Exception as e:
                logger.warning(f"Housing type breakdown failed: {e}")

        # Average value by type
        if type_col and value_col:
            safe_ty = f'"{type_col}"'
            safe_v = f'"{value_col}"'
            try:
                avg_val = conn.execute(f"""
                    SELECT {safe_ty} as permit_type,
                           ROUND(AVG({safe_v}), 2) as avg_value,
                           COUNT(*) as count
                    FROM dataset
                    WHERE {safe_v} IS NOT NULL
                    GROUP BY {safe_ty}
                    ORDER BY avg_value DESC
                """).df().to_dict(orient="records")
                extensions["avg_value_by_type"] = avg_val
            except Exception as e:
                logger.warning(f"Housing avg value by type failed: {e}")

        # District activity
        if district_col:
            safe_d = f'"{district_col}"'
            try:
                activity = conn.execute(f"""
                    SELECT {safe_d} as district, COUNT(*) as permits
                    FROM dataset
                    GROUP BY {safe_d}
                    ORDER BY permits DESC
                """).df().to_dict(orient="records")
                extensions["district_activity"] = activity
            except Exception as e:
                logger.warning(f"Housing district activity failed: {e}")

        # Seasonal pattern (monthly)
        if time_col:
            safe_t = f'"{time_col}"'
            try:
                seasonal = conn.execute(f"""
                    SELECT EXTRACT(MONTH FROM CAST({safe_t} AS TIMESTAMP)) as month,
                           COUNT(*) as count
                    FROM dataset
                    GROUP BY month
                    ORDER BY month
                """).df().to_dict(orient="records")
                extensions["seasonal_pattern"] = seasonal
            except Exception as e:
                logger.debug(f"Housing seasonal pattern failed: {e}")

        result.metrics["domain_extensions"] = extensions
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # Roles that represent measure columns in any template
    _MEASURE_ROLES = {
        "measure", "financial_measure", "env_measure", "traffic_measure",
        "population_measure", "value", "capacity", "severity",
        "secondary_measure",
    }

    # Labels used for aggregate/total rows that should be excluded when
    # computing per-period trends on grouped data.
    _AGGREGATE_LABELS = {
        "alls", "samtals", "total", "all", "heild", "sum",
        "allt", "samtala", "totalt", "yhteensä", "yhteensa",
    }

    def _find_measure_columns(self, df: pd.DataFrame, columns: dict) -> list[str]:
        """Find all numerical columns suitable as measures.

        Always includes columns the template matcher already mapped as
        measure roles (these passed structural matching, so trust them).
        Then discovers additional numerical columns, filtering out IDs,
        boolean/binary, constant, and high-null columns.
        """
        from data.profiling.profiler import infer_semantic_type, is_viable_measure

        # Start with matcher-confirmed measures (preserving order)
        confirmed = []
        for role, col_name in columns.items():
            if role in self._MEASURE_ROLES and col_name in df.columns:
                if col_name not in confirmed:
                    confirmed.append(col_name)

        # Discover additional measures from remaining columns
        total_rows = len(df)
        for col in df.columns:
            if col in confirmed:
                continue
            if infer_semantic_type(df[col], col) == "numerical":
                if is_viable_measure(df[col], col, total_rows):
                    confirmed.append(col)

        return confirmed

    def _find_geo_columns(self, df: pd.DataFrame) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Find latitude, longitude, and geometry columns."""
        from data.profiling.profiler import _column_name_words

        lat_col = None
        lon_col = None
        geom_col = None

        for col in df.columns:
            words = _column_name_words(col)
            if words & {"lat", "latitude", "y_coord"}:
                lat_col = col
            elif words & {"lon", "lng", "longitude", "x_coord"}:
                lon_col = col
            elif words & {"geom", "geometry", "wkt"}:
                geom_col = col

        return lat_col, lon_col, geom_col
