"""
Visualization type selection and Vega-Lite specification generation.
Deterministic logic selects chart types based on the matched template,
then generates validated Vega-Lite specs for the client to render.
"""

import logging
import unicodedata
from typing import Optional

from pydantic import BaseModel, Field

from data.profiling.template_definitions import TemplateType, TEMPLATE_MAP

from datetime import datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column label helpers — translate raw column names to readable English
# ---------------------------------------------------------------------------

def _build_column_labels(columns: dict[str, str], metrics: dict | None = None) -> dict[str, str]:
    """
    Build a mapping from raw column names to English-readable labels.

    Uses the keyword dictionary's resolve_column() to find canonical English
    terms for non-English column names (e.g. "ar" → "Year", "fjoldi" → "Count").
    Falls back to title-casing the raw name.
    """
    try:
        from data.profiling.keyword_dictionary import resolve_column
    except ImportError:
        logger.warning("keyword_dictionary not available; using raw column names")
        return {}

    labels: dict[str, str] = {}
    # Collect all column names from both the role mapping and the measure list
    all_col_names = set(columns.values())
    if metrics:
        for mc in metrics.get("measure_columns", []):
            all_col_names.add(mc)

    for col_name in all_col_names:
        if not col_name:
            continue
        signal = resolve_column(col_name)
        if signal.matched_canonicals:
            # Use the first matched canonical, title-cased, underscores → spaces
            label = " ".join(c.replace("_", " ").title() for c in signal.matched_canonicals[:2])
            labels[col_name] = label
        else:
            # Fallback: title-case the raw name, replacing underscores
            labels[col_name] = col_name.replace("_", " ").title()

    return labels


def _label(col_name: str, metrics: dict) -> str:
    """Get the English label for a column, falling back to the raw name."""
    labels = metrics.get("column_labels", {})
    return labels.get(col_name, col_name)


def _raw_title_case(col_name: str) -> str:
    """Normalize a raw column name into its title-cased fallback form."""
    return (col_name or "").replace("_", " ").title()


def _is_fallback_label(col_name: str, label: str) -> bool:
    """Return True when a label is just the raw fallback (not translated)."""
    return (label or "").strip() == _raw_title_case(col_name).strip()


def _to_ascii(text: str) -> str:
    """ASCII-fold for robust keyword matching."""
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _english_label(col_name: str, metrics: dict, fallback: str) -> str:
    """
    Prefer translated English labels; otherwise use a stable English fallback.
    This keeps chart UI text consistent even when raw column names are local-language.
    """
    if not col_name:
        return fallback
    label = _label(col_name, metrics)
    if _is_fallback_label(col_name, label):
        raw = _to_ascii(col_name)
        if "year" in raw or raw == "ar":
            return "Year"
        if "date" in raw or "time" in raw or "manud" in raw or "month" in raw:
            return "Time"
        return fallback
    return label


# Mapping from domain-specific column roles to archetype roles.
# Spec generators only know about archetype roles (category, time_axis, etc.),
# so domain templates with roles like "area", "demographic_group", "facility_type"
# need to be resolved before chart generation.
_ROLE_TO_CATEGORY = {
    "area", "demographic_group", "facility_type", "incident_type",
    "route", "service_type", "pollutant", "species",
    "series_group",  # can also serve as the grouping dimension
}
_ROLE_TO_TIME_AXIS = {"year", "date", "incident_date"}
_ROLE_TO_MEASURE = {
    "population_measure", "financial_measure", "env_measure",
    "traffic_measure", "capacity", "severity", "value",
    "measure", "secondary_measure",
}


def _resolve_columns(columns: dict[str, str]) -> dict[str, str]:
    """
    Normalise domain-specific column roles into archetype roles that
    the spec generators understand.

    Returns a NEW dict with standard keys (category, time_axis, series_group)
    alongside any originals so that domain info is preserved.
    """
    resolved = dict(columns)  # start with a copy

    # Already has the standard keys? Nothing to do.
    if "category" in resolved or "time_axis" in resolved:
        return resolved

    # Resolve category
    if "category" not in resolved:
        for role in _ROLE_TO_CATEGORY:
            if role in columns and role != "series_group":
                resolved["category"] = columns[role]
                break
        # Fallback: use area as category
        if "category" not in resolved:
            for role in ("area", "demographic_group", "facility_type", "incident_type"):
                if role in columns:
                    resolved["category"] = columns[role]
                    break

    # Resolve time_axis
    if "time_axis" not in resolved:
        for role in _ROLE_TO_TIME_AXIS:
            if role in columns:
                resolved["time_axis"] = columns[role]
                break

    return resolved


def _is_year_only(data: list[dict], field: str) -> bool:
    """Check if a field contains bare year integers (e.g. 1999, 2012)."""
    for row in data[:20]:
        val = row.get(field)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            if not (1900 <= val <= 2100):
                return False
        else:
            return False
    return True


def _coerce_year_data(data: list[dict], field: str) -> list[dict]:
    """Convert year-integer values to ISO date strings for Vega-Lite temporal parsing."""
    return [
        {**row, field: f"{int(row[field])}-01-01"} if row.get(field) is not None else row
        for row in data
    ]


def _infer_time_unit(metrics: dict) -> str | None:
    """
    Infer the Vega-Lite timeUnit from date_range and total_periods.

    Returns a timeUnit string (e.g. 'yearmonth', 'yearmonthdate', 'year')
    or None if indeterminate.
    """
    date_range = metrics.get("date_range", {})
    total_periods = metrics.get("total_periods", 0)
    if not date_range or total_periods < 2:
        return None

    min_str = str(date_range.get("min", ""))
    max_str = str(date_range.get("max", ""))

    # Handle bare year values (e.g. "1999", "2012")
    try:
        min_year = int(float(min_str))
        max_year = int(float(max_str))
        if 1900 <= min_year <= 2100 and 1900 <= max_year <= 2100:
            return "year"
    except (ValueError, TypeError):
        pass

    try:
        min_dt = datetime.fromisoformat(min_str)
        max_dt = datetime.fromisoformat(max_str)
    except (ValueError, KeyError):
        return None

    span_days = (max_dt - min_dt).days
    if span_days == 0:
        return None

    avg_gap_days = span_days / (total_periods - 1)

    if avg_gap_days >= 300:        # ~yearly
        return "year"
    elif avg_gap_days >= 20:       # ~monthly
        return "yearmonth"
    elif avg_gap_days >= 5:        # ~weekly
        return "yearmonthdate"
    else:                          # daily or finer
        return "yearmonthdate"


# ---------------------------------------------------------------------------
# Scale analysis helpers
# ---------------------------------------------------------------------------

def _measures_have_different_scales(
    measures: list[str], stats: dict, threshold: float = 10.0
) -> bool:
    """
    Check whether multiple measures have vastly different magnitudes.

    Uses each column's max (fallback: mean) so totals (e.g. vehicle stock)
    vs rates or fatalities (single digits) split into separate charts.
    Means alone miss this when some stats are null or means round oddly.
    """
    if len(measures) < 2:
        return False

    magnitudes: list[float] = []
    for m in measures:
        m_stats = stats.get(m, {}) or {}
        raw = m_stats.get("max")
        if raw is None:
            raw = m_stats.get("mean")
        if raw is None:
            continue
        try:
            v = abs(float(raw))
        except (TypeError, ValueError):
            continue
        if v > 0:
            magnitudes.append(v)

    if len(magnitudes) < 2:
        return False

    return max(magnitudes) / min(magnitudes) > threshold


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class VegaLiteSpec(BaseModel):
    """A validated Vega-Lite specification."""
    schema_url: str = Field(
        default="https://vega.github.io/schema/vega-lite/v5.json",
        alias="$schema",
    )
    title: str = ""
    description: str = ""
    width: str | int = "container"
    height: int = 400
    data: dict = Field(default_factory=dict)
    mark: dict = Field(default_factory=dict)
    encoding: dict = Field(default_factory=dict)
    layer: Optional[list[dict]] = None
    selection: Optional[dict] = None
    config: dict = Field(default_factory=dict)

    class Config:
        populate_by_name = True

    def to_dict(self) -> dict:
        """Export as a clean dict for JSON serialization."""
        d = self.model_dump(by_alias=True, exclude_none=True)
        # Remove empty dicts
        return {k: v for k, v in d.items() if v}


class ChartEntry(BaseModel):
    """A single chart in a multi-chart selection."""
    chart_type: str
    title_suffix: str = ""          # appended to the base title, e.g. "by district"
    description: str = ""           # why this chart adds value
    spec_overrides: dict = Field(default_factory=dict)  # extra hints for spec gen


class ChartSelection(BaseModel):
    """Result of the chart type selection process."""
    charts: list[ChartEntry] = Field(default_factory=list)
    reason: str

    @property
    def primary_chart(self) -> str:
        return self.charts[0].chart_type if self.charts else "bar"

    @property
    def secondary_chart(self) -> Optional[str]:
        return self.charts[1].chart_type if len(self.charts) > 1 else None


# ---------------------------------------------------------------------------
# Chart type selection
# ---------------------------------------------------------------------------

def select_chart_type(
    template_type: TemplateType,
    metrics: dict,
) -> ChartSelection:
    """
    Select the best chart type based on the template and metrics.
    Uses deterministic rules — no LLM involved.

    Args:
        template_type: The matched template type.
        metrics: The analytics metrics dict.

    Returns:
        ChartSelection with primary and optional secondary chart type.
    """
    template = TEMPLATE_MAP.get(template_type)

    # Archetype selection
    if template_type == TemplateType.TIME_SERIES:
        return _select_time_series_chart(metrics)
    elif template_type == TemplateType.CATEGORICAL:
        return _select_categorical_chart(metrics)
    elif template_type == TemplateType.GEOSPATIAL:
        return _select_geospatial_chart(metrics)

    # Domain template selection
    elif template_type == TemplateType.BUDGET:
        return _select_budget_chart(metrics)
    elif template_type == TemplateType.ENVIRONMENTAL:
        return _select_environmental_chart(metrics)
    elif template_type == TemplateType.TRANSPORT:
        return _select_transport_chart(metrics)
    elif template_type == TemplateType.DEMOGRAPHIC:
        return _select_demographic_chart(metrics)
    elif template_type == TemplateType.FACILITY:
        return _select_facility_chart(metrics)
    elif template_type == TemplateType.INCIDENT:
        return _select_incident_chart(metrics)
    elif template_type == TemplateType.HOUSING:
        return _select_housing_chart(metrics)

    else:
        # Fallback to template's primary recommendation
        if template:
            primary = template.chart_recommendations[0]
            return ChartSelection(
                primary_chart=primary.chart_type,
                reason=primary.description,
            )
        return ChartSelection(primary_chart="bar", reason="Default fallback.")


def _select_time_series_chart(metrics: dict) -> ChartSelection:
    """Select chart type for time-series data."""
    group_col = metrics.get("group_column")
    measure_count = len(metrics.get("measure_columns", []))
    total_periods = metrics.get("total_periods", 0)
    charts: list[ChartEntry] = []

    if group_col and measure_count == 1:
        charts.append(ChartEntry(
            chart_type="line",
            description="Trend lines per group over time.",
        ))
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(latest comparison)",
            description="Bar chart comparing groups for the most recent period.",
            spec_overrides={"latest_period_only": True},
        ))
        if total_periods > 6:
            charts.append(ChartEntry(
                chart_type="area",
                title_suffix="(cumulative)",
                description="Stacked area showing combined total and each group's share.",
                spec_overrides={"stacked": True},
            ))
    elif measure_count > 1:
        # Check if measures have vastly different scales
        # (e.g. 14000 students vs 10.8 students-per-teacher)
        measures = metrics.get("measure_columns", [])
        stats = metrics.get("summary_stats", {})
        scales_differ = _measures_have_different_scales(measures, stats)

        if scales_differ:
            # Individual chart per measure — each gets its own y-axis scale
            # Try to use English labels for the title suffix
            try:
                from data.profiling.keyword_dictionary import resolve_column
                _resolve = lambda c: (
                    " ".join(s.title() for s in resolve_column(c).matched_canonicals[:2])
                    or c
                )
            except ImportError:
                _resolve = lambda c: c

            for m in measures[:5]:  # cap at 5 charts
                english_name = _resolve(m)
                mmax = (stats.get(m) or {}).get("max")
                try:
                    # Small annual counts (fatalities, etc.) read well as period bars.
                    prefer_bar = mmax is not None and float(mmax) <= 30.0
                except (TypeError, ValueError):
                    prefer_bar = False
                charts.append(ChartEntry(
                    chart_type="bar" if prefer_bar else "line",
                    title_suffix=f"— {english_name}",
                    description=(
                        f"{'Counts' if prefer_bar else 'Trend'} over time for {english_name}."
                    ),
                    spec_overrides={"single_measure": m},
                ))
            # Add a latest-year bar chart for the primary measure if grouped
            if group_col:
                primary_name = _resolve(measures[0])
                charts.append(ChartEntry(
                    chart_type="bar",
                    title_suffix=f"— {primary_name} (latest)",
                    description=f"Bar chart comparing groups for {primary_name} in the most recent period.",
                    spec_overrides={
                        "latest_period_only": True,
                        "single_measure": measures[0],
                    },
                ))
        else:
            # Measures are on similar scales — combined charts work well
            charts.append(ChartEntry(
                chart_type="line",
                description="Parallel trend lines for each measure.",
            ))
            charts.append(ChartEntry(
                chart_type="bar",
                title_suffix="(side-by-side)",
                description="Grouped bars comparing measures per period.",
            ))
            charts.append(ChartEntry(
                chart_type="area",
                title_suffix="(cumulative)",
                description="Stacked area showing combined volume over time.",
                spec_overrides={"stacked": True},
            ))
    else:
        if total_periods <= 12:
            charts.append(ChartEntry(
                chart_type="bar",
                description="Bar chart for period comparison.",
            ))
            charts.append(ChartEntry(
                chart_type="line",
                title_suffix="(trend)",
                description="Line chart showing the overall trend.",
            ))
        else:
            charts.append(ChartEntry(
                chart_type="line",
                description="Line chart showing trend over time.",
            ))
            charts.append(ChartEntry(
                chart_type="bar",
                title_suffix="(period comparison)",
                description="Bar chart for direct period comparison.",
            ))
        charts.append(ChartEntry(
            chart_type="area",
            title_suffix="(volume)",
            description="Area chart emphasizing cumulative volume.",
        ))

    return ChartSelection(charts=charts, reason="Time-series multi-view.")


def _select_categorical_chart(metrics: dict) -> ChartSelection:
    """Select chart type for categorical data."""
    total_cats = metrics.get("total_categories", 0)
    measure_count = len(metrics.get("measure_columns", []))
    charts: list[ChartEntry] = []

    # Primary bar chart (horizontal if many categories)
    charts.append(ChartEntry(
        chart_type="bar",
        description="Ranked bar chart comparing categories.",
        spec_overrides={"horizontal": True} if total_cats > 10 else {},
    ))

    if measure_count >= 2:
        charts.append(ChartEntry(
            chart_type="scatter",
            title_suffix="(correlation)",
            description="Scatter plot showing relationship between two measures.",
        ))
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(grouped)",
            description="Grouped bars comparing multiple measures side-by-side.",
            spec_overrides={"grouped": True},
        ))
    else:
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(top & bottom)",
            description="Focused bar chart highlighting extremes.",
            spec_overrides={"top_n": 5, "bottom_n": 5} if total_cats > 10 else {},
        ))

    # Heatmap if there are multiple measures to compare across categories
    if measure_count >= 2:
        charts.append(ChartEntry(
            chart_type="heatmap",
            title_suffix="(heatmap)",
            description="Heatmap comparing all measures across categories.",
        ))

    return ChartSelection(charts=charts, reason="Categorical multi-view.")


def _select_geospatial_chart(metrics: dict) -> ChartSelection:
    """Select chart type for geospatial data."""
    has_measure = metrics.get("measure_column") is not None
    has_category = metrics.get("category_column") is not None
    has_geometry = metrics.get("geometry_column") is not None
    charts: list[ChartEntry] = []

    # Map view — always first
    if has_geometry:
        charts.append(ChartEntry(
            chart_type="choropleth",
            description="Choropleth map using geometry boundaries.",
        ))
    elif has_measure:
        charts.append(ChartEntry(
            chart_type="bubble_map",
            description="Bubble map with size encoding measure values.",
        ))
    else:
        charts.append(ChartEntry(
            chart_type="map",
            description="Point map showing locations.",
        ))

    # Bar breakdown by category
    if has_category:
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(count by category)",
            description="Bar chart showing count per category.",
            spec_overrides={"aggregate": "count", "by": "category"},
        ))

    # If both measure and category, add a second map and a bar-by-measure
    if has_measure and has_category:
        charts.append(ChartEntry(
            chart_type="map",
            title_suffix="(by type)",
            description="Point map colored by category.",
        ))
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(measure by category)",
            description="Bar chart comparing measure totals per category.",
            spec_overrides={"aggregate": "sum", "by": "category"},
        ))
    elif has_measure:
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(ranked)",
            description="Bar chart ranking locations by measure.",
        ))

    # Ensure at least 3 charts
    if len(charts) < 3:
        charts.append(ChartEntry(
            chart_type="map",
            title_suffix="(overview)",
            description="Overview point map of all locations.",
        ))

    return ChartSelection(charts=charts, reason="Geospatial multi-view.")


# ---------------------------------------------------------------------------
# Domain template chart selection
# ---------------------------------------------------------------------------

def _select_budget_chart(metrics: dict) -> ChartSelection:
    """Select chart type for budget/financial data."""
    group_col = metrics.get("group_column")
    measure_count = len(metrics.get("measure_columns", []))
    extensions = metrics.get("domain_extensions", {})
    charts: list[ChartEntry] = []

    # 1. Trend line — always
    charts.append(ChartEntry(
        chart_type="line",
        description="Financial trend over time.",
    ))

    # 2. Composition — if department/category grouping
    if extensions.get("department_share") or group_col:
        charts.append(ChartEntry(
            chart_type="area",
            title_suffix="(composition)",
            description="Stacked area showing spending composition over time.",
            spec_overrides={"stacked": True},
        ))
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(by department)",
            description="Bar chart comparing department/category totals.",
        ))

    # 3. Budget vs actual — if two measures
    if measure_count > 1 or extensions.get("budget_vs_actual"):
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(budget vs actual)",
            description="Grouped bar comparing budget to actual spending.",
            spec_overrides={"grouped": True},
        ))

    # 4. Year-over-year change
    if extensions.get("year_over_year_change"):
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(year-over-year change)",
            description="Bar chart showing percentage change between periods.",
            spec_overrides={"yoy": True},
        ))

    # Ensure at least 3
    if len(charts) < 3:
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(period comparison)",
            description="Bar chart for direct period-to-period comparison.",
        ))
    if len(charts) < 3:
        charts.append(ChartEntry(
            chart_type="area",
            title_suffix="(volume)",
            description="Area chart emphasizing total financial volume.",
        ))

    return ChartSelection(charts=charts, reason="Budget multi-view.")


def _select_environmental_chart(metrics: dict) -> ChartSelection:
    """Select chart type for environmental data."""
    extensions = metrics.get("domain_extensions", {})
    station_col = metrics.get("group_column")
    has_exceedance = any(k.startswith("exceedance_") for k in extensions)
    charts: list[ChartEntry] = []

    # 1. Line with readings over time (always)
    charts.append(ChartEntry(
        chart_type="line",
        description="Environmental readings over time" + (
            " with WHO threshold reference." if has_exceedance else "."
        ),
        spec_overrides={"threshold_line": True} if has_exceedance else {},
    ))

    # 2. Bar — average by station or summary by period
    if station_col:
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(station comparison)",
            description="Bar chart comparing average readings across stations.",
        ))
    else:
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(period averages)",
            description="Bar chart of average readings per period.",
        ))

    # 3. Heatmap — station × time or measure × time
    if station_col:
        charts.append(ChartEntry(
            chart_type="heatmap",
            title_suffix="(station × time)",
            description="Heatmap showing reading intensity by station and time.",
        ))

    # 4. Area — seasonal / cumulative pattern
    charts.append(ChartEntry(
        chart_type="area",
        title_suffix="(seasonal pattern)",
        description="Area chart highlighting seasonal or cumulative patterns.",
    ))

    return ChartSelection(charts=charts, reason="Environmental multi-view.")


def _select_transport_chart(metrics: dict) -> ChartSelection:
    """Select chart type for transport / mobility / road-safety time series."""
    extensions = metrics.get("domain_extensions", {})
    route_col = metrics.get("group_column")

    # Without a route/line grouping column, use the same line/bar/area mix as
    # generic time series (multi-measure scale split, temporal bars, etc.).
    if not route_col:
        ts_sel = _select_time_series_chart(metrics)
        charts = list(ts_sel.charts)
        if extensions.get("peak_hour_pattern"):
            charts.append(ChartEntry(
                chart_type="heatmap",
                title_suffix="(hourly pattern)",
                description="Heatmap showing traffic intensity by hour and day.",
            ))
        while len(charts) < 3:
            charts.append(ChartEntry(
                chart_type="area",
                title_suffix="(volume)",
                description="Area chart emphasizing volume over time.",
                spec_overrides={"stacked": False},
            ))
        return ChartSelection(
            charts=charts[:8],
            reason="Transport — time-series views plus optional hourly heatmap.",
        )

    charts: list[ChartEntry] = []

    # 1. Line — traffic volume over time
    charts.append(ChartEntry(
        chart_type="line",
        description="Traffic volume trend over time.",
    ))

    # 2. Bar — route or mode ranking
    if extensions.get("route_ranking") or route_col:
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(route ranking)",
            description="Bar chart ranking routes or modes by total volume.",
            spec_overrides={"horizontal": True},
        ))
    else:
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(period comparison)",
            description="Bar chart comparing traffic across periods.",
        ))

    # 3. Area — stacked by route/mode if available
    charts.append(ChartEntry(
        chart_type="area",
        title_suffix="(composition by route)",
        description="Stacked area showing volume contribution by route.",
        spec_overrides={"stacked": True},
    ))

    # 4. Heatmap — peak hour pattern if available
    if extensions.get("peak_hour_pattern"):
        charts.append(ChartEntry(
            chart_type="heatmap",
            title_suffix="(hourly pattern)",
            description="Heatmap showing traffic intensity by hour and day.",
        ))

    if len(charts) < 3:
        charts.append(ChartEntry(
            chart_type="area",
            title_suffix="(volume)",
            description="Area chart emphasizing total traffic volume.",
        ))

    return ChartSelection(charts=charts, reason="Transport multi-view (routed).")


def _select_demographic_chart(metrics: dict) -> ChartSelection:
    """Select chart type for demographic data."""
    total_cats = metrics.get("total_categories", 0)
    measure_count = len(metrics.get("measure_columns", []))
    extensions = metrics.get("domain_extensions", {})
    charts: list[ChartEntry] = []

    # 1. Ranked bar — population by area
    charts.append(ChartEntry(
        chart_type="bar",
        description="Bar chart ranking areas by population.",
        spec_overrides={"horizontal": True} if total_cats > 10 else {},
    ))

    # 2. Stacked bar — demographic composition
    if extensions.get("group_composition"):
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(composition)",
            description="Stacked bar showing demographic group composition per area.",
            spec_overrides={"stacked": True},
        ))

    # 3. Scatter — if multiple measures (density vs count, etc.)
    if measure_count >= 2:
        charts.append(ChartEntry(
            chart_type="scatter",
            title_suffix="(correlation)",
            description="Scatter plot showing relationship between demographic measures.",
        ))

    # 4. Population share
    if extensions.get("population_share"):
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(share %)",
            description="Bar chart showing each area's percentage share of total population.",
            spec_overrides={"percentage": True},
        ))

    # Ensure at least 3
    if len(charts) < 3:
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(comparison)",
            description="Bar chart for direct area-to-area comparison.",
        ))
    if len(charts) < 3:
        charts.append(ChartEntry(
            chart_type="heatmap",
            title_suffix="(overview)",
            description="Heatmap of measures across areas.",
        ))

    return ChartSelection(charts=charts, reason="Demographic multi-view.")


def _select_facility_chart(metrics: dict) -> ChartSelection:
    """Select chart type for facility/infrastructure data."""
    has_measure = metrics.get("measure_column") is not None
    extensions = metrics.get("domain_extensions", {})
    charts: list[ChartEntry] = []

    # 1. Map — always
    if has_measure:
        charts.append(ChartEntry(
            chart_type="bubble_map",
            description="Bubble map of facilities, sized by capacity.",
        ))
    else:
        charts.append(ChartEntry(
            chart_type="map",
            description="Point map of facilities, colored by type.",
        ))

    # 2. Bar — count by type
    if extensions.get("type_counts"):
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(count by type)",
            description="Bar chart showing number of facilities per type.",
        ))

    # 3. Bar — count by district
    if extensions.get("district_distribution") or extensions.get("coverage_ratio"):
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(by district)",
            description="Bar chart showing facility distribution across districts.",
        ))

    # 4. Stacked bar — type × district
    if extensions.get("type_counts") and (extensions.get("district_distribution") or extensions.get("coverage_ratio")):
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(type × district)",
            description="Stacked bar showing facility types within each district.",
            spec_overrides={"stacked": True},
        ))

    # Ensure at least 3
    if len(charts) < 3:
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(summary)",
            description="Summary bar chart of facility counts.",
        ))
    if len(charts) < 3:
        charts.append(ChartEntry(
            chart_type="map",
            title_suffix="(overview)",
            description="Overview map showing all facility locations.",
        ))

    return ChartSelection(charts=charts, reason="Facility multi-view.")


def _select_incident_chart(metrics: dict) -> ChartSelection:
    """Select chart type for incident/event data."""
    extensions = metrics.get("domain_extensions", {})
    charts: list[ChartEntry] = []

    # 1. Map — incident locations colored by type
    charts.append(ChartEntry(
        chart_type="map",
        description="Point map of incident locations colored by type.",
    ))

    # 2. Line — volume trend over time
    if extensions.get("volume_trend"):
        charts.append(ChartEntry(
            chart_type="line",
            title_suffix="(volume over time)",
            description="Line chart showing incident volume trend.",
        ))

    # 3. Bar — breakdown by type
    if extensions.get("type_breakdown"):
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(by type)",
            description="Bar chart breaking down incidents by type.",
        ))

    # 4. Bar — hotspot areas
    if extensions.get("hotspot_areas"):
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(by area)",
            description="Bar chart showing incident concentration by area.",
            spec_overrides={"horizontal": True},
        ))

    # 5. Heatmap — temporal patterns
    if extensions.get("temporal_patterns"):
        charts.append(ChartEntry(
            chart_type="heatmap",
            title_suffix="(time patterns)",
            description="Heatmap showing when incidents occur (hour × day).",
        ))

    # Ensure at least 3
    if len(charts) < 3:
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(summary)",
            description="Summary bar chart of incident counts.",
        ))
    if len(charts) < 3:
        charts.append(ChartEntry(
            chart_type="line",
            title_suffix="(trend)",
            description="Trend line of incidents over time.",
        ))

    return ChartSelection(charts=charts, reason="Incident multi-view.")


def _select_housing_chart(metrics: dict) -> ChartSelection:
    """Select chart type for housing/permits data."""
    extensions = metrics.get("domain_extensions", {})
    charts: list[ChartEntry] = []

    # 1. Line — permit volume trend
    charts.append(ChartEntry(
        chart_type="line",
        description="Permit volume trend over time.",
    ))

    # 2. Bar — by type
    if extensions.get("type_breakdown"):
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(by type)",
            description="Bar chart breaking down permits by type.",
        ))

    # 3. Bar — by district
    if extensions.get("district_activity"):
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(by district)",
            description="Bar chart showing permit activity across districts.",
            spec_overrides={"horizontal": True},
        ))

    # 4. Area — stacked by type over time
    if extensions.get("type_breakdown") and extensions.get("volume_trend"):
        charts.append(ChartEntry(
            chart_type="area",
            title_suffix="(composition over time)",
            description="Stacked area showing permit type composition over time.",
            spec_overrides={"stacked": True},
        ))

    # 5. Average value by type
    if extensions.get("avg_value_by_type"):
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(avg value by type)",
            description="Bar chart comparing average permit value across types.",
        ))

    # Ensure at least 3
    if len(charts) < 3:
        charts.append(ChartEntry(
            chart_type="bar",
            title_suffix="(period comparison)",
            description="Bar chart for period-to-period comparison.",
        ))
    if len(charts) < 3:
        charts.append(ChartEntry(
            chart_type="area",
            title_suffix="(volume)",
            description="Area chart showing total permit volume.",
        ))

    return ChartSelection(charts=charts, reason="Housing multi-view.")


# ---------------------------------------------------------------------------
# Vega-Lite spec generation
# ---------------------------------------------------------------------------

# Vega-Lite → Vega compilation treats "." in `field` as nested property access.
# Icelandic CSV headers like "Fjöldi ökutækja 31. desember" or
# "Fjöldi slasaðra á 1.000 ökutæki" break into bogus sub-expressions; replace
# literal ASCII periods in chart field names only (display titles unchanged).
_VEGA_SAFE_DOT_SUBSTITUTE = "\u00b7"  # MIDDLE DOT


def vega_safe_field_name(name: str) -> str:
    if not name or "." not in name:
        return name
    return name.replace(".", _VEGA_SAFE_DOT_SUBSTITUTE)


def _dot_field_renames_from_rows(data: list[dict]) -> dict[str, str]:
    """Map original column names → Vega-safe names when they contain '.'."""
    if not data:
        return {}
    out: dict[str, str] = {}
    for k in data[0].keys():
        if k and "." in k:
            out[k] = vega_safe_field_name(k)
    return out


def _apply_vega_field_renames(
    data: list[dict],
    columns: dict[str, str],
    metrics: dict,
    rename: dict[str, str],
) -> tuple[list[dict], dict[str, str], dict]:
    if not rename:
        return data, columns, metrics

    def r(name: str) -> str:
        return rename.get(name, name)

    new_rows = [
        {r(k): v for k, v in row.items()}
        for row in data
    ]
    new_columns = {role: r(col) for role, col in columns.items()}
    new_metrics = dict(metrics)

    mc = new_metrics.get("measure_columns")
    if isinstance(mc, list):
        new_metrics["measure_columns"] = [r(m) for m in mc]

    for mk in (
        "time_column",
        "group_column",
        "category_column",
        "lat_column",
        "lon_column",
        "geometry_column",
        "measure_column",
    ):
        if mk in new_metrics and isinstance(new_metrics[mk], str):
            new_metrics[mk] = r(new_metrics[mk])

    cl = new_metrics.get("column_labels")
    if isinstance(cl, dict):
        new_metrics["column_labels"] = {r(k): v for k, v in cl.items()}

    ss = new_metrics.get("summary_stats")
    if isinstance(ss, dict):
        new_metrics["summary_stats"] = {r(k): v for k, v in ss.items()}

    tr = new_metrics.get("trend")
    if isinstance(tr, dict):
        new_metrics["trend"] = {r(k): v for k, v in tr.items()}

    so = new_metrics.get("_spec_overrides")
    if isinstance(so, dict):
        new_so = dict(so)
        sm = new_so.get("single_measure")
        if isinstance(sm, str):
            new_so["single_measure"] = r(sm)
        new_metrics["_spec_overrides"] = new_so

    return new_rows, new_columns, new_metrics


def generate_spec(
    chart_type: str,
    data: list[dict],
    columns: dict[str, str],
    metrics: dict,
    title: str = "",
) -> dict:
    """
    Generate a Vega-Lite specification.

    Args:
        chart_type: The chart type (line, bar, scatter, area, map, etc.).
        data: The aggregation table rows.
        columns: Matched column mapping {role: column_name}.
        metrics: The analytics metrics.
        title: Chart title.

    Returns:
        Vega-Lite spec as a dict.
    """
    generators = {
        "line": _gen_line_spec,
        "bar": _gen_bar_spec,
        "area": _gen_area_spec,
        "scatter": _gen_scatter_spec,
        "map": _gen_point_map_spec,
        "bubble_map": _gen_bubble_map_spec,
        "choropleth": _gen_bubble_map_spec,  # fallback until GeoJSON topology is supported
        "heatmap": _gen_heatmap_spec,
    }

    # Resolve domain-specific column roles to archetype roles
    columns = _resolve_columns(columns)

    metrics = {**metrics}
    # Inject column_labels so generators can use _label() (keys = raw column names)
    if "column_labels" not in metrics:
        metrics["column_labels"] = _build_column_labels(columns, metrics)

    rename = _dot_field_renames_from_rows(data)
    if rename:
        data, columns, metrics = _apply_vega_field_renames(data, columns, metrics, rename)

    generator = generators.get(chart_type, _gen_bar_spec)
    spec = generator(data, columns, metrics, title)

    logger.info(f"Generated {chart_type} Vega-Lite spec: '{title}'")
    return spec


_AGGREGATE_LABELS = {
    "alls", "samtals", "total", "all", "heild", "sum",
    "allt", "samtala", "totalt", "yhteensä", "yhteensa",
}


def _filter_aggregate_transform(group_col: str, data: list[dict]) -> list[dict]:
    """Return Vega-Lite filter transforms that remove aggregate/total rows.

    Checks the actual data values to avoid injecting a useless filter
    when no aggregate rows are present.
    """
    present = set()
    for row in data:
        val = str(row.get(group_col, "")).strip().lower()
        if val in _AGGREGATE_LABELS:
            present.add(val)
    if not present:
        return []
    conditions = " && ".join(
        f"lower(datum['{group_col}']) !== '{lbl}'" for lbl in sorted(present)
    )
    return [{"filter": conditions}]


def _sanitize_data(data: list[dict]) -> list[dict]:
    """Replace NaN/Infinity values with None for JSON-safe serialization."""
    import math

    clean = []
    for row in data:
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean_row[k] = None
            else:
                clean_row[k] = v
        clean.append(clean_row)
    return clean


def _base_spec(data: list[dict], title: str) -> dict:
    """Create a base Vega-Lite spec."""
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title,
        "width": "container",
        "height": 400,
        "data": {"values": _sanitize_data(data)},
        "config": {
            "view": {"stroke": "transparent"},
            "axis": {"labelFontSize": 12, "titleFontSize": 13},
            "title": {"fontSize": 16, "anchor": "start", "orient": "top"},
        },
    }


def _build_measure_label_map(measure_cols: list[str], metrics: dict) -> dict[str, str]:
    """Build a mapping from raw measure column names to English labels."""
    labels = metrics.get("column_labels", {})
    return {col: labels.get(col, col) for col in measure_cols}


def _translate_measure_transforms(label_map: dict[str, str]) -> list[dict]:
    """
    Build Vega-Lite calculate transforms that translate raw measure names
    in the fold output to English labels for the legend.

    Generates a chained conditional expression:
      datum.measure == 'fjoldi' ? 'Count' : datum.measure == 'tegund' ? 'Type' : datum.measure
    """
    if not label_map or all(k == v for k, v in label_map.items()):
        # No translation needed — labels are already the same as column names
        return []

    # Build a nested ternary expression
    expr = "datum.measure"
    for raw, english in reversed(list(label_map.items())):
        if raw != english:
            expr = f"datum.measure == '{raw}' ? '{english}' : {expr}"

    return [{"calculate": expr, "as": "measure_label"}]


def _gen_line_spec(data: list[dict], columns: dict, metrics: dict, title: str) -> dict:
    """Generate a line chart spec."""
    time_col = columns.get("time_axis", "")
    measure_cols = metrics.get("measure_columns", [])
    group_col = columns.get("series_group")

    # Check for single_measure override (individual chart for one measure)
    single_measure = metrics.get("_spec_overrides", {}).get("single_measure")
    if single_measure and single_measure in measure_cols:
        measure_cols = [single_measure]

    # Coerce bare year integers (1999 → "1999-01-01") so Vega-Lite parses them correctly
    chart_data = data
    if time_col and _is_year_only(data, time_col):
        chart_data = _coerce_year_data(data, time_col)

    spec = _base_spec(chart_data, title)

    # Build x-axis encoding with proper timeUnit for clean labels
    x_enc = {
        "field": time_col,
        "type": "temporal",
        "title": _english_label(time_col, metrics, "Time"),
    }
    time_unit = _infer_time_unit(metrics)
    if time_unit:
        x_enc["timeUnit"] = time_unit

    if group_col and len(measure_cols) == 1:
        # Multi-series: color by group
        agg_transforms = _filter_aggregate_transform(group_col, chart_data)
        if agg_transforms:
            spec.setdefault("transform", []).extend(agg_transforms)
        spec["mark"] = {"type": "line", "point": True, "tooltip": True}
        spec["encoding"] = {
            "x": x_enc,
            "y": {
                "field": measure_cols[0],
                "type": "quantitative",
                "title": _english_label(measure_cols[0], metrics, "Value"),
            },
            "color": {
                "field": group_col,
                "type": "nominal",
                "title": _english_label(group_col, metrics, "Category"),
            },
        }
    elif len(measure_cols) > 1:
        # Multi-measure: fold into long format with English legend labels
        label_map = _build_measure_label_map(measure_cols, metrics)
        spec["transform"] = [
            {"fold": measure_cols, "as": ["measure", "value"]},
            *_translate_measure_transforms(label_map),
        ]
        spec["mark"] = {"type": "line", "point": True, "tooltip": True}
        spec["encoding"] = {
            "x": x_enc,
            "y": {"field": "value", "type": "quantitative", "title": "Value"},
            "color": {"field": "measure_label", "type": "nominal", "title": "Measure"},
        }
    else:
        y_field = measure_cols[0] if measure_cols else ""
        spec["mark"] = {"type": "line", "point": True, "tooltip": True}
        spec["encoding"] = {
            "x": x_enc,
            "y": {
                "field": y_field,
                "type": "quantitative",
                "title": _english_label(y_field, metrics, "Value"),
            },
        }

    return spec


def _gen_bar_spec(data: list[dict], columns: dict, metrics: dict, title: str) -> dict:
    """Generate a bar chart spec."""
    cat_col = columns.get("category", columns.get("time_axis", ""))
    measure_cols = metrics.get("measure_columns", [])
    group_col = columns.get("series_group")
    overrides = metrics.get("_spec_overrides", {})

    # single_measure override for per-measure charts
    single_measure = overrides.get("single_measure")
    if single_measure and single_measure in measure_cols:
        measure_cols = [single_measure]

    # latest_period_only: pivot to a categorical bar chart of the last period
    latest_only = overrides.get("latest_period_only", False)
    if latest_only and group_col:
        return _gen_latest_period_bar(
            data, columns, metrics, measure_cols, group_col, title,
        )

    # Detect if x-axis is temporal (time-series shown as bars)
    is_temporal = "time_axis" in columns and cat_col == columns["time_axis"]
    time_unit = _infer_time_unit(metrics) if is_temporal else None

    # Coerce bare year integers for temporal bar charts
    chart_data = data
    if is_temporal and _is_year_only(data, cat_col):
        chart_data = _coerce_year_data(data, cat_col)

    spec = _base_spec(chart_data, title)

    def _x_enc() -> dict:
        """Build the x/y encoding for the category/time axis."""
        if is_temporal:
            enc = {
                "field": cat_col,
                "type": "temporal",
                "title": _english_label(cat_col, metrics, "Time"),
            }
            if time_unit:
                enc["timeUnit"] = time_unit
        else:
            enc = {
                "field": cat_col,
                "type": "nominal",
                "title": _english_label(cat_col, metrics, "Category"),
            }
        return enc

    total_cats = metrics.get("total_categories", len(data))
    MAX_CHART_CATEGORIES = 30

    if group_col and len(measure_cols) == 1:
        # Grouped time-series: color bars by group with legend
        agg_transforms = _filter_aggregate_transform(group_col, chart_data)
        if agg_transforms:
            spec.setdefault("transform", []).extend(agg_transforms)
        y_field = measure_cols[0]
        spec["mark"] = {"type": "bar", "tooltip": True}
        spec["encoding"] = {
            "x": _x_enc(),
            "y": {
                "field": y_field,
                "type": "quantitative",
                "title": _english_label(y_field, metrics, "Value"),
            },
            "color": {
                "field": group_col,
                "type": "nominal",
                "title": _english_label(group_col, metrics, "Category"),
                "legend": {"orient": "top"},
            },
        }
        if not is_temporal:
            spec["encoding"]["xOffset"] = {"field": group_col}
    elif not is_temporal and total_cats > 10:
        # Horizontal bars for many categories (not for temporal data)
        measure_field = measure_cols[0] if measure_cols else ""
        spec["mark"] = {"type": "bar", "tooltip": True}
        spec["encoding"] = {
            "y": {
                "field": cat_col,
                "type": "nominal",
                "sort": "-x",
                "title": _english_label(cat_col, metrics, "Category"),
            },
            "x": {
                "field": measure_field,
                "type": "quantitative",
                "title": _english_label(measure_field, metrics, "Value"),
            },
        }
        # Cap to top N categories for readability
        if total_cats > MAX_CHART_CATEGORIES and measure_field:
            spec["transform"] = [
                {
                    "window": [{"op": "rank", "as": "_rank"}],
                    "sort": [{"field": measure_field, "order": "descending"}],
                },
                {"filter": f"datum._rank <= {MAX_CHART_CATEGORIES}"},
            ]
            spec["title"] = f"{spec.get('title', '')} (top {MAX_CHART_CATEGORIES})"
    elif len(measure_cols) > 1:
        # Grouped bars with English legend labels
        label_map = _build_measure_label_map(measure_cols, metrics)
        translate_transforms = _translate_measure_transforms(label_map)
        color_field = "measure_label" if translate_transforms else "measure"
        transforms: list[dict] = [
            {"fold": measure_cols, "as": ["measure", "value"]},
            *translate_transforms,
        ]
        x_enc_bar: dict
        if is_temporal:
            # Temporal x + fold without xOffset stacks every measure on the same
            # tick (looks broken). Use a nominal period label + xOffset instead.
            esc = cat_col.replace("\\", "\\\\").replace("'", "\\'")
            transforms.append({
                "calculate": f"timeFormat(toDate(datum['{esc}']), '%Y')",
                "as": "_period_nominal",
            })
            x_enc_bar = {
                "field": "_period_nominal",
                "type": "ordinal",
                "sort": "ascending",
                "title": _english_label(cat_col, metrics, "Time"),
            }
        else:
            x_enc_bar = _x_enc()

        spec["transform"] = transforms
        spec["mark"] = {"type": "bar", "tooltip": True}
        spec["encoding"] = {
            "x": x_enc_bar,
            "y": {"field": "value", "type": "quantitative", "title": "Value"},
            "color": {"field": color_field, "type": "nominal", "title": "Measure"},
            "xOffset": {"field": color_field},
        }
    else:
        y_field = measure_cols[0] if measure_cols else ""
        spec["mark"] = {"type": "bar", "tooltip": True}
        spec["encoding"] = {
            "x": _x_enc(),
            "y": {
                "field": y_field,
                "type": "quantitative",
                "title": _english_label(y_field, metrics, "Value"),
            },
        }

    return spec


def _gen_latest_period_bar(
    data: list[dict],
    columns: dict,
    metrics: dict,
    measure_cols: list[str],
    group_col: str,
    title: str,
) -> dict:
    """Bar chart showing only the most recent time period, grouped by category.

    Filters data to the maximum value of the time axis, removes aggregate
    rows, and renders a horizontal bar chart sorted by the measure.
    """
    time_col = columns.get("time_axis", "")
    date_range = metrics.get("date_range", {})
    max_period = date_range.get("max")

    # Pre-filter data to the latest period (Python-side for simplicity)
    if time_col and max_period is not None:
        max_str = str(max_period)
        latest_data = [
            row for row in data if str(row.get(time_col, "")) == max_str
        ]
        if not latest_data:
            latest_data = data[-20:]  # fallback: last N rows
    else:
        latest_data = data

    # Remove aggregate rows
    latest_data = [
        row for row in latest_data
        if str(row.get(group_col, "")).strip().lower() not in _AGGREGATE_LABELS
    ]

    y_field = measure_cols[0] if measure_cols else ""
    spec = _base_spec(latest_data, title)
    spec["mark"] = {"type": "bar", "tooltip": True}
    spec["encoding"] = {
        "y": {
            "field": group_col,
            "type": "nominal",
            "sort": "-x",
            "title": _english_label(group_col, metrics, "Category"),
        },
        "x": {
            "field": y_field,
            "type": "quantitative",
            "title": _english_label(y_field, metrics, "Value"),
        },
        "color": {
            "field": group_col,
            "type": "nominal",
            "legend": None,
        },
    }
    return spec


def _gen_area_spec(data: list[dict], columns: dict, metrics: dict, title: str) -> dict:
    """Generate an area chart spec."""
    stacked = metrics.get("_spec_overrides", {}).get("stacked", False)
    spec = _gen_line_spec(data, columns, metrics, title)
    # Convert line to area
    if isinstance(spec.get("mark"), dict):
        spec["mark"]["type"] = "area"
        spec["mark"]["opacity"] = 0.6
        spec["mark"]["line"] = True
    else:
        spec["mark"] = {"type": "area", "opacity": 0.6, "line": True, "tooltip": True}

    if stacked and "encoding" in spec and "y" in spec["encoding"]:
        spec["encoding"]["y"]["stack"] = True

    return spec


def _gen_scatter_spec(data: list[dict], columns: dict, metrics: dict, title: str) -> dict:
    """Generate a scatter plot spec."""
    spec = _base_spec(data, title)
    measure_cols = metrics.get("measure_columns", [])
    cat_col = columns.get("category")

    if len(measure_cols) >= 2:
        x_field = measure_cols[0]
        y_field = measure_cols[1]
    elif len(measure_cols) == 1 and cat_col:
        # Only 1 measure: use category on x, measure on y (dot strip)
        x_field = None
        y_field = measure_cols[0]
    else:
        x_field = measure_cols[0] if measure_cols else ""
        y_field = x_field

    spec["mark"] = {"type": "circle", "tooltip": True, "size": 100}

    if x_field is None:
        # Dot strip: category vs measure
        spec["encoding"] = {
            "x": {"field": cat_col, "type": "nominal", "title": _label(cat_col, metrics)},
            "y": {"field": y_field, "type": "quantitative", "title": _label(y_field, metrics)},
        }
    else:
        spec["encoding"] = {
            "x": {"field": x_field, "type": "quantitative", "title": _label(x_field, metrics)},
            "y": {"field": y_field, "type": "quantitative", "title": _label(y_field, metrics)},
        }
        if cat_col:
            spec["encoding"]["color"] = {"field": cat_col, "type": "nominal", "title": _label(cat_col, metrics)}

    return spec


def _gen_point_map_spec(data: list[dict], columns: dict, metrics: dict, title: str) -> dict:
    """Generate a point map spec."""
    spec = _base_spec(data, title)
    # Prefer columns dict, fall back to metrics for lat/lon/category
    lat_col = columns.get("latitude", metrics.get("lat_column", "latitude"))
    lon_col = columns.get("longitude", metrics.get("lon_column", "longitude"))
    cat_col = columns.get("category", metrics.get("category_column"))

    spec["mark"] = {"type": "circle", "tooltip": True, "size": 80}
    spec["encoding"] = {
        "longitude": {"field": lon_col, "type": "quantitative"},
        "latitude": {"field": lat_col, "type": "quantitative"},
    }
    if cat_col:
        spec["encoding"]["color"] = {"field": cat_col, "type": "nominal", "title": _label(cat_col, metrics)}

    # Map projection
    spec["projection"] = {"type": "mercator"}
    spec["height"] = 500

    return spec


def _gen_bubble_map_spec(data: list[dict], columns: dict, metrics: dict, title: str) -> dict:
    """Generate a bubble map spec."""
    spec = _gen_point_map_spec(data, columns, metrics, title)
    measure_col = metrics.get("measure_column")

    if measure_col:
        spec["encoding"]["size"] = {
            "field": measure_col,
            "type": "quantitative",
            "scale": {"range": [50, 500]},
        }

    return spec


def _gen_heatmap_spec(data: list[dict], columns: dict, metrics: dict, title: str) -> dict:
    """Generate a heatmap spec."""
    spec = _base_spec(data, title)
    cat_col = columns.get("category", "")
    measure_cols = metrics.get("measure_columns", [])

    label_map = _build_measure_label_map(measure_cols, metrics)
    translate_transforms = _translate_measure_transforms(label_map)
    x_field = "measure_label" if translate_transforms else "measure"
    spec["transform"] = [
        {"fold": measure_cols, "as": ["measure", "value"]},
        *translate_transforms,
    ]
    spec["mark"] = {"type": "rect", "tooltip": True}
    spec["encoding"] = {
        "x": {"field": x_field, "type": "nominal", "title": "Measure"},
        "y": {"field": cat_col, "type": "nominal", "title": _label(cat_col, metrics)},
        "color": {"field": "value", "type": "quantitative", "scale": {"scheme": "blues"}, "title": "Value"},
    }

    return spec