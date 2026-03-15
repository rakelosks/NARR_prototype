"""
Visualization type selection and Vega-Lite specification generation.
Deterministic logic selects chart types based on the matched template,
then generates validated Vega-Lite specs for the client to render.
"""

import logging
from typing import Optional

from pydantic import BaseModel, Field

from data.profiling.template_definitions import TemplateType, TEMPLATE_MAP

from datetime import datetime

logger = logging.getLogger(__name__)


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

    try:
        min_dt = datetime.fromisoformat(str(date_range["min"]))
        max_dt = datetime.fromisoformat(str(date_range["max"]))
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


class ChartSelection(BaseModel):
    """Result of the chart type selection process."""
    primary_chart: str
    secondary_chart: Optional[str] = None
    reason: str


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
    template = TEMPLATE_MAP[template_type]

    if template_type == TemplateType.TIME_SERIES:
        return _select_time_series_chart(metrics)
    elif template_type == TemplateType.CATEGORICAL:
        return _select_categorical_chart(metrics)
    elif template_type == TemplateType.GEOSPATIAL:
        return _select_geospatial_chart(metrics)
    else:
        # Fallback to template's primary recommendation
        primary = template.chart_recommendations[0]
        return ChartSelection(
            primary_chart=primary.chart_type,
            reason=primary.description,
        )


def _select_time_series_chart(metrics: dict) -> ChartSelection:
    """Select chart type for time-series data."""
    group_col = metrics.get("group_column")
    measure_count = len(metrics.get("measure_columns", []))

    if group_col and measure_count == 1:
        return ChartSelection(
            primary_chart="line",
            secondary_chart="area",
            reason="Multi-series line chart grouped by category over time.",
        )
    elif measure_count > 1:
        return ChartSelection(
            primary_chart="line",
            secondary_chart="bar",
            reason="Multi-measure line chart showing parallel trends.",
        )
    else:
        total_periods = metrics.get("total_periods", 0)
        if total_periods <= 12:
            return ChartSelection(
                primary_chart="bar",
                secondary_chart="line",
                reason="Bar chart for small number of time periods.",
            )
        return ChartSelection(
            primary_chart="line",
            reason="Line chart showing trend over time.",
        )


def _select_categorical_chart(metrics: dict) -> ChartSelection:
    """Select chart type for categorical data."""
    total_cats = metrics.get("total_categories", 0)
    measure_count = len(metrics.get("measure_columns", []))

    if total_cats > 15:
        return ChartSelection(
            primary_chart="bar",
            secondary_chart="scatter",
            reason="Horizontal bar chart for many categories.",
        )
    elif measure_count >= 2:
        return ChartSelection(
            primary_chart="bar",
            secondary_chart="scatter",
            reason="Grouped bar chart for multi-measure comparison.",
        )
    else:
        return ChartSelection(
            primary_chart="bar",
            reason="Bar chart comparing values across categories.",
        )


def _select_geospatial_chart(metrics: dict) -> ChartSelection:
    """Select chart type for geospatial data."""
    has_measure = metrics.get("measure_column") is not None
    has_category = metrics.get("category_column") is not None
    has_geometry = metrics.get("geometry_column") is not None

    if has_geometry:
        return ChartSelection(
            primary_chart="choropleth",
            reason="Choropleth map using geometry boundaries.",
        )
    elif has_measure and has_category:
        return ChartSelection(
            primary_chart="bubble_map",
            secondary_chart="map",
            reason="Bubble map with size from measure and color from category.",
        )
    elif has_measure:
        return ChartSelection(
            primary_chart="bubble_map",
            reason="Bubble map with size encoding measure values.",
        )
    else:
        return ChartSelection(
            primary_chart="map",
            reason="Point map showing locations.",
        )


# ---------------------------------------------------------------------------
# Vega-Lite spec generation
# ---------------------------------------------------------------------------

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

    generator = generators.get(chart_type, _gen_bar_spec)
    spec = generator(data, columns, metrics, title)

    logger.info(f"Generated {chart_type} Vega-Lite spec: '{title}'")
    return spec


def _base_spec(data: list[dict], title: str) -> dict:
    """Create a base Vega-Lite spec."""
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title,
        "width": "container",
        "height": 400,
        "data": {"values": data},
        "config": {
            "view": {"stroke": "transparent"},
            "axis": {"labelFontSize": 12, "titleFontSize": 13},
            "title": {"fontSize": 16},
        },
    }


def _gen_line_spec(data: list[dict], columns: dict, metrics: dict, title: str) -> dict:
    """Generate a line chart spec."""
    spec = _base_spec(data, title)
    time_col = columns.get("time_axis", "")
    measure_cols = metrics.get("measure_columns", [])
    group_col = columns.get("series_group")

    # Build x-axis encoding with proper timeUnit for clean labels
    x_enc = {"field": time_col, "type": "temporal", "title": time_col}
    time_unit = _infer_time_unit(metrics)
    if time_unit:
        x_enc["timeUnit"] = time_unit

    if group_col and len(measure_cols) == 1:
        # Multi-series: color by group
        spec["mark"] = {"type": "line", "point": True, "tooltip": True}
        spec["encoding"] = {
            "x": x_enc,
            "y": {"field": measure_cols[0], "type": "quantitative", "title": measure_cols[0]},
            "color": {"field": group_col, "type": "nominal", "title": group_col},
        }
    elif len(measure_cols) > 1:
        # Multi-measure: fold into long format
        spec["transform"] = [
            {"fold": measure_cols, "as": ["measure", "value"]}
        ]
        spec["mark"] = {"type": "line", "point": True, "tooltip": True}
        spec["encoding"] = {
            "x": x_enc,
            "y": {"field": "value", "type": "quantitative", "title": "Value"},
            "color": {"field": "measure", "type": "nominal", "title": "Measure"},
        }
    else:
        y_field = measure_cols[0] if measure_cols else ""
        spec["mark"] = {"type": "line", "point": True, "tooltip": True}
        spec["encoding"] = {
            "x": x_enc,
            "y": {"field": y_field, "type": "quantitative", "title": y_field},
        }

    return spec


def _gen_bar_spec(data: list[dict], columns: dict, metrics: dict, title: str) -> dict:
    """Generate a bar chart spec."""
    spec = _base_spec(data, title)
    cat_col = columns.get("category", columns.get("time_axis", ""))
    measure_cols = metrics.get("measure_columns", [])

    # Detect if x-axis is temporal (time-series shown as bars)
    is_temporal = "time_axis" in columns and cat_col == columns["time_axis"]
    time_unit = _infer_time_unit(metrics) if is_temporal else None

    def _x_enc() -> dict:
        """Build the x/y encoding for the category/time axis."""
        if is_temporal:
            enc = {"field": cat_col, "type": "temporal", "title": cat_col}
            if time_unit:
                enc["timeUnit"] = time_unit
        else:
            enc = {"field": cat_col, "type": "nominal", "title": cat_col}
        return enc

    total_cats = metrics.get("total_categories", len(data))
    MAX_CHART_CATEGORIES = 30

    if not is_temporal and total_cats > 10:
        # Horizontal bars for many categories (not for temporal data)
        measure_field = measure_cols[0] if measure_cols else ""
        spec["mark"] = {"type": "bar", "tooltip": True}
        spec["encoding"] = {
            "y": {"field": cat_col, "type": "nominal", "sort": "-x", "title": cat_col},
            "x": {"field": measure_field, "type": "quantitative"},
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
        # Grouped bars
        spec["transform"] = [
            {"fold": measure_cols, "as": ["measure", "value"]}
        ]
        spec["mark"] = {"type": "bar", "tooltip": True}
        enc = {
            "x": _x_enc(),
            "y": {"field": "value", "type": "quantitative", "title": "Value"},
            "color": {"field": "measure", "type": "nominal"},
        }
        # xOffset only works with nominal/ordinal x-axis, not temporal
        if not is_temporal:
            enc["xOffset"] = {"field": "measure"}
        spec["encoding"] = enc
    else:
        y_field = measure_cols[0] if measure_cols else ""
        spec["mark"] = {"type": "bar", "tooltip": True}
        spec["encoding"] = {
            "x": _x_enc(),
            "y": {"field": y_field, "type": "quantitative", "title": y_field},
        }

    return spec


def _gen_area_spec(data: list[dict], columns: dict, metrics: dict, title: str) -> dict:
    """Generate an area chart spec."""
    spec = _gen_line_spec(data, columns, metrics, title)
    # Convert line to area
    if isinstance(spec.get("mark"), dict):
        spec["mark"]["type"] = "area"
        spec["mark"]["opacity"] = 0.6
        spec["mark"]["line"] = True
    else:
        spec["mark"] = {"type": "area", "opacity": 0.6, "line": True, "tooltip": True}
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
            "x": {"field": cat_col, "type": "nominal", "title": cat_col},
            "y": {"field": y_field, "type": "quantitative", "title": y_field},
        }
    else:
        spec["encoding"] = {
            "x": {"field": x_field, "type": "quantitative", "title": x_field},
            "y": {"field": y_field, "type": "quantitative", "title": y_field},
        }
        if cat_col:
            spec["encoding"]["color"] = {"field": cat_col, "type": "nominal", "title": cat_col}

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
        spec["encoding"]["color"] = {"field": cat_col, "type": "nominal", "title": cat_col}

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

    spec["transform"] = [
        {"fold": measure_cols, "as": ["measure", "value"]}
    ]
    spec["mark"] = {"type": "rect", "tooltip": True}
    spec["encoding"] = {
        "x": {"field": "measure", "type": "nominal", "title": "Measure"},
        "y": {"field": cat_col, "type": "nominal", "title": cat_col},
        "color": {"field": "value", "type": "quantitative", "scale": {"scheme": "blues"}, "title": "Value"},
    }

    return spec