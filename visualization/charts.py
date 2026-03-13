"""
Vega-Lite visualization specification generator.
Uses deterministic logic to select chart types based on dataset templates.
"""

from typing import Optional


def select_chart_type(column_types: dict) -> str:
    """Select the appropriate chart type based on column types.

    Uses deterministic logic — no LLM involved in chart selection.
    """
    has_temporal = any(t == "temporal" for t in column_types.values())
    has_numerical = any(t == "numerical" for t in column_types.values())
    has_categorical = any(t == "categorical" for t in column_types.values())
    has_geospatial = any(t == "geospatial" for t in column_types.values())

    if has_geospatial:
        return "map"
    elif has_temporal and has_numerical:
        return "line"
    elif has_categorical and has_numerical:
        return "bar"
    elif has_numerical:
        return "scatter"
    else:
        return "table"


def generate_vega_spec(
    chart_type: str,
    data: list[dict],
    x_field: str,
    y_field: str,
    title: Optional[str] = None,
) -> dict:
    """Generate a Vega-Lite specification for the given chart type and data."""
    base_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title or "",
        "width": "container",
        "height": 400,
        "data": {"values": data},
        "mark": {"type": chart_type, "tooltip": True},
        "encoding": {
            "x": {"field": x_field, "type": "nominal"},
            "y": {"field": y_field, "type": "quantitative"},
        },
    }

    # Adjust encoding based on chart type
    if chart_type == "line":
        base_spec["encoding"]["x"]["type"] = "temporal"
    elif chart_type == "scatter":
        base_spec["encoding"]["x"]["type"] = "quantitative"

    return base_spec
