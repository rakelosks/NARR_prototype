"""
Dataset template definitions.
Each template describes a class of datasets and the visualization
strategies that apply to it.

Templates are matched against DatasetProfile objects by the
template matching algorithm.
"""

from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class TemplateType(str, Enum):
    """Supported dataset template types."""
    # Structural archetypes
    TIME_SERIES = "time_series"
    CATEGORICAL = "categorical"
    GEOSPATIAL = "geospatial"
    # Domain-specialized templates
    BUDGET = "budget"
    ENVIRONMENTAL = "environmental"
    TRANSPORT = "transport"
    DEMOGRAPHIC = "demographic"
    FACILITY = "facility"
    INCIDENT = "incident"
    HOUSING = "housing"


class ColumnRequirement(BaseModel):
    """Defines a column requirement for a template."""
    semantic_type: str  # numerical, categorical, temporal, geospatial
    role: str  # e.g. "time_axis", "measure", "category", "latitude", "longitude"
    required: bool = True
    description: str = ""


class ChartRecommendation(BaseModel):
    """A recommended chart type for this template."""
    chart_type: str  # line, bar, scatter, map, heatmap, etc.
    description: str = ""
    priority: int = 1  # 1 = primary, 2 = secondary, 3 = alternative


class NarrativeHint(BaseModel):
    """Hints for narrative generation tailored to this template."""
    focus: str  # What the narrative should emphasize
    example_questions: list[str] = []  # Questions the narrative should answer


class DatasetTemplate(BaseModel):
    """
    A dataset template defines a class of datasets and how
    they should be visualized and narrated.
    """
    id: TemplateType
    name: str
    description: str
    parent_archetype: Optional[TemplateType] = None
    column_requirements: list[ColumnRequirement]
    domain_keywords: list[str] = Field(default_factory=list)
    chart_recommendations: list[ChartRecommendation]
    narrative_hints: NarrativeHint
    min_rows: int = 2  # Minimum rows for the template to apply


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

TIME_SERIES_TEMPLATE = DatasetTemplate(
    id=TemplateType.TIME_SERIES,
    name="Time series",
    description=(
        "Datasets with a temporal dimension and one or more numerical measures. "
        "Suitable for showing trends, patterns, and changes over time. "
        "Common in city data: budget time series, traffic counts, air quality readings."
    ),
    column_requirements=[
        ColumnRequirement(
            semantic_type="temporal",
            role="time_axis",
            required=True,
            description="A date or timestamp column that defines the time axis.",
        ),
        ColumnRequirement(
            semantic_type="numerical",
            role="measure",
            required=True,
            description="One or more numerical columns representing the measured values.",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="series_group",
            required=False,
            description="Optional grouping column to split into multiple series.",
        ),
    ],
    chart_recommendations=[
        ChartRecommendation(
            chart_type="line",
            description="Line chart showing trend over time.",
            priority=1,
        ),
        ChartRecommendation(
            chart_type="area",
            description="Area chart for cumulative or stacked time series.",
            priority=2,
        ),
        ChartRecommendation(
            chart_type="bar",
            description="Bar chart for periodic aggregations (monthly, yearly).",
            priority=3,
        ),
    ],
    narrative_hints=NarrativeHint(
        focus="Trends, patterns, and significant changes over time.",
        example_questions=[
            "What is the overall trend?",
            "Are there seasonal patterns?",
            "When did significant changes occur?",
            "What is the most recent value compared to historical average?",
        ],
    ),
    min_rows=3,
)


CATEGORICAL_TEMPLATE = DatasetTemplate(
    id=TemplateType.CATEGORICAL,
    name="Categorical comparison",
    description=(
        "Datasets comparing numerical measures across categories. "
        "Suitable for rankings, distributions, and comparisons. "
        "Common in city data: budget by department, complaints by district, "
        "services by type."
    ),
    column_requirements=[
        ColumnRequirement(
            semantic_type="categorical",
            role="category",
            required=True,
            description="A column defining the categories to compare.",
        ),
        ColumnRequirement(
            semantic_type="numerical",
            role="measure",
            required=True,
            description="One or more numerical columns to compare across categories.",
        ),
    ],
    chart_recommendations=[
        ChartRecommendation(
            chart_type="bar",
            description="Bar chart comparing values across categories.",
            priority=1,
        ),
        ChartRecommendation(
            chart_type="scatter",
            description="Scatter plot for two-measure comparisons.",
            priority=2,
        ),
        ChartRecommendation(
            chart_type="heatmap",
            description="Heatmap for multi-category, multi-measure data.",
            priority=3,
        ),
    ],
    narrative_hints=NarrativeHint(
        focus="Comparisons, rankings, and notable differences between categories.",
        example_questions=[
            "Which category has the highest/lowest value?",
            "How do categories compare to each other?",
            "Are there outliers or surprising values?",
            "What is the distribution across categories?",
        ],
    ),
    min_rows=2,
)


GEOSPATIAL_TEMPLATE = DatasetTemplate(
    id=TemplateType.GEOSPATIAL,
    name="Geospatial",
    description=(
        "Datasets with geographic coordinates or geometry data. "
        "Suitable for maps, spatial patterns, and location-based analysis. "
        "Common in city data: service locations, infrastructure, zoning, "
        "incident reports with coordinates."
    ),
    column_requirements=[
        ColumnRequirement(
            semantic_type="geospatial",
            role="location",
            required=True,
            description="Latitude/longitude columns or geometry objects.",
        ),
        ColumnRequirement(
            semantic_type="numerical",
            role="measure",
            required=False,
            description="Optional numerical column for choropleth/bubble sizing.",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="category",
            required=False,
            description="Optional category for color coding points on the map.",
        ),
    ],
    chart_recommendations=[
        ChartRecommendation(
            chart_type="map",
            description="Point map showing locations.",
            priority=1,
        ),
        ChartRecommendation(
            chart_type="choropleth",
            description="Choropleth map with area-based coloring.",
            priority=2,
        ),
        ChartRecommendation(
            chart_type="bubble_map",
            description="Bubble map with size encoding a numerical measure.",
            priority=3,
        ),
    ],
    narrative_hints=NarrativeHint(
        focus="Spatial patterns, clusters, and geographic distribution.",
        example_questions=[
            "Where are the hotspots or clusters?",
            "How is the data distributed geographically?",
            "Are there spatial patterns or correlations?",
            "Which areas have the highest/lowest values?",
        ],
    ),
    min_rows=2,
)


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

# Archetype templates (structural fallbacks)
ARCHETYPE_TEMPLATES: list[DatasetTemplate] = [
    TIME_SERIES_TEMPLATE,
    CATEGORICAL_TEMPLATE,
    GEOSPATIAL_TEMPLATE,
]

# Domain templates are registered after import to avoid circular deps
DOMAIN_TEMPLATES: list[DatasetTemplate] = []

ALL_TEMPLATES: list[DatasetTemplate] = list(ARCHETYPE_TEMPLATES)

TEMPLATE_MAP: dict[TemplateType, DatasetTemplate] = {
    t.id: t for t in ALL_TEMPLATES
}


def register_domain_templates(templates: list[DatasetTemplate]) -> None:
    """Register domain-specialized templates into the global registry."""
    DOMAIN_TEMPLATES.clear()
    DOMAIN_TEMPLATES.extend(templates)
    ALL_TEMPLATES.clear()
    ALL_TEMPLATES.extend(ARCHETYPE_TEMPLATES)
    ALL_TEMPLATES.extend(DOMAIN_TEMPLATES)
    TEMPLATE_MAP.clear()
    TEMPLATE_MAP.update({t.id: t for t in ALL_TEMPLATES})


def get_template(template_type: TemplateType) -> DatasetTemplate:
    """Get a template by type."""
    return TEMPLATE_MAP[template_type]


def get_parent_archetype(template_type: TemplateType) -> Optional[TemplateType]:
    """Get the parent archetype for a domain template, or None for archetypes."""
    t = TEMPLATE_MAP.get(template_type)
    return t.parent_archetype if t else None
