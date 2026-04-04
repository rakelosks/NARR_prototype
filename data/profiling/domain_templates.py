"""
Domain-specialized template definitions.

Seven domain templates that extend the three structural archetypes
with domain-specific keyword matching, analytics extensions,
chart recommendations, and narrative hints.

These are registered into the global template registry on import.
"""

from data.profiling.template_definitions import (
    DatasetTemplate,
    TemplateType,
    ColumnRequirement,
    ChartRecommendation,
    NarrativeHint,
    register_domain_templates,
)


# ---------------------------------------------------------------------------
# Budget / financial
# ---------------------------------------------------------------------------

BUDGET_TEMPLATE = DatasetTemplate(
    id=TemplateType.BUDGET,
    name="Budget / financial",
    description=(
        "Datasets tracking public budgets, expenditures, revenues, and fiscal allocations "
        "over time. Extends the time-series archetype with financial domain analytics."
    ),
    parent_archetype=TemplateType.TIME_SERIES,
    column_requirements=[
        ColumnRequirement(
            semantic_type="temporal",
            role="time_axis",
            required=True,
            description="Fiscal year, quarter, or month column.",
        ),
        ColumnRequirement(
            semantic_type="numerical",
            role="financial_measure",
            required=True,
            description="Budget amount, expenditure, revenue, or other financial figure.",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="department",
            required=False,
            description="Department, ministry, or organizational unit.",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="budget_category",
            required=False,
            description="Budget line item, spending category, or fund type.",
        ),
        ColumnRequirement(
            semantic_type="numerical",
            role="secondary_measure",
            required=False,
            description="Second financial measure for budget-vs-actual comparison.",
        ),
    ],
    domain_keywords=[
        "budget", "expenditure", "revenue", "spending", "allocation",
        "fiscal", "cost", "income", "grant", "tax",
    ],
    chart_recommendations=[
        ChartRecommendation(
            chart_type="area",
            description="Stacked area chart showing spending composition over time.",
            priority=1,
        ),
        ChartRecommendation(
            chart_type="bar",
            description="Grouped bar chart comparing budget categories or departments.",
            priority=2,
        ),
        ChartRecommendation(
            chart_type="line",
            description="Line chart for year-over-year financial trends.",
            priority=3,
        ),
    ],
    narrative_hints=NarrativeHint(
        focus="Fiscal accountability — how public money is allocated and spent over time.",
        example_questions=[
            "How has spending changed year-over-year?",
            "Which departments receive the largest allocations?",
            "Is actual spending tracking the budget?",
        ],
    ),
    min_rows=3,
)


# ---------------------------------------------------------------------------
# Environmental monitoring
# ---------------------------------------------------------------------------

ENVIRONMENTAL_TEMPLATE = DatasetTemplate(
    id=TemplateType.ENVIRONMENTAL,
    name="Environmental monitoring",
    description=(
        "Datasets with environmental sensor readings over time: air quality, "
        "temperature, noise, emissions. Extends time-series with threshold analysis."
    ),
    parent_archetype=TemplateType.TIME_SERIES,
    column_requirements=[
        ColumnRequirement(
            semantic_type="temporal",
            role="time_axis",
            required=True,
            description="Timestamp or date of the environmental reading.",
        ),
        ColumnRequirement(
            semantic_type="numerical",
            role="env_measure",
            required=True,
            description="Environmental measurement (pollutant level, temperature, etc.).",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="station",
            required=False,
            description="Monitoring station or sensor identifier.",
        ),
        ColumnRequirement(
            semantic_type="geospatial",
            role="location",
            required=False,
            description="Station coordinates for spatial context.",
        ),
        ColumnRequirement(
            semantic_type="numerical",
            role="secondary_measure",
            required=False,
            description="Additional environmental parameter.",
        ),
    ],
    domain_keywords=[
        "air_quality", "pm2_5", "pm10", "no2", "co2", "ozone",
        "temperature", "humidity", "noise", "emission", "pollution", "sensor",
        "concentration", "substance", "exceedance", "measurement",
    ],
    chart_recommendations=[
        ChartRecommendation(
            chart_type="line",
            description="Line chart with WHO threshold reference band.",
            priority=1,
        ),
        ChartRecommendation(
            chart_type="heatmap",
            description="Heatmap showing readings by time-of-day or station.",
            priority=2,
        ),
        ChartRecommendation(
            chart_type="area",
            description="Area chart for cumulative or seasonal patterns.",
            priority=3,
        ),
    ],
    narrative_hints=NarrativeHint(
        focus="Public health implications of environmental readings over time.",
        example_questions=[
            "Are readings within safe limits?",
            "When do exceedances occur?",
            "Are conditions improving or worsening?",
        ],
    ),
    min_rows=3,
)


# ---------------------------------------------------------------------------
# Transport / mobility
# ---------------------------------------------------------------------------

TRANSPORT_TEMPLATE = DatasetTemplate(
    id=TemplateType.TRANSPORT,
    name="Transport / mobility & road safety",
    description=(
        "Datasets tracking traffic volumes, transit ridership, bike counters, "
        "parking, and road safety over time (collisions, injuries, rates per "
        "vehicle, fatalities). Extends time-series with optional peak-hour analysis."
    ),
    parent_archetype=TemplateType.TIME_SERIES,
    column_requirements=[
        ColumnRequirement(
            semantic_type="temporal",
            role="time_axis",
            required=True,
            description="Timestamp or date of the traffic observation.",
        ),
        ColumnRequirement(
            semantic_type="numerical",
            role="traffic_measure",
            required=True,
            description="Traffic count, ridership figure, or speed measurement.",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="route",
            required=False,
            description="Route, line, or road identifier.",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="direction",
            required=False,
            description="Direction of travel or flow.",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="mode",
            required=False,
            description="Transport mode (bus, bike, car, etc.).",
        ),
        ColumnRequirement(
            semantic_type="geospatial",
            role="location",
            required=False,
            description="Counter or station location.",
        ),
    ],
    domain_keywords=[
        "traffic", "vehicle", "transit", "passenger", "ridership",
        "bike", "bicycle", "parking", "speed", "flow", "counter",
        "bus", "train",
        "accident", "collision", "casualty", "fatal", "injury",
        "safety", "road safety", "crash",
    ],
    chart_recommendations=[
        ChartRecommendation(
            chart_type="line",
            description="Line chart showing traffic volume over time.",
            priority=1,
        ),
        ChartRecommendation(
            chart_type="bar",
            description=(
                "Bar chart by route/direction/mode, period comparison, or "
                "year-by-year counts (e.g. injuries, fatalities)."
            ),
            priority=2,
        ),
        ChartRecommendation(
            chart_type="area",
            description="Area chart for cumulative ridership or flow.",
            priority=3,
        ),
    ],
    narrative_hints=NarrativeHint(
        focus=(
            "How people and vehicles move through the network, and how safe roads are: "
            "traffic volumes, transit use, and road safety (collisions, injuries, "
            "rates per vehicle, serious and fatal outcomes when present)."
        ),
        example_questions=[
            "When are peak hours or busiest periods?",
            "How do collisions, injuries, or fatalities trend over time?",
            "How do rates per vehicle or per distance compare to raw counts?",
            "Which routes or areas show the highest flow or risk?",
        ],
    ),
    min_rows=3,
)


# ---------------------------------------------------------------------------
# Demographic breakdown
# ---------------------------------------------------------------------------

DEMOGRAPHIC_TEMPLATE = DatasetTemplate(
    id=TemplateType.DEMOGRAPHIC,
    name="Demographic breakdown",
    description=(
        "Datasets with population, age, gender, or household statistics "
        "broken down by area or group. Extends the categorical archetype."
    ),
    parent_archetype=TemplateType.CATEGORICAL,
    column_requirements=[
        ColumnRequirement(
            semantic_type="categorical",
            role="area",
            required=True,
            description="Geographic area, district, or neighbourhood.",
        ),
        ColumnRequirement(
            semantic_type="numerical",
            role="population_measure",
            required=True,
            description="Population count, density, or demographic figure.",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="demographic_group",
            required=False,
            description="Age group, gender, nationality, or household type.",
        ),
        ColumnRequirement(
            semantic_type="temporal",
            role="year",
            required=False,
            description="Year for longitudinal demographic comparisons.",
        ),
        ColumnRequirement(
            semantic_type="numerical",
            role="secondary_measure",
            required=False,
            description="Additional demographic measure (area_km2, growth rate).",
        ),
    ],
    domain_keywords=[
        "population", "age", "gender", "household", "census",
        "inhabitants", "residents", "density", "birth", "death", "migration",
    ],
    chart_recommendations=[
        ChartRecommendation(
            chart_type="bar",
            description="Horizontal bar chart ranking areas by population.",
            priority=1,
        ),
        ChartRecommendation(
            chart_type="bar",
            description="Stacked bar chart showing demographic composition.",
            priority=2,
        ),
        ChartRecommendation(
            chart_type="choropleth",
            description="Choropleth map of population density by area.",
            priority=3,
        ),
    ],
    narrative_hints=NarrativeHint(
        focus="How the city's population is distributed and composed.",
        example_questions=[
            "Which areas are most/least populated?",
            "How does demographic composition vary?",
            "Where is population growing?",
        ],
    ),
    min_rows=2,
)


# ---------------------------------------------------------------------------
# Facility / infrastructure
# ---------------------------------------------------------------------------

FACILITY_TEMPLATE = DatasetTemplate(
    id=TemplateType.FACILITY,
    name="Facility / infrastructure",
    description=(
        "Datasets listing public facilities (schools, hospitals, parks) "
        "with geographic locations. Extends the geospatial archetype."
    ),
    parent_archetype=TemplateType.GEOSPATIAL,
    column_requirements=[
        ColumnRequirement(
            semantic_type="geospatial",
            role="location",
            required=True,
            description="Facility coordinates (lat/lon).",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="facility_type",
            required=True,
            description="Type of facility (school, hospital, library, etc.).",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="name",
            required=False,
            description="Name of the individual facility.",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="district",
            required=False,
            description="District or neighbourhood the facility belongs to.",
        ),
        ColumnRequirement(
            semantic_type="numerical",
            role="capacity",
            required=False,
            description="Facility capacity or size measure.",
        ),
    ],
    domain_keywords=[
        "school", "hospital", "library", "park", "station",
        "facility", "building", "capacity", "shelter", "playground",
    ],
    chart_recommendations=[
        ChartRecommendation(
            chart_type="map",
            description="Point map colored by facility type.",
            priority=1,
        ),
        ChartRecommendation(
            chart_type="bar",
            description="Bar chart showing facility counts by type or district.",
            priority=2,
        ),
        ChartRecommendation(
            chart_type="bubble_map",
            description="Bubble map sized by capacity.",
            priority=3,
        ),
    ],
    narrative_hints=NarrativeHint(
        focus="Service coverage and accessibility of public infrastructure across the city.",
        example_questions=[
            "How are facilities distributed?",
            "Which areas have gaps?",
            "What types are most common?",
        ],
    ),
    min_rows=2,
)


# ---------------------------------------------------------------------------
# Incident / event map (composite: geospatial + time_series)
# ---------------------------------------------------------------------------

INCIDENT_TEMPLATE = DatasetTemplate(
    id=TemplateType.INCIDENT,
    name="Incident / event map",
    description=(
        "Spatiotemporal event datasets: incidents, accidents, complaints, "
        "inspections with location and timestamp. Composite of geospatial + time_series."
    ),
    parent_archetype=TemplateType.GEOSPATIAL,  # primary parent
    column_requirements=[
        ColumnRequirement(
            semantic_type="geospatial",
            role="location",
            required=True,
            description="Incident coordinates.",
        ),
        ColumnRequirement(
            semantic_type="temporal",
            role="event_time",
            required=True,
            description="When the incident occurred.",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="event_type",
            required=True,
            description="Type or category of incident.",
        ),
        ColumnRequirement(
            semantic_type="numerical",
            role="severity",
            required=False,
            description="Severity score or damage amount.",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="status",
            required=False,
            description="Resolution status (open, closed, pending).",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="district",
            required=False,
            description="District or area of the incident.",
        ),
    ],
    domain_keywords=[
        "incident", "accident", "crime", "complaint", "report",
        "violation", "inspection", "permit", "ticket", "emergency", "fire",
    ],
    chart_recommendations=[
        ChartRecommendation(
            chart_type="map",
            description="Point map showing incident locations.",
            priority=1,
        ),
        ChartRecommendation(
            chart_type="line",
            description="Line chart showing incident volume over time.",
            priority=2,
        ),
        ChartRecommendation(
            chart_type="bar",
            description="Bar chart breaking down incidents by type.",
            priority=3,
        ),
    ],
    narrative_hints=NarrativeHint(
        focus="Where and when incidents occur, patterns, and hotspots.",
        example_questions=[
            "Where are incidents concentrated?",
            "Are there temporal patterns?",
            "Which types are most common?",
        ],
    ),
    min_rows=2,
)


# ---------------------------------------------------------------------------
# Housing / permits (composite: geospatial + time_series)
# ---------------------------------------------------------------------------

HOUSING_TEMPLATE = DatasetTemplate(
    id=TemplateType.HOUSING,
    name="Housing / permits",
    description=(
        "Datasets tracking building permits, construction activity, and housing data "
        "with temporal trends and optional location. Composite of geospatial + time_series."
    ),
    parent_archetype=TemplateType.TIME_SERIES,  # primary parent
    column_requirements=[
        ColumnRequirement(
            semantic_type="temporal",
            role="date",
            required=True,
            description="Permit date, approval date, or construction period.",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="type",
            required=True,
            description="Permit type, construction type, or housing category.",
        ),
        ColumnRequirement(
            semantic_type="numerical",
            role="value",
            required=True,
            description="Monetary value, unit count, or area measurement.",
        ),
        ColumnRequirement(
            semantic_type="geospatial",
            role="location",
            required=False,
            description="Location of the construction or property.",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="district",
            required=False,
            description="District or neighbourhood.",
        ),
        ColumnRequirement(
            semantic_type="categorical",
            role="status",
            required=False,
            description="Permit status (approved, pending, denied).",
        ),
    ],
    domain_keywords=[
        "housing", "building", "permit", "construction", "rent",
        "property", "dwelling", "zoning", "renovation", "planning",
    ],
    chart_recommendations=[
        ChartRecommendation(
            chart_type="line",
            description="Line chart showing permit volume trends.",
            priority=1,
        ),
        ChartRecommendation(
            chart_type="bar",
            description="Bar chart breaking down by permit/construction type.",
            priority=2,
        ),
        ChartRecommendation(
            chart_type="map",
            description="Point map of construction locations.",
            priority=3,
        ),
    ],
    narrative_hints=NarrativeHint(
        focus="Construction and development activity, permit trends, and housing patterns.",
        example_questions=[
            "Are permits increasing or declining?",
            "Which areas see the most activity?",
            "What types of construction dominate?",
        ],
    ),
    min_rows=2,
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

ALL_DOMAIN_TEMPLATES: list[DatasetTemplate] = [
    BUDGET_TEMPLATE,
    ENVIRONMENTAL_TEMPLATE,
    TRANSPORT_TEMPLATE,
    DEMOGRAPHIC_TEMPLATE,
    FACILITY_TEMPLATE,
    INCIDENT_TEMPLATE,
    HOUSING_TEMPLATE,
]

# Register into the global template registry on import
register_domain_templates(ALL_DOMAIN_TEMPLATES)
