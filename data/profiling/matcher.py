"""
Template matching algorithm.
Maps a DatasetProfile to the best-fitting DatasetTemplate
based on column types, requirements, and scoring.

Four-stage matching:
  1. Structural scoring (column type matching)
  2. Domain keyword scoring (column names via keyword dictionary)
  3. Metadata tiebreaker (CKAN title/description/tags)
  4. Selection (viability → score → specificity)
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field

from data.profiling.profiler import DatasetProfile, _column_name_words
from data.profiling.keyword_dictionary import column_suggests_air_quality_context
from data.profiling.template_definitions import (
    DatasetTemplate,
    TemplateType,
    ALL_TEMPLATES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TemplateMatch(BaseModel):
    """Result of matching a profile against a template."""
    template_id: TemplateType
    template_name: str
    score: float = Field(description="Match score from 0.0 to 1.0")
    matched_columns: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of template role -> matched column name",
    )
    missing_required: list[str] = Field(
        default_factory=list,
        description="Required roles that could not be matched",
    )
    is_viable: bool = Field(
        default=False,
        description="True if all required columns are matched",
    )
    domain_keyword_hits: int = Field(
        default=0,
        description="Number of domain keyword matches from column names",
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Why this domain template was rejected (for fallback transparency)",
    )


class MatchResult(BaseModel):
    """Complete result of the template matching process."""
    dataset_id: str
    best_match: Optional[TemplateMatch] = None
    all_matches: list[TemplateMatch] = []
    profile_summary: dict = Field(default_factory=dict)
    fallback_note: Optional[str] = Field(
        default=None,
        description="Transparency note when falling back to archetype",
    )


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def _score_template(profile: DatasetProfile, template: DatasetTemplate) -> TemplateMatch:
    """
    Score how well a dataset profile matches a template.

    Scoring:
        - Each matched required column: +0.3
        - Each matched optional column: +0.15
        - Bonus for multiple measures: +0.05 per extra
        - Penalty for missing required: marks as non-viable
        - Row count check: must meet min_rows

    Returns:
        TemplateMatch with score and column mappings.
    """
    matched_columns: dict[str, str] = {}
    missing_required: list[str] = []
    score = 0.0

    # Check minimum row count
    if profile.row_count < template.min_rows:
        return TemplateMatch(
            template_id=template.id,
            template_name=template.name,
            score=0.0,
            matched_columns={},
            missing_required=[r.role for r in template.column_requirements if r.required],
            is_viable=False,
        )

    # Track which profile columns have been claimed
    claimed_columns: set[str] = set()

    for req in template.column_requirements:
        # Find candidate columns matching this requirement's semantic type
        candidates = [
            col for col in profile.columns
            if col.semantic_type == req.semantic_type
            and col.name not in claimed_columns
        ]

        if candidates:
            # Pick the best candidate
            best = _pick_best_candidate(candidates, req.role)
            matched_columns[req.role] = best.name
            claimed_columns.add(best.name)

            if req.required:
                score += 0.3
            else:
                score += 0.15

            # Bonus for additional measures (e.g. multiple numerical columns)
            if req.role == "measure":
                extra_measures = [
                    c for c in profile.columns
                    if c.semantic_type == req.semantic_type
                    and c.name not in claimed_columns
                ]
                score += len(extra_measures) * 0.05

        elif req.required:
            missing_required.append(req.role)

    is_viable = len(missing_required) == 0

    # Normalize score to 0-1 range
    max_possible = sum(0.3 if r.required else 0.15 for r in template.column_requirements)
    if max_possible > 0:
        score = min(score / max_possible, 1.0)
    else:
        score = 0.0

    match = TemplateMatch(
        template_id=template.id,
        template_name=template.name,
        score=round(score, 3),
        matched_columns=matched_columns,
        missing_required=missing_required,
        is_viable=is_viable,
    )

    # Geospatial: resolve generic "location" into lat/lon or geometry
    # Applies to the geospatial archetype and any domain template with a
    # "location" role requirement (facility, incident, housing, etc.)
    if "location" in matched_columns:
        match = _enrich_geospatial_match(profile, match)

    return match


ROLE_NAME_HINTS: dict[str, set[str]] = {
    "measure": {
        "amount", "count", "total", "value", "budget", "spending", "revenue",
        "population", "score", "price", "cost", "rate", "salary", "income",
        "quantity", "sum", "avg", "average",
    },
    "financial_measure": {
        "budget", "expenditure", "revenue", "spending", "allocation",
        "cost", "income", "grant", "tax", "amount",
    },
    "env_measure": {
        "pm2_5", "pm10", "no2", "co2", "ozone", "temperature",
        "humidity", "noise", "emission", "pollution",
        "concentration", "exceedance", "measurement",
    },
    "traffic_measure": {
        "traffic", "vehicle", "passenger", "ridership", "speed", "flow",
        "parking", "counter",
    },
    "population_measure": {
        "population", "inhabitants", "residents", "density",
        "birth", "death", "migration",
    },
    "time_axis": {
        "date", "time", "timestamp", "period", "year", "month", "day",
        "created", "updated",
    },
    "event_time": {
        "date", "time", "timestamp", "created", "reported", "occurred",
    },
    "category": {
        "type", "category", "group", "class", "district", "department",
        "region", "name", "status", "sector",
    },
    "facility_type": {
        "school", "hospital", "library", "park", "station",
        "facility", "building", "shelter", "playground",
    },
    "event_type": {
        "incident", "accident", "crime", "complaint", "report",
        "violation", "inspection", "ticket", "emergency", "fire",
    },
    "area": {
        "district", "area", "neighbourhood", "neighborhood", "region",
        "zone", "municipality",
    },
    "department": {
        "department", "ministry", "division", "unit", "office",
    },
    "location": {
        "lat", "lon", "lng", "latitude", "longitude", "geom", "geometry",
        "coord", "location",
    },
    "route": {
        "route", "line", "road", "street", "path",
    },
    "direction": {
        "direction", "dir", "heading", "inbound", "outbound",
    },
    "mode": {
        "mode", "type", "vehicle", "bus", "train", "bike", "car",
    },
    "station": {
        "sensor", "station", "counter", "monitor", "detector",
    },
}


def _pick_best_candidate(candidates: list, role: str):
    """
    Pick the best column candidate for a given role.

    Scores by:
      - Name hint bonus (-0.5 if column name words match the role's hints)
      - Keyword-dictionary role hints (-0.45 when the column resolves to this role)
      - env_measure: prefer concentration / pollutant units over exceedance counts
      - financial_measure: deprioritize obvious environmental columns
      - Null rate (lower is better)
    """
    hints = ROLE_NAME_HINTS.get(role, set())

    def _env_measure_priority(col) -> float:
        if role != "env_measure":
            return 0.0
        sig = col.keyword_signal
        if not sig:
            return 0.0
        mc = set(sig.matched_canonicals or [])
        if "concentration" in mc:
            return -0.4
        if mc & {"pm10", "pm2_5", "no2", "ozone", "co2"}:
            return -0.3
        if "exceedance" in mc:
            return -0.12
        return 0.0

    def _sig_bonus(col) -> float:
        sig = col.keyword_signal
        if sig and role in (sig.role_hints or []):
            return -0.45
        return 0.0

    def _financial_env_penalty(col) -> float:
        if role != "financial_measure":
            return 0.0
        sig = col.keyword_signal
        if not sig:
            return 0.0
        rh = sig.role_hints or []
        if "env_measure" in rh or (
            column_suggests_air_quality_context(col.name)
            and "concentration" in (sig.matched_canonicals or [])
        ):
            return 0.55
        return 0.0

    def _score(col):
        w = _column_name_words(col.name)
        name_bonus = -0.5 if (hints & w) else 0
        return (
            col.null_rate
            + name_bonus
            + _sig_bonus(col)
            + _env_measure_priority(col)
            + _financial_env_penalty(col)
        )

    return min(candidates, key=_score)


def _enrich_geospatial_match(
    profile: DatasetProfile,
    match: TemplateMatch,
) -> TemplateMatch:
    """
    Post-process a geospatial match to resolve lat/lon pair vs geometry column.

    Replaces the generic "location" role with specific "latitude"/"longitude"
    or "geometry" roles based on column name analysis.
    """
    geo_cols = [c for c in profile.columns if c.semantic_type == "geospatial"]

    lat_kw = {"lat", "latitude", "y_coord", "y"}
    lon_kw = {"lon", "lng", "longitude", "x_coord", "x"}
    geom_kw = {"geom", "geometry", "wkt"}

    lat_col = lon_col = geom_col = None
    for col in geo_cols:
        words = _column_name_words(col.name)
        if words & lat_kw:
            lat_col = col.name
        elif words & lon_kw:
            lon_col = col.name
        elif words & geom_kw:
            geom_col = col.name

    # Build new column mapping, removing generic "location"
    new_columns = {k: v for k, v in match.matched_columns.items() if k != "location"}

    if lat_col and lon_col:
        new_columns["latitude"] = lat_col
        new_columns["longitude"] = lon_col
        match.matched_columns = new_columns
        match.is_viable = True
    elif geom_col:
        new_columns["geometry"] = geom_col
        match.matched_columns = new_columns
        match.is_viable = True
    # else: keep whatever the standard matching produced

    return match


# ---------------------------------------------------------------------------
# Domain keyword scoring (stage 2)
# ---------------------------------------------------------------------------

def _compute_domain_bonus(
    profile: DatasetProfile,
    template: DatasetTemplate,
    match: TemplateMatch,
) -> TemplateMatch:
    """
    For domain templates, scan column names via the keyword dictionary.
    Add +0.20 bonus if at least one column produces a domain signal
    matching the template's domain. Mark non-viable if zero hits.
    """
    if not template.domain_keywords:
        # Archetype — no domain scoring needed
        return match

    template_domain = template.id.value  # e.g. "budget", "environmental"
    hits = 0

    for col in profile.columns:
        if col.keyword_signal and col.keyword_signal.domain_signals:
            if template_domain in col.keyword_signal.domain_signals:
                hits += 1

    match.domain_keyword_hits = hits

    if hits > 0:
        # Add domain bonus (0.20 added to raw score before normalization)
        match.score = min(match.score + 0.20, 1.0)
    else:
        # Domain template with no keyword hits is not viable
        match.is_viable = False
        match.rejection_reason = (
            f"No {template_domain} keywords detected in column names"
        )

    return match


# ---------------------------------------------------------------------------
# Metadata tiebreaker (stage 3)
# ---------------------------------------------------------------------------

def _metadata_tiebreak_score(
    template: DatasetTemplate,
    metadata_title: str = "",
    metadata_description: str = "",
    metadata_tags: Optional[list[str]] = None,
) -> int:
    """
    Scan CKAN metadata for domain keyword hits.
    Returns hit count for use as a tiebreaker.
    """
    if not template.domain_keywords:
        return 0

    try:
        from data.profiling.keyword_dictionary import resolve_metadata_domains
        domain_hits = resolve_metadata_domains(
            title=metadata_title,
            description=metadata_description,
            tags=metadata_tags,
        )
        return domain_hits.get(template.id.value, 0)
    except ImportError:
        return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_template(
    profile: DatasetProfile,
    templates: Optional[list[DatasetTemplate]] = None,
    metadata_title: str = "",
    metadata_description: str = "",
    metadata_tags: Optional[list[str]] = None,
) -> MatchResult:
    """
    Match a dataset profile against all templates and return the best fit.

    Four-stage process:
      1. Structural scoring (column type matching)
      2. Domain keyword scoring (column names via keyword dictionary)
      3. Metadata tiebreaker (CKAN title/description/tags)
      4. Selection (viability → score → specificity)

    Args:
        profile: The DatasetProfile to match.
        templates: Optional custom template list. Defaults to ALL_TEMPLATES.
        metadata_title: Dataset title from CKAN for tiebreaking.
        metadata_description: Dataset description from CKAN for tiebreaking.
        metadata_tags: Dataset tags from CKAN for tiebreaking.

    Returns:
        MatchResult with best match and all scored matches.
    """
    if templates is None:
        templates = ALL_TEMPLATES

    logger.info(
        f"Matching dataset '{profile.dataset_id}' "
        f"({profile.column_types_summary}) against {len(templates)} templates"
    )

    all_matches = []
    for template in templates:
        # Stage 1: structural scoring
        match = _score_template(profile, template)

        # Stage 2: domain keyword scoring
        match = _compute_domain_bonus(profile, template, match)

        all_matches.append(match)
        logger.debug(
            f"  {template.name}: score={match.score:.3f}, "
            f"viable={match.is_viable}, domain_hits={match.domain_keyword_hits}, "
            f"matched={match.matched_columns}"
        )

    # Stage 3: metadata tiebreaker (computed lazily during sort)
    metadata_scores: dict[TemplateType, int] = {}
    for m in all_matches:
        template = next((t for t in templates if t.id == m.template_id), None)
        if template and template.domain_keywords:
            metadata_scores[m.template_id] = _metadata_tiebreak_score(
                template, metadata_title, metadata_description, metadata_tags,
            )

    # Stage 4: selection
    def _specificity(m: TemplateMatch) -> int:
        """Count distinct semantic types among the matched columns."""
        if not m.matched_columns:
            return 0
        matched_col_names = set(m.matched_columns.values())
        return len({
            col.semantic_type for col in profile.columns
            if col.name in matched_col_names
        })

    # Domain templates with keyword evidence are preferred over
    # archetypes when structurally viable (keyword gating ensures
    # domain relevance). A viable domain template with ≥1 keyword
    # hit ranks above an archetype even if the archetype's
    # structural score is slightly higher.
    all_matches.sort(
        key=lambda m: (
            m.is_viable,
            1 if m.domain_keyword_hits > 0 else 0,  # Domain match > archetype
            m.score,
            m.domain_keyword_hits,
            metadata_scores.get(m.template_id, 0),
            _specificity(m),
        ),
        reverse=True,
    )

    best = all_matches[0] if all_matches and all_matches[0].is_viable else None

    # Build fallback transparency note
    fallback_note = None
    if best and not best.template_id.value.startswith(("time_series", "categorical", "geospatial")):
        pass  # Domain template matched — no fallback needed
    elif best:
        # Best is an archetype — note which domain templates were considered
        rejected_domains = [
            m for m in all_matches
            if m.rejection_reason and m.template_id != best.template_id
        ]
        if rejected_domains:
            reasons = "; ".join(
                f"{m.template_name}: {m.rejection_reason}"
                for m in rejected_domains[:3]
            )
            fallback_note = (
                f"Matched as generic {best.template_name}. "
                f"Domain templates considered but rejected: {reasons}"
            )

    if best:
        logger.info(f"Best match: {best.template_name} (score={best.score:.3f})")
    else:
        logger.warning(f"No viable template match for dataset '{profile.dataset_id}'")

    return MatchResult(
        dataset_id=profile.dataset_id,
        best_match=best,
        all_matches=all_matches,
        profile_summary=profile.column_types_summary,
        fallback_note=fallback_note,
    )
