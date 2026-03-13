"""
Template matching algorithm.
Maps a DatasetProfile to the best-fitting DatasetTemplate
based on column types, requirements, and scoring.
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field

from data.profiling.profiler import DatasetProfile
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


class MatchResult(BaseModel):
    """Complete result of the template matching process."""
    dataset_id: str
    best_match: Optional[TemplateMatch] = None
    all_matches: list[TemplateMatch] = []
    profile_summary: dict = Field(default_factory=dict)


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

    return TemplateMatch(
        template_id=template.id,
        template_name=template.name,
        score=round(score, 3),
        matched_columns=matched_columns,
        missing_required=missing_required,
        is_viable=is_viable,
    )


def _pick_best_candidate(candidates: list, role: str):
    """
    Pick the best column candidate for a given role.
    Prefers columns with lower null rates and names that
    hint at the role.
    """
    # Simple heuristic: prefer lower null rate
    return min(candidates, key=lambda c: c.null_rate)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_template(
    profile: DatasetProfile,
    templates: Optional[list[DatasetTemplate]] = None,
) -> MatchResult:
    """
    Match a dataset profile against all templates and return the best fit.

    Args:
        profile: The DatasetProfile to match.
        templates: Optional custom template list. Defaults to ALL_TEMPLATES.

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
        match = _score_template(profile, template)
        all_matches.append(match)
        logger.debug(
            f"  {template.name}: score={match.score:.3f}, "
            f"viable={match.is_viable}, matched={match.matched_columns}"
        )

    # Sort by viability first, then score, then specificity (number of
    # distinct semantic types matched) to break ties in favour of more
    # specialised templates (e.g. geospatial over categorical).
    def _specificity(m: TemplateMatch) -> int:
        """Count distinct semantic types among the matched columns."""
        if not m.matched_columns:
            return 0
        matched_col_names = set(m.matched_columns.values())
        return len({
            col.semantic_type for col in profile.columns
            if col.name in matched_col_names
        })

    all_matches.sort(
        key=lambda m: (m.is_viable, m.score, _specificity(m)),
        reverse=True,
    )

    best = all_matches[0] if all_matches and all_matches[0].is_viable else None

    if best:
        logger.info(f"Best match: {best.template_name} (score={best.score:.3f})")
    else:
        logger.warning(f"No viable template match for dataset '{profile.dataset_id}'")

    return MatchResult(
        dataset_id=profile.dataset_id,
        best_match=best,
        all_matches=all_matches,
        profile_summary=profile.column_types_summary,
    )
