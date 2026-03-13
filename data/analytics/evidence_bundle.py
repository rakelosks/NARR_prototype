"""
Evidence bundle builder.
Packages analytics metrics, visualization specs, and metadata
into a structured bundle that feeds into narrative generation.

An evidence bundle is the complete context that the LLM needs
to generate a data narrative for a dataset.
"""

import logging
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from data.profiling.profiler import DatasetProfile
from data.profiling.matcher import MatchResult
from data.profiling.template_definitions import TemplateType, TEMPLATE_MAP
from data.analytics.analytics import AnalyticsEngine, AnalyticsResult
from visualization.charts import select_chart_type, generate_spec, ChartSelection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class VisualizationBundle(BaseModel):
    """A single visualization with its spec and metadata."""
    chart_type: str
    title: str
    description: str
    vega_lite_spec: dict
    is_primary: bool = True


class NarrativeContext(BaseModel):
    """Context hints for narrative generation."""
    template_name: str
    focus: str
    suggested_questions: list[str] = []
    key_findings: list[str] = []


class EvidenceBundle(BaseModel):
    """
    Complete evidence bundle for narrative generation.
    Contains everything the LLM needs to produce a data narrative.
    """
    dataset_id: str
    source: str = ""
    template_type: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # Dataset overview
    row_count: int = 0
    column_count: int = 0
    column_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Column count by semantic type",
    )
    matched_columns: dict[str, str] = Field(
        default_factory=dict,
        description="Template role → column name mapping",
    )

    # Analytics
    metrics: dict = Field(default_factory=dict)

    # Visualizations
    visualizations: list[VisualizationBundle] = []

    # Narrative generation context
    narrative_context: Optional[NarrativeContext] = None

    def to_llm_context(self) -> str:
        """
        Format the bundle as a text context string for the LLM prompt.
        This is the structured summary the LLM receives to generate narratives.
        """
        lines = [
            f"Dataset: {self.dataset_id}",
            f"Source: {self.source}",
            f"Template: {self.template_type}",
            f"Size: {self.row_count} rows × {self.column_count} columns",
            f"Column types: {self.column_summary}",
            f"Mapped columns: {self.matched_columns}",
            "",
            "--- Metrics ---",
        ]

        # Format metrics based on template type
        if self.template_type == "time_series":
            lines.extend(self._format_time_series_metrics())
        elif self.template_type == "categorical":
            lines.extend(self._format_categorical_metrics())
        elif self.template_type == "geospatial":
            lines.extend(self._format_geospatial_metrics())

        if self.narrative_context:
            lines.append("")
            lines.append("--- Narrative guidance ---")
            lines.append(f"Focus: {self.narrative_context.focus}")
            if self.narrative_context.key_findings:
                lines.append("Key findings:")
                for f in self.narrative_context.key_findings:
                    lines.append(f"  - {f}")
            if self.narrative_context.suggested_questions:
                lines.append("Questions to address:")
                for q in self.narrative_context.suggested_questions:
                    lines.append(f"  - {q}")

        return "\n".join(lines)

    def _format_time_series_metrics(self) -> list[str]:
        lines = []
        m = self.metrics
        lines.append(f"Time range: {m.get('date_range', {}).get('min')} to {m.get('date_range', {}).get('max')}")
        lines.append(f"Total periods: {m.get('total_periods')}")

        for col, trend in m.get("trend", {}).items():
            lines.append(
                f"Trend ({col}): {trend['direction']}, "
                f"from {trend['first']} to {trend['last']} "
                f"({trend['pct_change']:+.1f}%)"
            )

        for col, stats in m.get("summary_stats", {}).items():
            lines.append(
                f"Stats ({col}): mean={stats['mean']}, "
                f"min={stats['min']}, max={stats['max']}, total={stats['total']}"
            )
        return lines

    def _format_categorical_metrics(self) -> list[str]:
        lines = []
        m = self.metrics
        lines.append(f"Total categories: {m.get('total_categories')}")

        for col, ranking in m.get("rankings", {}).items():
            if ranking:
                top = ranking[0]
                bottom = ranking[-1]
                lines.append(
                    f"Ranking ({col}): highest = {top['category']} ({top['value']}), "
                    f"lowest = {bottom['category']} ({bottom['value']})"
                )

        for col, stats in m.get("summary_stats", {}).items():
            lines.append(
                f"Stats ({col}): mean={stats['mean']}, "
                f"min={stats['min']}, max={stats['max']}"
            )
        return lines

    def _format_geospatial_metrics(self) -> list[str]:
        lines = []
        m = self.metrics
        lines.append(f"Total points: {m.get('total_points')}")
        bb = m.get("bounding_box", {})
        if bb:
            lines.append(
                f"Bounding box: lat [{bb.get('min_lat')}, {bb.get('max_lat')}], "
                f"lon [{bb.get('min_lon')}, {bb.get('max_lon')}]"
            )
        dist = m.get("category_distribution", [])
        if dist:
            lines.append("Category distribution:")
            for d in dist[:10]:
                lines.append(f"  {d['category']}: {d['count']}")
        return lines


# ---------------------------------------------------------------------------
# Bundle builder
# ---------------------------------------------------------------------------

class BundleBuilder:
    """
    Builds evidence bundles by orchestrating the full pipeline:
    profile → match → analyze → select chart → generate spec → bundle.
    """

    def __init__(self):
        self.analytics = AnalyticsEngine()

    def build(
        self,
        df,
        profile: DatasetProfile,
        match: MatchResult,
        title: Optional[str] = None,
    ) -> EvidenceBundle:
        """
        Build a complete evidence bundle for a dataset.

        Args:
            df: The source DataFrame.
            profile: The dataset profile.
            match: The template match result.
            title: Optional title for the visualizations.

        Returns:
            EvidenceBundle ready for narrative generation.
        """
        if not match.best_match or not match.best_match.is_viable:
            raise ValueError(f"No viable template match for '{match.dataset_id}'")

        template_type = match.best_match.template_id
        template = TEMPLATE_MAP[template_type]
        columns = match.best_match.matched_columns

        logger.info(f"Building evidence bundle for '{match.dataset_id}' ({template_type.value})")

        # Step 1: Run analytics
        analytics_result = self.analytics.analyze(df, match)

        # Step 2: Select chart type
        chart_selection = select_chart_type(template_type, analytics_result.metrics)

        # Step 3: Generate visualization specs
        viz_title = title or f"{profile.dataset_id} — {template.name}"
        visualizations = []

        # Primary chart
        primary_spec = generate_spec(
            chart_type=chart_selection.primary_chart,
            data=analytics_result.aggregation_table,
            columns=columns,
            metrics=analytics_result.metrics,
            title=viz_title,
        )
        visualizations.append(VisualizationBundle(
            chart_type=chart_selection.primary_chart,
            title=viz_title,
            description=chart_selection.reason,
            vega_lite_spec=primary_spec,
            is_primary=True,
        ))

        # Secondary chart if available
        if chart_selection.secondary_chart:
            secondary_spec = generate_spec(
                chart_type=chart_selection.secondary_chart,
                data=analytics_result.aggregation_table,
                columns=columns,
                metrics=analytics_result.metrics,
                title=f"{viz_title} (alternative view)",
            )
            visualizations.append(VisualizationBundle(
                chart_type=chart_selection.secondary_chart,
                title=f"{viz_title} (alternative view)",
                description=f"Alternative: {chart_selection.secondary_chart} chart",
                vega_lite_spec=secondary_spec,
                is_primary=False,
            ))

        # Step 4: Build narrative context
        key_findings = self._extract_key_findings(template_type, analytics_result.metrics)

        narrative_context = NarrativeContext(
            template_name=template.name,
            focus=template.narrative_hints.focus,
            suggested_questions=template.narrative_hints.example_questions,
            key_findings=key_findings,
        )

        # Step 5: Assemble the bundle
        bundle = EvidenceBundle(
            dataset_id=match.dataset_id,
            source=profile.source,
            template_type=template_type.value,
            row_count=profile.row_count,
            column_count=profile.column_count,
            column_summary=profile.column_types_summary,
            matched_columns=columns,
            metrics=analytics_result.metrics,
            visualizations=visualizations,
            narrative_context=narrative_context,
        )

        logger.info(
            f"Bundle complete: {len(visualizations)} visualizations, "
            f"{len(key_findings)} key findings"
        )
        return bundle

    def _extract_key_findings(self, template_type: TemplateType, metrics: dict) -> list[str]:
        """Extract human-readable key findings from metrics."""
        findings = []

        if template_type == TemplateType.TIME_SERIES:
            for col, trend in metrics.get("trend", {}).items():
                direction = trend["direction"]
                pct = trend["pct_change"]
                findings.append(
                    f"{col} is {direction} ({pct:+.1f}% change from "
                    f"{trend['first']} to {trend['last']})"
                )
            date_range = metrics.get("date_range", {})
            if date_range:
                findings.append(
                    f"Data covers {date_range.get('min')} to {date_range.get('max')} "
                    f"({metrics.get('total_periods', 0)} periods)"
                )

        elif template_type == TemplateType.CATEGORICAL:
            for col, ranking in metrics.get("rankings", {}).items():
                if ranking:
                    top = ranking[0]
                    bottom = ranking[-1]
                    findings.append(f"Highest {col}: {top['category']} ({top['value']})")
                    findings.append(f"Lowest {col}: {bottom['category']} ({bottom['value']})")

        elif template_type == TemplateType.GEOSPATIAL:
            findings.append(f"{metrics.get('total_points', 0)} locations mapped")
            dist = metrics.get("category_distribution", [])
            if dist:
                top_cat = dist[0]
                findings.append(f"Most common type: {top_cat['category']} ({top_cat['count']})")

        return findings
