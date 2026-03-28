"""
Evidence bundle builder.
Packages analytics metrics, visualization specs, and metadata
into a structured bundle that feeds into narrative generation.

An evidence bundle is the complete context that the LLM needs
to generate a data narrative for a dataset.

Includes three-tiered metadata provenance so the narrative
generator knows which facts are authoritative (portal),
inferred (profiler), or missing (disclaimer needed).
"""

import logging
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from data.profiling.profiler import DatasetProfile
from data.profiling.matcher import MatchResult
from data.profiling.template_definitions import TemplateType, TEMPLATE_MAP, get_parent_archetype
from data.analytics.analytics import AnalyticsEngine, AnalyticsResult
from data.metadata_normalize import NormalizedMetadata, MetadataTier
from visualization.charts import select_chart_type, generate_spec, ChartSelection, ChartEntry

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
    all_columns: dict[str, str] = Field(
        default_factory=dict,
        description="All column names → semantic type",
    )

    # Three-tiered metadata provenance
    normalized_metadata: Optional[NormalizedMetadata] = None

    # Analytics
    metrics: dict = Field(default_factory=dict)

    # Visualizations
    visualizations: list[VisualizationBundle] = []

    # Narrative generation context
    narrative_context: Optional[NarrativeContext] = None

    # Sample data rows for LLM context (first few rows of aggregation table)
    data_sample: list[dict] = Field(default_factory=list)

    # Unique values per categorical column for LLM context
    categorical_values: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Column name → list of unique values (for categorical columns)",
    )

    # Fallback transparency note (when archetype matched instead of domain)
    fallback_note: Optional[str] = None

    def to_llm_context(self) -> str:
        """
        Format the bundle as a text context string for the LLM prompt.
        This is the structured summary the LLM receives to generate narratives.

        Includes metadata provenance so the LLM knows which facts are
        authoritative (portal), inferred, or missing (needs disclaimer).
        """
        meta = self.normalized_metadata
        lines = []

        # --- Dataset identity (use normalized metadata when available) ---
        if meta and meta.title.available:
            lines.append(f"Dataset: {meta.title.value}")
        else:
            lines.append(f"Dataset: {self.dataset_id}")

        if meta and meta.organization.available:
            lines.append(f"Publisher: {meta.organization.value}")

        if meta and meta.source_url.available:
            lines.append(f"Source: {meta.source_url.value}")
        else:
            lines.append(f"Source: {self.source}")

        if meta and meta.licence.available:
            lines.append(f"Licence: {meta.licence.value}")

        if meta and meta.last_modified.available:
            lines.append(f"Last updated: {meta.last_modified.value}")

        if meta and meta.description.available:
            lines.append(f"Description: {meta.description.value}")

        if meta and meta.tags:
            lines.append(f"Tags: {', '.join(meta.tags)}")

        if meta and meta.portal_language.available:
            lines.append(f"Data language: {meta.portal_language.value}")

        lines.append(f"Template: {self.template_type}")
        lines.append(f"Size: {self.row_count} rows × {self.column_count} columns")
        lines.append(f"Column types: {self.column_summary}")
        lines.append(f"Mapped columns: {self.matched_columns}")

        # --- Metadata provenance notice ---
        if meta:
            tier = meta.overall_tier
            if tier == MetadataTier.PORTAL:
                lines.append("Metadata quality: Complete (from open data portal)")
            elif tier == MetadataTier.INFERRED:
                lines.append("Metadata quality: Partial (some fields inferred from data)")
            else:
                lines.append("Metadata quality: Limited (key fields unavailable)")

            disclaimer = meta.disclaimer
            if disclaimer:
                lines.append(f"IMPORTANT — {disclaimer}")
                lines.append(
                    "Include a brief data provenance note in the narrative "
                    "acknowledging the missing information."
                )

        # List all columns with their unique values for categorical ones
        if self.all_columns:
            lines.append("All columns:")
            for col_name, col_type in self.all_columns.items():
                vals = self.categorical_values.get(col_name)
                if vals:
                    vals_str = ", ".join(f'"{v}"' for v in vals[:15])
                    lines.append(f"  - {col_name} ({col_type}) — values: {vals_str}")
                else:
                    lines.append(f"  - {col_name} ({col_type})")

        # Sample data rows so the LLM sees actual values
        if self.data_sample:
            lines.append("")
            lines.append("Sample data (first rows):")
            for row in self.data_sample[:5]:
                row_str = ", ".join(f"{k}={v}" for k, v in row.items())
                lines.append(f"  {row_str}")

        # Fallback transparency
        if self.fallback_note:
            lines.append(f"Note: {self.fallback_note}")

        lines.extend(["", "--- Metrics ---"])

        # Determine base formatter from parent archetype or template type
        parent = get_parent_archetype(TemplateType(self.template_type))
        base_type = parent.value if parent else self.template_type

        # Format base metrics
        if base_type == "time_series":
            lines.extend(self._format_time_series_metrics())
        elif base_type == "categorical":
            lines.extend(self._format_categorical_metrics())
        elif base_type == "geospatial":
            lines.extend(self._format_geospatial_metrics())

        # Format domain-specific extensions
        domain_ext = self.metrics.get("domain_extensions", {})
        if domain_ext:
            lines.extend(self._format_domain_extensions(domain_ext))

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
            # Peak and shape information — critical for accurate narrative
            if trend.get("peak") is not None:
                lines.append(
                    f"  Peak: {trend['peak']} (at {trend.get('peak_period', '?')})"
                )
            if trend.get("shape") == "rise_then_fall":
                lines.append(
                    f"  ⚠ IMPORTANT: Values rose to a peak of {trend['peak']} "
                    f"at {trend.get('peak_period', '?')}, then DROPPED {trend.get('drop_from_peak_pct', 0):.0f}% "
                    f"to {trend['last']}. The overall trend is NOT a steady increase — "
                    f"describe both the rise AND the subsequent decline."
                )
            if trend.get("recovery_from_trough_pct"):
                lines.append(
                    f"  Trough: {trend['trough']} at {trend.get('trough_period', '?')}, "
                    f"then recovered {trend['recovery_from_trough_pct']:.0f}%"
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

    def _format_domain_extensions(self, extensions: dict) -> list[str]:
        """Format domain-specific extension metrics for the LLM context."""
        lines = ["", "--- Domain-specific insights ---"]

        # Budget extensions
        if "department_share" in extensions:
            lines.append("Department/category shares:")
            for item in extensions["department_share"][:10]:
                lines.append(
                    f"  {item.get('department', 'N/A')}: "
                    f"{item.get('pct_share', 0):.1f}% ({item.get('total', 0):,.0f})"
                )

        if "budget_vs_actual" in extensions:
            lines.append("Budget vs actual variance:")
            for item in extensions["budget_vs_actual"][:5]:
                var_pct = item.get("variance_pct")
                var_str = f"{var_pct:+.1f}%" if var_pct is not None else "N/A"
                lines.append(f"  {item.get('period', 'N/A')}: {var_str}")

        if "year_over_year_change" in extensions:
            yoy = extensions["year_over_year_change"]
            # Show last few periods with YoY change
            recent = [r for r in yoy if r.get("yoy_pct") is not None][-5:]
            if recent:
                lines.append("Year-over-year change (recent):")
                for item in recent:
                    lines.append(
                        f"  {item.get('period', 'N/A')}: {item['yoy_pct']:+.1f}%"
                    )

        # Environmental extensions
        for key, val in extensions.items():
            if key.startswith("exceedance_") and isinstance(val, dict):
                lines.append(
                    f"WHO threshold analysis ({val.get('pollutant', 'N/A')}):"
                )
                lines.append(
                    f"  Threshold: {val.get('who_threshold')} {val.get('unit', '')}"
                )
                lines.append(
                    f"  Exceedances: {val.get('exceedances', 0)} of "
                    f"{val.get('total_readings', 0)} readings "
                    f"({val.get('exceedance_pct', 0):.1f}%)"
                )
                lines.append(
                    "  Note: Thresholds are WHO guidelines, not local regulations."
                )

        if "station_comparison" in extensions:
            lines.append("Station comparison (average readings):")
            for item in extensions["station_comparison"][:10]:
                lines.append(
                    f"  {item.get('station', 'N/A')}: "
                    f"avg={item.get('avg_reading', 'N/A')}, "
                    f"range=[{item.get('min_reading', 'N/A')}, {item.get('max_reading', 'N/A')}]"
                )

        # Transport extensions
        if "peak_hour_pattern" in extensions:
            peak = extensions["peak_hour_pattern"]
            if peak:
                sorted_hours = sorted(peak, key=lambda x: x.get("avg_value", 0), reverse=True)
                top_hours = sorted_hours[:3]
                lines.append("Peak hours (highest average):")
                for h in top_hours:
                    lines.append(f"  Hour {int(h.get('hour', 0)):02d}: {h.get('avg_value', 0)}")

        if "weekday_vs_weekend" in extensions:
            lines.append("Weekday vs weekend:")
            for item in extensions["weekday_vs_weekend"]:
                lines.append(
                    f"  {item.get('day_type', 'N/A')}: "
                    f"avg={item.get('avg_value', 0)}, obs={item.get('observations', 0)}"
                )

        if "route_ranking" in extensions:
            lines.append("Route ranking (by total):")
            for item in extensions["route_ranking"][:10]:
                lines.append(
                    f"  {item.get('route', 'N/A')}: total={item.get('total_value', 0):,.0f}"
                )

        # Demographic extensions
        if "population_share" in extensions:
            lines.append("Population share by area:")
            for item in extensions["population_share"][:10]:
                lines.append(
                    f"  {item.get('area', 'N/A')}: {item.get('pct_share', 0):.1f}%"
                )

        if "growth_rate" in extensions:
            growth = extensions["growth_rate"]
            recent = [r for r in growth if r.get("growth_pct") is not None][-10:]
            if recent:
                lines.append("Growth rates (recent):")
                for item in recent:
                    lines.append(
                        f"  {item.get('area', 'N/A')} ({item.get('year', '')}): "
                        f"{item['growth_pct']:+.1f}%"
                    )

        # Facility extensions
        if "type_counts" in extensions:
            lines.append("Facility type counts:")
            for item in extensions["type_counts"][:10]:
                lines.append(f"  {item.get('facility_type', 'N/A')}: {item.get('count', 0)}")

        if "coverage_ratio" in extensions:
            lines.append("Service coverage by district:")
            for item in extensions["coverage_ratio"][:10]:
                lines.append(
                    f"  {item.get('district', 'N/A')}: "
                    f"{item.get('distinct_types', 0)} types, "
                    f"{item.get('total_facilities', 0)} facilities"
                )

        # Incident extensions
        if "type_breakdown" in extensions:
            lines.append("Incident type breakdown:")
            for item in extensions["type_breakdown"][:10]:
                lines.append(
                    f"  {item.get('event_type', item.get('permit_type', 'N/A'))}: "
                    f"{item.get('count', 0)}"
                )

        if "hotspot_areas" in extensions:
            lines.append("Hotspot areas:")
            for item in extensions["hotspot_areas"][:10]:
                lines.append(f"  {item.get('area', 'N/A')}: {item.get('count', 0)} incidents")

        if "resolution_stats" in extensions:
            lines.append("Resolution status:")
            for item in extensions["resolution_stats"]:
                lines.append(f"  {item.get('status', 'N/A')}: {item.get('count', 0)}")

        # Housing extensions
        if "district_activity" in extensions:
            lines.append("District activity (permits):")
            for item in extensions["district_activity"][:10]:
                lines.append(f"  {item.get('district', 'N/A')}: {item.get('permits', 0)}")

        if "avg_value_by_type" in extensions:
            lines.append("Average value by type:")
            for item in extensions["avg_value_by_type"][:10]:
                lines.append(
                    f"  {item.get('permit_type', 'N/A')}: "
                    f"avg={item.get('avg_value', 0):,.0f} (n={item.get('count', 0)})"
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
        metadata: Optional[NormalizedMetadata] = None,
    ) -> EvidenceBundle:
        """
        Build a complete evidence bundle for a dataset.

        Args:
            df: The source DataFrame.
            profile: The dataset profile.
            match: The template match result.
            title: Optional title for the visualizations.
            metadata: Optional NormalizedMetadata with tiered provenance.

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
        viz_title = title or self._generate_title(
            profile, template_type, columns, analytics_result.metrics, metadata
        )
        visualizations = []

        for idx, entry in enumerate(chart_selection.charts):
            chart_title = viz_title
            if entry.title_suffix:
                chart_title = f"{viz_title} {entry.title_suffix}"

            spec = generate_spec(
                chart_type=entry.chart_type,
                data=analytics_result.aggregation_table,
                columns=columns,
                metrics=analytics_result.metrics,
                title=chart_title,
            )
            visualizations.append(VisualizationBundle(
                chart_type=entry.chart_type,
                title=chart_title,
                description=entry.description,
                vega_lite_spec=spec,
                is_primary=(idx == 0),
            ))

        # Step 4: Build narrative context
        key_findings = self._extract_key_findings(template_type, analytics_result.metrics)

        narrative_context = NarrativeContext(
            template_name=template.name,
            focus=template.narrative_hints.focus,
            suggested_questions=template.narrative_hints.example_questions,
            key_findings=key_findings,
        )

        # Step 5: Extract sample data and categorical values for LLM context
        all_columns = {col.name: col.semantic_type for col in profile.columns}

        # Sample rows from aggregation table
        data_sample = analytics_result.aggregation_table[:5]

        # Unique values for categorical columns (crucial for interpretation)
        categorical_values: dict[str, list[str]] = {}
        for col in profile.columns:
            if col.semantic_type == "categorical" and col.name in df.columns:
                unique_vals = df[col.name].dropna().unique().tolist()
                categorical_values[col.name] = [str(v) for v in unique_vals[:20]]

        # Step 6: Assemble the bundle
        bundle = EvidenceBundle(
            dataset_id=match.dataset_id,
            source=profile.source,
            template_type=template_type.value,
            row_count=profile.row_count,
            column_count=profile.column_count,
            column_summary=profile.column_types_summary,
            matched_columns=columns,
            all_columns=all_columns,
            normalized_metadata=metadata,
            metrics=analytics_result.metrics,
            visualizations=visualizations,
            narrative_context=narrative_context,
            data_sample=data_sample,
            categorical_values=categorical_values,
            fallback_note=match.fallback_note,
        )

        logger.info(
            f"Bundle complete: {len(visualizations)} visualizations, "
            f"{len(key_findings)} key findings"
        )
        return bundle

    def _generate_title(
        self,
        profile,
        template_type: TemplateType,
        columns: dict[str, str],
        metrics: dict,
        metadata: Optional[NormalizedMetadata] = None,
    ) -> str:
        """Generate a descriptive chart title from the data context.

        Uses the CKAN metadata title when available (it's more descriptive
        than raw column names, especially for non-English datasets).
        """
        measure_cols = metrics.get("measure_columns", [])
        measure_label = measure_cols[0] if measure_cols else ""

        # Prefer metadata title over raw column name for the base label
        base_label = ""
        if metadata and metadata.title.available:
            base_label = metadata.title.value
        elif measure_label:
            base_label = measure_label
        else:
            base_label = profile.dataset_id

        # Resolve to parent archetype for title formatting
        parent = get_parent_archetype(template_type)
        base_type = parent if parent else template_type

        if base_type == TemplateType.TIME_SERIES:
            date_range = metrics.get("date_range", {})
            time_span = ""
            if date_range.get("min") and date_range.get("max"):
                min_str = str(date_range["min"])[:4]
                max_str = str(date_range["max"])[:4]
                if min_str != max_str:
                    time_span = f" ({min_str}\u2013{max_str})"
                else:
                    time_span = f" ({min_str})"
            return f"{base_label}{time_span}"

        elif base_type == TemplateType.CATEGORICAL:
            cat_col = columns.get("category", "")
            if cat_col:
                return f"{base_label} — {cat_col}"
            return base_label

        elif base_type == TemplateType.GEOSPATIAL:
            total = metrics.get("total_points", 0)
            return f"{base_label} — {total} locations"

        return base_label

    def _extract_key_findings(self, template_type: TemplateType, metrics: dict) -> list[str]:
        """Extract human-readable key findings from metrics."""
        findings = []

        # Base findings from parent archetype
        parent = get_parent_archetype(template_type)
        base_type = parent if parent else template_type

        if base_type in (TemplateType.TIME_SERIES,):
            for col, trend in metrics.get("trend", {}).items():
                direction = trend["direction"]
                pct = trend["pct_change"]
                findings.append(
                    f"{col} is {direction} ({pct:+.1f}% change from "
                    f"{trend['first']} to {trend['last']})"
                )
                # Highlight peak and drop if present
                if trend.get("shape") == "rise_then_fall":
                    findings.append(
                        f"{col} peaked at {trend['peak']} ({trend.get('peak_period', '?')}) "
                        f"then dropped {trend.get('drop_from_peak_pct', 0):.0f}% to {trend['last']}"
                    )
                elif trend.get("peak") and trend.get("peak_period"):
                    findings.append(f"{col} peak: {trend['peak']} at {trend['peak_period']}")
            date_range = metrics.get("date_range", {})
            if date_range:
                findings.append(
                    f"Data covers {date_range.get('min')} to {date_range.get('max')} "
                    f"({metrics.get('total_periods', 0)} periods)"
                )

        elif base_type == TemplateType.CATEGORICAL:
            for col, ranking in metrics.get("rankings", {}).items():
                if ranking:
                    top = ranking[0]
                    bottom = ranking[-1]
                    findings.append(f"Highest {col}: {top['category']} ({top['value']})")
                    findings.append(f"Lowest {col}: {bottom['category']} ({bottom['value']})")

        elif base_type == TemplateType.GEOSPATIAL:
            findings.append(f"{metrics.get('total_points', 0)} locations mapped")
            dist = metrics.get("category_distribution", [])
            if dist:
                top_cat = dist[0]
                findings.append(f"Most common type: {top_cat['category']} ({top_cat['count']})")

        # Domain-specific findings from extensions
        extensions = metrics.get("domain_extensions", {})
        if extensions:
            findings.extend(self._extract_domain_findings(template_type, extensions))

        return findings

    def _extract_domain_findings(
        self, template_type: TemplateType, extensions: dict
    ) -> list[str]:
        """Extract key findings from domain-specific extension metrics."""
        findings = []

        # Budget
        if template_type == TemplateType.BUDGET:
            shares = extensions.get("department_share", [])
            if shares:
                top = shares[0]
                findings.append(
                    f"Largest allocation: {top.get('department', 'N/A')} "
                    f"({top.get('pct_share', 0):.1f}% of total)"
                )
            bva = extensions.get("budget_vs_actual", [])
            if bva:
                latest = bva[-1]
                var = latest.get("variance_pct")
                if var is not None:
                    direction = "over" if var > 0 else "under"
                    findings.append(
                        f"Latest period: {abs(var):.1f}% {direction} budget"
                    )

        # Environmental
        elif template_type == TemplateType.ENVIRONMENTAL:
            for key, val in extensions.items():
                if key.startswith("exceedance_") and isinstance(val, dict):
                    exc_pct = val.get("exceedance_pct", 0)
                    findings.append(
                        f"{val.get('pollutant', 'N/A')} exceeded WHO guideline "
                        f"({val.get('who_threshold')} {val.get('unit', '')}) "
                        f"in {exc_pct:.1f}% of readings"
                    )

        # Transport
        elif template_type == TemplateType.TRANSPORT:
            peak = extensions.get("peak_hour_pattern", [])
            if peak:
                sorted_hours = sorted(peak, key=lambda x: x.get("avg_value", 0), reverse=True)
                if sorted_hours:
                    top_h = sorted_hours[0]
                    findings.append(
                        f"Peak hour: {int(top_h.get('hour', 0)):02d}:00 "
                        f"(avg {top_h.get('avg_value', 0):,.0f})"
                    )
            ww = extensions.get("weekday_vs_weekend", [])
            if len(ww) == 2:
                wd = next((w for w in ww if w.get("day_type") == "weekday"), {})
                we = next((w for w in ww if w.get("day_type") == "weekend"), {})
                if wd and we:
                    ratio = wd.get("avg_value", 0) / we.get("avg_value", 1) if we.get("avg_value") else 0
                    findings.append(f"Weekday traffic is {ratio:.1f}x weekend average")

        # Demographic
        elif template_type == TemplateType.DEMOGRAPHIC:
            shares = extensions.get("population_share", [])
            if shares:
                top = shares[0]
                findings.append(
                    f"Most populated area: {top.get('area', 'N/A')} "
                    f"({top.get('pct_share', 0):.1f}%)"
                )

        # Facility
        elif template_type == TemplateType.FACILITY:
            counts = extensions.get("type_counts", [])
            if counts:
                top = counts[0]
                findings.append(
                    f"Most common facility type: {top.get('facility_type', 'N/A')} "
                    f"({top.get('count', 0)})"
                )

        # Incident
        elif template_type == TemplateType.INCIDENT:
            breakdown = extensions.get("type_breakdown", [])
            if breakdown:
                top = breakdown[0]
                findings.append(
                    f"Most common incident type: {top.get('event_type', 'N/A')} "
                    f"({top.get('count', 0)})"
                )
            hotspots = extensions.get("hotspot_areas", [])
            if hotspots:
                top = hotspots[0]
                findings.append(
                    f"Highest incident concentration: {top.get('area', 'N/A')} "
                    f"({top.get('count', 0)})"
                )

        # Housing
        elif template_type == TemplateType.HOUSING:
            breakdown = extensions.get("type_breakdown", [])
            if breakdown:
                top = breakdown[0]
                findings.append(
                    f"Most common permit type: {top.get('permit_type', 'N/A')} "
                    f"({top.get('count', 0)})"
                )

        return findings
