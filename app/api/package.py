"""
Package builder.
Assembles the final deliverable package combining the generated narrative,
visualization specs, dataset metadata, and provenance information.

A package is the complete output that gets served to the frontend.
"""

import logging
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from data.analytics.evidence_bundle import EvidenceBundle, VisualizationBundle
from llm.narrative import GeneratedNarrative, GenerationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ProvenanceInfo(BaseModel):
    """Tracks how a package was generated for transparency and reproducibility."""
    dataset_id: str
    dataset_source: str = ""
    template_type: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    generation_attempts: int = 0
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    pipeline_version: str = "0.1.0"


class DatasetSummary(BaseModel):
    """Lightweight dataset metadata included in the package."""
    dataset_id: str
    source: str = ""
    row_count: int = 0
    column_count: int = 0
    column_types: dict[str, int] = Field(default_factory=dict)
    matched_columns: dict[str, str] = Field(default_factory=dict)
    template_type: str = ""


class NarrativePackage(BaseModel):
    """
    The complete deliverable package.
    Contains everything the frontend needs to render an editorial data story.
    """
    package_id: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # Narrative content — editorial data story format
    headline: str = ""
    lede: str = ""
    story_blocks: list[dict] = Field(default_factory=list)
    data_note: str = ""
    followup_question: str = ""

    # Visualizations
    visualizations: list[dict] = Field(default_factory=list)

    # Metadata
    dataset: DatasetSummary
    provenance: ProvenanceInfo

    def to_api_response(self) -> dict:
        """Format as a clean API response dict."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class PackageBuilder:
    """
    Builds NarrativePackage objects from generation results and evidence bundles.
    """

    def build(
        self,
        generation_result: GenerationResult,
        bundle: EvidenceBundle,
        llm_provider_name: str = "",
        llm_model_name: str = "",
    ) -> NarrativePackage:
        """
        Build a complete package from a generation result and evidence bundle.

        Args:
            generation_result: The result from NarrativeGenerator.
            bundle: The evidence bundle used for generation.
            llm_provider_name: Name of the LLM provider used.
            llm_model_name: Name of the model used.

        Returns:
            NarrativePackage ready for the API/frontend.
        """
        narrative = generation_result.narrative
        package_id = f"{bundle.dataset_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        dataset_source = self._resolve_dataset_source(bundle)

        # Build dataset summary
        dataset_summary = DatasetSummary(
            dataset_id=bundle.dataset_id,
            source=dataset_source,
            row_count=bundle.row_count,
            column_count=bundle.column_count,
            column_types=bundle.column_summary,
            matched_columns=bundle.matched_columns,
            template_type=bundle.template_type,
        )

        # Build provenance
        provenance = ProvenanceInfo(
            dataset_id=bundle.dataset_id,
            dataset_source=dataset_source,
            template_type=bundle.template_type,
            llm_provider=llm_provider_name,
            llm_model=llm_model_name,
            generation_attempts=generation_result.attempts,
        )

        # Build visualization dicts
        viz_dicts = []
        for viz in bundle.visualizations:
            viz_dicts.append({
                "chart_type": viz.chart_type,
                "title": viz.title,
                "description": viz.description,
                "vega_lite_spec": viz.vega_lite_spec,
                "is_primary": viz.is_primary,
            })

        # Assemble story blocks
        story_blocks = []
        if narrative:
            for block in narrative.story_blocks:
                story_blocks.append(block.model_dump(exclude_none=True))

        package = NarrativePackage(
            package_id=package_id,
            headline=narrative.headline if narrative else f"Analysis of {bundle.dataset_id}",
            lede=narrative.lede if narrative else "",
            story_blocks=story_blocks,
            data_note=narrative.data_note if narrative else "",
            followup_question=narrative.followup_question if narrative else "",
            visualizations=viz_dicts,
            dataset=dataset_summary,
            provenance=provenance,
        )

        logger.info(
            f"Package built: {package_id} — "
            f"{len(story_blocks)} story blocks, {len(viz_dicts)} visualizations"
        )
        return package

    def build_without_narrative(
        self,
        bundle: EvidenceBundle,
    ) -> NarrativePackage:
        """
        Build a package with visualizations and metrics but no LLM narrative.
        Useful when the LLM is unavailable or for preview mode.

        Args:
            bundle: The evidence bundle.

        Returns:
            NarrativePackage with viz specs and metrics only.
        """
        package_id = f"{bundle.dataset_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        dataset_source = self._resolve_dataset_source(bundle)

        dataset_summary = DatasetSummary(
            dataset_id=bundle.dataset_id,
            source=dataset_source,
            row_count=bundle.row_count,
            column_count=bundle.column_count,
            column_types=bundle.column_summary,
            matched_columns=bundle.matched_columns,
            template_type=bundle.template_type,
        )

        provenance = ProvenanceInfo(
            dataset_id=bundle.dataset_id,
            dataset_source=dataset_source,
            template_type=bundle.template_type,
            llm_provider="none",
            generation_attempts=0,
        )

        viz_dicts = [
            {
                "chart_type": viz.chart_type,
                "title": viz.title,
                "description": viz.description,
                "vega_lite_spec": viz.vega_lite_spec,
                "is_primary": viz.is_primary,
            }
            for viz in bundle.visualizations
        ]

        # Auto-generate story blocks from key findings + chart references
        story_blocks = []
        findings_text = ""
        if bundle.narrative_context and bundle.narrative_context.key_findings:
            findings_text = " ".join(bundle.narrative_context.key_findings)

        # Primary chart block
        if viz_dicts:
            story_blocks.append({
                "type": "narrative",
                "heading": "Overview",
                "body": findings_text or f"Automated analysis of {bundle.row_count} rows.",
                "viz_index": 0,
            })
        # Secondary chart block (if available)
        if len(viz_dicts) > 1:
            story_blocks.append({
                "type": "narrative",
                "heading": "Alternative view",
                "body": viz_dicts[1].get("description", "An alternative visualization of the data."),
                "viz_index": 1,
            })

        # Ensure at least a basic block if no charts
        if not story_blocks:
            story_blocks.append({
                "type": "narrative",
                "heading": "Key findings",
                "body": findings_text or f"Dataset contains {bundle.row_count} rows across {bundle.column_count} columns.",
            })

        return NarrativePackage(
            package_id=package_id,
            headline=f"Analysis of {bundle.dataset_id}",
            lede=f"Automated analysis of {bundle.row_count} rows across {bundle.column_count} columns.",
            story_blocks=story_blocks,
            visualizations=viz_dicts,
            dataset=dataset_summary,
            provenance=provenance,
        )

    def _resolve_dataset_source(self, bundle: EvidenceBundle) -> str:
        """
        Resolve a non-empty dataset source for provenance.
        Priority: explicit bundle source → normalized metadata source_url.
        """
        if bundle.source and bundle.source.strip():
            return bundle.source.strip()
        if (
            bundle.normalized_metadata
            and bundle.normalized_metadata.source_url.available
            and bundle.normalized_metadata.source_url.value
        ):
            return bundle.normalized_metadata.source_url.value.strip()
        return ""
