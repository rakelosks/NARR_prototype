"""
Three-tiered metadata normalization.

Merges metadata from multiple sources with provenance tracking:

    Tier 1 — Portal metadata (CKAN): title, description, tags,
             organization, licence, last modified. Structured and
             authoritative.

    Tier 2 — Inferred from data (profiler): column types, frequency,
             row/column counts, null rates. Cannot recover licence or
             publisher but captures what the data *is*.

    Tier 3 — Transparent disclaimer: fields that could not be resolved
             by either tier. The narrative generator surfaces these as
             explicit caveats rather than guessing.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------

class MetadataTier(str, Enum):
    """Provenance tier for a metadata field."""
    PORTAL = "portal"       # Tier 1: from the CKAN portal API
    INFERRED = "inferred"   # Tier 2: derived from profiling the data
    MISSING = "missing"     # Tier 3: not available — needs disclaimer


# ---------------------------------------------------------------------------
# Per-field provenance wrapper
# ---------------------------------------------------------------------------

class TieredField(BaseModel):
    """A metadata value with its provenance tier."""
    value: Optional[str] = None
    tier: MetadataTier = MetadataTier.MISSING

    @property
    def available(self) -> bool:
        return self.tier != MetadataTier.MISSING and self.value is not None


# ---------------------------------------------------------------------------
# Normalized metadata model
# ---------------------------------------------------------------------------

class NormalizedMetadata(BaseModel):
    """
    Unified dataset metadata with per-field provenance tracking.

    Every field records *where* its value came from so downstream
    consumers (narrative generator, UI) can decide whether to
    present the value confidently or attach a disclaimer.
    """
    dataset_id: str

    # --- Tier 1 fields (portal) or Tier 3 (missing) -----------------------
    title: TieredField = Field(default_factory=TieredField)
    description: TieredField = Field(default_factory=TieredField)
    organization: TieredField = Field(default_factory=TieredField)
    licence: TieredField = Field(default_factory=TieredField)
    tags: list[str] = []
    tags_tier: MetadataTier = MetadataTier.MISSING
    source_url: TieredField = Field(default_factory=TieredField)
    last_modified: TieredField = Field(default_factory=TieredField)
    portal_language: TieredField = Field(default_factory=TieredField)

    # --- Tier 2 fields (inferred from data) --------------------------------
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    column_types: dict[str, str] = Field(
        default_factory=dict,
        description="column name → semantic type",
    )
    inferred_frequency: TieredField = Field(default_factory=TieredField)
    total_null_rate: Optional[float] = None

    # --- Aggregate tier assessment -----------------------------------------
    @property
    def overall_tier(self) -> MetadataTier:
        """
        The *worst* tier across key provenance fields.

        If any critical field is MISSING, the overall tier is MISSING
        (the narrative generator should include a disclaimer).
        If all critical fields come from the portal, overall is PORTAL.
        Otherwise INFERRED.
        """
        critical = [self.title, self.description, self.organization, self.source_url]
        tiers = {f.tier for f in critical}
        if MetadataTier.MISSING in tiers:
            return MetadataTier.MISSING
        if MetadataTier.INFERRED in tiers:
            return MetadataTier.INFERRED
        return MetadataTier.PORTAL

    @property
    def missing_fields(self) -> list[str]:
        """List field names that are at Tier 3 (missing)."""
        check = {
            "title": self.title,
            "description": self.description,
            "organization": self.organization,
            "licence": self.licence,
            "source_url": self.source_url,
            "last_modified": self.last_modified,
        }
        return [name for name, field in check.items() if field.tier == MetadataTier.MISSING]

    @property
    def disclaimer(self) -> Optional[str]:
        """
        Generate a human-readable disclaimer for Tier 3 gaps.
        Returns None if no disclaimer is needed.
        """
        missing = self.missing_fields
        if not missing:
            return None

        field_labels = {
            "title": "dataset title",
            "description": "dataset description",
            "organization": "publisher/source organization",
            "licence": "licence information",
            "source_url": "original source URL",
            "last_modified": "last updated date",
        }
        gaps = [field_labels.get(f, f) for f in missing]

        if len(gaps) >= 4:
            return (
                "Source metadata unavailable — this data was loaded without "
                "structured provenance information. Title, publisher, licence, "
                "and update history could not be determined."
            )

        return f"Note: {', '.join(gaps)} could not be determined from the source."

    def to_dict(self) -> dict:
        """Serialize to a flat dict suitable for storage and LLM context."""
        return {
            "dataset_id": self.dataset_id,
            "overall_tier": self.overall_tier.value,
            "title": self.title.value,
            "title_tier": self.title.tier.value,
            "description": self.description.value,
            "description_tier": self.description.tier.value,
            "organization": self.organization.value,
            "organization_tier": self.organization.tier.value,
            "licence": self.licence.value,
            "licence_tier": self.licence.tier.value,
            "tags": self.tags,
            "tags_tier": self.tags_tier.value,
            "source_url": self.source_url.value,
            "source_url_tier": self.source_url.tier.value,
            "last_modified": self.last_modified.value,
            "last_modified_tier": self.last_modified.tier.value,
            "portal_language": self.portal_language.value,
            "portal_language_tier": self.portal_language.tier.value,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "column_types": self.column_types,
            "inferred_frequency": self.inferred_frequency.value,
            "total_null_rate": self.total_null_rate,
            "missing_fields": self.missing_fields,
            "disclaimer": self.disclaimer,
        }


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def _tiered(value: Optional[str], tier: MetadataTier) -> TieredField:
    """Helper: create a TieredField, falling back to MISSING if value is empty."""
    if value and str(value).strip():
        return TieredField(value=str(value).strip(), tier=tier)
    return TieredField()


def from_ckan(
    dataset,                       # CKANDataset
    portal_language: str = "",
) -> NormalizedMetadata:
    """
    Build NormalizedMetadata from CKAN portal data (Tier 1).
    Profiler fields are left empty to be filled by `enrich_with_profile`.
    """
    # Tags
    tags = dataset.tags if dataset.tags else []
    tags_tier = MetadataTier.PORTAL if tags else MetadataTier.MISSING

    # Organization
    org = dataset.organization if hasattr(dataset, "organization") else None

    # Licence — CKAN stores this in the raw dict; may not be in the model yet
    licence = getattr(dataset, "licence_title", None) or getattr(dataset, "license_title", None)

    # Source URL — first resource with a URL
    source_url = None
    if dataset.resources:
        source_url = dataset.resources[0].url

    return NormalizedMetadata(
        dataset_id=dataset.id,
        title=_tiered(dataset.title, MetadataTier.PORTAL),
        description=_tiered(dataset.notes, MetadataTier.PORTAL),
        organization=_tiered(org, MetadataTier.PORTAL),
        licence=_tiered(licence, MetadataTier.PORTAL),
        tags=tags,
        tags_tier=tags_tier,
        source_url=_tiered(source_url, MetadataTier.PORTAL),
        last_modified=_tiered(dataset.metadata_modified, MetadataTier.PORTAL),
        portal_language=_tiered(portal_language or None, MetadataTier.PORTAL),
    )


def from_url(url: str) -> NormalizedMetadata:
    """
    Build NormalizedMetadata for a direct URL load (Tier 3 for most fields).
    Only the source URL is known.
    """
    return NormalizedMetadata(
        dataset_id=url,
        source_url=TieredField(value=url, tier=MetadataTier.PORTAL),
        # Everything else stays MISSING — Tier 3
    )


def from_file(filepath: str) -> NormalizedMetadata:
    """
    Build NormalizedMetadata for a local file load (Tier 3 for most fields).
    """
    import os
    name = os.path.basename(filepath)
    return NormalizedMetadata(
        dataset_id=filepath,
        title=TieredField(value=name, tier=MetadataTier.INFERRED),
        source_url=TieredField(value=filepath, tier=MetadataTier.PORTAL),
    )


def enrich_with_profile(
    metadata: NormalizedMetadata,
    profile,                      # DatasetProfile
) -> NormalizedMetadata:
    """
    Fill Tier 2 (inferred) fields from a DatasetProfile.
    Does not overwrite existing Tier 1 values.
    """
    metadata.row_count = profile.row_count
    metadata.column_count = profile.column_count
    metadata.total_null_rate = profile.total_null_rate
    metadata.column_types = {
        col.name: col.semantic_type for col in profile.columns
    }

    # Infer title from dataset_id if still missing
    if not metadata.title.available:
        # Clean up dataset_id into a readable title
        name = profile.dataset_id.replace("_", " ").replace("-", " ").title()
        metadata.title = TieredField(value=name, tier=MetadataTier.INFERRED)

    # Infer frequency from temporal columns
    for col in profile.columns:
        if (
            col.semantic_type == "temporal"
            and col.temporal_stats
            and col.temporal_stats.inferred_frequency
        ):
            metadata.inferred_frequency = TieredField(
                value=col.temporal_stats.inferred_frequency,
                tier=MetadataTier.INFERRED,
            )
            break

    return metadata
