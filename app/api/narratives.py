"""
API routes for narrative generation.
Handles synchronous narrative generation requests.
For async/long-running jobs, use the /jobs endpoint instead.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from data.cache.parquet_cache import load_snapshot, snapshot_exists
from data.profiling.profiler import profile_dataset
from data.profiling.matcher import match_template
from data.analytics.evidence_bundle import BundleBuilder
from app.api.package import PackageBuilder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/narratives", tags=["narratives"])


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    dataset_id: str
    user_message: Optional[str] = None
    title: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/preview")
async def preview_narrative(request: GenerateRequest):
    """
    Generate a narrative preview WITHOUT LLM.
    Returns the evidence bundle with visualizations and key findings.
    Useful for testing the pipeline or when the LLM is unavailable.
    """
    try:
        # Load dataset
        if not snapshot_exists(request.dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found")

        df = load_snapshot(request.dataset_id)

        # Profile → match → bundle
        profile = profile_dataset(df, dataset_id=request.dataset_id)
        match = match_template(profile)

        if not match.best_match or not match.best_match.is_viable:
            raise HTTPException(
                status_code=422,
                detail=f"No viable template match for dataset. Types found: {profile.column_types_summary}",
            )

        builder = BundleBuilder()
        bundle = builder.build(df, profile, match, title=request.title)

        # Build package without narrative
        pkg_builder = PackageBuilder()
        package = pkg_builder.build_without_narrative(bundle)

        return package.to_api_response()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preview generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def generate_narrative(request: GenerateRequest):
    """
    Generate a full narrative WITH LLM.
    This is a synchronous endpoint — for long-running generation,
    use POST /jobs/generate instead.

    Requires an LLM provider (Ollama) to be running.
    """
    try:
        # Load dataset
        if not snapshot_exists(request.dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found")

        df = load_snapshot(request.dataset_id)

        # Profile → match → bundle
        profile = profile_dataset(df, dataset_id=request.dataset_id)
        match = match_template(profile)

        if not match.best_match or not match.best_match.is_viable:
            raise HTTPException(
                status_code=422,
                detail=f"No viable template match. Types found: {profile.column_types_summary}",
            )

        builder = BundleBuilder()
        bundle = builder.build(df, profile, match, title=request.title)

        # Generate narrative with LLM
        from llm.interface import get_provider
        from llm.narrative import NarrativeGenerator

        provider = get_provider()
        generator = NarrativeGenerator(provider)
        result = await generator.generate(bundle, user_message=request.user_message)

        if not result.success:
            raise HTTPException(
                status_code=502,
                detail=f"Narrative generation failed: {result.error}",
            )

        # Build full package
        pkg_builder = PackageBuilder()
        package = pkg_builder.build(result, bundle, llm_provider_name="ollama")

        return package.to_api_response()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Narrative generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))