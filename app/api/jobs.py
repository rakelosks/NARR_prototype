"""
Async job execution for long-running narrative generation.
Uses FastAPI BackgroundTasks with an in-memory job store.
"""

import uuid
import logging
from datetime import datetime
from typing import Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from config import settings
from data.cache.parquet_cache import load_snapshot
from data.profiling.profiler import profile_dataset
from data.profiling.matcher import match_template
from data.analytics.evidence_bundle import BundleBuilder
from data.storage.metadata import MetadataStore
from app.api.package import PackageBuilder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])

_metadata_store = MetadataStore(db_path=settings.metadata_db_path)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(BaseModel):
    job_id: str
    dataset_id: str
    status: JobStatus = JobStatus.PENDING
    user_message: Optional[str] = None
    title: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class CreateJobRequest(BaseModel):
    dataset_id: str
    user_message: Optional[str] = None
    title: Optional[str] = None


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

_jobs: dict[str, Job] = {}


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def list_jobs(limit: int = 20) -> list[Job]:
    sorted_jobs = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
    return sorted_jobs[:limit]


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

async def run_generation_job(job_id: str):
    """Background task that runs the full narrative generation pipeline."""
    job = _jobs.get(job_id)
    if not job:
        return

    job.status = JobStatus.RUNNING
    logger.info(f"Job {job_id} started for dataset '{job.dataset_id}'")

    try:
        config = _metadata_store.get_config(job.dataset_id)
        ru = (config or {}).get("resource_url") or None
        df = load_snapshot(job.dataset_id, ru) if ru else None
        if df is None:
            df = load_snapshot(job.dataset_id)
        if df is None:
            raise FileNotFoundError(f"Dataset '{job.dataset_id}' not found or cache expired")
        profile_source = ""
        if config:
            profile_source = config.get("resource_url", "")
        if not profile_source:
            profile_source = settings.ckan_portal_url

        # Profile and match (reuse config if available)
        profile = profile_dataset(df, dataset_id=job.dataset_id, source=profile_source)
        if config:
            from app.api.narratives import _reconstruct_match
            match = _reconstruct_match(config, profile)
            _metadata_store.touch_config(job.dataset_id)
        else:
            match = match_template(profile)

        if not match.best_match or not match.best_match.is_viable:
            raise ValueError(
                f"No viable template match. Types found: {profile.column_types_summary}"
            )

        builder = BundleBuilder()
        bundle = builder.build(df, profile, match, title=job.title)

        # Try LLM generation
        pkg_builder = PackageBuilder()
        try:
            from llm.interface import get_providers
            from llm.intent import IntentParser
            from llm.narrative import NarrativeGenerator

            intent_provider, generation_provider = get_providers()
            parsed_intent = None
            if job.user_message:
                try:
                    parser = IntentParser(intent_provider)
                    intent_result = await parser.parse(
                        job.user_message,
                        portal_language=settings.ckan_portal_language,
                    )
                    parsed_intent = intent_result.intent
                except Exception as e:
                    logger.warning(f"Job {job_id}: intent parse for chart labels failed: {e}")

            generator = NarrativeGenerator(generation_provider, intent_llm_provider=intent_provider)
            result = await generator.generate(
                bundle,
                user_message=job.user_message,
                intent=parsed_intent,
            )

            if result.success:
                llm_model_name = getattr(generation_provider, "model", "")
                package = pkg_builder.build(
                    result,
                    bundle,
                    llm_provider_name=settings.llm_provider,
                    llm_model_name=llm_model_name,
                )
            else:
                logger.warning(f"Job {job_id}: LLM failed, falling back to preview mode")
                package = pkg_builder.build_without_narrative(bundle)

        except Exception as llm_error:
            logger.warning(f"Job {job_id}: LLM unavailable ({llm_error}), using preview mode")
            package = pkg_builder.build_without_narrative(bundle)

        job.result = package.to_api_response()
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow().isoformat()
        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.utcnow().isoformat()
        logger.error(f"Job {job_id} failed: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/generate", status_code=202)
async def create_generation_job(
    request: CreateJobRequest,
    background_tasks: BackgroundTasks,
):
    """
    Create an async narrative generation job.
    Returns immediately with a job ID that can be polled for status.
    """
    config = _metadata_store.get_config(request.dataset_id)
    ru = (config or {}).get("resource_url") or None
    df = load_snapshot(request.dataset_id, ru) if ru else None
    if df is None:
        df = load_snapshot(request.dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found or cache expired")

    job_id = str(uuid.uuid4())[:8]
    job = Job(
        job_id=job_id,
        dataset_id=request.dataset_id,
        user_message=request.user_message,
        title=request.title,
    )
    _jobs[job_id] = job

    background_tasks.add_task(run_generation_job, job_id)

    logger.info(f"Created job {job_id} for dataset '{request.dataset_id}'")
    return {"job_id": job_id, "status": job.status.value}


@router.get("/{job_id}")
async def get_job_status(job_id: str):
    """Get the status and result of a generation job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    response = {
        "job_id": job.job_id,
        "dataset_id": job.dataset_id,
        "status": job.status.value,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }

    if job.status == JobStatus.COMPLETED:
        response["result"] = job.result
    elif job.status == JobStatus.FAILED:
        response["error"] = job.error

    return response


@router.get("/")
async def list_all_jobs(limit: int = 20):
    """List recent jobs."""
    jobs = list_jobs(limit=limit)
    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "dataset_id": j.dataset_id,
                "status": j.status.value,
                "created_at": j.created_at,
                "completed_at": j.completed_at,
            }
            for j in jobs
        ]
    }
