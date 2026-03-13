"""
API routes for visualization operations.
Returns Vega-Lite specs for a profiled dataset.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from data.cache.parquet_cache import load_snapshot, snapshot_exists
from data.profiling.profiler import profile_dataset
from data.profiling.matcher import match_template
from data.analytics.analytics import AnalyticsEngine
from visualization.charts import select_chart_type, generate_spec

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/visualizations", tags=["visualizations"])


class VizRequest(BaseModel):
    dataset_id: str
    chart_type: Optional[str] = None
    title: Optional[str] = None


@router.post("/generate")
async def generate_visualization(request: VizRequest):
    """Generate a Vega-Lite visualization spec for a dataset."""
    try:
        if not snapshot_exists(request.dataset_id):
            raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found")

        df = load_snapshot(request.dataset_id)

        # Profile and match
        profile = profile_dataset(df, dataset_id=request.dataset_id)
        match = match_template(profile)

        if not match.best_match or not match.best_match.is_viable:
            raise HTTPException(
                status_code=422,
                detail=f"No viable template match. Types: {profile.column_types_summary}",
            )

        # Run analytics
        engine = AnalyticsEngine()
        analytics = engine.analyze(df, match)

        # Select chart type (or use override)
        if request.chart_type:
            chart_type = request.chart_type
        else:
            selection = select_chart_type(analytics.template_type, analytics.metrics)
            chart_type = selection.primary_chart

        # Generate spec
        spec = generate_spec(
            chart_type=chart_type,
            data=analytics.aggregation_table,
            columns=analytics.matched_columns,
            metrics=analytics.metrics,
            title=request.title or f"{request.dataset_id} visualization",
        )

        return {
            "chart_type": chart_type,
            "template_type": analytics.template_type.value,
            "vega_lite_spec": spec,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))