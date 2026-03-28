"""
API routes for visualization operations.
Returns Vega-Lite specs for a profiled dataset.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from data.cache.parquet_cache import load_snapshot
from data.profiling.profiler import profile_dataset
from data.profiling.matcher import match_template
from data.analytics.analytics import AnalyticsEngine
from data.storage.metadata import MetadataStore
from visualization.charts import select_chart_type, generate_spec

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/visualizations", tags=["visualizations"])

_metadata_store = MetadataStore()


class VizRequest(BaseModel):
    dataset_id: str
    chart_type: Optional[str] = None
    title: Optional[str] = None


@router.post("/generate")
async def generate_visualization(request: VizRequest):
    """Generate a Vega-Lite visualization spec for a dataset."""
    try:
        df = load_snapshot(request.dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found or cache expired")

        # Profile and match (reuse config if available)
        profile = profile_dataset(df, dataset_id=request.dataset_id)

        config = _metadata_store.get_config(request.dataset_id)
        if config:
            from app.api.narratives import _reconstruct_match
            match = _reconstruct_match(config, profile)
            _metadata_store.touch_config(request.dataset_id)
        else:
            match = match_template(profile)

        if not match.best_match or not match.best_match.is_viable:
            raise HTTPException(
                status_code=422,
                detail=f"No viable template match. Types: {profile.column_types_summary}",
            )

        # Run analytics
        engine = AnalyticsEngine()
        analytics = engine.analyze(df, match)

        # Select chart types (or use single override)
        selection = select_chart_type(analytics.template_type, analytics.metrics)
        base_title = request.title or f"{request.dataset_id} visualization"

        if request.chart_type:
            # Single chart override — backward compatible
            spec = generate_spec(
                chart_type=request.chart_type,
                data=analytics.aggregation_table,
                columns=analytics.matched_columns,
                metrics=analytics.metrics,
                title=base_title,
            )
            return {
                "chart_type": request.chart_type,
                "template_type": analytics.template_type.value,
                "vega_lite_spec": spec,
            }

        # Generate all selected charts
        charts = []
        for entry in selection.charts:
            chart_title = base_title
            if entry.title_suffix:
                chart_title = f"{base_title} {entry.title_suffix}"
            spec = generate_spec(
                chart_type=entry.chart_type,
                data=analytics.aggregation_table,
                columns=analytics.matched_columns,
                metrics=analytics.metrics,
                title=chart_title,
            )
            charts.append({
                "chart_type": entry.chart_type,
                "title": chart_title,
                "description": entry.description,
                "vega_lite_spec": spec,
            })

        return {
            "chart_type": selection.primary_chart,
            "template_type": analytics.template_type.value,
            "vega_lite_spec": charts[0]["vega_lite_spec"] if charts else {},
            "all_charts": charts,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
