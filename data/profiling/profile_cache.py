"""
Cache for profiling and template matching results.
Stores results as Parquet files for fast retrieval and reproducibility.
"""

import os
import json
import logging
from typing import Optional

import pandas as pd
from pydantic import BaseModel

from data.profiling.profiler import DatasetProfile
from data.profiling.matcher import MatchResult

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = os.path.join("data", "cache", "profiles")


class ProfileCache:
    """
    Caches DatasetProfile and MatchResult objects as Parquet files.

    Storage layout:
        {cache_dir}/
            {dataset_id}_profile.parquet   — column-level profile data
            {dataset_id}_match.parquet     — template match results
    """

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _profile_path(self, dataset_id: str) -> str:
        return os.path.join(self.cache_dir, f"{dataset_id}_profile.parquet")

    def _match_path(self, dataset_id: str) -> str:
        return os.path.join(self.cache_dir, f"{dataset_id}_match.parquet")

    def _meta_path(self, dataset_id: str) -> str:
        return os.path.join(self.cache_dir, f"{dataset_id}_meta.json")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_profile(self, profile: DatasetProfile):
        """
        Save a DatasetProfile to Parquet.
        Column profiles are stored as rows for easy querying.
        Dataset-level metadata is stored as a JSON sidecar.
        """
        # Column-level data as a flat table
        rows = []
        for col in profile.columns:
            row = {
                "column_name": col.name,
                "semantic_type": col.semantic_type,
                "pandas_dtype": col.pandas_dtype,
                "total_count": col.total_count,
                "null_count": col.null_count,
                "null_rate": col.null_rate,
                "sample_values": json.dumps(col.sample_values),
            }

            # Flatten type-specific stats
            if col.numeric_stats:
                for key, val in col.numeric_stats.model_dump().items():
                    row[f"num_{key}"] = val
            if col.categorical_stats:
                row["cat_unique_count"] = col.categorical_stats.unique_count
                row["cat_is_unique"] = col.categorical_stats.is_unique
                row["cat_top_values"] = json.dumps(col.categorical_stats.top_values)
            if col.temporal_stats:
                for key, val in col.temporal_stats.model_dump().items():
                    row[f"temp_{key}"] = val
            if col.geospatial_stats:
                for key, val in col.geospatial_stats.model_dump().items():
                    row[f"geo_{key}"] = val

            rows.append(row)

        df = pd.DataFrame(rows)
        filepath = self._profile_path(profile.dataset_id)
        df.to_parquet(filepath, index=False)

        # Save dataset-level metadata as JSON sidecar
        meta = {
            "dataset_id": profile.dataset_id,
            "source": profile.source,
            "row_count": profile.row_count,
            "column_count": profile.column_count,
            "total_null_rate": profile.total_null_rate,
            "profiled_at": profile.profiled_at,
            "column_types_summary": profile.column_types_summary,
        }
        with open(self._meta_path(profile.dataset_id), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved profile for '{profile.dataset_id}' to {filepath}")

    def save_match(self, match_result: MatchResult):
        """Save a MatchResult to Parquet."""
        rows = []
        for m in match_result.all_matches:
            rows.append({
                "dataset_id": match_result.dataset_id,
                "template_id": m.template_id.value,
                "template_name": m.template_name,
                "score": m.score,
                "is_viable": m.is_viable,
                "matched_columns": json.dumps(m.matched_columns),
                "missing_required": json.dumps(m.missing_required),
            })

        df = pd.DataFrame(rows)
        filepath = self._match_path(match_result.dataset_id)
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved match results for '{match_result.dataset_id}' to {filepath}")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_profile_table(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load cached profile as a DataFrame."""
        filepath = self._profile_path(dataset_id)
        if not os.path.exists(filepath):
            return None
        return pd.read_parquet(filepath)

    def load_profile_meta(self, dataset_id: str) -> Optional[dict]:
        """Load cached profile metadata."""
        filepath = self._meta_path(dataset_id)
        if not os.path.exists(filepath):
            return None
        with open(filepath) as f:
            return json.load(f)

    def load_match_table(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load cached match results as a DataFrame."""
        filepath = self._match_path(dataset_id)
        if not os.path.exists(filepath):
            return None
        return pd.read_parquet(filepath)

    def get_best_match(self, dataset_id: str) -> Optional[dict]:
        """Load the best match for a dataset from cache."""
        df = self.load_match_table(dataset_id)
        if df is None or len(df) == 0:
            return None

        viable = df[df["is_viable"] == True]
        if len(viable) == 0:
            return None

        best = viable.sort_values("score", ascending=False).iloc[0]
        return {
            "template_id": best["template_id"],
            "template_name": best["template_name"],
            "score": best["score"],
            "matched_columns": json.loads(best["matched_columns"]),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def has_profile(self, dataset_id: str) -> bool:
        """Check if a profile is cached."""
        return os.path.exists(self._profile_path(dataset_id))

    def has_match(self, dataset_id: str) -> bool:
        """Check if match results are cached."""
        return os.path.exists(self._match_path(dataset_id))

    def list_cached(self) -> list[str]:
        """List all dataset IDs with cached profiles."""
        ids = set()
        for filename in os.listdir(self.cache_dir):
            if filename.endswith("_profile.parquet"):
                ids.add(filename.replace("_profile.parquet", ""))
        return sorted(ids)

    def clear(self, dataset_id: Optional[str] = None):
        """Clear cache for a dataset or all datasets."""
        if dataset_id:
            for path in [
                self._profile_path(dataset_id),
                self._match_path(dataset_id),
                self._meta_path(dataset_id),
            ]:
                if os.path.exists(path):
                    os.remove(path)
            logger.info(f"Cleared cache for '{dataset_id}'")
        else:
            for f in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, f))
            logger.info("Cleared all profile cache")
