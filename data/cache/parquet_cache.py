"""
Parquet-based data caching.
Provides columnar, reproducible storage optimized for analytical workloads.
"""

import os
import pandas as pd


CACHE_DIR = os.path.join(os.path.dirname(__file__), "snapshots")


def save_snapshot(df: pd.DataFrame, dataset_id: str) -> str:
    """Save a DataFrame as a Parquet snapshot."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    filepath = os.path.join(CACHE_DIR, f"{dataset_id}.parquet")
    df.to_parquet(filepath, index=False)
    return filepath


def load_snapshot(dataset_id: str) -> pd.DataFrame:
    """Load a Parquet snapshot by dataset ID."""
    filepath = os.path.join(CACHE_DIR, f"{dataset_id}.parquet")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No snapshot found for dataset: {dataset_id}")
    return pd.read_parquet(filepath)


def snapshot_exists(dataset_id: str) -> bool:
    """Check if a snapshot exists for a dataset."""
    filepath = os.path.join(CACHE_DIR, f"{dataset_id}.parquet")
    return os.path.exists(filepath)
