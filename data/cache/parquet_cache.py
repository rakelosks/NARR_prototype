"""
Parquet-based data caching with TTL expiration.
Provides columnar storage optimized for analytical workloads.
Files auto-expire after a configurable TTL (default 24 hours).
"""

import os
import time
import logging
from typing import Optional

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "snapshots")


def _filepath(dataset_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{dataset_id}.parquet")


def _is_expired(filepath: str) -> bool:
    """Check if a file has exceeded the TTL based on its modification time."""
    if settings.cache_ttl_hours <= 0:
        return True
    age_hours = (time.time() - os.path.getmtime(filepath)) / 3600
    return age_hours > settings.cache_ttl_hours


def save_snapshot(df: pd.DataFrame, dataset_id: str) -> str:
    """Save a DataFrame as a Parquet snapshot."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _filepath(dataset_id)
    df.to_parquet(path, index=False)
    return path


def load_snapshot(dataset_id: str) -> Optional[pd.DataFrame]:
    """Load a Parquet snapshot if it exists and is fresh. Returns None if missing or expired."""
    path = _filepath(dataset_id)
    if not os.path.exists(path):
        return None
    if _is_expired(path):
        logger.info(f"Snapshot expired for {dataset_id}, removing")
        os.remove(path)
        return None
    return pd.read_parquet(path)


def snapshot_exists(dataset_id: str) -> bool:
    """Check if a fresh snapshot exists for a dataset."""
    path = _filepath(dataset_id)
    if not os.path.exists(path):
        return False
    if _is_expired(path):
        logger.info(f"Snapshot expired for {dataset_id}, removing")
        os.remove(path)
        return False
    return True


def cleanup_expired() -> int:
    """Remove all expired snapshots. Returns count of files removed."""
    if not os.path.isdir(CACHE_DIR):
        return 0
    removed = 0
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith(".parquet"):
            continue
        path = os.path.join(CACHE_DIR, fname)
        if _is_expired(path):
            os.remove(path)
            removed += 1
    if removed:
        logger.info(f"Cleaned up {removed} expired snapshot(s)")
    return removed
