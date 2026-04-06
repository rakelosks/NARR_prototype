"""
Parquet-based data caching with TTL expiration.
Provides columnar storage optimized for analytical workloads.
Files auto-expire after a configurable TTL (default 24 hours).

Snapshots are keyed by dataset_id plus an optional resource URL hash so
different CSV resources under the same CKAN package do not overwrite each
other (legacy files named ``{dataset_id}.parquet`` are still supported).
"""

import hashlib
import os
import time
import logging
from typing import Optional

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

# Read from central config to keep local and Docker behavior consistent.
CACHE_DIR = settings.cache_dir


def _snapshot_stem(dataset_id: str, resource_url: Optional[str] = None) -> str:
    if not resource_url:
        return dataset_id
    digest = hashlib.sha256(resource_url.encode("utf-8")).hexdigest()[:16]
    return f"{dataset_id}__{digest}"


def _filepath(dataset_id: str, resource_url: Optional[str] = None) -> str:
    stem = _snapshot_stem(dataset_id, resource_url)
    return os.path.join(CACHE_DIR, f"{stem}.parquet")


def _is_expired(filepath: str) -> bool:
    """Check if a file has exceeded the TTL based on its modification time."""
    if settings.cache_ttl_hours <= 0:
        return True
    age_hours = (time.time() - os.path.getmtime(filepath)) / 3600
    return age_hours > settings.cache_ttl_hours


def save_snapshot(
    df: pd.DataFrame, dataset_id: str, resource_url: Optional[str] = None,
) -> str:
    """Save a DataFrame as a Parquet snapshot."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _filepath(dataset_id, resource_url)
    df.to_parquet(path, index=False)
    return path


def load_snapshot(
    dataset_id: str, resource_url: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Load a Parquet snapshot if it exists and is fresh. Returns None if missing or expired.

    When *resource_url* is set, loads the composite key snapshot. When it is
    omitted, only the legacy ``{dataset_id}.parquet`` file is considered.
    """
    if resource_url:
        path = _filepath(dataset_id, resource_url)
        if os.path.exists(path):
            if _is_expired(path):
                logger.info(f"Snapshot expired for {dataset_id} (resource hash), removing")
                os.remove(path)
                return None
            return pd.read_parquet(path)
        return None

    path = _filepath(dataset_id, None)
    if not os.path.exists(path):
        return None
    if _is_expired(path):
        logger.info(f"Snapshot expired for {dataset_id}, removing")
        os.remove(path)
        return None
    return pd.read_parquet(path)


def snapshot_exists(
    dataset_id: str, resource_url: Optional[str] = None,
) -> bool:
    """Check if a fresh snapshot exists for a dataset (optionally for one resource)."""
    path = _filepath(dataset_id, resource_url)
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
