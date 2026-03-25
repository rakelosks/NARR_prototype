"""
SQLite metadata store.
Manages template configurations, processing logs,
and three-tiered metadata provenance.
"""

import json
import sqlite3
from typing import Optional


class MetadataStore:
    """SQLite-based metadata management."""

    def __init__(self, db_path: str = "metadata.sqlite"):
        self.db_path = db_path
        self._init_tables()

    def _init_tables(self):
        """Initialize database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS template_configurations (
                    dataset_id TEXT PRIMARY KEY,
                    portal_url TEXT,
                    resource_url TEXT,
                    template_type TEXT NOT NULL,
                    column_mappings TEXT NOT NULL,
                    profiling_summary TEXT NOT NULL,
                    metadata_tier TEXT DEFAULT 'missing',
                    normalized_metadata TEXT,
                    last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT,
                    action TEXT,
                    status TEXT,
                    message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

            # Migration: add columns to existing tables
            self._migrate(conn)

    def _migrate(self, conn):
        """Add new columns if they don't exist (safe for existing DBs)."""
        cursor = conn.execute("PRAGMA table_info(template_configurations)")
        existing = {row[1] for row in cursor.fetchall()}
        if "metadata_tier" not in existing:
            conn.execute(
                "ALTER TABLE template_configurations ADD COLUMN metadata_tier TEXT DEFAULT 'missing'"
            )
        if "normalized_metadata" not in existing:
            conn.execute(
                "ALTER TABLE template_configurations ADD COLUMN normalized_metadata TEXT"
            )
        conn.commit()

    def save_config(
        self,
        dataset_id: str,
        portal_url: str,
        resource_url: str,
        template_type: str,
        column_mappings: dict,
        profiling_summary: dict,
        metadata_tier: str = "missing",
        normalized_metadata: Optional[dict] = None,
    ):
        """Save or update a template configuration for a dataset."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO template_configurations
                   (dataset_id, portal_url, resource_url, template_type,
                    column_mappings, profiling_summary,
                    metadata_tier, normalized_metadata, last_used_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                (
                    dataset_id,
                    portal_url,
                    resource_url,
                    template_type,
                    json.dumps(column_mappings),
                    json.dumps(profiling_summary),
                    metadata_tier,
                    json.dumps(normalized_metadata) if normalized_metadata else None,
                ),
            )
            conn.commit()

    def get_config(self, dataset_id: str) -> Optional[dict]:
        """Fetch a template configuration by dataset ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM template_configurations WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            result["column_mappings"] = json.loads(result["column_mappings"])
            result["profiling_summary"] = json.loads(result["profiling_summary"])
            if result.get("normalized_metadata"):
                result["normalized_metadata"] = json.loads(result["normalized_metadata"])
            return result

    def has_config(self, dataset_id: str) -> bool:
        """Check if a template configuration exists for a dataset."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM template_configurations WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
            return row is not None

    def touch_config(self, dataset_id: str):
        """Update last_used_at timestamp for a dataset configuration."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE template_configurations SET last_used_at = CURRENT_TIMESTAMP WHERE dataset_id = ?",
                (dataset_id,),
            )
            conn.commit()

    def list_configs(self) -> list[dict]:
        """List all template configurations, most recently used first."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM template_configurations ORDER BY last_used_at DESC"
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["column_mappings"] = json.loads(d["column_mappings"])
                d["profiling_summary"] = json.loads(d["profiling_summary"])
                if d.get("normalized_metadata"):
                    d["normalized_metadata"] = json.loads(d["normalized_metadata"])
                results.append(d)
            return results
