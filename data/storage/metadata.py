"""
SQLite metadata store.
Manages dataset registries, template configurations, and processing logs.
"""

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
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_url TEXT,
                    description TEXT,
                    row_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT,
                    action TEXT,
                    status TEXT,
                    message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
                )
            """)
            conn.commit()

    def register_dataset(self, dataset_id: str, name: str, **kwargs):
        """Register a new dataset in the metadata store."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO datasets (id, name, source_url, description, row_count) VALUES (?, ?, ?, ?, ?)",
                (dataset_id, name, kwargs.get("source_url"), kwargs.get("description"), kwargs.get("row_count")),
            )
            conn.commit()

    def list_datasets(self) -> list[dict]:
        """List all registered datasets."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]
