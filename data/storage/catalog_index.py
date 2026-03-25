"""
Catalog index storage and refresh logic.
Stores CKAN catalog entries in SQLite for fast local querying.
"""

import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional

from data.ingestion.ckan_client import CKANCatalogEntry, CKANClient

logger = logging.getLogger(__name__)


class CatalogIndex:
    """
    SQLite-backed catalog index for CKAN datasets.
    Stores lightweight catalog entries for fast search and filtering
    without hitting the remote API repeatedly.
    """

    def __init__(self, db_path: str = "metadata.sqlite"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_tables()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_tables(self):
        """Create catalog tables if they don't exist."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS catalog_entries (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    title TEXT DEFAULT '',
                    description TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]',
                    organization TEXT,
                    resource_formats TEXT DEFAULT '[]',
                    num_resources INTEGER DEFAULT 0,
                    metadata_modified TEXT,
                    portal_url TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS catalog_sync_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portal_url TEXT NOT NULL,
                    datasets_indexed INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    message TEXT DEFAULT '',
                    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_catalog_portal
                ON catalog_entries(portal_url)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_catalog_org
                ON catalog_entries(organization)
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert_entry(self, entry: CKANCatalogEntry, portal_url: str):
        """Insert or update a single catalog entry."""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO catalog_entries
                    (dataset_id, name, title, description, tags, organization,
                     resource_formats, num_resources, metadata_modified, portal_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.dataset_id,
                entry.name,
                entry.title,
                entry.description,
                json.dumps(entry.tags),
                entry.organization,
                json.dumps(entry.resource_formats),
                entry.num_resources,
                entry.metadata_modified,
                portal_url,
            ))
            conn.commit()

    def upsert_entries(self, entries: list[CKANCatalogEntry], portal_url: str):
        """Bulk insert or update catalog entries."""
        try:
            with self._get_conn() as conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO catalog_entries
                        (dataset_id, name, title, description, tags, organization,
                         resource_formats, num_resources, metadata_modified, portal_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (
                        e.dataset_id, e.name, e.title, e.description,
                        json.dumps(e.tags), e.organization,
                        json.dumps(e.resource_formats), e.num_resources,
                        e.metadata_modified, portal_url,
                    )
                    for e in entries
                ])
                conn.commit()
            logger.info(f"Upserted {len(entries)} catalog entries for {portal_url}")
        except sqlite3.Error as e:
            logger.error(f"Database error upserting catalog entries: {e}")
            raise

    def clear_portal(self, portal_url: str):
        """Remove all catalog entries for a specific portal."""
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM catalog_entries WHERE portal_url = ?",
                (portal_url,),
            )
            conn.commit()

    def log_sync(self, portal_url: str, datasets_indexed: int, status: str, message: str = ""):
        """Log a catalog sync event."""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO catalog_sync_log (portal_url, datasets_indexed, status, message)
                VALUES (?, ?, ?, ?)
            """, (portal_url, datasets_indexed, status, message))
            conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def search(
        self,
        query: Optional[str] = None,
        portal_url: Optional[str] = None,
        format_filter: Optional[str] = None,
        organization: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """
        Search the catalog index.

        Args:
            query: Text search across title, description, and tags.
            portal_url: Filter by specific portal.
            format_filter: Filter by resource format (e.g. 'csv').
            organization: Filter by organization name.
            limit: Max results.
            offset: Pagination offset.

        Returns:
            List of catalog entry dicts.
        """
        conditions = []
        params = []

        if query:
            conditions.append(
                "(title LIKE ? OR description LIKE ? OR tags LIKE ?)"
            )
            q = f"%{query}%"
            params.extend([q, q, q])

        if portal_url:
            conditions.append("portal_url = ?")
            params.append(portal_url)

        if format_filter:
            conditions.append("resource_formats LIKE ?")
            params.append(f'%"{format_filter.lower()}"%')

        if organization:
            conditions.append("organization LIKE ?")
            params.append(f"%{organization}%")

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT * FROM catalog_entries
            {where}
            ORDER BY metadata_modified DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            results = []
            for row in rows:
                entry = dict(row)
                try:
                    entry["tags"] = json.loads(entry["tags"])
                except (json.JSONDecodeError, TypeError):
                    entry["tags"] = []
                try:
                    entry["resource_formats"] = json.loads(entry["resource_formats"])
                except (json.JSONDecodeError, TypeError):
                    entry["resource_formats"] = []
                results.append(entry)
            return results

    def get_entry(self, dataset_id: str) -> Optional[dict]:
        """Get a single catalog entry by dataset ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM catalog_entries WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
            if row:
                entry = dict(row)
                try:
                    entry["tags"] = json.loads(entry["tags"])
                except (json.JSONDecodeError, TypeError):
                    entry["tags"] = []
                try:
                    entry["resource_formats"] = json.loads(entry["resource_formats"])
                except (json.JSONDecodeError, TypeError):
                    entry["resource_formats"] = []
                return entry
            return None

    def count(self, portal_url: Optional[str] = None) -> int:
        """Count total catalog entries, optionally filtered by portal."""
        if portal_url:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM catalog_entries WHERE portal_url = ?",
                    (portal_url,),
                ).fetchone()
        else:
            with self._get_conn() as conn:
                row = conn.execute("SELECT COUNT(*) FROM catalog_entries").fetchone()
        return row[0]

    def last_sync(self, portal_url: str) -> Optional[dict]:
        """Get the most recent sync log for a portal."""
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT * FROM catalog_sync_log
                WHERE portal_url = ?
                ORDER BY synced_at DESC
                LIMIT 1
            """, (portal_url,)).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # Refresh logic
    # ------------------------------------------------------------------

    async def refresh(self, client: CKANClient, portal_url: str, full: bool = False):
        """
        Refresh the catalog index from a CKAN portal.

        Args:
            client: An initialized CKANClient.
            portal_url: The portal base URL (used as the storage key).
            full: If True, clears existing entries before re-indexing.
        """
        try:
            if full:
                self.clear_portal(portal_url)
                logger.info(f"Cleared existing entries for {portal_url}")

            catalog = await client.build_catalog()
            self.upsert_entries(catalog, portal_url)

            self.log_sync(
                portal_url=portal_url,
                datasets_indexed=len(catalog),
                status="success",
            )
            logger.info(
                f"Catalog refresh complete: {len(catalog)} datasets indexed from {portal_url}"
            )

        except Exception as e:
            self.log_sync(
                portal_url=portal_url,
                datasets_indexed=0,
                status="error",
                message=str(e),
            )
            logger.error(f"Catalog refresh failed for {portal_url}: {e}")
            raise
