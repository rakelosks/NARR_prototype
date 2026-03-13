"""
DuckDB analytical query engine.
Handles SQL-based querying for ingested datasets.
"""

import duckdb
from typing import Optional


class QueryEngine:
    """DuckDB-based analytical query engine."""

    def __init__(self, db_path: str = ":memory:"):
        self.conn = duckdb.connect(db_path)

    def execute(self, query: str) -> list[dict]:
        """Execute a SQL query and return results as list of dicts."""
        result = self.conn.execute(query)
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def register_parquet(self, name: str, filepath: str):
        """Register a Parquet file as a virtual table."""
        self.conn.execute(
            f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{filepath}')"
        )

    def close(self):
        """Close the database connection."""
        self.conn.close()
