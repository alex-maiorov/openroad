"""Base database connection for OpenROAD tool analysis.

DbConnection provides:
- SQLite connection management (read-write or read-only)
- Automatic metadata and schema caching
- A raw query() method
"""

import sqlite3
import pandas as pd


class DbConnection:
    """Base class for tool-specific database connections.

    Caches the table list, column names, and metadata on init.
    Subclasses add query methods that are thin SQL wrappers with
    keyword-argument filtering.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    read_only : bool
        If True, open in read-only mode (safe for concurrent access).
        If False, open read-write (needed for preprocessing).
    """

    def __init__(self, db_path: str, read_only: bool = False):
        self.db_path = db_path

        if read_only:
            self.conn = sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
            )
        else:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)

        self.conn.row_factory = sqlite3.Row
        self._cache_schema()

    # ------------------------------------------------------------------
    # Schema / metadata
    # ------------------------------------------------------------------

    def _cache_schema(self):
        """Populate self.tables and self.metadata from the database."""
        self.tables = {}

        try:
            rows = self.conn.execute(
                "SELECT table_name, column_names FROM table_list"
            ).fetchall()
            for r in rows:
                self.tables[r["table_name"]] = r["column_names"].split(",")
        except sqlite3.OperationalError:
            pass

        # Scan sqlite_master for any tables not in table_list
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT IN ("
            "  SELECT table_name FROM table_list)"
        )
        for r in cursor:
            tname = r["name"]
            col_cursor = self.conn.execute(f"PRAGMA table_info('{tname}')")
            self.tables[tname] = [c["name"] for c in col_cursor]

        # Metadata key/value store
        self.metadata = {}
        try:
            rows = self.conn.execute(
                "SELECT key, value FROM metadata"
            ).fetchall()
            for r in rows:
                self.metadata.setdefault(r["key"], []).append(r["value"])
        except sqlite3.OperationalError:
            pass

    def get_metadata(self, key: str = None):
        """Return metadata values.

        Parameters
        ----------
        key : str or None
            If provided, return values for that key only.

        Returns
        -------
        dict or list
            All metadata, or the list of values for *key*.
        """
        if key:
            return self.metadata.get(key, [])
        return self.metadata

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, sql: str, params=()):
        """Execute a raw SQL query and return a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query string.  May contain ``?`` placeholders.
        params : tuple or list
            Parameters bound to the placeholders.

        Returns
        -------
        pd.DataFrame
        """
        return pd.read_sql_query(sql, self.conn, params=params)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
