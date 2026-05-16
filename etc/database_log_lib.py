#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2025, The OpenROAD Authors

"""
Python library for reading OpenROAD SQLite database logs.

Usage:

    from database_log_lib import DatabaseLog

    with DatabaseLog("/path/to/db.log") as db:
        # Discover registered schemas
        for schema in db.schemas():
            print(f"{schema['table_name']}: {schema['column_names']}")

        # Query a specific data table
        rows = db.get_table("EXA", 10)
        for row in rows:
            print(row)

        # Or use subscript syntax:
        rows = db["EXA_10"]

        # Read metadata
        for m in db.metadata():
            print(f"{m['key']} = {m['value']}")

See docs/SQLITE_LOG_FORMAT.md for the full format specification.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, Iterator, List, Optional, Tuple


class DatabaseLog:
    """Read-only interface to an OpenROAD SQLite database log.

    Opens the database with WAL-mode compatibility, meaning the caller
    gets a consistent snapshot even if the database is being actively
    written to by an OpenROAD process.

    Use as a context manager::

        with DatabaseLog(path) as db:
            ...

    Or manage the lifecycle manually::

        db = DatabaseLog(path)
        rows = db.get_table("EXA", 6)
        db.close()
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._conn = sqlite3.connect(path)
        # Enable WAL-mode coexistence: readers see a consistent snapshot.
        self._conn.execute("PRAGMA query_only = ON;")
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    @property
    def path(self) -> str:
        """Path to the database file."""
        return self._path

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying ``sqlite3.Connection`` for custom queries."""
        return self._conn

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "DatabaseLog":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------
    def tools(self) -> List[Dict[str, Any]]:
        """Return all tool entries as a list of dicts with keys
        ``tool_id`` and ``name``.

        The list is in order of ascending ``tool_id``, which matches
        the order of the ``FOREACH_TOOL`` macro in Logger.h.
        """
        rows = self._conn.execute(
            "SELECT tool_id, name FROM tool_names ORDER BY tool_id"
        ).fetchall()
        return [dict(r) for r in rows]

    def tool_id(self, name: str) -> Optional[int]:
        """Look up the numeric ID for a tool by its three-letter name
        (case-insensitive).  Returns ``None`` if not found.
        """
        row = self._conn.execute(
            "SELECT tool_id FROM tool_names WHERE name = ?",
            (name.upper(),),
        ).fetchone()
        return row["tool_id"] if row is not None else None

    def tool_name(self, tool_id: int) -> Optional[str]:
        """Look up the three-letter tool name for a numeric ID.
        Returns ``None`` if not found.
        """
        row = self._conn.execute(
            "SELECT name FROM tool_names WHERE tool_id = ?",
            (tool_id,),
        ).fetchone()
        return row["name"] if row is not None else None

    def schemas(self) -> List[Dict[str, Any]]:
        """Return every registered schema from ``table_list``.

        Each entry has keys:

        ``tool_id``
            Numeric tool ID (foreign key into ``tool_names``).
        ``tool_name``
            Three-letter tool name, resolved from ``tool_names``.
        ``message_id``
            The message/ID number (second component of the table name).
        ``table_name``
            The full SQLite table name, e.g. ``"EXA_10"``.
        ``column_types``
            Comma-separated type string, e.g. ``"INTEGER,REAL"``.
        ``column_names``
            Comma-separated column name string, e.g. ``"id,val"``.
        ``columns``
            A list of ``(name, type)`` tuples parsed from the above.
        """
        rows = self._conn.execute("""
            SELECT
                tl.tool_id,
                tn.name AS tool_name,
                tl.message_id,
                tl.column_types,
                tl.column_names
            FROM table_list tl
            JOIN tool_names tn ON tl.tool_id = tn.tool_id
            ORDER BY tn.name, tl.message_id
        """).fetchall()

        result: List[Dict[str, Any]] = []
        for r in rows:
            raw = dict(r)
            raw["table_name"] = f"{raw['tool_name']}_{raw['message_id']}"
            type_list = [
                t.strip()
                for t in raw["column_types"].split(",")
                if t.strip()
            ]
            name_list = [
                n.strip()
                for n in raw["column_names"].split(",")
                if n.strip()
            ]
            raw["columns"] = list(zip(name_list, type_list))
            result.append(raw)
        return result

    def table_names(self) -> List[str]:
        """Return the names of all registered data tables.

        Example: ``["EXA_6", "EXA_7", "EXA_8", ...]``

        *Note:* This reads from ``table_list``, not ``sqlite_master``,
        because a table may be registered even if no rows have been
        written to it yet (or the data was rolled back).
        """
        rows = self._conn.execute("""
            SELECT tn.name || '_' || tl.message_id AS table_name
            FROM table_list tl
            JOIN tool_names tn ON tl.tool_id = tn.tool_id
            ORDER BY tn.name, tl.message_id
        """).fetchall()
        return [r["table_name"] for r in rows]

    def schema_for(self, tool_name: str, message_id: int) -> Optional[Dict[str, Any]]:
        """Get the schema for a specific ``(tool_name, message_id)``
        pair, or ``None`` if it is not registered.

        The returned dict has the same keys as each entry in
        :meth:`schemas`.
        """
        row = self._conn.execute("""
            SELECT
                tl.tool_id,
                tn.name AS tool_name,
                tl.message_id,
                tl.column_types,
                tl.column_names
            FROM table_list tl
            JOIN tool_names tn ON tl.tool_id = tn.tool_id
            WHERE tn.name = ? AND tl.message_id = ?
        """, (tool_name.upper(), message_id)).fetchone()

        if row is None:
            return None

        raw = dict(row)
        raw["table_name"] = f"{raw['tool_name']}_{raw['message_id']}"
        type_list = [
            t.strip()
            for t in raw["column_types"].split(",")
            if t.strip()
        ]
        name_list = [
            n.strip()
            for n in raw["column_names"].split(",")
            if n.strip()
        ]
        raw["columns"] = list(zip(name_list, type_list))
        return raw

    def table_exists(self, table_name: str) -> bool:
        """Check whether a data table (e.g. ``"EXA_10"``) actually
        exists in the SQLite schema.  This queries ``sqlite_master``.

        A table may be registered in ``table_list`` but not yet created
        (although in practice creation is synchronous with schema
        registration).  This method checks the physical table existence.
        """
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Metadata access
    # ------------------------------------------------------------------
    def metadata(
        self,
        tool: Optional[str] = None,
        key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return metadata rows.

        Results are ordered by ``tool_id`` then ``key`` for consistency.

        Parameters
        ----------
        tool : str, optional
            If given, filter to rows for this tool (e.g. ``"EXA"``).
            Case-insensitive.
        key : str, optional
            If given, filter to rows with this exact key name.
        """
        parts: List[str] = [
            "SELECT tn.name AS tool_name, m.key, m.value"
            " FROM metadata m"
            " JOIN tool_names tn ON m.tool_id = tn.tool_id"
        ]
        params: List[Any] = []
        if tool is not None:
            parts.append("WHERE tn.name = ?")
            params.append(tool.upper())
            if key is not None:
                parts.append("AND m.key = ?")
                params.append(key)
        elif key is not None:
            parts.append("WHERE m.key = ?")
            params.append(key)

        parts.append("ORDER BY tn.name, m.key")
        sql = " ".join(parts)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Data table access
    # ------------------------------------------------------------------
    def get_table(
        self,
        tool_name: str,
        message_id: int,
    ) -> List[Dict[str, Any]]:
        """Query all rows from the data table for
        ``(tool_name, message_id)``.

        Raises ``ValueError`` if the schema is not registered (i.e.
        no ``logToDb``/``logToDbBulk`` call with this pair was ever
        made).

        Returns a list of dicts keyed by column name.
        """
        schema = self.schema_for(tool_name, message_id)
        if schema is None:
            raise ValueError(
                f"No schema registered for ({tool_name}, {message_id}). "
                f"Known schemas: {self.table_names()}"
            )
        return self.get_table_by_name(schema["table_name"])

    def get_table_by_name(self, table_name: str) -> List[Dict[str, Any]]:
        """Query all rows from a data table by its SQLite name
        (e.g. ``"EXA_10"``).

        Returns a list of dicts keyed by column name.  If the table
        has no rows, returns an empty list.
        """
        if not self.table_exists(table_name):
            # The table may not exist if no rows were ever inserted.
            # Return empty rather than fail.
            return []
        rows = self._conn.execute(
            f"SELECT * FROM [{table_name}]"
        ).fetchall()
        return [dict(r) for r in rows]

    def __getitem__(self, key: str) -> List[Dict[str, Any]]:
        """``db["EXA_10"]`` is equivalent to
        ``db.get_table_by_name("EXA_10")``.
        """
        return self.get_table_by_name(key)

    def query(self, tool_name: str, message_id: int, **filters: Any) -> List[Dict[str, Any]]:
        """Query rows with optional WHERE filters.

        Example::

            # All rows where id == 1
            db.query("EXA", 8, id=1)

        Only supports equality filters.  Column names must match
        the schema.  If no filters are given, this is equivalent to
        :meth:`get_table`.

        Raises ``ValueError`` if a filter column is not in the schema.
        """
        schema = self.schema_for(tool_name, message_id)
        if schema is None:
            raise ValueError(
                f"No schema registered for ({tool_name}, {message_id})."
            )

        col_names = [c[0] for c in schema["columns"]]
        for col in filters:
            if col not in col_names:
                raise ValueError(
                    f"Column '{col}' is not in schema ({tool_name}, {message_id}). "
                    f"Available columns: {col_names}"
                )

        if not filters:
            return self.get_table(tool_name, message_id)

        where_clauses = [f"[{col}] = ?" for col in filters]
        where_sql = " AND ".join(where_clauses)
        params = tuple(filters[col] for col in filters)

        table = schema["table_name"]
        sql = f"SELECT * FROM [{table}] WHERE {where_sql}"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<DatabaseLog path={self._path!r}>"

    def __len__(self) -> int:
        """Return the number of registered data tables."""
        return len(self.table_names())


# -----------------------------------------------------------------------
# Standalone helper: quick inspection from the command line
# -----------------------------------------------------------------------
def inspect(db_path: str) -> None:
    """Print a human-readable summary of a database log file."""
    with DatabaseLog(db_path) as db:
        print(f"Database: {db.path}")
        print()

        # Tools
        print("=== Tool Names ===")
        for t in db.tools():
            print(f"  {t['tool_id']}: {t['name']}")
        print()

        # Schemas
        print("=== Registered Schemas ===")
        for s in db.schemas():
            cols = ", ".join(f"{n} {t}" for n, t in s["columns"])
            print(f"  {s['table_name']}: ({cols})")
        print()

        # Data
        print("=== Data Summary ===")
        for s in db.schemas():
            rows = db.get_table_by_name(s["table_name"])
            print(f"  {s['table_name']}: {len(rows)} row(s)")
            if rows:
                for r in rows[:5]:
                    print(f"    {dict(r)}")
                if len(rows) > 5:
                    print(f"    ... ({len(rows) - 5} more)")

        print()

        # Metadata
        meta = db.metadata()
        if meta:
            print("=== Metadata ===")
            for m in meta:
                print(f"  [{m['tool_name']}] {m['key']} = {m['value']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path-to-db-log>", file=sys.stderr)
        sys.exit(1)

    inspect(sys.argv[1])
