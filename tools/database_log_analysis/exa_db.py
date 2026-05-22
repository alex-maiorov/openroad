"""EXA (Example) database interface.

``ExaDb`` follows the same pattern as ``GplDb`` but for the example
tool tables.  There is no shared base-class logic beyond
``DbConnection`` — each tool has its own thin-SQL-wrapper class.
"""

import pandas as pd
from .core import DbConnection


class ExaDb(DbConnection):
    """Database interface for the EXA (example) tool.

    Parameters
    ----------
    db_path : str
        Path to an EXA SQLite database.
    """

    def __init__(self, db_path: str):
        # EXA has no preprocessing — open read-only.
        super().__init__(db_path, read_only=True)

    # ── Raw table access ───────────────────────────────────────────

    def st_nonbulk(self) -> pd.DataFrame:
        """Single-threaded, non-bulk logged values."""
        return self.query("SELECT * FROM exa_st_nonbulk ORDER BY id")

    def st_bulk(self) -> pd.DataFrame:
        """Single-threaded, bulk logged values."""
        return self.query("SELECT * FROM exa_st_bulk ORDER BY id")

    def mt_nonbulk(self) -> pd.DataFrame:
        """Multi-threaded, non-bulk logged values."""
        return self.query("SELECT * FROM exa_mt_nonbulk ORDER BY thread_id, iter")

    def mt_bulk(self) -> pd.DataFrame:
        """Multi-threaded, bulk logged values."""
        return self.query("SELECT * FROM exa_mt_bulk ORDER BY thread_id, iter")

    # ── SQL-computed derived data ──────────────────────────────────

    def average_st_bulk_val(self) -> float:
        """Average ``val`` across all single-threaded bulk logs."""
        df = self.query("SELECT AVG(val) AS avg_val FROM exa_st_bulk")
        return float(df["avg_val"].iloc[0]) if not df.empty else 0.0

    def magic_metric(self) -> pd.DataFrame:
        """Group single-threaded bulk logs by ``type`` and compute
        ``magic_score = mean(val) * 100``."""
        return self.query("""
            SELECT type,
                   ROUND(AVG(val) * 100, 4) AS magic_score
            FROM exa_st_bulk
            GROUP BY type
            ORDER BY type
        """)
