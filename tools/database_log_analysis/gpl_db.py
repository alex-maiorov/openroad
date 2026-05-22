"""GPL (Global Placement) database interface.

``GplDb`` is a fully self-contained class that:
  1. Opens a GPL SQLite database,
  2. Runs any missing preprocessing automatically (gradient metrics,
     density forces, path signatures),
  3. Provides thin SQL-wrapper methods for all data access.

Usage
-----
>>> from database_log_analysis import GplDb
>>> gpl = GplDb("tmp/test_gpl.sqlite")      # auto-preprocess
>>> df = gpl.cell_dense_gradients(iter_range=(0, 25))
>>> df = gpl.cell_gradient_metrics(cell_ids=[0, 1, 2])
>>> gpl.close()

>>> gpl = GplDb("tmp/test_gpl.sqlite", must_be_preprocessed=True)  # read-only
"""

import sqlite3
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from .core import DbConnection

# ── Derived table names (single source of truth) ───────────────────
GRADIENT_METRICS_TABLE = "gpl_cell_gradient_metrics"
DENSITY_FORCES_TABLE = "gpl_cell_density_forces"
PATH_SIGNATURES_TABLE = "gpl_path_signatures"


class GplDb(DbConnection):
    """Database interface for GPL (Global Placement) analysis.

    Parameters
    ----------
    db_path : str
        Path to a GPL SQLite database.
    must_be_preprocessed : bool
        If True, open read-only and raise when derived tables are
        missing.  If False (default), open read-write and run any
        missing preprocessing steps.
    batch_size : int
        Iterations per batch during preprocessing (default 25).
        Smaller values reduce peak RAM at the cost of more passes.
    """

    def __init__(
        self,
        db_path: str,
        must_be_preprocessed: bool = False,
        batch_size: int = 25,
    ):
        if must_be_preprocessed:
            super().__init__(db_path, read_only=True)
            self._ensure_preprocessed()
        else:
            super().__init__(db_path, read_only=False)
            self.preprocess(batch_size=batch_size)

    # ================================================================
    #  PREPROCESSING  (inlined — no separate module needed)
    # ================================================================

    def preprocess(self, force_rebuild: bool = False, batch_size: int = 25):
        """Run all missing preprocessing steps.

        Each step checks whether its target table already exists and
        has data, so the method is **idempotent**.  Pass
        ``force_rebuild=True`` to drop and re-create all three tables.

        Parameters
        ----------
        force_rebuild : bool
            Drop existing derived tables before rebuilding.
        batch_size : int
            Iterations per batch (default 25).
        """
        print(f"  [GplDb.preprocess]  batch_size={batch_size}")

        if force_rebuild:
            self._drop_derived_tables()

        self._preprocess_gradient_metrics(batch_size)
        self._preprocess_density_forces(batch_size)
        self._preprocess_path_signatures(batch_size)

        self.conn.commit()
        print("  [GplDb.preprocess] done.\n")

    def _table_has_data(self, name: str) -> bool:
        """True when *name* exists in sqlite_master and has rows."""
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name=?",
            (name,),
        )
        if not cur.fetchone():
            return False
        cur = self.conn.execute(f"SELECT COUNT(*) FROM [{name}]")
        return cur.fetchone()[0] > 0

    def _drop_derived_tables(self):
        print("    Force rebuild — dropping tables …")
        for t in (GRADIENT_METRICS_TABLE, DENSITY_FORCES_TABLE,
                  PATH_SIGNATURES_TABLE):
            self.conn.execute(f"DROP TABLE IF EXISTS [{t}]")
        self.conn.commit()
        # Re-scan schema so _table_has_data sees the changes
        self._cache_schema()

    def _ensure_preprocessed(self):
        """Raise ``RuntimeError`` if any derived table is missing."""
        required = [
            GRADIENT_METRICS_TABLE,
            DENSITY_FORCES_TABLE,
            PATH_SIGNATURES_TABLE,
        ]
        missing = [t for t in required if not self._table_has_data(t)]
        if missing:
            raise RuntimeError(
                f"Database is not preprocessed — missing tables: {missing}. "
                "Open GplDb with must_be_preprocessed=False to auto-build."
            )

    # ── Step 1: Gradient metrics ───────────────────────────────────

    def _preprocess_gradient_metrics(self, batch_size: int):
        """Create ``gpl_cell_gradient_metrics`` if missing."""
        if self._table_has_data(GRADIENT_METRICS_TABLE):
            print(f"    [{GRADIENT_METRICS_TABLE}] exists — skip.")
            return

        cur = self.conn.execute(
            "SELECT COALESCE(MIN(Iter),0), COALESCE(MAX(Iter),0) "
            "FROM gpl_cell_dense_gradients"
        )
        row = cur.fetchone()
        if not row or row[1] == 0:
            print("    gpl_cell_dense_gradients empty — skip gradient metrics.")
            return
        imin, imax = int(row[0]), int(row[1])
        print(f"    [{GRADIENT_METRICS_TABLE}]  iters {imin}–{imax}  "
              f"batch={batch_size}")

        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS [{GRADIENT_METRICS_TABLE}] (
                Iter INTEGER,
                CellId INTEGER,
                mag_wl REAL,
                mag_tim REAL,
                dot_wl_tim REAL,
                opposition REAL
            )
        """)
        self.conn.commit()

        total = 0
        for start in range(imin, imax + 1, batch_size):
            end = min(start + batch_size - 1, imax)

            dense = pd.read_sql_query(
                "SELECT Iter, CellId, WlX, WlY "
                "FROM gpl_cell_dense_gradients WHERE Iter BETWEEN ? AND ?",
                self.conn, params=(start, end),
            )
            timing = pd.read_sql_query(
                "SELECT Iter, CellId, TimX, TimY "
                "FROM gpl_cell_timing_gradients WHERE Iter BETWEEN ? AND ?",
                self.conn, params=(start, end),
            )

            df = pd.merge(dense, timing, on=["Iter", "CellId"], how="left")
            df["TimX"] = pd.to_numeric(df["TimX"], errors="coerce").fillna(0.0)
            df["TimY"] = pd.to_numeric(df["TimY"], errors="coerce").fillna(0.0)

            df["mag_wl"] = np.sqrt(df["WlX"] ** 2 + df["WlY"] ** 2)
            df["mag_tim"] = np.sqrt(df["TimX"] ** 2 + df["TimY"] ** 2)
            df["dot_wl_tim"] = df["WlX"] * df["TimX"] + df["WlY"] * df["TimY"]

            prod = df["mag_wl"] * df["mag_tim"]
            df["opposition"] = np.where(prod > 0, -df["dot_wl_tim"] / prod, 0.0)

            out = df[["Iter", "CellId", "mag_wl", "mag_tim",
                       "dot_wl_tim", "opposition"]]
            out.to_sql(GRADIENT_METRICS_TABLE, self.conn,
                       if_exists="append", index=False)
            total += len(out)
            print(f"      batch {start:>5d}–{end:<5d}  "
                  f"wrote {len(out):>8d}  (total {total})")
            del dense, timing, df, out

        _create_indexes(self.conn, GRADIENT_METRICS_TABLE,
                        ["Iter", "CellId", "Iter, CellId"])
        print(f"    [{GRADIENT_METRICS_TABLE}] done — {total} rows.")

    # ── Step 2: Density forces ─────────────────────────────────────

    def _preprocess_density_forces(self, batch_size: int):
        """Create ``gpl_cell_density_forces`` if missing."""
        if self._table_has_data(DENSITY_FORCES_TABLE):
            print(f"    [{DENSITY_FORCES_TABLE}] exists — skip.")
            return

        # Grid metadata
        try:
            _get = lambda k: float(self.conn.execute(
                "SELECT value FROM metadata WHERE key=?", (k,)
            ).fetchone()[0])
            lx = _get("region_core_lx")
            ly = _get("region_core_ly")
            bin_size_x = _get("region_core_binSizeX")
            bin_size_y = _get("region_core_binSizeY")
            bin_cnt_x = int(_get("region_core_binCntX"))
            bin_cnt_y = int(_get("region_core_binCntY"))
        except (TypeError, IndexError):
            print("    Missing region_core_* metadata — skip density forces.")
            return

        cur = self.conn.execute(
            "SELECT COALESCE(MIN(Iter),0), COALESCE(MAX(Iter),0) "
            "FROM gpl_cell_dense_gradients"
        )
        row = cur.fetchone()
        if not row or row[1] == 0:
            print("    gpl_cell_dense_gradients empty — skip density forces.")
            return
        imin, imax = int(row[0]), int(row[1])
        print(f"    [{DENSITY_FORCES_TABLE}]  iters {imin}–{imax}  "
              f"batch={batch_size}")

        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS [{DENSITY_FORCES_TABLE}] (
                Iter INTEGER,
                CellId INTEGER,
                EstDensityForceX REAL,
                EstDensityForceY REAL,
                EstDensityForceMag REAL
            )
        """)
        self.conn.commit()

        static = pd.read_sql_query(
            "SELECT CellId, Width, Height FROM gpl_cell_static_info",
            self.conn,
        )
        if static.empty:
            print("    gpl_cell_static_info empty — skip density forces.")
            return

        bin_area = bin_size_x * bin_size_y
        total = 0

        for start in range(imin, imax + 1, batch_size):
            end = min(start + batch_size - 1, imax)
            n_iters = end - start + 1

            bins = pd.read_sql_query(
                "SELECT Iter, BinIdx, ElectroFieldX, ElectroFieldY "
                "FROM gpl_bin_grid WHERE Iter BETWEEN ? AND ? "
                "ORDER BY Iter, BinIdx",
                self.conn, params=(start, end),
            )
            if bins.empty:
                continue

            fx = bins["ElectroFieldX"].values.astype(np.float32).reshape(
                n_iters, bin_cnt_y, bin_cnt_x
            )
            fy = bins["ElectroFieldY"].values.astype(np.float32).reshape(
                n_iters, bin_cnt_y, bin_cnt_x
            )

            sat_x = np.zeros((n_iters, bin_cnt_y + 1, bin_cnt_x + 1),
                             dtype=np.float64)
            sat_y = np.zeros((n_iters, bin_cnt_y + 1, bin_cnt_x + 1),
                             dtype=np.float64)
            sat_x[:, 1:, 1:] = np.cumsum(np.cumsum(fx, axis=1), axis=2) * bin_area
            sat_y[:, 1:, 1:] = np.cumsum(np.cumsum(fy, axis=1), axis=2) * bin_area

            cells = pd.read_sql_query(
                "SELECT g.Iter, g.CellId, g.PosX, g.PosY, i.DensityPenalty "
                "FROM gpl_cell_dense_gradients g "
                "JOIN gpl_iteration_scalars i ON g.Iter = i.Iter "
                "WHERE g.Iter BETWEEN ? AND ?",
                self.conn, params=(start, end),
            )
            if cells.empty:
                continue

            cells = pd.merge(cells, static, on="CellId")
            pos_x = cells["PosX"].values
            pos_y = cells["PosY"].values
            w = cells["Width"].values
            h = cells["Height"].values
            pen = cells["DensityPenalty"].values

            iter_to_local = {it: i for i, it in
                             enumerate(range(start, end + 1))}
            li = np.array([iter_to_local[it] for it in cells["Iter"]])

            lx_b = (pos_x - w / 2 - lx) / bin_size_x
            ly_b = (pos_y - h / 2 - ly) / bin_size_y
            ux_b = (pos_x + w / 2 - lx) / bin_size_x
            uy_b = (pos_y + h / 2 - ly) / bin_size_y

            lx_b = np.clip(lx_b, 0, bin_cnt_x)
            ly_b = np.clip(ly_b, 0, bin_cnt_y)
            ux_b = np.clip(ux_b, 0, bin_cnt_x)
            uy_b = np.clip(uy_b, 0, bin_cnt_y)

            def _bilerp(sat, idx, x, y):
                ix = np.floor(x).astype(int)
                iy = np.floor(y).astype(int)
                fx = x - ix
                fy = y - iy
                ix = np.clip(ix, 0, bin_cnt_x - 1)
                iy = np.clip(iy, 0, bin_cnt_y - 1)
                v00 = sat[idx, iy, ix]
                v10 = sat[idx, iy, ix + 1]
                v01 = sat[idx, iy + 1, ix]
                v11 = sat[idx, iy + 1, ix + 1]
                return ((1 - fx) * (1 - fy) * v00 +
                        fx * (1 - fy) * v10 +
                        (1 - fx) * fy * v01 +
                        fx * fy * v11)

            fxl = _bilerp(sat_x, li, lx_b, ly_b)
            fxu = _bilerp(sat_x, li, ux_b, ly_b)
            fxl2 = _bilerp(sat_x, li, lx_b, uy_b)
            fxu2 = _bilerp(sat_x, li, ux_b, uy_b)
            force_x = (fxu2 - fxl2 - fxu + fxl) * pen

            fyl = _bilerp(sat_y, li, lx_b, ly_b)
            fyu = _bilerp(sat_y, li, ux_b, ly_b)
            fyl2 = _bilerp(sat_y, li, lx_b, uy_b)
            fyu2 = _bilerp(sat_y, li, ux_b, uy_b)
            force_y = (fyu2 - fyl2 - fyu + fyl) * pen

            out = pd.DataFrame({
                "Iter": cells["Iter"],
                "CellId": cells["CellId"],
                "EstDensityForceX": force_x.astype(np.float32),
                "EstDensityForceY": force_y.astype(np.float32),
                "EstDensityForceMag": np.sqrt(
                    force_x ** 2 + force_y ** 2
                ).astype(np.float32),
            })
            out.to_sql(DENSITY_FORCES_TABLE, self.conn,
                       if_exists="append", index=False)
            total += len(out)
            print(f"      batch {start:>5d}–{end:<5d}  "
                  f"wrote {len(out):>8d}  (total {total})")
            del bins, cells, out, sat_x, sat_y

        _create_indexes(self.conn, DENSITY_FORCES_TABLE,
                        ["Iter", "CellId", "Iter, CellId"])
        print(f"    [{DENSITY_FORCES_TABLE}] done — {total} rows.")

    # ── Step 3: Path signatures ────────────────────────────────────

    def _preprocess_path_signatures(self, batch_size: int):
        """Create ``gpl_path_signatures`` via two-pass batch."""
        if self._table_has_data(PATH_SIGNATURES_TABLE):
            print(f"    [{PATH_SIGNATURES_TABLE}] exists — skip.")
            return

        try:
            cur = self.conn.execute(
                "SELECT COALESCE(MIN(Iter),0), COALESCE(MAX(Iter),0) "
                "FROM gpl_path_cells"
            )
        except pd.errors.DatabaseError:
            print("    gpl_path_cells not found — skip path signatures.")
            return
        row = cur.fetchone()
        if not row or row[1] == 0:
            print("    gpl_path_cells empty — skip path signatures.")
            return
        imin, imax = int(row[0]), int(row[1])
        print(f"    [{PATH_SIGNATURES_TABLE}]  iters {imin}–{imax}  "
              f"batch={batch_size}")

        # ── Pass 1: collect unique cell-id sequences ───────────────
        print("      Pass 1 — collecting unique sequences …")
        unique_seqs: set = set()
        for start in range(imin, imax + 1, batch_size):
            end = min(start + batch_size - 1, imax)
            df = pd.read_sql_query(
                "SELECT PathId, CellId, Iter, PathSeq "
                "FROM gpl_path_cells WHERE Iter BETWEEN ? AND ? "
                "ORDER BY Iter, PathId, PathSeq",
                self.conn, params=(start, end),
            )
            if df.empty:
                continue
            seqs = df.groupby(["PathId", "Iter"])["CellId"].apply(tuple)
            unique_seqs.update(seqs)

        if not unique_seqs:
            print("      No sequences found — skip.")
            return

        seq_to_id = {seq: i for i, seq in enumerate(unique_seqs)}
        del unique_seqs
        print(f"      Found {len(seq_to_id)} unique physical paths.")

        # ── Pass 2: write mapping ──────────────────────────────────
        self.conn.execute(f"DROP TABLE IF EXISTS [{PATH_SIGNATURES_TABLE}]")
        self.conn.execute(f"""
            CREATE TABLE [{PATH_SIGNATURES_TABLE}] (
                Iter INTEGER,
                PathId INTEGER,
                PhysicalPathId INTEGER
            )
        """)
        self.conn.commit()

        total = 0
        for start in range(imin, imax + 1, batch_size):
            end = min(start + batch_size - 1, imax)
            df = pd.read_sql_query(
                "SELECT PathId, CellId, Iter, PathSeq "
                "FROM gpl_path_cells WHERE Iter BETWEEN ? AND ? "
                "ORDER BY Iter, PathId, PathSeq",
                self.conn, params=(start, end),
            )
            if df.empty:
                continue
            seqs = df.groupby(["PathId", "Iter"])["CellId"].apply(tuple)
            mapping = seqs.apply(lambda x: seq_to_id[x]).reset_index(
                name="PhysicalPathId"
            )
            mapping.to_sql(PATH_SIGNATURES_TABLE, self.conn,
                           if_exists="append", index=False)
            total += len(mapping)

        _create_indexes(self.conn, PATH_SIGNATURES_TABLE,
                        ["Iter, PathId", "PhysicalPathId"])
        print(f"    [{PATH_SIGNATURES_TABLE}] done — {total} rows.")

    # ================================================================
    #  QUERY HELPERS
    # ================================================================

    def _exists(self, table: str) -> bool:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        return cur.fetchone() is not None

    @staticmethod
    def _make_select(
        table: str,
        iter_range: Optional[Tuple[int, int]] = None,
        cell_ids: Optional[List[int]] = None,
        path_ids: Optional[List[int]] = None,
        order_by: Optional[str] = None,
    ) -> Tuple[str, tuple]:
        """Build a parameterised ``SELECT * FROM table WHERE ``.

        Returns ``("", ())`` when no rows can match (e.g. empty IN list)
        so the caller can short-circuit to an empty DataFrame.
        """
        clauses: List[str] = []
        params: list = []

        if iter_range is not None:
            clauses.append("Iter BETWEEN ? AND ?")
            params.extend(iter_range)

        if cell_ids is not None:
            if not cell_ids:
                return "", ()
            placeholders = ",".join("?" for _ in cell_ids)
            clauses.append(f"CellId IN ({placeholders})")
            params.extend(cell_ids)

        if path_ids is not None:
            if not path_ids:
                return "", ()
            placeholders = ",".join("?" for _ in path_ids)
            clauses.append(f"PathId IN ({placeholders})")
            params.extend(path_ids)

        sql = f"SELECT * FROM [{table}]"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if order_by:
            sql += f" ORDER BY {order_by}"
        return sql, tuple(params)

    # ================================================================
    #  RAW TABLE ACCESS  (thin SQL wrappers)
    # ================================================================

    def cell_static_info(self) -> pd.DataFrame:
        """``gpl_cell_static_info`` — cell dimensions, macro/locked flags."""
        return self.query("SELECT * FROM gpl_cell_static_info")

    def cell_dense_gradients(
        self,
        iter_range: Optional[Tuple[int, int]] = None,
        cell_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Wirelength gradient components (dense — every cell every iter)."""
        sql, params = self._make_select(
            "gpl_cell_dense_gradients",
            iter_range=iter_range,
            cell_ids=cell_ids,
            order_by="Iter, CellId",
        )
        if not sql:
            return pd.DataFrame()
        return self.query(sql, params)

    def cell_timing_gradients(
        self,
        iter_range: Optional[Tuple[int, int]] = None,
        cell_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Timing gradient components (sparse — missing == zero force)."""
        if not self._exists("gpl_cell_timing_gradients"):
            return pd.DataFrame()
        sql, params = self._make_select(
            "gpl_cell_timing_gradients",
            iter_range=iter_range,
            cell_ids=cell_ids,
            order_by="Iter, CellId",
        )
        if not sql:
            return pd.DataFrame()
        return self.query(sql, params)

    def iteration_scalars(self) -> pd.DataFrame:
        """Per-iteration scalar values (step length, penalty, overflow)."""
        return self.query(
            "SELECT * FROM gpl_iteration_scalars ORDER BY Iter"
        )

    def bin_grid(
        self, iter_range: Optional[Tuple[int, int]] = None
    ) -> pd.DataFrame:
        """Electrostatic bin grid (field + density) per iteration."""
        sql, params = self._make_select(
            "gpl_bin_grid",
            iter_range=iter_range,
            order_by="Iter, BinIdx",
        )
        if not sql:
            return pd.DataFrame()
        return self.query(sql, params)

    # ================================================================
    #  PRECOMPUTED DERIVED TABLES
    # ================================================================

    def cell_gradient_metrics(
        self,
        iter_range: Optional[Tuple[int, int]] = None,
        cell_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Precomputed gradient metrics (mag_wl, mag_tim, dot_wl_tim,
        opposition)."""
        if not self._exists(GRADIENT_METRICS_TABLE):
            return pd.DataFrame()
        sql, params = self._make_select(
            GRADIENT_METRICS_TABLE,
            iter_range=iter_range,
            cell_ids=cell_ids,
            order_by="Iter, CellId",
        )
        if not sql:
            return pd.DataFrame()
        return self.query(sql, params)

    def cell_density_forces(
        self,
        iter_range: Optional[Tuple[int, int]] = None,
        cell_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Precomputed estimated density (electrostatic) forces per cell."""
        if not self._exists(DENSITY_FORCES_TABLE):
            return pd.DataFrame()
        sql, params = self._make_select(
            DENSITY_FORCES_TABLE,
            iter_range=iter_range,
            cell_ids=cell_ids,
            order_by="Iter, CellId",
        )
        if not sql:
            return pd.DataFrame()
        return self.query(sql, params)

    def path_slacks(self) -> pd.DataFrame:
        """Overall path slack per (PathId, Iter)."""
        if not self._exists("gpl_path_slacks"):
            return pd.DataFrame()
        return self.query("SELECT * FROM gpl_path_slacks ORDER BY Iter, PathId")

    def path_cells(
        self,
        iter_range: Optional[Tuple[int, int]] = None,
        path_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Cell membership along timing paths (CellId, PathSeq, per-stage slack)."""
        if not self._exists("gpl_path_cells"):
            return pd.DataFrame()
        sql, params = self._make_select(
            "gpl_path_cells",
            iter_range=iter_range,
            path_ids=path_ids,
            order_by="Iter, PathId, PathSeq",
        )
        if not sql:
            return pd.DataFrame()
        return self.query(sql, params)

    def path_signatures(
        self,
        iter_range: Optional[Tuple[int, int]] = None,
        path_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Mapping from unstable (PathId, Iter) to stable PhysicalPathId."""
        if not self._exists(PATH_SIGNATURES_TABLE):
            return pd.DataFrame()
        sql, params = self._make_select(
            PATH_SIGNATURES_TABLE,
            iter_range=iter_range,
            path_ids=path_ids,
            order_by="Iter, PathId",
        )
        if not sql:
            return pd.DataFrame()
        return self.query(sql, params)

    def physical_paths(self) -> pd.DataFrame:
        """Unique physical paths with their cell-id sequences.

        Returns one row per distinct physical path with a comma-separated
        ``CellSequence`` column (ordered by PathSeq).
        """
        if not self._exists(PATH_SIGNATURES_TABLE):
            return pd.DataFrame()
        return self.query(f"""
            SELECT sm.PhysicalPathId,
                   GROUP_CONCAT(pc.CellId ORDER BY pc.PathSeq) AS CellSequence
            FROM [{PATH_SIGNATURES_TABLE}] sm
            JOIN gpl_path_cells pc
              ON sm.PathId = pc.PathId AND sm.Iter = pc.Iter
            GROUP BY sm.PhysicalPathId
            ORDER BY sm.PhysicalPathId
        """)

    # ================================================================
    #  SQL-COMPUTED DERIVED DATA
    #  (Only methods whose logic genuinely benefits from being close to
    #   the database engine.  Simple aggregates are left to the user
    #   via the raw accessors above.)
    # ================================================================

    def cell_movements(
        self,
        iter_range: Optional[Tuple[int, int]] = None,
        cell_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Displacement of each cell between consecutive logged iterations.

        Uses the SQL ``LAG()`` window function so the computation stays
        in the engine (no need to load 22M rows into pandas just to
        call ``.diff()``).  Returns columns ``DeltaX``, ``DeltaY``,
        ``Distance``.
        """
        where, params = "", ()
        if iter_range is not None or cell_ids is not None:
            _sql, params = self._make_select(
                "gpl_cell_dense_gradients",
                iter_range=iter_range,
                cell_ids=cell_ids,
            )
            if not _sql:
                return pd.DataFrame()
            where = _sql[len("SELECT * FROM [gpl_cell_dense_gradients]"):]

        return self.query(f"""
            SELECT Iter, CellId, PosX, PosY,
                   COALESCE(
                       PosX - LAG(PosX) OVER (PARTITION BY CellId ORDER BY Iter),
                       0
                   ) AS DeltaX,
                   COALESCE(
                       PosY - LAG(PosY) OVER (PARTITION BY CellId ORDER BY Iter),
                       0
                   ) AS DeltaY,
                   ROUND(SQRT(
                       COALESCE(POWER(PosX - LAG(PosX) OVER
                           (PARTITION BY CellId ORDER BY Iter), 2), 0) +
                       COALESCE(POWER(PosY - LAG(PosY) OVER
                           (PARTITION BY CellId ORDER BY Iter), 2), 0)
                   ), 6) AS Distance
            FROM [gpl_cell_dense_gradients]{where}
            ORDER BY CellId, Iter
        """, params)

    def worst_paths_history(
        self,
        top_n: int = 10,
        iter_range: Optional[Tuple[int, int]] = None,
        min_separation: float = 0.0,
        max_similarity: float = 1.0,
    ) -> List[pd.DataFrame]:
        """Return per-path DataFrames for the most critical timing paths.

        Paths are identified by their stable ``PhysicalPathId`` (when
        the signature table exists) or fall back to the unstable
        ``PathId``.

        Parameters
        ----------
        top_n : int
            Maximum number of paths to return.
        iter_range : tuple of (int, int) or None
            Iteration window over which to evaluate "worst" (i.e. the
            slack used for ranking is the minimum within this window).
            None means use all iterations.
        min_separation : float
            Minimum slack difference (in ps) between any two returned
            paths.  If a candidate's worst slack is within this
            threshold of an already-selected path, the candidate is
            skipped.  0 means no filtering.
        max_similarity : float
            Maximum ratio of shared nodes (CellIds) between a
            candidate path and any already-selected path.  Computed as
            ``|cells(new) ∩ cells(existing)| / |cells(new)|``.
            Candidates exceeding this threshold are skipped.  1.0
            means no filtering.

        Returns
        -------
        list of pd.DataFrame
            One DataFrame per selected path, each containing columns
            ``(PathId, Iter, Slack, PhysicalPathId)`` sorted by Iter.
            The list is ordered from worst (most negative) slack to
            least critical.  Empty list when no data is available.
        """
        if not self._exists("gpl_path_slacks") or top_n <= 0:
            return []

        has_sig = self._exists(PATH_SIGNATURES_TABLE)

        # ── 1. Fetch all (PathId, Iter, Slack, PhysicalPathId) ─────
        if has_sig:
            query = """
                SELECT ps.PathId, ps.Iter, ps.Slack, sm.PhysicalPathId
                FROM gpl_path_slacks ps
                JOIN [{}] sm ON ps.PathId = sm.PathId AND ps.Iter = sm.Iter
            """.format(PATH_SIGNATURES_TABLE)
        else:
            query = """
                SELECT ps.PathId, ps.Iter, ps.Slack,
                       ps.PathId AS PhysicalPathId
                FROM gpl_path_slacks ps
            """

        params = []
        if iter_range is not None:
            query += " WHERE ps.Iter BETWEEN ? AND ?"
            params = list(iter_range)

        df = self.query(query, params)
        if df.empty:
            return []

        # ── 2. Worst slack per physical path ────────────────────────
        worst = df.groupby("PhysicalPathId")["Slack"].min().sort_values()

        # ── 3. Apply min_separation filter ──────────────────────────
        # Walk paths from worst → best, keep only those whose slack
        # differs by at least min_separation from the last kept path.
        kept: list = []
        for phys_id in worst.index:
            if min_separation > 0 and kept:
                last_worst = worst[kept[-1]]
                if abs(worst[phys_id] - last_worst) < min_separation:
                    continue
            kept.append(phys_id)
            if len(kept) >= top_n:
                break

        if not kept:
            return []

        # ── 4. Apply max_similarity (node overlap) filter ───────────
        if max_similarity < 1.0 and has_sig:
            placeholders = ",".join("?" for _ in kept)
            cell_df = self.query(f"""
                SELECT sm.PhysicalPathId, pc.CellId
                FROM [{PATH_SIGNATURES_TABLE}] sm
                JOIN gpl_path_cells pc
                  ON sm.PathId = pc.PathId AND sm.Iter = pc.Iter
                WHERE sm.PhysicalPathId IN ({placeholders})
            """, kept)

            path_cells = {
                pid: set(grp["CellId"].values)
                for pid, grp in cell_df.groupby("PhysicalPathId")
            }

            filtered: list = []
            for phys_id in kept:
                new_set = path_cells.get(phys_id, set())
                too_similar = False
                for existing_id in filtered:
                    existing_set = path_cells.get(existing_id, set())
                    if len(new_set) > 0:
                        overlap = len(new_set & existing_set) / len(new_set)
                        if overlap > max_similarity:
                            too_similar = True
                            break
                if not too_similar:
                    filtered.append(phys_id)
                    if len(filtered) >= top_n:
                        break
            kept = filtered

        if not kept:
            return []

        # ── 5. Fetch full history for kept paths ────────────────────
        placeholders = ",".join("?" for _ in kept)

        if has_sig:
            hist_query = f"""
                SELECT ps.PathId, ps.Iter, ps.Slack, sm.PhysicalPathId
                FROM gpl_path_slacks ps
                JOIN [{PATH_SIGNATURES_TABLE}] sm
                  ON ps.PathId = sm.PathId AND ps.Iter = sm.Iter
                WHERE sm.PhysicalPathId IN ({placeholders})
                ORDER BY sm.PhysicalPathId, ps.Iter
            """
        else:
            hist_query = f"""
                SELECT ps.PathId, ps.Iter, ps.Slack,
                       ps.PathId AS PhysicalPathId
                FROM gpl_path_slacks ps
                WHERE ps.PathId IN ({placeholders})
                ORDER BY ps.PathId, ps.Iter
            """

        df_hist = self.query(hist_query, kept)

        # ── 6. Split into per-path DataFrames ──────────────────────
        result = []
        for phys_id in kept:
            pdf = df_hist[df_hist["PhysicalPathId"] == phys_id].copy()
            if not pdf.empty:
                result.append(pdf)

        return result


# ====================================================================
#  Module-level helpers
# ====================================================================

def _create_indexes(conn: sqlite3.Connection, table: str, col_groups):
    """Create indexes in *table* for each group in *col_groups*."""
    for cols in col_groups:
        name = f"idx_{table}_{cols.replace(', ', '_')}"
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS [{name}] ON [{table}] ({cols})"
        )
