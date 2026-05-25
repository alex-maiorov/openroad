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
import time
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

        Indexing is separated from data generation:
          Step 0 — indexes on raw source tables
          Step 1 — gradient metrics table
          Step 2 — density forces table
          Step 3 — path signatures table
          Step 4 — indexes on derived tables

        Parameters
        ----------
        force_rebuild : bool
            Drop existing derived tables before rebuilding.
        batch_size : int
            Iterations per batch (default 25).
        """
        t_total = time.time()
        print(f"  [GplDb.preprocess]  batch_size={batch_size}")

        if force_rebuild:
            self._drop_derived_tables()

        # ── Step 0: source indexes ────────────────────────────────
        t0 = time.time()
        self._preprocess_source_indexes()
        print(f"    [source_index] done — {time.time()-t0:.1f}s")

        # ── Step 1: gradient metrics ──────────────────────────────
        t0 = time.time()
        self._preprocess_gradient_metrics(batch_size)
        print(f"    [gradient_metrics] done — {time.time()-t0:.1f}s")

        # ── Step 2: density forces ────────────────────────────────
        t0 = time.time()
        self._preprocess_density_forces(batch_size)
        print(f"    [density_forces] done — {time.time()-t0:.1f}s")

        # ── Step 3: path signatures ───────────────────────────────
        t0 = time.time()
        self._preprocess_path_signatures(batch_size)
        print(f"    [path_signatures] done — {time.time()-t0:.1f}s")

        # ── Step 4: derived table indexes ─────────────────────────
        t0 = time.time()
        self._preprocess_derived_indexes()
        print(f"    [derived_index] done — {time.time()-t0:.1f}s")

        self.conn.commit()
        print(f"  [GplDb.preprocess] done — {time.time()-t_total:.1f}s total.\n")

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
        print("    [preprocess] force rebuild — dropping derived tables …")
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

    # ── Step 0: Source table indexes ───────────────────────────────

    def _preprocess_source_indexes(self):
        """Create indexes on every raw source table for fast lookup.

        Creates an index on ``Iter`` for every table that has an
        ``Iter`` column, as well as composite indexes on the natural
        lookup key (e.g. ``CellId``, ``PathId``, ``BinIdx``).  Static
        tables (netlist, cell_static_info) get indexes on their
        primary key only.

        Tables that do not exist in the database are silently skipped
        with a warning — this handles databases that are missing
        optional sources (e.g. timing/routability gradients).

        Uses ``CREATE INDEX IF NOT EXISTS``, so calling this
        repeatedly is idempotent and cheap.

        Each table's indexing is timed independently — there are no
        inter-dependencies between index operations.
        """
        def _maybe_index(table: str, col_groups):
            if not self._exists(table):
                print(f"    [source_index] table '{table}' not found "
                      f"— skipping.")
                return
            print(f"    [source_index] creating indexes on {table} …")
            t0 = time.time()
            _create_indexes(self.conn, table, col_groups)
            print(f"    [source_index]   • {table}: {time.time()-t0:.1f}s")

        # ── Tables WITH an Iter column ──────────────────────────
        _maybe_index("gpl_iteration_scalars",
                     ["Iter"])
        _maybe_index("gpl_bin_grid",
                     ["Iter", "BinIdx", "Iter, BinIdx"])
        _maybe_index("gpl_cell_dense_gradients",
                     ["Iter", "CellId", "Iter, CellId"])
        _maybe_index("gpl_cell_timing_gradients",
                     ["Iter", "CellId", "Iter, CellId"])
        _maybe_index("gpl_cell_routability_gradients",
                     ["Iter", "CellId", "Iter, CellId"])
        _maybe_index("gpl_cell_positions",
                     ["Iter", "CellId", "Iter, CellId"])
        _maybe_index("gpl_path_slacks",
                     ["Iter", "PathId", "Iter, PathId"])
        _maybe_index("gpl_path_cells",
                     ["Iter", "PathId", "CellId",
                      "Iter, PathId", "Iter, CellId"])
        _maybe_index("gpl_netlist_cells",
                     ["Iter", "CellId", "Iter, CellId"])
        _maybe_index("gpl_netlist_nets",
                     ["Iter", "NetId", "Iter, NetId"])
        _maybe_index("gpl_netlist_connectivity",
                     ["Iter", "PinId", "NetId", "CellId",
                      "Iter, PinId", "Iter, NetId", "Iter, CellId"])

        # ── Static tables (no Iter column) ──────────────────────
        _maybe_index("gpl_cell_static_info",
                     ["CellId"])

    # ── Step 1: Gradient metrics ───────────────────────────────────

    def _preprocess_gradient_metrics(self, batch_size: int):
        """Create ``gpl_cell_gradient_metrics`` if missing."""
        if self._table_has_data(GRADIENT_METRICS_TABLE):
            print(f"    [gradient_metrics] table already populated — skip.")
            return

        cur = self.conn.execute(
            "SELECT COALESCE(MIN(Iter),0), COALESCE(MAX(Iter),0) "
            "FROM gpl_cell_dense_gradients"
        )
        row = cur.fetchone()
        if not row or row[1] == 0:
            print("    [gradient_metrics] source table empty — skip.")
            return
        imin, imax = int(row[0]), int(row[1])
        print(f"    [gradient_metrics]  iters {imin}–{imax}  "
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
        has_timing = self._exists("gpl_cell_timing_gradients")
        for start in range(imin, imax + 1, batch_size):
            end = min(start + batch_size - 1, imax)

            dense = pd.read_sql_query(
                "SELECT Iter, CellId, WlX, WlY "
                "FROM gpl_cell_dense_gradients WHERE Iter BETWEEN ? AND ?",
                self.conn, params=(start, end),
            )
            if has_timing:
                timing = pd.read_sql_query(
                    "SELECT Iter, CellId, TimX, TimY "
                    "FROM gpl_cell_timing_gradients WHERE Iter BETWEEN ? AND ?",
                    self.conn, params=(start, end),
                )
                df = pd.merge(dense, timing, on=["Iter", "CellId"], how="left")
                df["TimX"] = pd.to_numeric(df["TimX"], errors="coerce").fillna(0.0)
                df["TimY"] = pd.to_numeric(df["TimY"], errors="coerce").fillna(0.0)
            else:
                df = dense.copy()
                df["TimX"] = 0.0
                df["TimY"] = 0.0

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
            print(f"    [gradient_metrics] batch {start:>5d}–{end:<5d}  "
                  f"wrote {len(out):>8d}  (total {total})")
            del dense, df, out
            if has_timing:
                del timing

        print(f"    [gradient_metrics] done — {total} rows.")

    # ── Step 2: Density forces ─────────────────────────────────────

    def _dense_has_column(self, col: str) -> bool:
        """Check whether ``gpl_cell_dense_gradients`` has column *col*."""
        if not self._exists("gpl_cell_dense_gradients"):
            return False
        cols = {row[1] for row in self.conn.execute(
            "PRAGMA table_info(gpl_cell_dense_gradients)"
        ).fetchall()}
        return col in cols

    def _preprocess_density_forces(self, batch_size: int):
        """Create ``gpl_cell_density_forces`` if missing.

        If the source table ``gpl_cell_dense_gradients`` already
        contains ``DensX`` / ``DensY`` columns (dumped directly from
        the placer), a fast SQL-only path is used.  Otherwise falls
        back to the Summed-Area-Table reconstruction.
        """
        if self._table_has_data(DENSITY_FORCES_TABLE):
            print(f"    [density_forces] table already populated — skip.")
            return

        if self._dense_has_column("DensX") and self._dense_has_column("DensY"):
            self._preprocess_density_forces_fast(batch_size)
        else:
            self._preprocess_density_forces_sat(batch_size)

    def _preprocess_density_forces_fast(self, batch_size: int):
        """Populate ``gpl_cell_density_forces`` directly from the
        ``DensX`` / ``DensY`` columns already dumped by the placer.

        The placer writes the *raw* density gradient (without the
        ``DensityPenalty`` multiplier) — consistent with how ``WlX``
        / ``WlY`` are raw.  We multiply by ``DensityPenalty`` from
        ``gpl_iteration_scalars`` to obtain the effective force.
        """
        cur = self.conn.execute(
            "SELECT COALESCE(MIN(Iter),0), COALESCE(MAX(Iter),0) "
            "FROM gpl_cell_dense_gradients"
        )
        row = cur.fetchone()
        if not row or row[1] == 0:
            print("    [density_forces] source table empty — skip.")
            return
        imin, imax = int(row[0]), int(row[1])
        print(f"    [density_forces]  fast-path  iters {imin}–{imax}  "
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

        total = 0
        for start in range(imin, imax + 1, batch_size):
            end = min(start + batch_size - 1, imax)
            _t0 = time.time()
            n = self.conn.execute(
                f"""
                INSERT INTO [{DENSITY_FORCES_TABLE}]
                    (Iter, CellId, EstDensityForceX,
                     EstDensityForceY, EstDensityForceMag)
                SELECT
                    g.Iter,
                    g.CellId,
                    g.DensX * s.DensityPenalty,
                    g.DensY * s.DensityPenalty,
                    SQRT(g.DensX * g.DensX + g.DensY * g.DensY)
                        * s.DensityPenalty
                FROM gpl_cell_dense_gradients g
                JOIN gpl_iteration_scalars s ON g.Iter = s.Iter
                WHERE g.Iter BETWEEN ? AND ?
                """,
                (start, end),
            ).rowcount
            total += n
            self.conn.commit()
            print(f"    [density_forces] batch {start:>5d}–{end:<5d}  "
                  f"wrote {n:>8d}   ({time.time() - _t0:.2f}s)")
        print(f"    [density_forces] done — {total} rows (fast-path).")

    def _preprocess_density_forces_sat(self, batch_size: int):
        """Fallback: SAT reconstruction from ``gpl_bin_grid``.

        Used when the source ``gpl_cell_dense_gradients`` table was
        written by an older placer that does not include ``DensX`` /
        ``DensY`` columns.
        """
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
            print("    [density_forces] missing region metadata — skip.")
            return

        cur = self.conn.execute(
            "SELECT COALESCE(MIN(Iter),0), COALESCE(MAX(Iter),0) "
            "FROM gpl_cell_dense_gradients"
        )
        row = cur.fetchone()
        if not row or row[1] == 0:
            print("    [density_forces] source table empty — skip.")
            return
        imin, imax = int(row[0]), int(row[1])
        print(f"    [density_forces]  SAT-fallback  iters {imin}–{imax}  "
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
            print("    [density_forces] cell static info empty — skip.")
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
            print(f"    [density_forces] batch {start:>5d}–{end:<5d}  "
                  f"wrote {len(out):>8d}  (total {total})")
            del bins, cells, out, sat_x, sat_y

        print(f"    [density_forces] done — {total} rows (SAT-fallback).")

    # ── Step 3: Path signatures ────────────────────────────────────

    def _preprocess_path_signatures(self, batch_size: int):
        """Create ``gpl_path_signatures`` via two-pass batch."""
        if self._table_has_data(PATH_SIGNATURES_TABLE):
            print(f"    [path_signatures] table already populated — skip.")
            return

        if not self._exists("gpl_path_cells"):
            print("    [path_signatures] source table gpl_path_cells not found — skip.")
            return
        try:
            cur = self.conn.execute(
                "SELECT COALESCE(MIN(Iter),0), COALESCE(MAX(Iter),0) "
                "FROM gpl_path_cells"
            )
        except pd.errors.DatabaseError:
            print("    [path_signatures] source table not found — skip.")
            return
        row = cur.fetchone()
        if not row or row[1] == 0:
            print("    [path_signatures] source table empty — skip.")
            return
        imin, imax = int(row[0]), int(row[1])
        print(f"    [path_signatures]  iters {imin}–{imax}  "
              f"batch={batch_size}")

        # ── Pass 1: collect unique cell-id sequences ───────────────
        print("    [path_signatures] pass 1 — collecting unique sequences …")
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
            print("    [path_signatures] no sequences found — skip.")
            return

        seq_to_id = {seq: i for i, seq in enumerate(unique_seqs)}
        del unique_seqs
        print(f"    [path_signatures]   • {len(seq_to_id)} unique physical paths.")

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

        print(f"    [path_signatures] done — {total} rows.")

    # ── Step 4: Derived table indexes ────────────────────────────

    def _preprocess_derived_indexes(self):
        """Create indexes on all three derived tables.

        Separated from the data-generation steps so that indexing is
        batched and timed independently.  Uses ``CREATE INDEX IF NOT
        EXISTS`` so repeated calls are idempotent and cheap.

        Each table's indexing is timed independently — there are no
        inter-dependencies between index operations.
        """
        for table, col_groups in [
            (GRADIENT_METRICS_TABLE,
             ["Iter", "CellId", "Iter, CellId"]),
            (DENSITY_FORCES_TABLE,
             ["Iter", "CellId", "Iter, CellId"]),
            (PATH_SIGNATURES_TABLE,
             ["Iter, PathId", "PhysicalPathId"]),
        ]:
            if not self._table_has_data(table):
                print(f"    [derived_index] {table} empty — skip.")
                continue
            print(f"    [derived_index] creating indexes on {table} …")
            t0 = time.time()
            _create_indexes(self.conn, table, col_groups)
            print(f"    [derived_index]   • {table}: {time.time()-t0:.1f}s")

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
        """Per-cell dense vectors: position, WL/density/sum gradients.

        Columns (newer placer builds include all; older builds omit
        trailing columns):
        ``Iter, CellId, PosX, PosY, WlX, WlY[, DensX, DensY, SumX, SumY]``.
        """
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

    def top_timing_cells(
        self,
        top_n: int = 10,
        iter_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Return the top N cells ranked by total timing force magnitude.
        
        Returns a DataFrame with columns (CellId, TotalTimForce).
        """
        if not self._exists(GRADIENT_METRICS_TABLE):
            return pd.DataFrame()
            
        sql = f"SELECT CellId, SUM(mag_tim) AS TotalTimForce FROM [{GRADIENT_METRICS_TABLE}]"
        params = []
        if iter_range is not None:
            sql += " WHERE Iter BETWEEN ? AND ?"
            params.extend((int(iter_range[0]), int(iter_range[1])))
            
        sql += " GROUP BY CellId ORDER BY TotalTimForce DESC LIMIT ?"
        params.append(int(top_n))
        
        return self.query(sql, tuple(params))

    def cell_path_counts(
        self,
        cell_ids: List[int],
        iter_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Return the number of unique violating paths passing through each cell per iteration.
        
        Returns a DataFrame with columns (Iter, CellId, PathCount).
        """
        if not self._exists("gpl_path_cells") or not cell_ids:
            return pd.DataFrame()
            
        placeholders = ",".join("?" for _ in cell_ids)
        sql = "SELECT Iter, CellId, COUNT(DISTINCT PathId) AS PathCount FROM gpl_path_cells"
        
        clauses = [f"CellId IN ({placeholders})"]
        params = [int(c) for c in cell_ids]
        
        if iter_range is not None:
            clauses.append("Iter BETWEEN ? AND ?")
            params.extend((int(iter_range[0]), int(iter_range[1])))
            
        sql += " WHERE " + " AND ".join(clauses)
        sql += " GROUP BY Iter, CellId ORDER BY Iter, CellId"
        
        return self.query(sql, tuple(params))

    def cell_force_alignment(
        self,
        cell_ids: List[int],
        iter_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Return the dot product of the Effective force and Timing force over time.
        
        Effective force = Wirelength + Timing + Density.
        Returns a DataFrame with columns (Iter, CellId, Alignment).
        """
        if not cell_ids:
            return pd.DataFrame()
            
        placeholders = ",".join("?" for _ in cell_ids)
        
        # We use LEFT JOINs to construct the effective force completely inside SQLite
        sql = f"""
            SELECT 
                d.Iter, d.CellId,
                (d.WlX + COALESCE(t.TimX, 0.0) + COALESCE(df.EstDensityForceX, 0.0)) * COALESCE(t.TimX, 0.0) +
                (d.WlY + COALESCE(t.TimY, 0.0) + COALESCE(df.EstDensityForceY, 0.0)) * COALESCE(t.TimY, 0.0) AS Alignment
            FROM gpl_cell_dense_gradients d
            LEFT JOIN gpl_cell_timing_gradients t 
                ON d.Iter = t.Iter AND d.CellId = t.CellId
            LEFT JOIN [{DENSITY_FORCES_TABLE}] df 
                ON d.Iter = df.Iter AND d.CellId = df.CellId
            WHERE d.CellId IN ({placeholders})
        """
        params = [int(c) for c in cell_ids]
        
        if iter_range is not None:
            sql += " AND d.Iter BETWEEN ? AND ?"
            params.extend((int(iter_range[0]), int(iter_range[1])))
            
        sql += " ORDER BY d.Iter, d.CellId"
        
        return self.query(sql, tuple(params))

    def paths_containing_cells(
        self,
        cell_ids: List[int],
        iter_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Return the full cell sequences for all paths that contain at least one of the specified cells.
        
        Returns a DataFrame with columns (Iter, PathId, CellId, PathSeq).
        """
        if not self._exists("gpl_path_cells") or not cell_ids:
            return pd.DataFrame()
            
        placeholders = ",".join("?" for _ in cell_ids)
        params = [int(c) for c in cell_ids]
        
        iter_clause = ""
        if iter_range is not None:
            iter_clause = " AND Iter BETWEEN ? AND ?"
            params.extend((int(iter_range[0]), int(iter_range[1])))
            
        sql = f"""
            SELECT p.Iter, p.PathId, p.CellId, p.PathSeq
            FROM gpl_path_cells p
            JOIN (
                SELECT DISTINCT Iter, PathId 
                FROM gpl_path_cells 
                WHERE CellId IN ({placeholders}){iter_clause}
            ) target ON p.Iter = target.Iter AND p.PathId = target.PathId
        """
        
        # Add iter filter on outer query too for performance
        if iter_range is not None:
            sql += " WHERE p.Iter BETWEEN ? AND ?"
            params.extend((int(iter_range[0]), int(iter_range[1])))
            
        sql += " ORDER BY p.Iter, p.PathId, p.PathSeq"
        
        return self.query(sql, tuple(params))

    def similarity_path_data(
        self,
        cell_ids: List[int],
        iter_range: Optional[Tuple[int, int]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch optimized data for path similarity analysis.
        
        Returns:
            active_paths: DataFrame(Iter, PhysicalPathId)
            path_seqs: DataFrame(PhysicalPathId, CellSequence)
        """
        if not self._exists("gpl_path_cells") or not self._exists(PATH_SIGNATURES_TABLE) or not cell_ids:
            return pd.DataFrame(), pd.DataFrame()
            
        placeholders = ",".join("?" for _ in cell_ids)
        params = [int(c) for c in cell_ids]
        
        iter_clause = ""
        if iter_range is not None:
            iter_clause = " AND p.Iter BETWEEN ? AND ?"
            params.extend((int(iter_range[0]), int(iter_range[1])))

        # 1. Active Physical Paths per iteration
        sql_active = f"""
            SELECT DISTINCT sm.Iter, sm.PhysicalPathId
            FROM gpl_path_cells p
            JOIN [{PATH_SIGNATURES_TABLE}] sm
              ON p.Iter = sm.Iter AND p.PathId = sm.PathId
            WHERE p.CellId IN ({placeholders}){iter_clause}
            ORDER BY sm.Iter, sm.PhysicalPathId
        """
        active_paths = self.query(sql_active, tuple(params))
        
        if active_paths.empty:
            return active_paths, pd.DataFrame()

        # 2. Get unique physical path sequences
        unique_phys_ids = active_paths["PhysicalPathId"].unique().tolist()
        phys_placeholders = ",".join("?" for _ in unique_phys_ids)
        phys_params = [int(p) for p in unique_phys_ids]
        
        sql_seqs = f"""
            SELECT sm.PhysicalPathId,
                   GROUP_CONCAT(pc.CellId ORDER BY pc.PathSeq) AS CellSequence
            FROM (
                SELECT PhysicalPathId, MIN(Iter) as MinIter, MIN(PathId) as MinPathId
                FROM [{PATH_SIGNATURES_TABLE}]
                WHERE PhysicalPathId IN ({phys_placeholders})
                GROUP BY PhysicalPathId
            ) sm
            JOIN gpl_path_cells pc
              ON pc.Iter = sm.MinIter AND pc.PathId = sm.MinPathId
            GROUP BY sm.PhysicalPathId
        """
        path_seqs = self.query(sql_seqs, tuple(phys_params))
        
        return active_paths, path_seqs

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
            paths.  This check uses **OR** logic with
            ``max_similarity``: a candidate is only rejected when it
            fails **both** this check AND the similarity check against
            the same already-selected path.  0 means no filtering.
        max_similarity : float
            Maximum ratio of shared nodes (CellIds) between a
            candidate path and any already-selected path.  Computed as
            ``|cells(new) ∩ cells(existing)| / |cells(new)|``.
            This check uses **OR** logic with ``min_separation``: a
            candidate is only rejected when it fails **both** checks
            against the same already-selected path.  1.0 means no
            filtering.

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

        # ── 3. Combined OR filtering ──────────────────────────
        # A candidate is selected if it differs from ALL
        # already-selected paths in EITHER slack
        # (≥ min_separation) OR cell composition
        # (overlap ≤ max_similarity).  Only paths that fail
        # BOTH checks against SOME already-selected path are
        # rejected.  When only one criterion's threshold is
        # active (the other at its default) it works as a
        # single-dimension filter for backward compatibility.

        use_slack = (min_separation > 0)
        use_sim = (max_similarity < 1.0 and has_sig)

        # Shortcut: if neither filter is active, take top_n
        # directly (avoids the "both_violated stays True" trap).
        if not use_slack and not use_sim:
            kept = worst.index[:top_n].tolist()
        else:
            # Pre-fetch cell sets for all candidates when
            # similarity is enabled.
            if use_sim:
                all_ids = worst.index.tolist()
                ph = ",".join("?" for _ in all_ids)
                cell_df = self.query(f"""
                    SELECT sm.PhysicalPathId, pc.CellId
                    FROM [{PATH_SIGNATURES_TABLE}] sm
                    JOIN gpl_path_cells pc
                      ON sm.PathId = pc.PathId AND sm.Iter = pc.Iter
                    WHERE sm.PhysicalPathId IN ({ph})
                """, all_ids)
                path_cells = {
                    pid: set(grp["CellId"].values)
                    for pid, grp in cell_df.groupby("PhysicalPathId")
                }

            kept: list = []
            for phys_id in worst.index:
                if len(kept) >= top_n:
                    break

                reject = False
                for existing_id in kept:
                    # Start by assuming this candidate fails ALL
                    # dimensions vs *existing_id*.  If EITHER the
                    # slack or the cell check passes, we mark the
                    # pair as OK.
                    both_violated = True

                    if use_slack:
                        diff = abs(worst[phys_id] - worst[existing_id])
                        if diff >= min_separation:
                            both_violated = False

                    if use_sim:
                        new_set = path_cells.get(phys_id, set())
                        old_set = path_cells.get(existing_id, set())
                        if len(new_set) > 0:
                            overlap = len(new_set & old_set) / len(new_set)
                            if overlap <= max_similarity:
                                both_violated = False

                    if both_violated:
                        reject = True
                        break

                if not reject:
                    kept.append(phys_id)

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

    # ================================================================
    #  NETLIST GRAPH ACCESSORS  (gpl_netlist_* tables)
    # ================================================================

    def netlist_cells(
        self,
        iter_range: Optional[Tuple[int, int]] = None,
        cell_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Return ``gpl_netlist_cells`` rows.

        Parameters
        ----------
        iter_range : tuple of (int, int), optional
            Restrict to iterations in [min,max] inclusive.
        cell_ids : list of int, optional
            If provided, restrict to these cell IDs.

        Returns
        -------
        pd.DataFrame
            Columns: Iter, CellId, Cx, Cy, Width, Height, IsMacro,
            IsLocked, NumInstances, NumPins.
        """
        sql, params = self._make_select(
            "gpl_netlist_cells",
            iter_range=iter_range,
            cell_ids=cell_ids,
            order_by="Iter, CellId",
        )
        if not sql:
            return pd.DataFrame()
        return self.query(sql, params)

    def netlist_nets(
        self,
        iter_range: Optional[Tuple[int, int]] = None,
        net_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Return ``gpl_netlist_nets`` rows.

        Parameters
        ----------
        iter_range : tuple of (int, int), optional
            Restrict to iterations in [min,max] inclusive.
        net_ids : list of int, optional
            If provided, restrict to these net IDs.

        Returns
        -------
        pd.DataFrame
            Columns: Iter, NetId, NumPins, TimingWeight, CustomWeight.
        """
        sql, params = self._make_select(
            "gpl_netlist_nets",
            iter_range=iter_range,
            order_by="Iter, NetId",
        )
        if not sql:
            return pd.DataFrame()
        # _make_select doesn't support net_ids directly, append manually
        if net_ids is not None:
            if not net_ids:
                return pd.DataFrame()
            placeholders = ",".join("?" for _ in net_ids)
            sql += f" AND NetId IN ({placeholders})"
            params = tuple(params) + tuple(net_ids)
        return self.query(sql, params)

    def netlist_connectivity(
        self,
        iter_range: Optional[Tuple[int, int]] = None,
        pin_ids: Optional[List[int]] = None,
        cell_ids: Optional[List[int]] = None,
        net_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Return ``gpl_netlist_connectivity`` rows with optional filters.

        Parameters
        ----------
        iter_range : tuple of (int, int), optional
            Restrict to iterations in [min,max] inclusive.
        pin_ids : list of int, optional
            Restrict to these pin IDs.
        cell_ids : list of int, optional
            Restrict to pins belonging to these cells.
        net_ids : list of int, optional
            Restrict to pins on these nets.

        Returns
        -------
        pd.DataFrame
            Columns: Iter, PinId, NetId, CellId, PinCx, PinCy.
        """
        table = "gpl_netlist_connectivity"
        clauses: List[str] = []
        params: list = []

        if iter_range is not None:
            clauses.append("Iter BETWEEN ? AND ?")
            params.extend(iter_range)

        for col, ids in [("PinId", pin_ids),
                          ("CellId", cell_ids),
                          ("NetId", net_ids)]:
            if ids is not None:
                if not ids:
                    return pd.DataFrame()
                placeholders = ",".join("?" for _ in ids)
                clauses.append(f"{col} IN ({placeholders})")
                params.extend(ids)

        if not clauses:
            return self.query(
                f"SELECT * FROM [{table}] ORDER BY Iter, PinId")

        sql = (f"SELECT * FROM [{table}] WHERE {' AND '.join(clauses)} "
               f"ORDER BY Iter, PinId")
        return self.query(sql, params)

    def _has_netlist_tables(self) -> bool:
        """True when all three netlist tables exist and have rows."""
        for table in ("gpl_netlist_cells", "gpl_netlist_nets",
                      "gpl_netlist_connectivity"):
            if not self._table_has_data(table):
                return False
        return True

    # ── Graph traversal ───────────────────────────────────────────

    def netlist_cell_nets(
        self,
        cell_ids: List[int],
        iter_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Nets connected to the given cells (one row per pin).

        Returns columns: CellId, NetId, PinId, PinCx, PinCy.
        """
        if not cell_ids:
            return pd.DataFrame()
        return self.netlist_connectivity(
            cell_ids=cell_ids, iter_range=iter_range)

    def netlist_net_cells(
        self,
        net_ids: List[int],
        iter_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Cells connected to the given nets (one row per pin).

        Returns columns: NetId, CellId, PinId, PinCx, PinCy.
        """
        if not net_ids:
            return pd.DataFrame()
        return self.netlist_connectivity(
            net_ids=net_ids, iter_range=iter_range)

    def netlist_cell_neighbors(
        self,
        cell_ids: List[int],
        include_self: bool = False,
        iter_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Cells that share at least one net with the given cells.

        Uses a self-join on ``gpl_netlist_connectivity``, scoped to
        *iter_range* when provided (default: all iterations).

        Returns columns: CellId, NeighborCellId, SharedNetId.
        """
        if not cell_ids:
            return pd.DataFrame()

        placeholders = ",".join("?" for _ in cell_ids)
        op = "!=" if not include_self else "IS NOT"

        iter_clause = ""
        params = list(cell_ids)
        if iter_range is not None:
            iter_clause = " AND a.Iter BETWEEN ? AND ?"
            params.extend(iter_range)

        return self.query(f"""
            SELECT a.CellId, b.CellId AS NeighborCellId,
                   a.NetId AS SharedNetId
            FROM [gpl_netlist_connectivity] a
            JOIN [gpl_netlist_connectivity] b
              ON a.NetId = b.NetId AND a.Iter = b.Iter
            WHERE a.CellId IN ({placeholders})
              AND a.CellId {op} b.CellId{iter_clause}
            ORDER BY a.CellId, b.CellId
        """, params)

    def netlist_net_neighbors(
        self,
        net_ids: List[int],
        include_self: bool = False,
        iter_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Nets that share at least one cell with the given nets.

        Uses a self-join on ``gpl_netlist_connectivity``, scoped to
        *iter_range* when provided.

        Returns columns: NetId, NeighborNetId, SharedCellId.
        """
        if not net_ids:
            return pd.DataFrame()

        placeholders = ",".join("?" for _ in net_ids)
        op = "!=" if not include_self else "IS NOT"

        iter_clause = ""
        params = list(net_ids)
        if iter_range is not None:
            iter_clause = " AND a.Iter BETWEEN ? AND ?"
            params.extend(iter_range)

        return self.query(f"""
            SELECT a.NetId, b.NetId AS NeighborNetId,
                   a.CellId AS SharedCellId
            FROM [gpl_netlist_connectivity] a
            JOIN [gpl_netlist_connectivity] b
              ON a.CellId = b.CellId AND a.Iter = b.Iter
            WHERE a.NetId IN ({placeholders})
              AND a.NetId {op} b.NetId{iter_clause}
            ORDER BY a.NetId, b.NetId
        """, params)

    def netlist_adjacency(
        self,
        cell_ids: Optional[List[int]] = None,
        iter_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Cell-to-cell adjacency list (one row per edge).

        Two cells are adjacent if they share at least one net.
        Scoped to *iter_range* when provided.

        Returns columns: CellId, NeighborCellId, SharedNetId.
        """
        if cell_ids is not None and not cell_ids:
            return pd.DataFrame()

        iter_clause = ""
        iter_params: list = []
        if iter_range is not None:
            iter_clause = " AND a.Iter BETWEEN ? AND ?"
            iter_params = list(iter_range)

        if cell_ids is not None:
            placeholders = ",".join("?" for _ in cell_ids)
            return self.query(f"""
                SELECT a.CellId, b.CellId AS NeighborCellId,
                       a.NetId AS SharedNetId
                FROM [gpl_netlist_connectivity] a
                JOIN [gpl_netlist_connectivity] b
                  ON a.NetId = b.NetId AND a.Iter = b.Iter
                WHERE a.CellId IN ({placeholders})
                  AND a.CellId != b.CellId{iter_clause}
                ORDER BY a.CellId, b.CellId
            """, list(cell_ids) + iter_params)

        return self.query(f"""
            SELECT a.CellId, b.CellId AS NeighborCellId,
                   a.NetId AS SharedNetId
            FROM [gpl_netlist_connectivity] a
            JOIN [gpl_netlist_connectivity] b
              ON a.NetId = b.NetId AND a.Iter = b.Iter
            WHERE a.CellId != b.CellId{iter_clause}
            ORDER BY a.CellId, b.CellId
        """, iter_params)

    # ── Graph metrics ─────────────────────────────────────────────

    def netlist_cell_degree(
        self,
        cell_ids: Optional[List[int]] = None,
        iter_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Number of distinct nets connected to each cell.

        Scoped to *iter_range* when provided.

        Returns columns: Iter, CellId, Degree.
        """
        table = "gpl_netlist_connectivity"

        iter_clause = ""
        params: list = []
        if iter_range is not None:
            iter_clause = " AND Iter BETWEEN ? AND ?"
            params = list(iter_range)

        if cell_ids is not None:
            if not cell_ids:
                return pd.DataFrame()
            placeholders = ",".join("?" for _ in cell_ids)
            return self.query(f"""
                SELECT Iter, CellId, COUNT(DISTINCT NetId) AS Degree
                FROM [{table}]
                WHERE CellId IN ({placeholders}){iter_clause}
                GROUP BY Iter, CellId
                ORDER BY Iter, CellId
            """, list(cell_ids) + params)

        return self.query(f"""
            SELECT Iter, CellId, COUNT(DISTINCT NetId) AS Degree
            FROM [{table}]
            WHERE 1=1{iter_clause}
            GROUP BY Iter, CellId
            ORDER BY Iter, CellId
        """, params)

    def netlist_net_degree(
        self,
        net_ids: Optional[List[int]] = None,
        iter_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Pin count per net (computed from connectivity).

        Scoped to *iter_range* when provided.

        Returns columns: Iter, NetId, Degree.
        """
        table = "gpl_netlist_connectivity"

        iter_clause = ""
        params: list = []
        if iter_range is not None:
            iter_clause = " AND Iter BETWEEN ? AND ?"
            params = list(iter_range)

        if net_ids is not None:
            if not net_ids:
                return pd.DataFrame()
            placeholders = ",".join("?" for _ in net_ids)
            return self.query(f"""
                SELECT Iter, NetId, COUNT(*) AS Degree
                FROM [{table}]
                WHERE NetId IN ({placeholders}){iter_clause}
                GROUP BY Iter, NetId
                ORDER BY Iter, NetId
            """, list(net_ids) + params)

        return self.query(f"""
            SELECT Iter, NetId, COUNT(*) AS Degree
            FROM [{table}]
            WHERE 1=1{iter_clause}
            GROUP BY Iter, NetId
            ORDER BY Iter, NetId
        """, params)

    # ── Cross-table joins ─────────────────────────────────────────

    def netlist_cells_with_net_counts(
        self,
        iter_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Cell properties augmented with live net-degree.

        Scoped to *iter_range* when provided.

        Returns columns: Iter, CellId, Cx, Cy, Width, Height, IsMacro,
        IsLocked, NumInstances, NumPins, NetDegree.
        """
        iter_join = ""
        params: list = []
        if iter_range is not None:
            iter_join = " AND c.Iter = d.Iter"
            params = list(iter_range)

        where = ""
        if iter_range is not None:
            where = " WHERE c.Iter BETWEEN ? AND ?"

        return self.query(f"""
            SELECT c.Iter, c.CellId, c.Cx, c.Cy, c.Width, c.Height,
                   c.IsMacro, c.IsLocked, c.NumInstances,
                   c.NumPins, COALESCE(d.Degree, 0) AS NetDegree
            FROM [gpl_netlist_cells] c
            LEFT JOIN (
                SELECT Iter, CellId, COUNT(DISTINCT NetId) AS Degree
                FROM [gpl_netlist_connectivity]
                GROUP BY Iter, CellId
            ) d ON c.CellId = d.CellId{iter_join}
            {where}
            ORDER BY c.Iter, c.CellId
        """, params)

    def netlist_net_cell_coordinates(
        self,
        net_ids: List[int],
        iter_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Cells on the given nets, with coordinates.

        Scoped to *iter_range* when provided.

        Returns columns: NetId, CellId, Cx, Cy, Width, Height,
        PinId, PinCx, PinCy.
        """
        if not net_ids:
            return pd.DataFrame()

        placeholders = ",".join("?" for _ in net_ids)
        iter_clause = ""
        params = list(net_ids)
        if iter_range is not None:
            iter_clause = " AND conn.Iter BETWEEN ? AND ?"
            params.extend(iter_range)

        return self.query(f"""
            SELECT conn.NetId, conn.CellId, c.Cx, c.Cy,
                   c.Width, c.Height,
                   conn.PinId, conn.PinCx, conn.PinCy
            FROM [gpl_netlist_connectivity] conn
            JOIN [gpl_netlist_cells] c
              ON conn.CellId = c.CellId AND conn.Iter = c.Iter
            WHERE conn.NetId IN ({placeholders}){iter_clause}
            ORDER BY conn.NetId, conn.CellId
        """, params)


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


def _select_by_ids(
    db: DbConnection,
    table: str,
    col: str,
    ids: Optional[List[int]],
) -> pd.DataFrame:
    """``SELECT * FROM [table]`` optionally filtered by ``col IN (...)``."""
    if ids is not None:
        if not ids:
            return pd.DataFrame()
        placeholders = ",".join("?" for _ in ids)
        return db.query(
            f"SELECT * FROM [{table}] WHERE {col} IN ({placeholders}) "
            f"ORDER BY {col}",
            ids,
        )
    return db.query(f"SELECT * FROM [{table}] ORDER BY {col}")
