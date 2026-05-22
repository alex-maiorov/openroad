import sqlite3
import numpy as np
import pandas as pd
import os
import sys

# Add the parent directory to sys.path if run as a script
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from database_log_analysis.core import DatabaseLog
else:
    from .core import DatabaseLog

class GplPreprocessor:
    """Library module for preprocessing GPL database logs."""
    
    @staticmethod
    def run(db_path: str, force_rebuild=False, batch_size=25):
        """Perform preprocessing on the database at the given path.
        
        Args:
            db_path: Path to the SQLite database.
            force_rebuild: If True, drop and rebuild existing derived tables.
            batch_size: Number of iterations to process per batch.
                        Smaller values reduce peak RAM at the cost of more
                        passes over the data.  Default 25 keeps RAM well
                        under 1 GiB for typical designs (100k+ cells, 500+
                        iterations).  For very large designs (>500k cells)
                        or memory-constrained environments, use 10 or lower.
        """
        print(f"Opening {db_path} for R/W preprocessing...")
        print(f"Batch size: {batch_size} iterations/chunk")
        conn = sqlite3.connect(db_path)
        
        if force_rebuild:
            print("Force rebuild requested. Dropping existing derived tables...")
            conn.execute("DROP TABLE IF EXISTS gpl_derived_gradients")
            conn.execute("DROP TABLE IF EXISTS path_signature_map")
            conn.commit()
            
        # Check if gradient table already exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='gpl_derived_gradients'")
        if not cursor.fetchone():
            GplPreprocessor._generate_derived_gradients(conn, batch_size=batch_size)
        else:
            print("Table 'gpl_derived_gradients' already exists. Skipping.")

        # Check if path signature table already exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='path_signature_map'")
        if not cursor.fetchone():
            GplPreprocessor._generate_path_signatures(conn, batch_size=batch_size)
        else:
            print("Table 'path_signature_map' already exists. Skipping.")

        conn.commit()
        conn.close()
        print("Preprocessing complete.")
    
    @staticmethod
    def _generate_derived_gradients(conn, batch_size=25):
        """Compute derived gradient metrics in batched iteration chunks.
        
        Instead of loading ALL rows (potentially 22M+) into RAM at once, this
        method queries the distinct iteration range and processes a sliding
        window of *batch_size* iterations per round.  Each chunk reads only
        its dense + timing rows, merges in-memory, writes the result, and
        discards the DataFrames before advancing.
        
        Peak memory per batch is roughly:
          rows_per_iter * batch_size * (cols_dense + cols_timing) * 8 bytes
        plus pandas overhead (~2x).  With batch_size=25 and 100k cells that
        is ~150 MB, well within a 1 GiB budget.
        """
        # Get the iteration range from the dense table (always present)
        cursor = conn.execute(
            "SELECT COALESCE(MIN(Iter),0), COALESCE(MAX(Iter),0) "
            "FROM gpl_cell_dense_gradients"
        )
        row = cursor.fetchone()
        if not row or row[1] == 0:
            print("  gpl_cell_dense_gradients is empty. Skipping.")
            return
        iter_min, iter_max = int(row[0]), int(row[1])
        print(f"  Iteration range: {iter_min} to {iter_max}")
        print(f"  Processing in batches of {batch_size} iteration(s)...")

        # Create the target table (empty) up front; each batch appends rows.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gpl_derived_gradients (
                Iter INTEGER,
                CellId INTEGER,
                mag_wl REAL,
                mag_tim REAL,
                dot_wl_tim REAL,
                opposition REAL
            )
        """)
        conn.commit()

        total_rows = 0
        for start in range(iter_min, iter_max + 1, batch_size):
            end = min(start + batch_size - 1, iter_max)

            # --- Load batch of dense gradients ---
            print(f"  Batch iters {start:>5d}–{end:<5d}: reading dense ...", end=" ")
            df_dense = pd.read_sql_query(
                "SELECT * FROM gpl_cell_dense_gradients "
                "WHERE Iter BETWEEN ? AND ?",
                conn, params=(start, end)
            )
            print(f"{len(df_dense):>8d} rows", end="")

            # --- Load corresponding timing gradients (sparse) ---
            print(",  timing ...", end=" ")
            df_timing = pd.read_sql_query(
                "SELECT * FROM gpl_cell_timing_gradients "
                "WHERE Iter BETWEEN ? AND ?",
                conn, params=(start, end)
            )
            print(f"{len(df_timing):>8d} rows", end="")

            # --- Merge ---
            df = pd.merge(df_dense, df_timing, on=['Iter', 'CellId'], how='left')
            # Convert to numeric then fill NaN with 0.
            # When df_timing is empty for a batch, the columns come through as
            # object dtype and .fillna() would trigger a FutureWarning about
            # silent downcasting.  Using pd.to_numeric avoids the warning and
            # is explicit about the intended dtype.
            df['TimX'] = pd.to_numeric(df['TimX'], errors='coerce').fillna(0.0)
            df['TimY'] = pd.to_numeric(df['TimY'], errors='coerce').fillna(0.0)

            # --- Compute derived columns ---
            df['mag_wl'] = np.sqrt(df['WlX']**2 + df['WlY']**2)
            df['mag_tim'] = np.sqrt(df['TimX']**2 + df['TimY']**2)
            df['dot_wl_tim'] = df['WlX'] * df['TimX'] + df['WlY'] * df['TimY']
            mag_prod = df['mag_wl'] * df['mag_tim']
            df['opposition'] = np.where(mag_prod > 0, -df['dot_wl_tim'] / mag_prod, 0)

            derived_df = df[['Iter', 'CellId', 'mag_wl', 'mag_tim',
                             'dot_wl_tim', 'opposition']]

            # --- Write (append) ---
            derived_df.to_sql('gpl_derived_gradients', conn,
                              if_exists='append', index=False)
            total_rows += len(derived_df)
            print(f"  → wrote {len(derived_df):>8d} rows  (running total: {total_rows})")

            # Free batch memory before next iteration
            del df_dense, df_timing, df, derived_df

        print(f"  Done. Total rows written: {total_rows}.")
        print("  Creating indexes for gpl_derived_gradients...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_derived_iter "
                     "ON gpl_derived_gradients (Iter)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_derived_cell "
                     "ON gpl_derived_gradients (CellId)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_derived_iter_cell "
                     "ON gpl_derived_gradients (Iter, CellId)")

    @staticmethod
    def _generate_path_signatures(conn, batch_size=50):
        """Generate stable path signatures using a two-pass batched approach.
        
        Why two passes?
            Stable signatures must be globally consistent: the same physical
            path (ordered sequence of CellIds) must get the same ID regardless
            of which iteration it appears in.  A single in-memory pass loading
            all rows would be the simplest, but the gpl_path_cells table can
            be very large (e.g. 500 paths × 50 cells × 500 iters = 12.5M rows).
            
        Pass 1 -- collect distinct sequences:
            Iterate through iteration batches, read only the rows for that
            chunk, group by (PathId, Iter) to reconstruct the CellId sequence,
            and add each unique sequence to a global ``set``.  The set contains
            at most one entry per unique physical path, which is typically
            orders of magnitude smaller than the raw table.
            
        Pass 2 -- write mapping:
            Re-iterate the same batches, reconstructing sequences again, and
            look up the globally assigned PhysicalPathId from the dictionary
            built in pass 1.  Write each batch's mapping rows immediately.
        """
        # --- Determine iteration range ---
        try:
            cursor = conn.execute(
                "SELECT COALESCE(MIN(Iter),0), COALESCE(MAX(Iter),0) "
                "FROM gpl_path_cells"
            )
        except pd.errors.DatabaseError:
            print("  Table 'gpl_path_cells' not found. Skipping path signatures.")
            return
        row = cursor.fetchone()
        if not row or row[1] == 0:
            print("  Table 'gpl_path_cells' is empty. Skipping path signatures.")
            return
        iter_min, iter_max = int(row[0]), int(row[1])
        print(f"  Iteration range: {iter_min} to {iter_max}")

        # ======== PASS 1: Collect all unique physical path sequences ========
        print("  Pass 1: collecting unique path sequences across all iterations...")
        unique_sequences = set()
        for start in range(iter_min, iter_max + 1, batch_size):
            end = min(start + batch_size - 1, iter_max)
            df = pd.read_sql_query(
                "SELECT PathId, CellId, Iter, PathSeq "
                "FROM gpl_path_cells "
                "WHERE Iter BETWEEN ? AND ? "
                "ORDER BY Iter, PathId, PathSeq",
                conn, params=(start, end)
            )
            if df.empty:
                continue

            path_sequences = df.groupby(['PathId', 'Iter'])['CellId'].apply(tuple)
            for seq in path_sequences:
                unique_sequences.add(seq)

            print(f"    [{start:>5d}–{end:<5d}]  {len(path_sequences):>8d} sequences,  "
                  f"unique so far: {len(unique_sequences)}")

        if not unique_sequences:
            print("  No path sequences found. Skipping.")
            return

        print(f"  Found {len(unique_sequences)} unique physical path shapes.")

        # Assign stable integer IDs — the dict lives for all of pass 2.
        seq_to_id = {seq: i for i, seq in enumerate(unique_sequences)}
        del unique_sequences  # free the set memory early

        # ======== PASS 2: Write mapping table in batches ========
        print("  Pass 2: writing path_signature_map...")

        # Drop + recreate the table so we can append per batch.
        conn.execute("DROP TABLE IF EXISTS path_signature_map")
        conn.execute("""
            CREATE TABLE path_signature_map (
                Iter INTEGER,
                PathId INTEGER,
                PhysicalPathId INTEGER
            )
        """)
        conn.commit()

        total_rows = 0
        for start in range(iter_min, iter_max + 1, batch_size):
            end = min(start + batch_size - 1, iter_max)
            df = pd.read_sql_query(
                "SELECT PathId, CellId, Iter, PathSeq "
                "FROM gpl_path_cells "
                "WHERE Iter BETWEEN ? AND ? "
                "ORDER BY Iter, PathId, PathSeq",
                conn, params=(start, end)
            )
            if df.empty:
                continue

            path_sequences = df.groupby(['PathId', 'Iter'])['CellId'].apply(tuple)
            mapping = path_sequences.apply(
                lambda x: seq_to_id[x]
            ).reset_index(name='PhysicalPathId')

            mapping.to_sql('path_signature_map', conn,
                           if_exists='append', index=False)
            total_rows += len(mapping)
            print(f"    [{start:>5d}–{end:<5d}]  wrote {len(mapping):>8d} rows  "
                  f"(running total: {total_rows})")

        print(f"  Done. Total rows written: {total_rows}.")

        print("  Creating indexes for path_signature_map...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sig_iter_path "
                     "ON path_signature_map (Iter, PathId)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sig_phys "
                     "ON path_signature_map (PhysicalPathId)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess OpenROAD placement SQLite database.")
    parser.add_argument("db_path", help="Path to the SQLite database")
    parser.add_argument("--force", action="store_true", help="Force rebuild of derived tables")
    parser.add_argument("--batch-size", type=int, default=25,
                        help="Number of iterations to process per batch. "
                             "Smaller values reduce peak RAM usage. Default: 25.")
    args = parser.parse_args()
    
    GplPreprocessor.run(args.db_path, force_rebuild=args.force,
                        batch_size=args.batch_size)
