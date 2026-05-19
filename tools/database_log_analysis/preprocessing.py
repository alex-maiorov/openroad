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
    def run(db_path: str, force_rebuild=False):
        """Perform preprocessing on the database at the given path."""
        print(f"Opening {db_path} for R/W preprocessing...")
        conn = sqlite3.connect(db_path)
        
        if force_rebuild:
            print("Force rebuild requested. Dropping existing derived tables...")
            conn.execute("DROP TABLE IF EXISTS gpl_derived_gradients")
            conn.execute("DROP TABLE IF EXISTS path_signature_map")
            conn.commit()
            
        # Check if gradient table already exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='gpl_derived_gradients'")
        if not cursor.fetchone():
            GplPreprocessor._generate_derived_gradients(conn)
        else:
            print("Table 'gpl_derived_gradients' already exists. Skipping.")

        # Check if path signature table already exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='path_signature_map'")
        if not cursor.fetchone():
            GplPreprocessor._generate_path_signatures(conn)
        else:
            print("Table 'path_signature_map' already exists. Skipping.")

        conn.commit()
        conn.close()
        print("Preprocessing complete.")
    
    @staticmethod
    def _generate_derived_gradients(conn):
        print("Reading gpl_cell_dense_gradients...")
        df_dense = pd.read_sql_query("SELECT * FROM gpl_cell_dense_gradients", conn)
        print(f"  Read {len(df_dense)} rows.")
        
        print("Reading gpl_cell_timing_gradients...")
        df_timing = pd.read_sql_query("SELECT * FROM gpl_cell_timing_gradients", conn)
        print(f"  Read {len(df_timing)} rows.")
        
        print("Merging tables (using LEFT join because timing gradients are sparse)...")
        # Use a left merge to preserve all dense gradients. Missing timing gradients imply 0 force.
        df = pd.merge(df_dense, df_timing, on=['Iter', 'CellId'], how='left')
        
        # Fill missing sparse gradients with 0
        df['TimX'] = df['TimX'].fillna(0.0)
        df['TimY'] = df['TimY'].fillna(0.0)
        
        print("Calculating derived metrics...")
        df['mag_wl'] = np.sqrt(df['WlX']**2 + df['WlY']**2)
        df['mag_tim'] = np.sqrt(df['TimX']**2 + df['TimY']**2)
        df['dot_wl_tim'] = df['WlX'] * df['TimX'] + df['WlY'] * df['TimY']
        mag_prod = df['mag_wl'] * df['mag_tim']
        df['opposition'] = np.where(mag_prod > 0, -df['dot_wl_tim'] / mag_prod, 0)
        
        print("Writing to gpl_derived_gradients...")
        derived_df = df[['Iter', 'CellId', 'mag_wl', 'mag_tim', 'dot_wl_tim', 'opposition']]
        derived_df.to_sql('gpl_derived_gradients', conn, if_exists='replace', index=False)
        
        print("Creating indexes for gpl_derived_gradients...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_derived_iter ON gpl_derived_gradients (Iter)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_derived_cell ON gpl_derived_gradients (CellId)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_derived_iter_cell ON gpl_derived_gradients (Iter, CellId)")

    @staticmethod
    def _generate_path_signatures(conn):
        print("Generating stable signatures for timing paths...")
        try:
            print("  Reading gpl_path_cells...")
            df = pd.read_sql("SELECT PathId, CellId, Iter, PathSeq FROM gpl_path_cells", conn)
        except pd.errors.DatabaseError:
            print("  Table 'gpl_path_cells' not found or empty. Skipping path signatures.")
            return

        if df.empty:
            print("  Table 'gpl_path_cells' is empty. Skipping path signatures.")
            return

        print("  Generating stable signatures...")
        # Sort to ensure order
        df = df.sort_values(['Iter', 'PathId', 'PathSeq'])

        # Group by PathId and Iter to get the sequence of CellIds
        # This effectively gives us the physical path per iteration
        path_sequences = df.groupby(['PathId', 'Iter'])['CellId'].apply(tuple)

        # Get unique sequences
        unique_sequences = path_sequences.unique()

        # Assign stable IDs based on unique sequences
        # Use a dictionary for fast lookup
        seq_to_id = {seq: i for i, seq in enumerate(unique_sequences)}

        # Map back to PathId, Iter using apply
        print("  Mapping to stable IDs...")
        mapping = path_sequences.apply(lambda x: seq_to_id[x]).reset_index(name='PhysicalPathId')

        print(f"  Assigned {len(unique_sequences)} stable path IDs.")

        # Write to database
        print("  Writing mapping to database table 'path_signature_map'...")
        mapping.to_sql('path_signature_map', conn, if_exists='replace', index=False)
        
        print("  Creating indexes for path_signature_map...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sig_iter_path ON path_signature_map (Iter, PathId)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sig_phys ON path_signature_map (PhysicalPathId)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess OpenROAD placement SQLite database.")
    parser.add_argument("db_path", help="Path to the SQLite database")
    parser.add_argument("--force", action="store_true", help="Force rebuild of derived tables")
    args = parser.parse_args()
    
    GplPreprocessor.run(args.db_path, force_rebuild=args.force)
