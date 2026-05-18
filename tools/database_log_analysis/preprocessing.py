import sqlite3
import numpy as np
import pandas as pd
from .core import DatabaseLog

class GplPreprocessor:
    """Library module for preprocessing GPL database logs."""
    
    @staticmethod
    def run(db_path: str):
        """Perform preprocessing on the database at the given path."""
        print(f"Opening {db_path} for R/W preprocessing...")
        conn = sqlite3.connect(db_path)
        
        # Check if table already exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='gpl_derived_gradients'")
        if cursor.fetchone():
            print("Table 'gpl_derived_gradients' already exists. Skipping.")
            conn.close()
            return

        print("Reading gpl_cell_dense_gradients...")
        df_dense = pd.read_sql_query("SELECT * FROM gpl_cell_dense_gradients", conn)
        print(f"  Read {len(df_dense)} rows.")
        
        print("Reading gpl_cell_timing_gradients...")
        df_timing = pd.read_sql_query("SELECT * FROM gpl_cell_timing_gradients", conn)
        print(f"  Read {len(df_timing)} rows.")
        
        print("Merging tables...")
        df = pd.merge(df_dense, df_timing, on=['Iter', 'CellId'], suffixes=('_wl', '_tim'))
        
        print("Calculating derived metrics...")
        df['mag_wl'] = np.sqrt(df['WlX']**2 + df['WlY']**2)
        df['mag_tim'] = np.sqrt(df['TimX']**2 + df['TimY']**2)
        df['dot_wl_tim'] = df['WlX'] * df['TimX'] + df['WlY'] * df['TimY']
        mag_prod = df['mag_wl'] * df['mag_tim']
        df['opposition'] = np.where(mag_prod > 0, -df['dot_wl_tim'] / mag_prod, 0)
        
        print("Writing to gpl_derived_gradients...")
        derived_df = df[['Iter', 'CellId', 'mag_wl', 'mag_tim', 'dot_wl_tim', 'opposition']]
        derived_df.to_sql('gpl_derived_gradients', conn, if_exists='replace', index=False)
        
        print("Creating indexes...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_derived_iter ON gpl_derived_gradients (Iter)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_derived_cell ON gpl_derived_gradients (CellId)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_derived_iter_cell ON gpl_derived_gradients (Iter, CellId)")
        
        print("Vacuuming...")
        conn.execute("VACUUM")
        conn.commit()
        conn.close()
        print("Preprocessing complete.")
