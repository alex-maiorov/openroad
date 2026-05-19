import argparse
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add the current directory to sys.path to allow importing the library
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from .core import DatabaseLog
    from .gpl import GplAnalysis
    from .preprocessing import GplPreprocessor
    from .timing_plots import plot_worst_path_evolution
except ImportError:
    # This happens when run as a script without -m
    print("Please run this script using: ")
    print("PYTHONPATH=openroad_versions/OpenROAD_alex_gpl/tools python3 -m database_log_analysis.plot_worst_paths <db_path>")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Plot evolution of worst n slacked paths.")
    parser.add_argument("db_path", help="Path to the SQLite database")
    parser.add_argument("-n", "--top_n", type=int, default=10, help="Number of worst paths to track")
    parser.add_argument("--force", action="store_true", help="Force rebuild of derived tables")
    args = parser.parse_args()

    # 1. Run preprocessing to ensure stable path IDs exist
    GplPreprocessor.run(args.db_path, force_rebuild=args.force)

    # 2. Connect to the database
    db = DatabaseLog(args.db_path)
    gpl = GplAnalysis(db)

    # 3. Plot using the library function
    plot_worst_path_evolution(gpl, top_n=args.top_n, output_name=f"worst_{args.top_n}_paths_evolution.png")

if __name__ == "__main__":
    main()
