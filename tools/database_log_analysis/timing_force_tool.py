import argparse
import os
import sys
import pandas as pd

# Add the parent directory to sys.path so we can import database_log_analysis
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from database_log_analysis.core import DatabaseLog
from database_log_analysis.gpl import GplAnalysis
from database_log_analysis.preprocessing import GplPreprocessor
from database_log_analysis.timing_plots import plot_timing_force_effectiveness

def run_tool():
    parser = argparse.ArgumentParser(description="Analyze effectiveness of timing forces on worst paths.")
    parser.add_argument("db_path", help="Path to the SQLite database")
    parser.add_argument("-n", "--top_n", type=int, default=10, help="Number of worst paths to analyze")
    parser.add_argument("--interactive", type=str, default="True", help="True (default) to show plots, False to print CSV data")
    parser.add_argument("--force", action="store_true", help="Force rebuild of derived tables")
    args = parser.parse_args()

    is_interactive = args.interactive.lower() == "true"

    # 1. Preprocessing
    if is_interactive:
        print(f"Ensuring {args.db_path} is preprocessed...")
    GplPreprocessor.run(args.db_path, force_rebuild=args.force)

    # 2. Data Extraction
    db = DatabaseLog(args.db_path)
    gpl = GplAnalysis(db)
    
    # Extract detailed force analysis data
    force_data = gpl.get_path_force_analysis(top_n=args.top_n)

    if force_data.empty:
        print("Error: No force data found for worst paths.")
        sys.exit(1)

    # 3. Output logic
    if not is_interactive:
        # Print summary CSV for debugging
        print(force_data.to_csv(index=False))
    else:
        # Plotting
        plot_timing_force_effectiveness(
            force_df=force_data,
            top_n=args.top_n,
            output_prefix=f"timing_force_n{args.top_n}",
            show=True
        )

if __name__ == "__main__":
    run_tool()
