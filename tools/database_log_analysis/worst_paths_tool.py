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
from database_log_analysis.timing_plots import plot_worst_path_evolution

def run_tool():
    parser = argparse.ArgumentParser(description="Extract and/or plot evolution of worst n slacked paths.")
    parser.add_argument("db_path", help="Path to the SQLite database")
    parser.add_argument("-n", "--top_n", type=int, default=10, help="Number of worst paths to track")
    parser.add_argument("--interactive", type=str, default="True", help="True (default) to show plot, False to print CSV data")
    parser.add_argument("--force", action="store_true", help="Force rebuild of derived tables")
    args = parser.parse_args()

    is_interactive = args.interactive.lower() == "true"

    # 1. Preprocessing (Embedded into the workflow)
    if is_interactive:
        print(f"Ensuring {args.db_path} is preprocessed...")
    GplPreprocessor.run(args.db_path, force_rebuild=args.force)

    # 2. Data Extraction
    db = DatabaseLog(args.db_path)
    gpl = GplAnalysis(db)
    
    if is_interactive:
        # Fetch and print timing gradient parameters from metadata
        print("\n--- Placer Timing Parameters ---")
        keys = [
            "timingDrivenMode",
            "timingGradPassStaRunInterval",
            "timingGradPassFirstIter",
            "timingGradPassTopN",
            "timingGradPassWeight",
            "timingGradPassProjWeight",
            "timingGradPassEndToEndWeight",
            "timingGradPassSlackSharpness",
            "timingGradPassSlackOffset",
            "timingGradPassSlackUpper"
        ]
        for key in keys:
            val = db.get_metadata(key)
            if val:
                print(f"{key}: {val[0]}")
        print("----------------------------------\n")

    history = gpl.get_worst_paths_history(top_n=args.top_n)

    if history.empty:
        print("Error: No history found for worst paths.")
        sys.exit(1)

    # 3. Output logic
    if not is_interactive:
        # Request data and print in CSV format as requested for debugging
        print(history.to_csv(index=False))
    else:
        # Plotting (using the library's plotting code)
        plot_worst_path_evolution(
            history_df=history, 
            top_n=args.top_n, 
            output_name=f"worst_{args.top_n}_paths_evolution.png",
            show=True
        )

if __name__ == "__main__":
    run_tool()
