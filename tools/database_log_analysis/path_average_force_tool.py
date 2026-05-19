import argparse
import os
import sys
import pandas as pd

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from database_log_analysis.core import DatabaseLog
from database_log_analysis.gpl import GplAnalysis
from database_log_analysis.preprocessing import GplPreprocessor
from database_log_analysis.timing_plots import plot_path_average_forces

def run_tool():
    parser = argparse.ArgumentParser(description="Analyze evolution of average forces on worst starting paths.")
    parser.add_argument("db_path", help="Path to the SQLite database")
    parser.add_argument("-n", "--top_n", type=int, default=5, help="Number of worst starting paths to analyze")
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
    
    # We need a large enough top_n in the history to find the ones that were worst at the start
    force_data = gpl.get_path_force_analysis(top_n=max(args.top_n, 20))

    if force_data.empty:
        print("Error: No force data found.")
        sys.exit(1)

    # Propagate PhysicalPathId across all iterations for the same CellId
    # Since path memberships are sparse (every 10 iters) but gradients are dense (every 1 iter)
    print("Interpolating path memberships for high-resolution force analysis...")
    
    # Sort for forward filling
    force_data = force_data.sort_values(['CellId', 'Iter'])
    
    # Forward fill PhysicalPathId within each CellId group
    # This assumes if a cell was in a path at iter 250, it's still in that same path 
    # for iterations 251-259 until the next STA update.
    force_data['PhysicalPathId'] = force_data.groupby('CellId')['PhysicalPathId'].ffill()
    force_data['Slack_path'] = force_data.groupby('CellId')['Slack_path'].ffill()
    
    # Drop rows that don't belong to any path after filling
    force_data = force_data.dropna(subset=['PhysicalPathId'])

    # 3. Output logic
    if not is_interactive:
        # Use the first iteration in the force_data (which is the start of timing logging)
        start_iter = force_data['Iter'].min()
        start_data = force_data[force_data['Iter'] == start_iter]
        worst_at_start = start_data.groupby('PhysicalPathId')['Slack_path'].first().nsmallest(args.top_n).index.tolist()
        
        subset = force_data[force_data['PhysicalPathId'].isin(worst_at_start)]
        agg = subset.groupby(['Iter', 'PhysicalPathId']).agg({
            'TimMag': 'mean',
            'Slack_path': 'first'
        }).reset_index()
        print(agg.to_csv(index=False))
    else:
        # Plotting
        plot_path_average_forces(
            force_df=force_data,
            top_n=args.top_n,
            show=True
        )

if __name__ == "__main__":
    run_tool()
