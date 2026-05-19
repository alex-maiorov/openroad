import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    from .gpl import GplAnalysis
except ImportError:
    # Handle direct script execution
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from database_log_analysis.gpl import GplAnalysis

# Ensure the plots directory exists
PLOTS_DIR = os.path.join("tools", "database_log_analysis", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def compute_timing_conflict_trends(gpl: GplAnalysis):
    """
    Computes conflict metrics over ALL iterations, focusing on violating paths.
    Compatible with both old (PathId, CellId, Iter) and new (+PathSeq, Slack) gpl_path_cells formats.
    """
    # 1. Get Gradient/Force Data for all iterations
    wl_df, _ = gpl.get_wl_gradient_vectors()
    tim_df, _ = gpl.get_tim_gradient_vectors()
    dens_df, _ = gpl.get_density_force_vectors()
    
    # 2. Get path data
    path_slacks, _ = gpl.path_slacks
    path_cells, _ = gpl.path_cells
    
    # Filter for violating paths (slack < 0)
    violating_paths = path_slacks[path_slacks["Slack"] < 0]
    
    # 3. Detect path_cells schema (old vs new format)
    has_new_path_cells = "PathSeq" in path_cells.columns and "Slack" in path_cells.columns
    
    # 4. Merge data for cells on violating paths across all iterations
    # If using new format, use suffixes to disambiguate per-cell Slack from path-level Slack
    merge_suffixes = ("_cell", "_path") if has_new_path_cells else ("_x", "_y")
    analysis = pd.merge(
        path_cells, violating_paths, on=["PathId", "Iter"],
        suffixes=merge_suffixes
    )
    analysis = pd.merge(analysis, wl_df, on=["Iter", "CellId"])
    analysis = pd.merge(analysis, tim_df, on=["Iter", "CellId"])
    analysis = pd.merge(analysis, dens_df, on=["Iter", "CellId"])
    
    # 5. Calculate Conflict Metrics per cell
    analysis["TimVecX"] = analysis["TimUnitX"] * analysis["TimMag"]
    analysis["TimVecY"] = analysis["TimUnitY"] * analysis["TimMag"]
    
    # Competitor Force (WL + Density)
    analysis["OtherVecX"] = (analysis["WlUnitX"] * analysis["WlMag"]) + \
                            (analysis["EstDensityForceUnitX"] * analysis["EstDensityForceMag"])
    analysis["OtherVecY"] = (analysis["WlUnitY"] * analysis["WlMag"]) + \
                            (analysis["EstDensityForceUnitY"] * analysis["EstDensityForceMag"])
    
    # Alignment: Dot Product (positive = aligned, negative = opposed)
    analysis["Alignment"] = (analysis["TimVecX"] * analysis["OtherVecX"]) + \
                            (analysis["TimVecY"] * analysis["OtherVecY"])
                            
    # Force Dominance: Mag(Tim) / (Mag(Tim) + Mag(Other))
    analysis["TimDominance"] = analysis["TimMag"] / (analysis["TimMag"] + \
                                np.sqrt(analysis["OtherVecX"]**2 + analysis["OtherVecY"]**2) + 1e-9)
    
    # 6. Aggregate to Path level (per iteration)
    # Build aggregation dict dynamically based on available columns
    agg_dict = {
        "Alignment": "mean",
        "TimDominance": "mean",
    }
    
    if has_new_path_cells:
        # New format: path-level Slack is suffixed _path, per-cell Slack is _cell
        agg_dict["Slack_path"] = "first"
        agg_dict["Slack_cell"] = "mean"
        agg_dict["PathSeq"] = "count"
    else:
        # Old format: the only Slack column is path-level
        agg_dict["Slack"] = "first"
    
    path_metrics = analysis.groupby(["Iter", "PathId"]).agg(agg_dict).reset_index()
    
    # Rename for consistent output
    if has_new_path_cells:
        path_metrics = path_metrics.rename(columns={
            "Slack_path": "Slack",
            "Slack_cell": "MeanCellSlack",
            "PathSeq": "PathLength"
        })
    
    return path_metrics

def plot_timing_conflict_trends(path_metrics, interactive=False):
    """
    Plots the average, worst 10%, worst 1%, and worst path metrics over time.
    Optionally shows an interactive window.
    """
    iters = np.sort(path_metrics["Iter"].unique())
    
    metrics = ["Alignment", "TimDominance", "Slack"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Per iteration, find the categories
        avg_vals, w10_vals, w1_vals, worst_vals = [], [], [], []
        
        for iter_num in iters:
            data = path_metrics[path_metrics["Iter"] == iter_num]
            
            # Categories (worst is lowest slack)
            worst_path_idx = data["Slack"].idxmin()
            worst_val = data.loc[worst_path_idx, metric]
            
            # Percentiles based on Slack
            worst_10pct = data.nsmallest(max(1, int(len(data) * 0.1)), "Slack")[metric].mean()
            worst_1pct = data.nsmallest(max(1, int(len(data) * 0.01)), "Slack")[metric].mean()
            avg_val = data[metric].mean()
            
            avg_vals.append(avg_val)
            w10_vals.append(worst_10pct)
            w1_vals.append(worst_1pct)
            worst_vals.append(worst_val)
            
        ax.plot(iters, avg_vals, label="Average")
        ax.plot(iters, w10_vals, label="Worst 10%")
        ax.plot(iters, w1_vals, label="Worst 1%")
        ax.plot(iters, worst_vals, label="Worst Path")
        ax.set_title(f"{metric} Trends (Violating Paths Only)")
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(PLOTS_DIR, "timing_conflicts.png")
    plt.savefig(plot_file)
    print(f"Plot saved to: {plot_file}")
    
    # Interactive display
    if interactive:
        print("Opening interactive plot...")
        plt.show()
    else:
        plt.close(fig)

def plot_timing_force_effectiveness(force_df: pd.DataFrame, top_n: int = 10, output_prefix: str = "timing_force", show: bool = False):
    """
    Visualizes the effectiveness and impact of timing forces on the worst paths.
    """
    if force_df.empty:
        print("Warning: Force DataFrame is empty. Nothing to plot.")
        return

    # Aggregate to Path level per iteration
    path_iter_metrics = force_df.groupby(['Iter', 'PhysicalPathId']).agg({
        'Slack_path': 'first',
        'TimMag': 'mean',
        'WlMag': 'mean',
        'EstDensityForceMag': 'mean',
        'Opposition_Score': 'mean',
        'Alignment_Tim_WL': 'mean'
    }).reset_index()

    path_iter_metrics['TotalMag'] = path_iter_metrics['TimMag'] + path_iter_metrics['WlMag'] + path_iter_metrics['EstDensityForceMag']
    path_iter_metrics['TimDominance'] = path_iter_metrics['TimMag'] / (path_iter_metrics['TotalMag'] + 1e-12)

    # Plot 1: Timing Dominance & Opposition Score over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    for phys_id, data in path_iter_metrics.groupby('PhysicalPathId'):
        data = data.sort_values('Iter')
        ax1.plot(data['Iter'], data['TimDominance'], alpha=0.5, label=f"Path {phys_id}" if top_n <= 10 else None)
        ax2.plot(data['Iter'], data['Opposition_Score'], alpha=0.5)

    ax1.set_ylabel("Timing Force Dominance\n(Tim / Total Force)")
    ax1.set_title(f"Timing Force Impact Analysis (Top {top_n} Worst Paths)")
    ax1.grid(True)
    if top_n <= 10:
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax2.set_ylabel("Opposition Score\n(-Dot(TimUnit, OtherUnit))")
    ax2.set_xlabel("Iteration")
    ax2.grid(True)
    
    plt.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, f"{output_prefix}_impact_trends.png")
    plt.savefig(plot_file)
    print(f"Plot saved to: {plot_file}")

    # Plot 2: Scatter plot of Slack vs Opposition Score
    fig2, ax3 = plt.subplots(figsize=(10, 7))
    scatter = ax3.scatter(path_iter_metrics['Opposition_Score'], path_iter_metrics['Slack_path'], 
                          c=path_iter_metrics['Iter'], cmap='viridis', alpha=0.6)
    ax3.set_xlabel("Opposition Score (Opposition to WL+Density)")
    ax3.set_ylabel("Path Slack (ps)")
    ax3.set_title("Correlation: Force Opposition vs. Timing Slack")
    plt.colorbar(scatter, label='Iteration')
    ax3.grid(True)
    
    plot_file_2 = os.path.join(PLOTS_DIR, f"{output_prefix}_slack_vs_opposition.png")
    plt.savefig(plot_file_2)
    print(f"Plot saved to: {plot_file_2}")

def plot_path_average_forces(force_df: pd.DataFrame, top_n: int = 5, show: bool = False):
    """
    Plots the evolution of average Timing Force magnitude for the 
    top_n paths that were worst when timing data started.
    Color codes the points by the path slack.
    """
    if force_df.empty:
        print("Warning: Force DataFrame is empty.")
        return

    # 1. Identify the start of timing data
    start_iter = force_df['Iter'].min()
    start_data = force_df[force_df['Iter'] == start_iter]
    
    # Get unique paths at this start iteration sorted by slack
    worst_at_start = start_data.groupby('PhysicalPathId')['Slack_path'].first().nsmallest(top_n).index.tolist()
    
    print(f"Tracking average timing forces for {len(worst_at_start)} paths that were worst at iteration {start_iter}")

    # 2. Filter and Aggregate
    plot_data = force_df[force_df['PhysicalPathId'].isin(worst_at_start)]
    path_avg = plot_data.groupby(['Iter', 'PhysicalPathId']).agg({
        'TimMag': 'mean',
        'Slack_path': 'first'
    }).reset_index()

    # 3. Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Find global slack range for consistent color mapping
    vmin = path_avg['Slack_path'].min()
    vmax = path_avg['Slack_path'].max()
    
    sc = None
    for phys_id in worst_at_start:
        subset = path_avg[path_avg['PhysicalPathId'] == phys_id].sort_values('Iter')
        
        # Plot line for connectivity
        ax.plot(subset['Iter'], subset['TimMag'], alpha=0.2, color='black', linewidth=1)
        
        # Scatter points color-coded by slack
        # Using RdYlGn (Red-Yellow-Green) where red is worst slack
        sc = ax.scatter(subset['Iter'], subset['TimMag'], c=subset['Slack_path'], 
                        cmap='RdYlGn', vmin=vmin, vmax=vmax, s=25, label=f"Path {phys_id}")
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Timing Force Magnitude")
    ax.set_title(f"Evolution of Average Timing Force for Top {top_n} Worst Paths (Iter {start_iter}+)\nColor mapped to Path Slack")
    ax.grid(True)
    
    # Add a legend for Path IDs (using the scatter handles)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Path ID")

    # Add colorbar for slack
    if sc:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Path Slack (ps)")

    plt.tight_layout()
    
    plot_file = os.path.join(PLOTS_DIR, "path_timing_force_evolution.png")
    plt.savefig(plot_file)
    print(f"Plot saved to: {plot_file}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_worst_path_evolution(history_df: pd.DataFrame, top_n: int = 10, output_name: str = "worst_paths_evolution.png", show: bool = False):
    """
    Plots the slack evolution of paths using the provided history DataFrame.
    The DataFrame must contain 'Iter', 'Slack', and 'PhysicalPathId'.
    """
    if history_df.empty:
        print("Warning: History DataFrame is empty. Nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by PhysicalPathId and plot each
    num_paths = history_df['PhysicalPathId'].nunique()
    print(f"Plotting evolution for {num_paths} paths...")
    
    for phys_id, path_data in history_df.groupby("PhysicalPathId"):
        path_data = path_data.sort_values("Iter")
        ax.plot(path_data["Iter"], path_data["Slack"], marker='o', markersize=2, alpha=0.5, label=f"Path {phys_id}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Slack (ps)")
    ax.set_title(f"Evolution of Physical Paths that were in Top {top_n} Worst at any Point")
    ax.grid(True)
    
    if num_paths <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_file = os.path.join(PLOTS_DIR, output_name)
    plt.savefig(plot_file)
    print(f"Plot saved to: {plot_file}")
    
    if show:
        print("Displaying plot...")
        plt.show()
    else:
        plt.close(fig)
