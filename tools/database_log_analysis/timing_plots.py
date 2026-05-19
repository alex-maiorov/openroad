import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from .gpl import GplAnalysis

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
