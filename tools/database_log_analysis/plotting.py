import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .gpl import GplAnalysis

class GplPlotter:
    def __init__(self, gpl_analysis: GplAnalysis):
        self.gpl = gpl_analysis

    def plot_path_slack_evolution(self, top_k=20, interactive=True, output_file="slack_evolution.png"):
        """Plot: Iterative slack progress for the most critical paths."""
        df_history = self.gpl.get_critical_path_evolution(top_k=top_k)
        if df_history.empty:
            print("No path slack history available.")
            return

        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot each path
        for path_id, group in df_history.groupby("PathId"):
            group = group.sort_values("Iter")
            ax.plot(group["Iter"], group["Slack"], label=f"Path {path_id}", alpha=0.6)
        
        # Highlight WNS (Worst Negative Slack) trend
        wns_trend = df_history.groupby("Iter")["Slack"].min().reset_index()
        ax.plot(wns_trend["Iter"], wns_trend["Slack"], color='black', linewidth=3, label="WNS Trend", linestyle='--')

        ax.set_title(f"Iterative Slack Evolution (Top {top_k} Worst Paths)")
        ax.set_xlabel("Nesterov Iteration")
        ax.set_ylabel("Slack (ps)")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Only show legend if count is manageable
        if len(df_history["PathId"].unique()) <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()

        if interactive:
            plt.show()
        else:
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
            plt.close()

    def plot_timing_convergence_velocity(self, top_k=10, interactive=True, output_file="timing_velocity.png"):
        """
        Plot: Comparison of Slack improvement vs. Cell Displacement.
        High velocity + No slack improvement = Placer is struggling/fighting.
        Low velocity + No slack improvement = Placer has given up/stagnated.
        """
        df_history = self.gpl.get_critical_path_evolution(top_k=top_k)
        if df_history.empty: return
        
        # Get cells on these paths
        path_cells, _ = self.gpl.path_cells
        target_cells = path_cells[path_cells["PathId"].isin(df_history["PathId"].unique())]["CellId"].unique().tolist()
        
        # Get movements
        df_move, _ = self.gpl.get_cell_movements(cell_ids=target_cells)
        
        # Aggregate movement by path and iteration
        path_mapping = path_cells[path_cells["PathId"].isin(df_history["PathId"].unique())]
        df_combined = pd.merge(df_move, path_mapping, on=["Iter", "CellId"])
        path_velocity = df_combined.groupby(["PathId", "Iter"])["Distance"].mean().reset_index()
        
        # Join with slack
        df_final = pd.merge(df_history, path_velocity, on=["PathId", "Iter"])
        
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()

        # Aggregate for global view
        global_wns = df_final.groupby("Iter")["Slack"].min()
        global_velocity = df_final.groupby("Iter")["Distance"].mean()

        ax1.plot(global_wns.index, global_wns.values, color='blue', label="Worst Slack", linewidth=2)
        ax2.bar(global_velocity.index, global_velocity.values, color='gray', alpha=0.3, label="Avg Cell Velocity")

        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Slack (ps)", color='blue')
        ax2.set_ylabel("Avg Displacement (dbu)", color='gray')
        ax1.set_title("Timing Convergence: Slack vs. Critical Cell Velocity")
        
        plt.tight_layout()
        if interactive:
            plt.show()
        else:
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
            plt.close()

    def plot_force_magnitude_comparison(self, top_n_paths=5, interactive=True, output_file="force_comparison.png"):
        """Plot 4: Force magnitude comparison for critical cells using matplotlib."""
        # 1. Get critical paths
        path_slacks, _ = self.gpl.path_slacks
        if path_slacks.empty:
            print("No path slack data available.")
            return
        
        worst_paths = path_slacks.groupby("PathId")["Slack"].min().nsmallest(top_n_paths).index
        
        # 2. Get cells on these paths
        path_cells, _ = self.gpl.path_cells
        target_cells = path_cells[path_cells["PathId"].isin(worst_paths)]["CellId"].unique().tolist()
        
        # 3. Get forces (PIECEWISE)
        df_forces, _ = self.gpl.get_cell_derived_gradients(cell_ids=target_cells)
        
        # 4. Plot
        avg_forces = df_forces.groupby("Iter")[["mag_wl", "mag_tim"]].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(avg_forces["Iter"], avg_forces["mag_wl"], label="Avg WL Force Mag")
        ax.plot(avg_forces["Iter"], avg_forces["mag_tim"], label="Avg Timing Force Mag")
        
        ax.set_title("Average Force Magnitude vs Iteration (Critical Path Cells)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Force Magnitude")
        ax.legend()
        plt.tight_layout()
        
        if interactive:
            plt.show()
        else:
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
            plt.close()

    def plot_path_tortuosity(self, top_n_cells=20, interactive=True, output_file="tortuosity.png"):
        """
        Plot: Path Tortuosity (Actual Distance / Euclidean Distance from Start).
        Identifies cells that are oscillating or moving inefficiently.
        """
        path_slacks, _ = self.gpl.path_slacks
        if path_slacks.empty:
            print("No path slack data available.")
            return
        
        worst_paths = path_slacks.groupby("PathId")["Slack"].min().nsmallest(5).index
        path_cells, _ = self.gpl.path_cells
        target_cells = path_cells[path_cells["PathId"].isin(worst_paths)]["CellId"].unique().tolist()

        df_move, _ = self.gpl.get_cell_movements(cell_ids=target_cells)
        
        tort_data = []
        for cell_id, group in df_move.groupby("CellId"):
            group = group.sort_values("Iter")
            total_dist = group["Distance"].sum()
            
            start_pos = (group.iloc[0]["PosX"], group.iloc[0]["PosY"])
            end_pos = (group.iloc[-1]["PosX"], group.iloc[-1]["PosY"])
            euclidean_dist = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
            
            tortuosity = total_dist / euclidean_dist if euclidean_dist > 0 else 0
            tort_data.append({"CellId": cell_id, "Tortuosity": tortuosity, "TotalDist": total_dist})
            
        df_tort = pd.DataFrame(tort_data).sort_values("Tortuosity", ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(df_tort)), df_tort["Tortuosity"])
        ax.set_xticks(range(len(df_tort)))
        ax.set_xticklabels(df_tort["CellId"], rotation=45)
        ax.set_title("Path Tortuosity for Critical Cells (Total Path / Net Displacement)")
        ax.set_ylabel("Tortuosity Ratio")
        plt.tight_layout()

        if interactive:
            plt.show()
        else:
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
            plt.close()

    def plot_gradient_balance(self, iter_range=None, interactive=True, output_file="gradient_balance.png"):
        """Plot: Timing vs Wirelength Gradient magnitude ratio over time."""
        df_stats, _ = self.gpl.get_gradient_balance_stats(iter_range=iter_range)
        if df_stats.empty:
            print("No gradient data available.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_stats["Iter"], df_stats["MeanRatio"], label="Mean Ratio (Tim/WL)")
        ax.plot(df_stats["Iter"], df_stats["MedianRatio"], label="Median Ratio (Tim/WL)")
        ax.fill_between(df_stats["Iter"], 
                        df_stats["MeanRatio"] - df_stats["StdRatio"],
                        df_stats["MeanRatio"] + df_stats["StdRatio"], 
                        alpha=0.2, label="Std Dev")
        
        ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label="Parity")
        ax.set_title("Timing Force vs Wirelength Force Balance")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Magnitude Ratio")
        ax.legend()
        plt.tight_layout()

        if interactive:
            plt.show()
        else:
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
            plt.close()

    def plot_opposition_heatmap(self, iteration, region_name="core", interactive=True, output_file="opposition_heatmap.png"):
        """
        Plot: Spatial heatmap of WL/Timing opposition for a specific iteration.
        Aggregates by bin to handle large cell counts.
        """
        df_derived, _ = self.gpl.get_cell_derived_gradients(iter_range=(iteration, iteration))
        if df_derived.empty:
            print(f"No data for iteration {iteration}")
            return
            
        df_bins, _ = self.gpl.get_cell_bin_mapping(iter_range=(iteration, iteration), region_name=region_name)
        df = pd.merge(df_derived, df_bins, on="CellId")
        bin_agg = df.groupby("BinIdx")["opposition"].mean().reset_index()
        
        try:
            bin_cnt_x = int(self.gpl.db.get_metadata(f"region_{region_name}_binCntX")[0])
            bin_cnt_y = int(self.gpl.db.get_metadata(f"region_{region_name}_binCntY")[0])
        except:
            print("Missing bin grid metadata.")
            return
            
        grid = np.zeros((bin_cnt_y, bin_cnt_x))
        grid.flat[bin_agg["BinIdx"].values] = bin_agg["opposition"].values
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(grid, origin="lower", cmap="RdYlGn_r", vmin=-1, vmax=1)
        plt.colorbar(im, label="Opposition Score (-1: Helping, 1: Fighting)")
        ax.set_title(f"Force Opposition Heatmap (Iter {iteration})")
        ax.set_xlabel("Bin X")
        ax.set_ylabel("Bin Y")
        plt.tight_layout()

        if interactive:
            plt.show()
        else:
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
            plt.close()
