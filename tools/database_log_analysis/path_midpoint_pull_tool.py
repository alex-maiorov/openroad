#!/usr/bin/env python3
"""
Path Midpoint Pull Analysis Tool
=================================

Explores whether timing forces pull cells towards the weighted average
position of all cells in the same timing path.

For each iteration and each of the top *n* worst paths (at the final
iteration where slacks are logged):

  1. Compute the *weighted midpoint* of all cells in the path, weighting
     each cell's position by its timing-force magnitude |Tim|.  Cells with
     zero timing force contribute nothing to the midpoint.
  2. For each cell, calculate:
       pull = unit_vector(cell → midpoint) · timing_force_vector
     where ``timing_force_vector`` is the *raw* (non-normalised)
     (TimX, TimY) gradient.  A positive value means the timing force is
     pulling the cell toward the midpoint; a negative value means it is
     pushing away.
  3. Plot the per-path average pull over iterations, with colour-coded
     markers controlled by ``--point-interval``.  Marker colour reflects
     the path slack at that iteration (red = worst slack, green = best).

Usage
-----
::

    PYTHONPATH=openroad_versions/OpenROAD_alex_gpl/tools python3 -m \\
        database_log_analysis.path_midpoint_pull_tool \\
        temp/reports/manual-run-1779369786/placement.sqlite
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup  (same pattern as the other tools in this package)
# ---------------------------------------------------------------------------
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from database_log_analysis.core import DatabaseLog
from database_log_analysis.gpl import GplAnalysis
from database_log_analysis.preprocessing import GplPreprocessor

# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute_midpoint_pull(
    gpl: GplAnalysis,
    top_n: int = 10,
    chunk_size: int = 25,
    top_n_min_sep: float = 0.0,
) -> pd.DataFrame:
    """
    Main analysis routine.

    Parameters
    ----------
    gpl :
        Connected ``GplAnalysis`` instance.
    top_n :
        Number of worst paths at the final iteration to analyse.
    chunk_size :
        Number of iterations per chunk when fetching gradient data (memory
        management for large designs).
    top_n_min_sep :
        Minimum slack separation between successively selected worst paths.
        When > 0, the worst path is always selected; then the next path whose
        slack is *better* (higher / less negative) by at least *top_n_min_sep*
        is chosen, and so on until *top_n* paths are collected.  Default 0.0
        (no separation requirement — plain ``nsmallest``).

    Returns
    -------
    pd.DataFrame
        Columns: ``Iter``, ``PhysicalPathId``, ``CellId``, ``PullValue``,
        ``TimMag``, ``PullValuePerPathAvg``, ``Slack``.
    """
    # ---- 1.  Find the final iteration that has path slacks -----------------
    df_slacks, _ = gpl.path_slacks
    if df_slacks.empty:
        print("Error: no path slack data found.")
        return pd.DataFrame()

    final_iter = int(df_slacks["Iter"].max())
    print(f"Final iteration with path slacks: {final_iter}")

    # ---- 2.  Resolve stable physical path IDs -----------------------------
    df_sig, _ = gpl.path_signature_map
    has_signatures = not df_sig.empty

    # ---- 3.  Top *n* worst paths at the final iteration -------------------
    final_slacks = df_slacks[df_slacks["Iter"] == final_iter].copy()

    if has_signatures:
        final_slacks = pd.merge(final_slacks, df_sig, on=["PathId", "Iter"])
        path_id_col = "PhysicalPathId"
    else:
        print("Warning: path_signature_map missing — using unstable PathId.")
        final_slacks["PhysicalPathId"] = final_slacks["PathId"]
        path_id_col = "PathId"

    # Most negative slack = worst
    final_slacks_sorted = final_slacks.sort_values("Slack")  # ascending → worst first

    if top_n_min_sep > 0.0:
        # Greedy selection with minimum separation
        selected_ids = []
        last_slack = None
        for _, row in final_slacks_sorted.iterrows():
            if last_slack is None:
                selected_ids.append(row["PhysicalPathId"])
                last_slack = row["Slack"]
            elif row["Slack"] >= last_slack + top_n_min_sep:
                selected_ids.append(row["PhysicalPathId"])
                last_slack = row["Slack"]
            if len(selected_ids) >= top_n:
                break
        worst_phys_ids = selected_ids
    else:
        # Plain nsmallest (backward compatible, default)
        worst_final = final_slacks_sorted.head(top_n)
        worst_phys_ids = worst_final["PhysicalPathId"].unique().tolist()

    print(f"Selected {len(worst_phys_ids)} distinct worst paths "
          f"(min_sep={top_n_min_sep}):")
    for pid in worst_phys_ids:
        row = final_slacks_sorted[
            final_slacks_sorted["PhysicalPathId"] == pid
        ].iloc[0]
        print(f"  PhysicalPathId {pid}: Slack = {row['Slack']:.3e}")

    # ---- 4.  Cell membership for these paths across *all* iterations ------
    df_path_cells, _ = gpl.path_cells
    if df_path_cells.empty:
        print("Error: gpl_path_cells table is empty.")
        return pd.DataFrame()

    if has_signatures:
        df_path_sig = pd.merge(df_path_cells, df_sig, on=["PathId", "Iter"])
        df_path_sig = df_path_sig[
            df_path_sig["PhysicalPathId"].isin(worst_phys_ids)
        ]
    else:
        df_path_sig = df_path_cells[
            df_path_cells["PathId"].isin(worst_phys_ids)
        ].copy()
        df_path_sig["PhysicalPathId"] = df_path_sig["PathId"]

    if df_path_sig.empty:
        print("Error: no path cell data for worst paths.")
        return pd.DataFrame()

    path_cell_ids = df_path_sig["CellId"].unique().tolist()
    print(f"Unique cells on worst paths: {len(path_cell_ids)}")

    # ---- 5.  Determine full iteration range -------------------------------
    df_scalars, _ = gpl.iteration_scalars
    if df_scalars.empty:
        print("Error: no iteration scalars.")
        return pd.DataFrame()
    iter_min = int(df_scalars["Iter"].min())
    iter_max = int(df_scalars["Iter"].max())
    print(f"Iteration range: {iter_min} – {iter_max}")

    # ---- 6.  Fetch data in iteration chunks -------------------------------
    all_chunks = []

    for chunk_df, _desc in gpl.iter_chunked(
        gpl.get_cell_dense_gradients,
        iter_range=(iter_min, iter_max),
        chunk_size=chunk_size,
        cell_ids=path_cell_ids,
    ):
        all_chunks.append(chunk_df)

    if not all_chunks:
        print("Error: no dense gradient data retrieved.")
        return pd.DataFrame()

    df_pos = pd.concat(all_chunks, ignore_index=True)
    del all_chunks
    print(f"Dense gradient rows: {len(df_pos)}")

    # Timing gradients (sparse) — same chunked pattern
    tim_chunks = []
    for chunk_df, _desc in gpl.iter_chunked(
        gpl.get_cell_timing_gradients,
        iter_range=(iter_min, iter_max),
        chunk_size=chunk_size,
        cell_ids=path_cell_ids,
    ):
        tim_chunks.append(chunk_df)

    df_tim = pd.concat(tim_chunks, ignore_index=True) if tim_chunks else pd.DataFrame()
    print(f"Timing gradient rows: {len(df_tim)}")

    # ---- 7.  Merge positions, timing forces, and path membership ----------
    # Left-join dense positions with sparse timing forces (missing = 0)
    df = pd.merge(df_pos, df_tim, on=["Iter", "CellId"], how="left")
    df["TimX"] = df["TimX"].fillna(0.0)
    df["TimY"] = df["TimY"].fillna(0.0)

    # Inner-join with path membership — keep only cells on tracked paths.
    # A cell can appear multiple times with different PathSeq positions in
    # the same (Iter, PhysicalPathId) — deduplicate to one row per (Iter, CellId).
    df_path_sig_filt = (
        df_path_sig[["Iter", "CellId", "PhysicalPathId"]]
        .drop_duplicates(subset=["Iter", "CellId", "PhysicalPathId"])
    )
    df = pd.merge(df, df_path_sig_filt, on=["Iter", "CellId"], how="inner")
    print(f"Merged rows (cell + path + force): {len(df)}")

    # ---- 8.  Weighted midpoint and pull per (Iter, PhysicalPathId) --------
    df["TimMag"] = np.sqrt(df["TimX"] ** 2 + df["TimY"] ** 2)

    result_rows = []

    for (iter_val, phys_id), group in df.groupby(["Iter", "PhysicalPathId"]):
        weights = group["TimMag"].values
        total_weight = weights.sum()

        if total_weight > 0:
            mid_x = np.average(group["PosX"].values, weights=weights)
            mid_y = np.average(group["PosY"].values, weights=weights)
        else:
            # All forces zero — fall back to simple centroid
            mid_x = group["PosX"].mean()
            mid_y = group["PosY"].mean()

        # Vectorised per-cell pull computation
        dx = mid_x - group["PosX"].values
        dy = mid_y - group["PosY"].values
        dist = np.sqrt(dx ** 2 + dy ** 2)

        mask = dist > 0
        unit_x = np.zeros_like(dx)
        unit_y = np.zeros_like(dy)
        unit_x[mask] = dx[mask] / dist[mask]
        unit_y[mask] = dy[mask] / dist[mask]

        pull = unit_x * group["TimX"].values + unit_y * group["TimY"].values

        for i, (_, row) in enumerate(group.iterrows()):
            result_rows.append(
                {
                    "Iter": iter_val,
                    "PhysicalPathId": phys_id,
                    "CellId": row["CellId"],
                    "PullValue": pull[i],
                    "TimMag": row["TimMag"],
                }
            )

    df_result = pd.DataFrame(result_rows)
    print(f"Result rows (all cells × all iterations): {len(df_result)}")

    # ---- 9.  Attach path slack for colour-coding --------------------------
    if has_signatures:
        df_slack_lookup = pd.merge(df_slacks, df_sig, on=["PathId", "Iter"])
        df_slack_lookup = df_slack_lookup[
            df_slack_lookup["PhysicalPathId"].isin(worst_phys_ids)
        ]
    else:
        df_slack_lookup = df_slacks[
            df_slacks["PathId"].isin(worst_phys_ids)
        ].copy()
        df_slack_lookup["PhysicalPathId"] = df_slack_lookup["PathId"]

    df_result = pd.merge(
        df_result,
        df_slack_lookup[["PhysicalPathId", "Iter", "Slack"]],
        on=["PhysicalPathId", "Iter"],
        how="left",
    )

    # ---- 10.  Per-path average pull (useful for plotting) -----------------
    path_avg = (
        df_result.groupby(["Iter", "PhysicalPathId"])["PullValue"]
        .mean()
        .reset_index(name="PullValuePerPathAvg")
    )
    df_result = pd.merge(df_result, path_avg, on=["Iter", "PhysicalPathId"])

    return df_result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_midpoint_pull(
    df_result: pd.DataFrame,
    top_n: int = 10,
    point_interval: int = 10,
    output_name: str = "midpoint_pull.png",
    show: bool = False,
):
    """
    Zero-centred line plot of per-path average pull with colour-coded markers.

    Parameters
    ----------
    df_result :
        Output of :func:`compute_midpoint_pull`.
    top_n :
        Used only for the title.
    point_interval :
        Every *point_interval*-th iteration a coloured marker is drawn.
    output_name :
        Filename for the saved PNG (placed under ``plots/``).
    """
    if df_result.empty:
        print("Error: no data to plot.")
        return

    # Build per-path averages for the continuous lines
    path_avg = (
        df_result.groupby(["Iter", "PhysicalPathId"])["PullValue"]
        .mean()
        .reset_index()
    )

    # Slack range for colour mapping
    slack_min = df_result["Slack"].min()
    slack_max = df_result["Slack"].max()

    # Slack is typically all negative — *red* = lower/more negative (worst),
    # *green* = higher/less negative (best).
    cmap = plt.cm.RdYlGn

    fig, ax = plt.subplots(figsize=(14, 8))

    # Zero-reference line
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

    # Per-path data
    phys_ids = sorted(path_avg["PhysicalPathId"].unique())

    for phys_id in phys_ids:
        sub = path_avg[path_avg["PhysicalPathId"] == phys_id].sort_values("Iter")

        # Continuous line showing average pull
        ax.plot(
            sub["Iter"],
            sub["PullValue"],
            alpha=0.5,
            linewidth=1.5,
            label=f"Path {phys_id}",
        )

        # Colour-coded markers
        marker_sub = sub[sub["Iter"] % point_interval == 0].copy()
        if not marker_sub.empty:
            # Fetch slack for each marker iteration
            slack_sub = df_result[
                (df_result["PhysicalPathId"] == phys_id)
                & (df_result["Iter"].isin(marker_sub["Iter"]))
            ][["Iter", "Slack"]].drop_duplicates("Iter")

            marker_sub = pd.merge(
                marker_sub, slack_sub, on="Iter", how="left"
            )

            if not marker_sub.empty and marker_sub["Slack"].notna().any():
                ax.scatter(
                    marker_sub["Iter"],
                    marker_sub["PullValue"],
                    c=marker_sub["Slack"],
                    cmap=cmap,
                    vmin=slack_min,
                    vmax=slack_max,
                    s=45,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=5,
                )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Pull Value (unit_vec · timing_force)")
    ax.set_title(
        f"Timing Force Pull Towards Path Midpoint\n"
        f"(Top {top_n} worst paths, markers every {point_interval} iters)"
    )
    ax.grid(True, alpha=0.3)

    # Colour bar
    norm = plt.Normalize(vmin=slack_min, vmax=slack_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Path Slack (ps)  —  red = worst, green = best")

    if len(phys_ids) <= 15:
        ax.legend(bbox_to_anchor=(1.20, 1), loc="upper left")

    plt.tight_layout()

    # Save
    plots_dir = os.path.join(_current_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, output_name)
    plt.savefig(plot_path, bbox_inches="tight", dpi=800)
    print(f"Plot saved to: {plot_path}")

    if show:
        print("Displaying plot...")
        plt.draw()
        plt.show(block=True)
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_tool():
    parser = argparse.ArgumentParser(
        description="Analyse whether timing forces pull cells towards "
                    "the path midpoint (weighted by force magnitude)."
    )
    parser.add_argument("db_path", help="Path to the SQLite database")
    parser.add_argument(
        "-n", "--top-n", type=int, default=10, dest="top_n",
        help="Number of worst (final iteration) paths to analyse (default: 10)",
    )
    parser.add_argument(
        "--top-n-min-sep", type=float, default=0.0, dest="top_n_min_sep",
        help="Minimum slack separation between successively selected worst "
             "paths.  When > 0 the worst path is always taken, then each "
             "next path must be better (higher slack) by at least this "
             "amount.  Default 0.0 (no separation).",
    )
    parser.add_argument(
        "--point-interval", type=int, default=10,
        help="Plot a coloured marker every N iterations (default: 10)",
    )
    parser.add_argument(
        "--interactive", type=str, default="True",
        help="True → show plot; False → print CSV to stdout (default: True)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=25,
        help="Iterations per chunk for memory management (default: 25)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force rebuild of derived tables (gpl_derived_gradients, "
             "path_signature_map)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="midpoint_pull.png",
        help="Output plot filename (default: midpoint_pull.png)",
    )
    args = parser.parse_args()

    is_interactive = args.interactive.lower() == "true"

    # 1.  Preprocessing
    GplPreprocessor.run(args.db_path, force_rebuild=args.force)

    # 2.  Connect
    db = DatabaseLog(args.db_path)
    gpl = GplAnalysis(db)

    # 3.  Compute
    df_result = compute_midpoint_pull(
        gpl, top_n=args.top_n, chunk_size=args.chunk_size,
        top_n_min_sep=args.top_n_min_sep,
    )

    if df_result.empty:
        print("Error: no results computed.")
        sys.exit(1)

    # 4.  Output
    if not is_interactive:
        print(df_result.to_csv(index=False))
    else:
        try:
            plot_midpoint_pull(
                df_result,
                top_n=args.top_n,
                point_interval=args.point_interval,
                output_name=args.output,
                show=True,
            )
        except Exception:
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    run_tool()
