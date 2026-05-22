#!/usr/bin/env python3
"""
Path Force Vector Map Tool
==========================

For each of the top *n* worst paths (at the final logged iteration), find the
first iteration at which a non-zero timing force exists.  For that iteration,
plot the *physical* positions of every cell in that path, connected in path
order (by *PathSeq*), with labelled arrows showing all four force components
acting on each cell:

  - **WL** (wirelength gradient)  — blue
  - **Tim** (timing gradient)     — red
  - **Rt** (routability gradient) — green  *(sparse; omitted if zero)*
  - **Dens** (density force)      — purple *(estimated via summed-area table)*

All force vectors are **normalised to a uniform length** so that only
*direction* is shown.  Zero‑magnitude force vectors are omitted entirely.

Usage
-----
::

    ./path_force_map_tool.py <db_path> -n 5

    # or via PYTHONPATH:
    PYTHONPATH=…/tools python3 -m database_log_analysis.path_force_map_tool \\
        <db_path> --interactive False
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
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

# Force-configuration tuples: (label, x-col, y-col, colour, nice name)
FORCE_CONFIGS = [
    ("WL",   "WlX",                "WlY",                "tab:blue",   "Wirelength"),
    ("Tim",  "TimX",               "TimY",               "tab:red",    "Timing"),
    ("Rt",   "RtX",                "RtY",                "tab:green",  "Routability"),
    ("Dens", "EstDensityForceX",   "EstDensityForceY",   "tab:purple", "Density"),
]

# Labels that can be passed to ``--forces``
FORCE_LABELS = {entry[0] for entry in FORCE_CONFIGS}


def _filter_force_configs(wanted: list[str]) -> list[tuple]:
    """Return only the force-config entries whose label is in *wanted*."""
    if not wanted:
        return FORCE_CONFIGS  # show all
    wanted_set = set(wanted)
    return [e for e in FORCE_CONFIGS if e[0] in wanted_set]


def _greedy_select_worst_paths(final_slacks: pd.DataFrame,
                                top_n: int,
                                min_sep: float) -> list:
    """Greedy selection of worst paths with minimum slack separation."""
    sorted_df = final_slacks.sort_values("Slack")
    selected = []
    last_slack = None
    for _, row in sorted_df.iterrows():
        sk = row["Slack"]
        if last_slack is None:
            selected.append(row["PhysicalPathId"])
            last_slack = sk
        elif sk >= last_slack + min_sep:
            selected.append(row["PhysicalPathId"])
            last_slack = sk
        if len(selected) >= top_n:
            break
    return selected


def compute_path_force_map(
    gpl: GplAnalysis,
    top_n: int = 5,
    top_n_min_sep: float = 0.0,
) -> dict:
    """
    Collect cell positions and force vectors for the worst *top_n* paths.

    Returns a dict ``{PhysicalPathId: {...}}`` where each value contains:

    - **cells**       — ``DataFrame`` with columns ``CellId, PathSeq, PosX,
                        PosY, WlX, WlY, TimX, TimY, RtX, RtY,
                        EstDensityForceX, EstDensityForceY`` (sorted by
                        PathSeq)
    - **slack**       — path slack at the final iteration
    - **first_iter**  — the iteration at which the force snapshot was taken
    """
    # ---- 1.  Find the final iteration and worst paths ---------------------
    df_slacks, _ = gpl.path_slacks
    if df_slacks.empty:
        print("Error: no path slack data found.")
        return {}

    final_iter = int(df_slacks["Iter"].max())
    df_sig, _ = gpl.path_signature_map
    has_sig = not df_sig.empty

    final_slacks = df_slacks[df_slacks["Iter"] == final_iter].copy()
    if has_sig:
        final_slacks = pd.merge(final_slacks, df_sig, on=["PathId", "Iter"])
    else:
        print("Warning: path_signature_map missing — using unstable PathId.")
        final_slacks["PhysicalPathId"] = final_slacks["PathId"]

    worst_phys_ids = _greedy_select_worst_paths(final_slacks, top_n,
                                                 top_n_min_sep)
    print(f"Selected {len(worst_phys_ids)} worst paths at iter {final_iter}:")
    for pid in worst_phys_ids:
        r = final_slacks[final_slacks["PhysicalPathId"] == pid].iloc[0]
        print(f"  Path {pid}:  Slack = {r['Slack']:.3e}")

    # ---- 2.  Cell membership for these paths ------------------------------
    df_path_cells, _ = gpl.path_cells
    if df_path_cells.empty:
        print("Error: gpl_path_cells table is empty.")
        return {}

    if has_sig:
        df_sig2 = pd.merge(df_path_cells, df_sig, on=["PathId", "Iter"])
        df_sig2 = df_sig2[df_sig2["PhysicalPathId"].isin(worst_phys_ids)]
    else:
        df_sig2 = df_path_cells[
            df_path_cells["PathId"].isin(worst_phys_ids)
        ].copy()
        df_sig2["PhysicalPathId"] = df_sig2["PathId"]

    if df_sig2.empty:
        print("Error: no path cell data for worst paths.")
        return {}

    all_cell_ids = df_sig2["CellId"].unique().tolist()

    # ---- 3.  Timing gradients — find the first non-zero iteration ---------
    tim_chunks = []
    for chunk_df, _ in gpl.iter_chunked(gpl.get_cell_timing_gradients,
                                          cell_ids=all_cell_ids,
                                          chunk_size=50):
        tim_chunks.append(chunk_df)
    df_tim_all = pd.concat(tim_chunks, ignore_index=True) if tim_chunks \
                 else pd.DataFrame()

    if df_tim_all.empty:
        print("Error: no timing gradient data at all.")
        return {}

    # Merge with path membership
    df_tim_path = pd.merge(df_tim_all,
                           df_sig2[["Iter", "CellId", "PhysicalPathId"]],
                           on=["Iter", "CellId"], how="inner")
    df_tim_path["TimMag"] = np.sqrt(df_tim_path["TimX"] ** 2
                                    + df_tim_path["TimY"] ** 2)
    df_tim_nz = df_tim_path[df_tim_path["TimMag"] > 0]

    # First iteration with non‑zero timing force per path
    first_tim_iter = df_tim_nz.groupby("PhysicalPathId")["Iter"].min()

    # ---- 4.  Gather force snapshot for each path --------------------------
    results = {}
    for phys_id in worst_phys_ids:
        if phys_id not in first_tim_iter.index:
            print(f"  Path {phys_id}: no non-zero timing force — skipping")
            continue

        target_iter = int(first_tim_iter[phys_id])
        print(f"  Path {phys_id}: snapshot at iteration {target_iter}")

        # Cells in this path at this iteration, ordered by PathSeq
        path_cells = (
            df_sig2[(df_sig2["PhysicalPathId"] == phys_id)
                    & (df_sig2["Iter"] == target_iter)]
            .sort_values("PathSeq")
            [["CellId", "PathSeq"]]
            .drop_duplicates(subset="CellId")
        )
        if path_cells.empty:
            continue
        cids = path_cells["CellId"].tolist()

        # Positions + WL gradients
        df_pos, _ = gpl.get_cell_dense_gradients(
            iter_range=(target_iter, target_iter), cell_ids=cids)

        # Timing gradients
        df_tim, _ = gpl.get_cell_timing_gradients(
            iter_range=(target_iter, target_iter), cell_ids=cids)

        # Routability gradients (sparse — may be empty table)
        df_rt, _ = gpl.get_cell_routability_gradients(
            iter_range=(target_iter, target_iter), cell_ids=cids)

        # Density forces (estimated)
        df_dens = pd.DataFrame()
        try:
            df_dens, _ = gpl.get_estimated_density_forces(
                iter_range=(target_iter, target_iter), cell_ids=cids)
        except (ValueError, KeyError) as exc:
            print(f"    Density estimation unavailable: {exc}")

        # Merge into a single row per cell
        cells = path_cells.copy()
        for df_src in [df_pos, df_tim, df_rt, df_dens]:
            if not df_src.empty:
                keep = [c for c in df_src.columns if c != "Iter"]
                cells = pd.merge(cells, df_src[keep], on="CellId", how="left")

        # Fill sparsely missing force columns with 0
        for _label, fx_name, fy_name, _color, _desc in FORCE_CONFIGS:
            for col in (fx_name, fy_name):
                if col in cells.columns:
                    cells[col] = cells[col].fillna(0.0)

        slack_row = final_slacks[final_slacks["PhysicalPathId"] == phys_id]
        slack = slack_row["Slack"].iloc[0] if not slack_row.empty else float("nan")

        results[phys_id] = {
            "cells": cells.sort_values("PathSeq").reset_index(drop=True),
            "slack": slack,
            "first_iter": target_iter,
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_path_force_map(
    result: dict,
    top_n: int = 5,
    output_name: str = "path_force_map.png",
    show: bool = False,
    force_configs: list = FORCE_CONFIGS,
):
    """One sub‑plot per path showing cell positions and force arrows."""
    n_paths = len(result)
    if n_paths == 0:
        print("Nothing to plot.")
        return

    n_cols = min(2, n_paths)
    n_rows = (n_paths + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(10, 5 * n_rows))
    if n_paths == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    # Colour per path (tab10 cycles)
    path_colors = plt.cm.tab10(np.linspace(0, 1, n_paths))

    for idx, (phys_id, data) in enumerate(result.items()):
        ax = axes_flat[idx]
        cells = data["cells"]
        color = path_colors[idx]

        xs = cells["PosX"].values
        ys = cells["PosY"].values

        # ---- Connecting lines (path order) --------------------------------
        ax.plot(xs, ys, "-", color=color, alpha=0.5, linewidth=2,
                label=f"Path {phys_id}")

        # ---- Cell nodes ---------------------------------------------------
        ax.scatter(xs, ys, color=color, s=60, zorder=5, edgecolors="black",
                   linewidths=0.7)

        # ---- Cell labels (showing CellId) ---------------------------------
        for _, row in cells.iterrows():
            ax.annotate(str(int(row["CellId"])),
                        (row["PosX"], row["PosY"]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, alpha=0.8)

        # ---- Determine a uniform arrow length for this sub‑plot ------------
        plot_span = max(np.ptp(xs), np.ptp(ys)) if np.ptp(xs) > 0 or np.ptp(ys) > 0 else 1
        # fallback when all cells sit at the same point
        if plot_span < 1:
            plot_span = max(abs(xs).max(), abs(ys).max()) * 0.1 or 1

        arrow_len = plot_span * 0.15  # all non‑zero vectors get this length (direction only)

        # ---- Force arrows per force type (batch quiver) --------------------
        #
        # Vectors are **normalised** to a fixed length so that DIRECTION is
        # compared, not magnitude.  Zero‑force vectors are omitted entirely.
        for label, fx_name, fy_name, arr_color, _desc in force_configs:
            if fx_name not in cells.columns:
                continue

            fx_vals = cells[fx_name].values
            fy_vals = cells[fy_name].values
            mags    = np.sqrt(fx_vals ** 2 + fy_vals ** 2)

            # Use a very low threshold: timing forces can be as small as
            # 1e‑19, so the old 1e‑15 threshold discarded them entirely.
            eps = 1e-30
            mask = mags > eps
            if not mask.any():
                continue

            pos_x = cells["PosX"].values[mask]
            pos_y = cells["PosY"].values[mask]

            # Normalise → unit vector, then scale to fixed arrow length
            u = (fx_vals[mask] / mags[mask]) * arrow_len
            v = (fy_vals[mask] / mags[mask]) * arrow_len

            ax.quiver(pos_x, pos_y, u, v,
                      angles="xy", scale_units="xy", scale=1,
                      color=arr_color,
                      width=0.004,
                      headwidth=3,
                      headlength=4,
                      headaxislength=3,
                      alpha=0.85,
                      zorder=10)

            # Label at the midpoint of each shaft
            for i in range(len(pos_x)):
                ax.text(pos_x[i] + u[i] * 0.5,
                        pos_y[i] + v[i] * 0.5,
                        label,
                        fontsize=6, color=arr_color, fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.1",
                                  facecolor="white", edgecolor="none",
                                  alpha=0.7))

        ax.set_title(
            f"Path {phys_id}  (iter {data['first_iter']}, "
            f"slack {data['slack']:.2e})",
            fontsize=10,
        )
        ax.set_xlabel("X (dbu)")
        ax.set_ylabel("Y (dbu)")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal")

    # Hide unused sub‑plots
    for j in range(n_paths, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plots_dir = os.path.join(_current_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, output_name)
    plt.savefig(path, bbox_inches="tight", dpi=800)
    print(f"Plot saved to: {path}")

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
        description="Visualise cell positions and all four force components "
                    "for the worst timing paths at the first non‑zero timing "
                    "force iteration.",
    )
    parser.add_argument("db_path", help="Path to the SQLite database")
    parser.add_argument("-n", "--top-n", type=int, default=5, dest="top_n",
                        help="Number of worst paths (default: 5)")
    parser.add_argument("--top-n-min-sep", type=float, default=0.0,
                        help="Minimum slack separation between selected paths")
    parser.add_argument("--interactive", type=str, default="True",
                        help="True → plot; False → CSV (default: True)")
    parser.add_argument("--forces", type=str, default="",
                        help="Comma‑separated list of forces to plot: "
                             f"{', '.join(sorted(FORCE_LABELS))}.  "
                             "Default: all forces.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild derived tables")
    parser.add_argument("-o", "--output", type=str,
                        default="path_force_map.png",
                        help="Output plot filename")
    args = parser.parse_args()

    is_interactive = args.interactive.lower() == "true"

    # Parse force filter (case‑insensitive matching)
    if args.forces:
        label_map = {lbl.lower(): lbl for lbl in FORCE_LABELS}
        wanted_raw = [s.strip() for s in args.forces.split(",")]
        wanted = []
        bad = []
        for w in wanted_raw:
            canonical = label_map.get(w.lower())
            if canonical is None:
                bad.append(w)
            else:
                wanted.append(canonical)
        if bad:
            print(f"Unknown force label(s): {', '.join(bad)}. "
                  f"Choose from: {', '.join(sorted(FORCE_LABELS))}")
            sys.exit(1)
        active_configs = _filter_force_configs(wanted)
        print(f"Showing forces: {', '.join(sorted(wanted))}")
    else:
        active_configs = FORCE_CONFIGS
        print(f"Showing all forces: {', '.join(sorted(FORCE_LABELS))}")

    # 1. Preprocessing
    GplPreprocessor.run(args.db_path, force_rebuild=args.force)

    # 2. Connect
    db = DatabaseLog(args.db_path)
    gpl = GplAnalysis(db)

    # 3. Compute
    result = compute_path_force_map(gpl, top_n=args.top_n,
                                     top_n_min_sep=args.top_n_min_sep)
    if not result:
        print("No results.")
        sys.exit(1)

    # 4. Output
    if not is_interactive:
        rows = []
        for phys_id, data in result.items():
            for _, cell in data["cells"].iterrows():
                rows.append({
                    "PhysicalPathId": phys_id,
                    "Iter": data["first_iter"],
                    "CellId": cell["CellId"],
                    "PathSeq": cell["PathSeq"],
                    "PosX": cell["PosX"],
                    "PosY": cell["PosY"],
                    "WlX": cell.get("WlX", 0.0),
                    "WlY": cell.get("WlY", 0.0),
                    "TimX": cell.get("TimX", 0.0),
                    "TimY": cell.get("TimY", 0.0),
                    "RtX": cell.get("RtX", 0.0),
                    "RtY": cell.get("RtY", 0.0),
                    "DensX": cell.get("EstDensityForceX", 0.0),
                    "DensY": cell.get("EstDensityForceY", 0.0),
                    "Slack": data["slack"],
                })
        print(pd.DataFrame(rows).to_csv(index=False))
    else:
        try:
            plot_path_force_map(result, top_n=args.top_n,
                                output_name=args.output,
                                show=True,
                                force_configs=active_configs)
        except Exception:
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    run_tool()
