#!/usr/bin/env python3
"""
Path Trajectory Animation Tool
==============================

Generates an MP4 video animating the evolution of the worst *n* timing paths
across all placement iterations that contain them.

Each path is plotted in a distinct colour with connecting lines (path order).
Force arrows (WL, Tim, Rt, Dens) are overlaid on each cell, normalised to a
uniform arrow length so that only *direction* is compared.

All paths share the same physical canvas.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

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
# Constants
# ---------------------------------------------------------------------------

FORCE_CONFIGS = [
    ("WL",   "WlX",                "WlY",                "tab:blue",   "Wirelength"),
    ("Tim",  "TimX",               "TimY",               "tab:red",    "Timing"),
    ("Rt",   "RtX",                "RtY",                "tab:green",  "Routability"),
    ("Dens", "EstDensityForceX",   "EstDensityForceY",   "tab:purple", "Density"),
]
FORCE_LABELS = {entry[0] for entry in FORCE_CONFIGS}

# Margins around data extent (fraction of span)
_PAD_FRAC = 0.10


def _filter_force_configs(wanted: list[str]) -> list[tuple]:
    """Return only force-config entries whose label is in *wanted*."""
    if not wanted:
        return FORCE_CONFIGS
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


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_frames(
    gpl: GplAnalysis,
    worst_phys_ids: list,
    force_configs: list,
) -> list[dict]:
    """
    Build a list of frame dicts, one per iteration present in ``path_cells``
    for the tracked paths.

    Each frame::

        {"iter": int,
         "cells": DataFrame with columns
             PhysicalPathId, CellId, PathSeq, PosX, PosY,
             plus force columns (may be 0-filled)}
    """
    df_sig, _ = gpl.path_signature_map
    df_pc, _ = gpl.path_cells

    # Attach stable PhysicalPathId
    df_pc = pd.merge(df_pc, df_sig, on=["PathId", "Iter"])
    df_pc = df_pc[df_pc["PhysicalPathId"].isin(worst_phys_ids)]

    if df_pc.empty:
        return []

    # Collect all (Iter, CellId) pairs needed, then fetch bulk force data
    all_pairs = df_pc[["Iter", "CellId"]].drop_duplicates()
    all_cell_ids = all_pairs["CellId"].unique().tolist()
    iter_min = int(all_pairs["Iter"].min())
    iter_max = int(all_pairs["Iter"].max())

    # ---- Bulk-fetch all force data ---------------------------------------
    fetch_cfg = [
        ("dense", gpl.get_cell_dense_gradients,
         ["PosX", "PosY", "WlX", "WlY"]),
        ("tim",   gpl.get_cell_timing_gradients,
         ["TimX", "TimY"]),
        ("rt",    gpl.get_cell_routability_gradients,
         ["RtX", "RtY"]),
        ("dens",  gpl.get_estimated_density_forces,
         ["EstDensityForceX", "EstDensityForceY"]),
    ]

    bulk = {}  # name -> DataFrame
    for name, method, _cols in fetch_cfg:
        chunks = []
        try:
            for chunk_df, _ in gpl.iter_chunked(
                method, iter_range=(iter_min, iter_max),
                cell_ids=all_cell_ids, chunk_size=200,
            ):
                chunks.append(chunk_df)
        except (ValueError, pd.errors.DatabaseError, Exception):
            pass  # sparse / unavailable
        bulk[name] = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    # ---- Build frames ----------------------------------------------------
    frames = []
    for it in sorted(df_pc["Iter"].unique()):
        it_cells = (
            df_pc[df_pc["Iter"] == it]
            [["PhysicalPathId", "CellId", "PathSeq"]]
            .sort_values("PathSeq")
            .copy()
        )

        # Merge force data
        for name, _method, cols in fetch_cfg:
            df_src = bulk[name]
            if df_src.empty:
                continue
            sub = df_src[df_src["Iter"] == it]
            if sub.empty:
                continue
            keep = [c for c in sub.columns if c != "Iter"]
            it_cells = pd.merge(it_cells, sub[keep], on="CellId", how="left")

        # Drop duplicate rows (same CellId appears twice in path —
        # rise/fall edges), keep first occurrence (lowest PathSeq)
        it_cells = it_cells.drop_duplicates(subset=["PhysicalPathId", "CellId"])

        # Fill NaN forces with 0
        for _label, fx, fy, _color, _desc in FORCE_CONFIGS:
            for col in (fx, fy):
                if col in it_cells.columns:
                    it_cells[col] = it_cells[col].fillna(0.0)

        # Ensure PathSeq order after merge may have been scrambled
        it_cells = it_cells.sort_values("PathSeq").reset_index(drop=True)

        frames.append({"iter": int(it), "cells": it_cells})

    return frames


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def render_video(frames: list[dict],
                 worst_phys_ids: list,
                 force_configs: list,
                 output_path: str,
                 fps: float = 2.0):
    """
    Render frames into an MP4 video.  Each frame is rendered with
    matplotlib, then stitched via ffmpeg (lossless PNG codec).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_paths = len(worst_phys_ids)
    path_colors = plt.cm.tab10(np.linspace(0, 1, n_paths))
    path_color_map = dict(zip(worst_phys_ids, path_colors))

    temp_dir = tempfile.mkdtemp(prefix="path_traj_")
    print(f"Rendering {len(frames)} frames to {temp_dir}")

    # Pre-compute global data bounds so the axes stay fixed
    all_x, all_y = [], []
    for fr in frames:
        cf = fr["cells"]
        all_x.extend(cf["PosX"].values)
        all_y.extend(cf["PosY"].values)
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_pad = max((x_max - x_min) * _PAD_FRAC, 1)
    y_pad = max((y_max - y_min) * _PAD_FRAC, 1)

    fig, ax = plt.subplots(figsize=(10, 10))

    for idx, fr in enumerate(frames):
        ax.cla()
        it = fr["iter"]
        cells = fr["cells"]

        xs_all = cells["PosX"].values
        ys_all = cells["PosY"].values
        plot_span = max(np.ptp(xs_all), np.ptp(ys_all)) if len(xs_all) > 1 else 1
        if plot_span < 1:
            plot_span = max(abs(xs_all).max(), abs(ys_all).max()) * 0.1 or 1
        arrow_len = plot_span * 0.08  # shorter arrows when many paths overlap

        # ---- Draw each path ----------------------------------------------
        for phys_id in worst_phys_ids:
            sub = cells[cells["PhysicalPathId"] == phys_id].copy()
            if sub.empty:
                continue

            # Sort in path order (some cells have duplicate PathSeq; stable sort)
            sub = sub.sort_values("PathSeq").drop_duplicates(subset="PathSeq")
            xs = sub["PosX"].values
            ys = sub["PosY"].values
            color = path_color_map[phys_id]

            # Connecting line
            ax.plot(xs, ys, "-", color=color, alpha=0.45, linewidth=1.5,
                    label=f"Path {phys_id}")

            # Cell markers
            ax.scatter(xs, ys, color=color, s=30, zorder=5, edgecolors="black",
                       linewidths=0.5)

        # ---- Force arrows (single quiver per force type) -----------------
        for label, fx_name, fy_name, arr_color, _desc in force_configs:
            if fx_name not in cells.columns:
                continue
            fx_vals = cells[fx_name].values
            fy_vals = cells[fy_name].values
            mags = np.sqrt(fx_vals ** 2 + fy_vals ** 2)

            eps = 1e-30
            mask = mags > eps
            if not mask.any():
                continue

            pos_x = cells["PosX"].values[mask]
            pos_y = cells["PosY"].values[mask]
            u = (fx_vals[mask] / mags[mask]) * arrow_len
            v = (fy_vals[mask] / mags[mask]) * arrow_len

            ax.quiver(pos_x, pos_y, u, v,
                      angles="xy", scale_units="xy", scale=1,
                      color=arr_color,
                      width=0.003,
                      headwidth=2.5,
                      headlength=3.5,
                      headaxislength=2.5,
                      alpha=0.6,
                      zorder=10)

        # ---- Axes styling ------------------------------------------------
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal")
        ax.set_title(f"Iteration {it}", fontsize=14, fontweight="bold")

        # Legend: unique path labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(),
                      fontsize=7, loc="upper right", ncol=2)

        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{idx:04d}.png")
        fig.savefig(frame_path, dpi=150, bbox_inches="tight",
                    pad_inches=0.1)
        if (idx + 1) % 5 == 0 or idx == len(frames) - 1:
            print(f"  Frame {idx + 1}/{len(frames)}")

    plt.close(fig)

    # ---- Stitch with ffmpeg (lossless PNG codec) -------------------------
    # Verify ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"],
                       capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: ffmpeg not found.  Install ffmpeg to generate MP4.")
        print(f"Frames saved individually in {temp_dir}")
        sys.exit(1)

    print(f"Encoding {output_path} ...")
    pattern = os.path.join(temp_dir, "frame_%04d.png")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "png",         # lossless PNG codec in MP4 container
        "-pix_fmt", "rgb24",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stderr:")
        print(result.stderr)
        raise RuntimeError(f"ffmpeg exited with code {result.returncode}")

    # Cleanup
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)

    file_size = os.path.getsize(output_path)
    print(f"Done — {output_path}  ({file_size / 1024 / 1024:.1f} MiB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_tool():
    parser = argparse.ArgumentParser(
        description="Animate the worst timing paths across all iterations.  "
                    "Generates a lossless MP4 with force vectors overlaid.",
    )
    parser.add_argument("db_path", help="Path to the SQLite database")
    parser.add_argument("-n", "--top-n", type=int, default=5, dest="top_n",
                        help="Number of worst paths (default: 5)")
    parser.add_argument("--top-n-min-sep", type=float, default=0.0,
                        help="Minimum slack separation between selected paths")
    parser.add_argument("--forces", type=str, default="",
                        help="Comma‑separated list of forces to plot: "
                             f"{', '.join(sorted(FORCE_LABELS))}.  "
                             "Default: all forces.")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Frames per second (default: 2)")
    parser.add_argument("-o", "--output", type=str,
                        default="path_trajectory.mp4",
                        help="Output video path (default: path_trajectory.mp4)")
    parser.add_argument("--force", action="store_true", dest="force_rebuild",
                        help="Rebuild derived tables")
    args = parser.parse_args()

    # Force filter
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

    # 1. Preprocess
    GplPreprocessor.run(args.db_path, force_rebuild=args.force_rebuild)

    # 2. Connect
    db = DatabaseLog(args.db_path)
    gpl = GplAnalysis(db)

    # 3. Select worst paths at final iteration
    slacks, _ = gpl.path_slacks
    final_iter = int(slacks["Iter"].max())
    df_sig, _ = gpl.path_signature_map

    final_slacks = slacks[slacks["Iter"] == final_iter].copy()
    final_slacks = pd.merge(final_slacks, df_sig, on=["PathId", "Iter"])

    worst_ids = _greedy_select_worst_paths(final_slacks, args.top_n,
                                           args.top_n_min_sep)
    if not worst_ids:
        print("No paths selected.")
        sys.exit(1)

    print(f"Selected {len(worst_ids)} worst paths at iter {final_iter}:")
    for pid in worst_ids:
        row = final_slacks[final_slacks["PhysicalPathId"] == pid].iloc[0]
        print(f"  Path {pid}:  Slack = {row['Slack']:.3e}")

    # 4. Collect frames
    frames = collect_frames(gpl, worst_ids, active_configs)
    if not frames:
        print("No frame data collected.")
        sys.exit(1)

    print(f"Collected {len(frames)} frames: "
          f"iters {frames[0]['iter']} – {frames[-1]['iter']}")

    # 5. Render video
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    render_video(frames, worst_ids, active_configs,
                 output_path=args.output, fps=args.fps)


if __name__ == "__main__":
    run_tool()
