"""Convenience plotting functions for GPL analysis data.

These are ordinary user-code that calls ``GplDb`` accessor methods.
They live here so they can be shared by multiple analysis scripts
without duplication.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .gpl_db import GplDb


def path_slack_evolution(
    gpl: GplDb,
    top_n: int = 10,
    iter_range=None,
    min_separation=0.0,
    max_similarity=1.0,
    interactive: bool = True,
    output_file: str = "slack_evolution.png",
):
    """Plot slack progress for critical paths.

    See ``GplDb.worst_paths_history`` for the meaning of
    *iter_range*, *min_separation*, and *max_similarity*.
    """
    paths = gpl.worst_paths_history(
        top_n=top_n,
        iter_range=iter_range,
        min_separation=min_separation,
        max_similarity=max_similarity,
    )
    if not paths:
        print("No path slack history available.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    for pdf in paths:
        pdf = pdf.sort_values("Iter")
        pid = pdf["PhysicalPathId"].iloc[0]
        ax.plot(pdf["Iter"], pdf["Slack"], label=f"Path {pid}", alpha=0.6)

    # Worst-negative-slack trend across all selected paths
    combined = pd.concat(paths, ignore_index=True)
    wns = combined.groupby("Iter")["Slack"].min().reset_index()
    ax.plot(
        wns["Iter"], wns["Slack"],
        color="black", linewidth=3, label="WNS Trend", linestyle="--",
    )

    ax.set_title(f"Slack Evolution (Top {top_n} Worst Paths)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Slack (ps)")
    ax.grid(True, linestyle=":", alpha=0.6)

    if len(paths) <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    _show_or_save(interactive, output_file)


def force_magnitude_comparison(
    gpl: GplDb,
    top_n_paths: int = 5,
    interactive: bool = True,
    output_file: str = "force_comparison.png",
):
    """Average WL and timing force magnitudes for cells on critical paths."""
    slacks = gpl.path_slacks()
    if slacks.empty:
        print("No path slack data.")
        return
    worst = (
        slacks.groupby("PathId")["Slack"]
        .min()
        .nsmallest(top_n_paths)
        .index.tolist()
    )

    cells = gpl.path_cells(path_ids=worst)
    if cells.empty:
        return
    cids = cells["CellId"].unique().tolist()

    forces = gpl.cell_gradient_metrics(cell_ids=cids)
    if forces.empty:
        return
    avg = forces.groupby("Iter")[["mag_wl", "mag_tim"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(avg["Iter"], avg["mag_wl"], label="Avg WL Force")
    ax.plot(avg["Iter"], avg["mag_tim"], label="Avg Timing Force")
    ax.set_title("Average Force Magnitude (Critical Path Cells)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Magnitude")
    ax.legend()
    plt.tight_layout()
    _show_or_save(interactive, output_file)


def cell_path_tortuosity(
    gpl: GplDb,
    top_n_paths: int = 5,
    interactive: bool = True,
    output_file: str = "tortuosity.png",
):
    """Tortuosity (path distance / straight-line distance) for cells on
    critical paths."""
    slacks = gpl.path_slacks()
    if slacks.empty:
        return
    worst = (
        slacks.groupby("PathId")["Slack"]
        .min()
        .nsmallest(top_n_paths)
        .index.tolist()
    )

    cells = gpl.path_cells(path_ids=worst)
    if cells.empty:
        return
    cids = cells["CellId"].unique().tolist()

    moves = gpl.cell_movements(cell_ids=cids)
    if moves.empty:
        return

    rows = []
    for cid, grp in moves.groupby("CellId"):
        grp = grp.sort_values("Iter")
        total = grp["Distance"].sum()
        sx, sy = grp.iloc[0]["PosX"], grp.iloc[0]["PosY"]
        ex, ey = grp.iloc[-1]["PosX"], grp.iloc[-1]["PosY"]
        straight = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
        tort = total / straight if straight > 0 else 0.0
        rows.append({"CellId": cid, "Tortuosity": tort, "TotalDist": total})

    df = pd.DataFrame(rows).sort_values("Tortuosity", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(df)), df["Tortuosity"])
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["CellId"].astype(int), rotation=45)
    ax.set_title("Path Tortuosity for Critical Cells")
    ax.set_ylabel("Tortuosity (path distance / straight-line dist)")
    plt.tight_layout()
    _show_or_save(interactive, output_file)


# ── helpers ──────────────────────────────────────────────────────

def _show_or_save(interactive: bool, path: str):
    if interactive:
        plt.show()
    else:
        plt.savefig(path)
        print(f"Plot saved to {path}")
        plt.close()
