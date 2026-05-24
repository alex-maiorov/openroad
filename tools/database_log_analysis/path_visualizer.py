#!/usr/bin/env python3
"""Dash GUI for tracing worst-slacked timing paths in GPL placement.

Shows the full-iteration trajectory of every cell on the worst timing
paths, then overlays the path connectivity and force arrows at the user-
selected snapshot iteration.

Usage
-----
    python -m tools.database_log_analysis.path_visualizer \\
        [--db path/to/placement-visualization.sqlite] \\
        [--port 8050]

Requirements: dash, plotly, pandas, numpy
"""

import argparse
import sys
import os

import dash
from dash import dcc, html, Input, Output, State, DiskcacheManager
import diskcache
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import pandas as pd

# Ensure the tools/ package is importable
_TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from database_log_analysis import GplDb

# ══════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════

FORCE_CONFIG = {
    "wl":       {"label": "Wirelength", "color": "#1f77b4", "sym": "circle",
                 "xcol": "WlX", "ycol": "WlY"},
    "tim":      {"label": "Timing",     "color": "#d62728", "sym": "square",
                 "xcol": "TimX", "ycol": "TimY"},
    "density":  {"label": "Density",    "color": "#2ca02c", "sym": "diamond",
                 "xcol": "EstDensityForceX", "ycol": "EstDensityForceY"},
    "effective":{"label": "Effective",  "color": "#9467bd", "sym": "star",
                 "xcol": "EffectiveX", "ycol": "EffectiveY"},
}
FORCE_ORDER = ["wl", "tim", "density", "effective"]
_PATH_COLORS = px.colors.qualitative.Plotly + px.colors.qualitative.Set1


# ══════════════════════════════════════════════════════════════════
#  Data helpers  —  all use existing GplDb methods
# ══════════════════════════════════════════════════════════════════

def _get_iter_range(gpl):
    """Return (min_iter, max_iter) from the dense-gradients table."""
    df = gpl.query(
        "SELECT MIN(Iter) AS mi, MAX(Iter) AS ma FROM gpl_cell_dense_gradients"
    )
    return int(df["mi"].iloc[0]), int(df["ma"].iloc[0])


def _get_slack_range_ps(gpl):
    """Return (min_slack_ps, max_slack_ps) across all PhysicalPathIds."""
    if not gpl._exists("gpl_path_signatures"):
        return -500, 500
    df = gpl.query("""
        SELECT MIN(ps.Slack) AS mins, MAX(ps.Slack) AS maxs
        FROM gpl_path_slacks ps
        JOIN gpl_path_signatures sig
          ON ps.PathId = sig.PathId AND ps.Iter = sig.Iter
    """)
    if df.empty or df["mins"].isna().all():
        return -500, 500
    lo = float(df["mins"].iloc[0]) * 1e12
    hi = float(df["maxs"].iloc[0]) * 1e12
    return lo, hi


def _get_design_bounds(gpl):
    """Return (xmin, xmax, ymin, ymax) from all cell positions.

    Queries the global min/max of cell coordinates across all
    iterations in ``gpl_cell_dense_gradients``.
    """
    df = gpl.query(
        "SELECT MIN(PosX) AS xmin, MAX(PosX) AS xmax, "
        "MIN(PosY) AS ymin, MAX(PosY) AS ymax "
        "FROM gpl_cell_dense_gradients"
    )
    if df.empty or df["xmin"].isna().all():
        return None
    return (float(df["xmin"].iloc[0]), float(df["xmax"].iloc[0]),
            float(df["ymin"].iloc[0]), float(df["ymax"].iloc[0]))


def _compute_default_arrow_length(des_bounds):
    """Return a sensible default arrow length in data coordinates.

    Uses ~3 % of the design diagonal so arrows are visible but don't
    overwhelm the plot at the default multiplier of 1.0.
    """
    if des_bounds is None:
        return 1000.0
    dx = des_bounds[1] - des_bounds[0]
    dy = des_bounds[3] - des_bounds[2]
    diag = (dx * dx + dy * dy) ** 0.5
    return diag * 0.03


def _get_worst_phys_ids(gpl, top_n, iter_range, min_sep_ps, max_sim,
                         slack_range_ps=None):
    """Use GplDb.worst_paths_history to find top pathological paths.

    Returns list of (PhysicalPathId, min_slack_ps).
    """
    if not gpl._exists("gpl_path_slacks"):
        return []
    histories = gpl.worst_paths_history(
        top_n=top_n,
        iter_range=iter_range,
        min_separation=min_sep_ps * 1e-12,
        max_similarity=max_sim,
    )
    results = []
    for pdf in histories:
        pid = int(pdf["PhysicalPathId"].iloc[0])
        slack_ps = float(pdf["Slack"].min()) * 1e12
        if slack_range_ps:
            lo, hi = slack_range_ps
            if not (lo <= slack_ps <= hi):
                continue
        results.append((pid, slack_ps))
    # Deduplicate
    seen = set()
    deduped = []
    for pid, sps in results:
        if pid not in seen:
            seen.add(pid)
            deduped.append((pid, sps))
    return deduped


# ── PhysicalPathId → cell sequence ──────────────────────────────

def _get_path_cell_sequences(gpl, phys_ids):
    """Return dict {PhysicalPathId: [CellId, …]} in PathSeq order.

    The cell sequence for a PhysicalPathId is **stable** — it never
    changes across iterations (it *defines* the identity of the path).
    """
    if not phys_ids:
        return {}
    placeholders = ",".join("?" for _ in phys_ids)
    # Pick the first iteration where each PhysicalPathId appears, then
    # fetch its PathId + cells.
    df = gpl.query(f"""
        SELECT first.PhysicalPathId, pc.CellId, pc.PathSeq
        FROM (
            SELECT PhysicalPathId, PathId, MIN(Iter) AS Iter
            FROM gpl_path_signatures
            WHERE PhysicalPathId IN ({placeholders})
            GROUP BY PhysicalPathId
        ) first
        JOIN gpl_path_cells pc
          ON pc.PathId = first.PathId AND pc.Iter = first.Iter
        ORDER BY first.PhysicalPathId, pc.PathSeq
    """, phys_ids)
    if df.empty:
        return {}
    result = {}
    for ppid, grp in df.groupby("PhysicalPathId"):
        result[int(ppid)] = grp["CellId"].astype(int).tolist()
    return result


# ── Cell positions across ALL iterations ────────────────────────

def _get_cell_trajectories(gpl, all_cell_ids):
    """Return DataFrame with (CellId, Iter, PosX, PosY) for every
    iteration *all_cell_ids* appear in ``gpl_cell_dense_gradients``.

    Result is sorted by (CellId, Iter).
    """
    if not all_cell_ids:
        return pd.DataFrame()
    placeholders = ",".join("?" for _ in all_cell_ids)
    df = gpl.query(f"""
        SELECT CellId, Iter, PosX, PosY
        FROM gpl_cell_dense_gradients
        WHERE CellId IN ({placeholders})
        ORDER BY CellId, Iter
    """, all_cell_ids)
    return df


# ── Snapshot forces at a single iteration ────────────────────────

def _build_snapshot_force_df(gpl, cell_ids, iteration):
    """Return one-row-per-CellId DataFrame with all 4 force components
    at *iteration*."""
    dense = gpl.cell_dense_gradients(
        iter_range=(iteration, iteration), cell_ids=cell_ids
    )
    if dense.empty:
        return None
    df = dense[["CellId", "PosX", "PosY", "WlX", "WlY"]].copy()

    timing = gpl.cell_timing_gradients(
        iter_range=(iteration, iteration), cell_ids=cell_ids
    )
    if timing.empty:
        df["TimX"] = df["TimY"] = 0.0
    else:
        df = df.merge(timing[["CellId", "TimX", "TimY"]],
                       on="CellId", how="left")
        df[["TimX", "TimY"]] = df[["TimX", "TimY"]].fillna(0.0)

    dens = gpl.cell_density_forces(
        iter_range=(iteration, iteration), cell_ids=cell_ids
    )
    if dens.empty:
        df["EstDensityForceX"] = df["EstDensityForceY"] = 0.0
    else:
        df = df.merge(
            dens[["CellId", "EstDensityForceX", "EstDensityForceY"]],
            on="CellId", how="left"
        )
        df[["EstDensityForceX", "EstDensityForceY"]] = (
            df[["EstDensityForceX", "EstDensityForceY"]].fillna(0.0)
        )

    df["EffectiveX"] = df["WlX"] + df["TimX"] + df["EstDensityForceX"]
    df["EffectiveY"] = df["WlY"] + df["TimY"] + df["EstDensityForceY"]
    return df


# ══════════════════════════════════════════════════════════════════
#  Trace builders
# ══════════════════════════════════════════════════════════════════

def _build_trajectory_trace(cell_ids_in_path, traj_df, path_label,
                            path_color, iter_min, iter_max):
    """Build one Scattergl trace for a single path.

    Plots every (PosX, PosY) across ALL iterations for every cell in the
    path, coloring the markers by Iter so you can see the migration
    timeline.
    """
    sub = traj_df[traj_df["CellId"].isin(cell_ids_in_path)].copy()
    if sub.empty:
        return None

    sub = sub.sort_values(["CellId", "Iter"])

    return go.Scattergl(
        x=sub["PosX"].tolist(),
        y=sub["PosY"].tolist(),
        mode="markers",
        marker=dict(
            size=3,
            color=sub["Iter"].tolist(),
            colorscale=[
                [0.0, "rgba(200,200,255,0.15)"],
                [0.3, "rgba(100,100,255,0.4)"],
                [0.6, "rgba(255,100,100,0.6)"],
                [1.0, path_color],
            ],
            cmin=iter_min,
            cmax=iter_max,
            colorbar=dict(
                title="Iter",
                thickness=12,
                len=0.4,
                x=1.04,
                y=0.7,
            ),
            line=dict(width=0),
            opacity=0.6,
        ),
        name=f"{path_label} (trajectory)",
        legendgroup=path_label,
        showlegend=False,
        hoverinfo="skip",
    )


def _build_snapshot_path_trace(cell_ids_in_path, traj_df, iteration,
                                path_label, path_color, force_df=None):
    """Build traces for the path at *iteration*.

    Returns a list of up to two traces:
      1. lines+markers connecting cells in PathSeq order
      2. a dotted straight line from the first cell to the last cell
         (shows how direct the path is vs. its actual routing).
    """
    sub = traj_df[
        (traj_df["CellId"].isin(cell_ids_in_path)) &
        (traj_df["Iter"] == iteration)
    ].copy()
    if sub.empty:
        return []

    # Sort by cell index in the path sequence
    seq_order = {cid: i for i, cid in enumerate(cell_ids_in_path)}
    sub["_seq"] = sub["CellId"].map(seq_order)
    sub = sub.sort_values("_seq")

    # ── Build hover customdata (CellId, Seq, Pos, forces) ─────
    # Lookup table: CellId → (WlX, WlY, TimX, TimY, DensX, DensY, EffX, EffY)
    force_lookup = {}
    if force_df is not None and not force_df.empty:
        for _, r in force_df.iterrows():
            force_lookup[int(r["CellId"])] = (
                float(r.get("WlX", 0)), float(r.get("WlY", 0)),
                float(r.get("TimX", 0)), float(r.get("TimY", 0)),
                float(r.get("EstDensityForceX", 0)),
                float(r.get("EstDensityForceY", 0)),
                float(r.get("EffectiveX", 0)),
                float(r.get("EffectiveY", 0)),
            )

    custom_rows = []
    for _, row in sub.iterrows():
        cid = int(row["CellId"])
        f = force_lookup.get(cid, (0.0,) * 8)
        custom_rows.append((
            cid,
            int(row["_seq"]),
            float(row["PosX"]),
            float(row["PosY"]),
        ) + f)

    traces = []

    # ── Trace 1: the actual path ────────────────────────────
    traces.append(go.Scatter(
        x=sub["PosX"].tolist(),
        y=sub["PosY"].tolist(),
        mode="lines+markers",
        line=dict(color=path_color, width=3),
        marker=dict(
            size=9,
            color=path_color,
            symbol="circle",
            line=dict(width=1.5, color="white"),
        ),
        name=f"{path_label}",
        legendgroup=path_label,
        showlegend=True,
        hovertemplate=(
            f"<b>{path_label}</b><br>"
            f"Iter: {iteration}<br>"
            "Cell: %{customdata[0]}<br>"
            "Seq: %{customdata[1]}<br>"
            "Pos: (%{customdata[2]:.0f}, %{customdata[3]:.0f})<br>"
            "<b>Gradients</b><br>"
            "WL:  (%{customdata[4]:.4f}, %{customdata[5]:.4f})<br>"
            "Tim: (%{customdata[6]:.4f}, %{customdata[7]:.4f})<br>"
            "Den: (%{customdata[8]:.4f}, %{customdata[9]:.4f})<br>"
            "Eff: (%{customdata[10]:.4f}, %{customdata[11]:.4f})<extra></extra>"
        ),
        customdata=np.array(custom_rows, dtype=float),
    ))

    # ── Trace 2: straight line from first to last cell ──────
    if len(sub) >= 2:
        first = sub.iloc[0]
        last = sub.iloc[-1]
        traces.append(go.Scatter(
            x=[float(first["PosX"]), float(last["PosX"])],
            y=[float(first["PosY"]), float(last["PosY"])],
            mode="lines",
            line=dict(color=path_color, width=1.5, dash="dot"),
            name=f"{path_label} (end-to-end)",
            legendgroup=path_label,
            showlegend=False,
            hoverinfo="skip",
        ))

    return traces


def _build_arrow_traces(force_df, fkey, arrow_length, color, legendgroup):
    """Build quiver traces for one force type at the snapshot
    iteration.

    All arrows are normalised to the same length *arrow_length* so the
    user can compare directions without magnitude bias.  The hover
    tooltip shows the cell ID, position, and all four force components
    (WL, timing, density, effective) so the user can inspect the
    actual gradient values for any cell.
    """
    x = force_df["PosX"].values
    y = force_df["PosY"].values
    cfg = FORCE_CONFIG[fkey]
    fx = force_df[cfg["xcol"]].values
    fy = force_df[cfg["ycol"]].values

    mag = np.hypot(fx, fy)
    keep = mag > 1e-30
    if not keep.any():
        return []

    xn, yn = x[keep], y[keep]
    # Normalise to unit vectors, then scale uniformly to *arrow_length*
    fxn = fx[keep] / mag[keep] * arrow_length
    fyn = fy[keep] / mag[keep] * arrow_length

    fig_q = ff.create_quiver(
        x=xn, y=yn, u=fxn, v=fyn,
        scale=1.0,
        arrow_scale=0.15,
        angle=np.pi / 7,
        name=f"{cfg['label']} force",
        line_color=color,
        line_width=1.5
    )

    trace = fig_q.data[0]
    trace.legendgroup = legendgroup
    trace.showlegend = False

    # ── Build hover customdata ────────────────────────────────
    # Each arrow renders as 7 Plotly points: 3 for the shaft
    # (start, end, None) + 4 for the arrow-head lines.  We repeat
    # the same cell/force info for all 7 points of each arrow.
    n_arr = keep.sum()
    kept_df = force_df.iloc[np.where(keep)[0]]

    rows = []
    for i in range(n_arr):
        r = kept_df.iloc[i]
        rows.append((
            int(r["CellId"]),
            float(r["PosX"]), float(r["PosY"]),
            float(r["WlX"]), float(r["WlY"]),
            float(r["TimX"]), float(r["TimY"]),
            float(r["EstDensityForceX"]), float(r["EstDensityForceY"]),
            float(r["EffectiveX"]), float(r["EffectiveY"]),
        ))
    # Repeat each cell's info for every plotly point that makes up
    # one quiver arrow (the exact count varies by plotly version).
    n_per_arrow = len(trace.x) // n_arr
    trace.customdata = np.repeat(rows, n_per_arrow, axis=0)
    trace.hovertemplate = (
        f"<b>{cfg['label']} force</b><br>"
        "Cell %{customdata[0]}<br>"
        "Pos: (%{customdata[1]:.0f}, %{customdata[2]:.0f})<br>"
        "<b>Gradients</b><br>"
        "WL:  (%{customdata[3]:.4f}, %{customdata[4]:.4f})<br>"
        "Tim: (%{customdata[5]:.4f}, %{customdata[6]:.4f})<br>"
        "Den: (%{customdata[7]:.4f}, %{customdata[8]:.4f})<br>"
        "Eff: (%{customdata[9]:.4f}, %{customdata[10]:.4f})<extra></extra>"
    )

    return [trace]


# ══════════════════════════════════════════════════════════════════
#  Dash Application
# ══════════════════════════════════════════════════════════════════

def make_app(db_path, read_only=False):
    """Create and return a Dash application instance."""
    if read_only:
        gpl = GplDb(db_path, must_be_preprocessed=True)
    else:
        gpl = GplDb(db_path)

    iter_min, iter_max = _get_iter_range(gpl)
    slack_lo, slack_hi = _get_slack_range_ps(gpl)
    slack_lo = np.floor(slack_lo / 5) * 5
    slack_hi = np.ceil(slack_hi / 5) * 5
    if slack_lo >= slack_hi:
        slack_hi = slack_lo + 10

    # Design bounds (min/max of all cell positions) — computed once
    des_bounds = _get_design_bounds(gpl)

    # Default arrow length in data coordinates (~3 % of design diag.)
    _default_arrow_length = _compute_default_arrow_length(des_bounds)

    cache = diskcache.Cache("./tmp/dash_bg_cache")
    app = dash.Dash(
        __name__,
        background_callback_manager=DiskcacheManager(cache),
    )
    app.title = "GPL Path Visualizer"
    app._gpl = gpl

    # ── Force control row helper ──────────────────────────────
    def _force_row(fkey):
        cfg = FORCE_CONFIG[fkey]
        return html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "6px",
                   "flexWrap": "wrap", "marginBottom": "4px"},
            children=[
                dcc.Checklist(
                    id=f"force-{fkey}-toggle",
                    options=[{"label": "", "value": fkey}],
                    value=[fkey],
                    style={"display": "inline-block"},
                ),
                html.Span(cfg["label"], style={
                    "fontSize": "12px", "fontWeight": "bold",
                    "color": cfg["color"], "minWidth": "80px",
                }),
            ],
        )

    # ── Layout ───────────────────────────────────────────────
    app.layout = html.Div(
        style={"display": "flex", "height": "100vh", "margin": "0",
               "padding": "0", "fontFamily": "Segoe UI, Arial, sans-serif"},
        children=[
            # ── Sidebar ───────────────────────────────────────
            html.Div(
                id="sidebar",
                style={"width": "350px", "minWidth": "350px",
                       "padding": "16px 14px", "overflowY": "auto",
                       "backgroundColor": "#f8f9fa",
                       "borderRight": "1px solid #dee2e6",
                       "boxSizing": "border-box"},
                children=[
                    html.H3("GPL Path Visualizer",
                            style={"marginTop": "0", "marginBottom": "16px",
                                   "color": "#2c3e50"}),

                    html.Label("Iteration range for worst-path search",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div(f"Iters {iter_min}–{iter_max} (step 10 for path slacks)",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "4px"}),
                    dcc.RangeSlider(
                        id="iter-range-slider",
                        min=iter_min, max=iter_max, step=10,
                        value=[150, iter_max],
                        marks={i: str(i) for i in
                               range(iter_min, iter_max + 1, 50)},
                        tooltip={"placement": "bottom",
                                 "always_visible": False},
                    ),
                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Slack range (paths within this slack window)",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div(f"Slack: {slack_lo:.1f}–{slack_hi:.1f} ps",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "4px"}),
                    dcc.RangeSlider(
                        id="slack-range-slider",
                        min=slack_lo, max=slack_hi, step=5,
                        value=[slack_lo, slack_hi],
                        marks={s: f"{s:.0f}" for s in
                               np.linspace(slack_lo, slack_hi, 5)},
                        tooltip={"placement": "bottom",
                                 "always_visible": False},
                    ),
                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Path filtering",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div([
                        html.Div([
                            html.Label("Min slack separation (ps):",
                                       style={"fontSize": "12px"}),
                            dcc.Input(id="min-sep-input", type="number",
                                      value=0, step=1,
                                      style={"width": "100%",
                                             "boxSizing": "border-box"}),
                        ], style={"marginBottom": "8px"}),
                        html.Div([
                            html.Label("Max path similarity (0–1):",
                                       style={"fontSize": "12px"}),
                            dcc.Input(id="max-sim-input", type="number",
                                      value=1.0, min=0.0, max=1.0,
                                      step=0.05,
                                      style={"width": "100%",
                                             "boxSizing": "border-box"}),
                        ], style={"marginBottom": "8px"}),
                        html.Div([
                            html.Label("Top N paths:",
                                       style={"fontSize": "12px"}),
                            dcc.Input(id="top-n-input", type="number",
                                      value=5, min=1, max=50, step=1,
                                      style={"width": "100%",
                                             "boxSizing": "border-box"}),
                        ]),
                    ]),
                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Snapshot iteration",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div("Path connectivity + force arrows shown at this iter",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "4px"}),
                    dcc.Slider(
                        id="viz-iter-slider",
                        min=iter_min, max=iter_max, step=1,
                        value=min(iter_max, 300),
                        marks={i: str(i) for i in
                               range(iter_min, iter_max + 1, 50)},
                        tooltip={"placement": "bottom",
                                 "always_visible": True},
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "6px",
                               "justifyContent": "center",
                               "marginTop": "4px"},
                        children=[
                            html.Button("◀ −1", id="step-minus-btn",
                                        style={"padding": "2px 10px",
                                               "fontSize": "12px",
                                               "cursor": "pointer"}),
                            html.Button("+1 ▶", id="step-plus-btn",
                                        style={"padding": "2px 10px",
                                               "fontSize": "12px",
                                               "cursor": "pointer"}),
                        ],
                    ),
                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Force arrows",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div("Check to show force component",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "6px"}),
                    html.Div(children=[_force_row(k) for k in FORCE_ORDER]),
                    html.Div([
                        html.Label("Global arrow length multiplier:",
                                   style={"fontSize": "12px", "fontWeight": "bold",
                                          "marginTop": "8px"}),
                        html.Div("All arrows have the same length; "
                                 "hover to see actual force magnitudes.",
                                 style={"fontSize": "10px", "color": "#888",
                                        "marginBottom": "2px"}),
                        dcc.Slider(
                            id="global-force-multiplier",
                            min=0.01, max=10.0, step=0.01,
                            value=1.0,
                            marks={0.01: "0.01x", 0.1: "0.1x", 0.5: "0.5x",
                                   1.0: "1x", 2.0: "2x", 5.0: "5x",
                                   10.0: "10x"},
                            tooltip={"placement": "bottom",
                                     "always_visible": True},
                        ),
                    ]),
                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Visualization Options",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    dcc.Checklist(
                        id="show-trajectories-toggle",
                        options=[{"label": " Show full cell trajectories (all iterations)", "value": "show"}],
                        value=["show"],
                        style={"fontSize": "12px", "marginBottom": "6px", "marginTop": "4px"}
                    ),
                    html.Hr(style={"margin": "14px 0"}),

                    html.Button(
                        "Update",
                        id="update-btn", n_clicks=0,
                        style={"width": "100%", "padding": "10px 0",
                               "backgroundColor": "#2c3e50", "color": "white",
                               "border": "none", "borderRadius": "4px",
                               "fontSize": "15px", "fontWeight": "bold",
                               "cursor": "pointer"},
                    ),
                    html.Button("Cancel", id="cancel-btn",
                                style={"width": "100%", "marginTop": "6px",
                                       "padding": "8px 0",
                                       "backgroundColor": "#dc3545", "color": "white",
                                       "border": "none", "borderRadius": "4px",
                                       "cursor": "pointer", "display": "none"}),
                    html.Div(id="status-msg", style={"marginTop": "8px",
                                                       "fontSize": "12px",
                                                       "color": "#6c757d"}),
                    # Hidden store for caching path/trajectory data
                    # so the snapshot slider updates instantly without
                    # re-querying the database.
                    dcc.Store(id="cached-data", storage_type="memory"),
                ],
            ),

            # ── Plot area ─────────────────────────────────────
            html.Div(
                id="plot-area",
                style={"flex": "1", "min-height": "0",
                       "padding": "10px",
                       "boxSizing": "border-box", "overflow": "hidden"},
                children=[
                    dcc.Loading(
                        id="loading", type="circle",
                        parent_style={"height": "100%",
                                      "position": "relative"},
                        style={"height": "100%"},
                        children=[
                            dcc.Graph(
                                id="main-plot",
                                style={"height": "100%", "width": "100%"},
                                config={"scrollZoom": True,
                                        "displayModeBar": True},
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    # ═════════════════════════════════════════════════════════════
    #  Shared rendering helper  —  used by both callbacks below
    # ═════════════════════════════════════════════════════════════

    def _render_figure_from_cache(cached, viz_iter, arrow_length, toggles,
                                   show_traj):
        """Build a Plotly figure from *cached* path/trajectory data.

        Only ``_build_snapshot_force_df`` hits the database (for the
        requested *viz_iter*).  All path-discovery and trajectory data
        lives in the in-memory ``dcc.Store`` so the snapshot slider
        responds instantly.

        *arrow_length* is the uniform length (in data coords) of every
        force arrow, regardless of the underlying force magnitude.
        """
        path_cell_seqs = {int(k): v
                          for k, v in cached["path_cell_seqs"].items()}
        phys_with_slack = [(int(p), float(s))
                           for p, s in cached["phys_with_slack"]]
        all_cell_ids = cached["all_cell_ids"]
        label_lines = cached["label_lines"]
        iter_lo = cached["iter_lo"]
        iter_hi = cached["iter_hi"]

        traj_df = pd.DataFrame.from_records(cached["traj_records"])
        for col, dtype in [("CellId", int), ("Iter", int),
                           ("PosX", float), ("PosY", float)]:
            traj_df[col] = traj_df[col].astype(dtype)

        fig = go.Figure()
        fig.update_layout(
            title=(f"Cell trajectories for worst timing paths"
                   f"  —  snapshot at iter {viz_iter}"
                   f"  (ranked over iters {iter_lo}–{iter_hi})"),
            xaxis_title="X position (nm)",
            yaxis_title="Y position (nm)",
            template="plotly_white",
            hovermode="closest",
            legend=dict(orientation="v", yanchor="top", y=1,
                        xanchor="left", x=1.02, font=dict(size=9)),
            margin=dict(l=40, r=160, t=50, b=40),
            dragmode="pan",
        )

        if des_bounds is not None:
            _dx = des_bounds[1] - des_bounds[0]
            _dy = des_bounds[3] - des_bounds[2]
            if _dx > 0 and _dy > 0:
                fig.update_yaxes(scaleanchor="x", scaleratio=_dy / _dx)

        # Core region outline
        try:
            m = gpl.get_metadata()
            lx = float(m["region_core_lx"][0])
            ly = float(m["region_core_ly"][0])
            bx = float(m["region_core_binSizeX"][0])
            by = float(m["region_core_binSizeY"][0])
            bcx = int(m["region_core_binCntX"][0])
            bcy = int(m["region_core_binCntY"][0])
            fig.add_shape(
                type="rect",
                x0=lx, y0=ly, x1=lx + bx * bcx, y1=ly + by * bcy,
                line=dict(color="#ccc", width=1, dash="dot"),
                fillcolor="rgba(200,200,200,0.03)",
                layer="below",
            )
        except (IndexError, TypeError, ValueError, KeyError):
            pass

        # Design bounds outline
        if des_bounds is not None:
            xmin, xmax, ymin, ymax = des_bounds
            fig.add_shape(
                type="rect",
                x0=xmin, y0=ymin, x1=xmax, y1=ymax,
                line=dict(color="#e74c3c", width=2, dash="dash"),
                fillcolor="rgba(231, 76, 60, 0.03)",
                layer="below",
            )

        # ── Snapshot forces  (the ONLY database query here) ────
        print(f"[snapshot] Fetching forces for iter {viz_iter}...")
        force_df = _build_snapshot_force_df(gpl, all_cell_ids, viz_iter)

        for idx, (ppid, cids) in enumerate(path_cell_seqs.items()):
            color = _PATH_COLORS[idx % len(_PATH_COLORS)]
            label = f"Path {ppid}"

            if show_traj:
                tr = _build_trajectory_trace(
                    cids, traj_df, label, color, iter_min, iter_max,
                )
                if tr is not None:
                    fig.add_trace(tr)

            for tr in _build_snapshot_path_trace(
                cids, traj_df, viz_iter, label, color, force_df,
            ):
                fig.add_trace(tr)

            if force_df is not None:
                force_sub = force_df[force_df["CellId"].isin(cids)]
                if not force_sub.empty:
                    for fkey in FORCE_ORDER:
                        if not toggles.get(fkey):
                            continue
                        cfg = FORCE_CONFIG[fkey]
                        for tr_arrow in _build_arrow_traces(
                            force_sub, fkey, arrow_length,
                            cfg["color"], label
                        ):
                            fig.add_trace(tr_arrow)

        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=1.01,
            text="<br>".join(label_lines),
            showarrow=False,
            font=dict(size=11, color="#555"),
            align="left",
            bordercolor="#ddd",
            borderwidth=1,
            borderpad=4,
            bgcolor="rgba(255,255,255,0.9)",
        )
        return fig

    # ── Common State declarations (DRY) ──────────────────────
    _PATH_STATES = (
        State("iter-range-slider", "value"),
        State("slack-range-slider", "value"),
        State("min-sep-input", "value"),
        State("max-sim-input", "value"),
        State("top-n-input", "value"),
    )
    _FORCE_STATES = (
        State("global-force-multiplier", "value"),
        State("force-wl-toggle", "value"),
        State("force-tim-toggle", "value"),
        State("force-density-toggle", "value"),
        State("force-effective-toggle", "value"),
    )

    def _parse_params(iter_range_val, slack_range_val,
                      min_sep, max_sim, top_n,
                      global_mult,
                      t_wl, t_tim, t_dens, t_eff,
                      show_trajectories):
        """Parse raw Input values into typed dicts."""
        iter_lo, iter_hi = (int(v) for v in (iter_range_val or [0, 0]))
        slo_ps, shi_ps = (float(v) for v in (slack_range_val or [0, 1]))
        mult = float(global_mult or 1.0)
        return {
            "iter_lo": iter_lo,
            "iter_hi": iter_hi,
            "slo_ps": slo_ps,
            "shi_ps": shi_ps,
            "min_sep": float(min_sep or 0),
            "max_sim": float(max_sim or 1),
            "top_n": int(top_n or 5),
            "show_traj": bool(show_trajectories),
            "arrow_length": _default_arrow_length * mult,
            "toggles": {
                "wl": bool(t_wl),
                "tim": bool(t_tim),
                "density": bool(t_dens),
                "effective": bool(t_eff),
            },
        }

    # ═════════════════════════════════════════════════════════════
    #  Callback  A  —  Update button  (full database fetch)
    # ═════════════════════════════════════════════════════════════

    @app.callback(
        Output("main-plot", "figure"),
        Output("cached-data", "data"),
        Input("update-btn", "n_clicks"),
        State("viz-iter-slider", "value"),
        State("show-trajectories-toggle", "value"),
        *_PATH_STATES,
        *_FORCE_STATES,
        background=True,
        running=[
            (Output("update-btn", "disabled"), True, False),
            (Output("cancel-btn", "style"),
             {"width": "100%", "marginTop": "6px", "padding": "8px 0",
              "backgroundColor": "#dc3545", "color": "white",
              "border": "none", "borderRadius": "4px", "cursor": "pointer",
              "display": "block"},
             {"display": "none"}),
            (Output("status-msg", "children"), "Computing paths...", ""),
        ],
        cancel=[Input("cancel-btn", "n_clicks")],
        prevent_initial_call=False,
    )
    def update_plot(
        n_clicks, viz_iter_val, show_trajectories,
        iter_range_val, slack_range_val,
        min_sep, max_sim, top_n,
        global_mult,
        t_wl, t_tim, t_dens, t_eff,
    ):
        p = _parse_params(
            iter_range_val, slack_range_val,
            min_sep, max_sim, top_n,
            global_mult,
            t_wl, t_tim, t_dens, t_eff,
            show_trajectories,
        )
        viz_iter = int(viz_iter_val or 0)

        print(f"[update] Fetching worst paths (top_n={p['top_n']})...")
        phys_with_slack = _get_worst_phys_ids(
            gpl, p["top_n"], (p["iter_lo"], p["iter_hi"]),
            p["min_sep"], p["max_sim"],
            slack_range_ps=(p["slo_ps"], p["shi_ps"]),
        )
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No paths match the current filters.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#888"),
        )
        if not phys_with_slack:
            return empty_fig, {}

        print(f"[update] Gathering cell sequences...")
        path_cell_seqs = _get_path_cell_sequences(
            gpl, [p[0] for p in phys_with_slack]
        )
        if not path_cell_seqs:
            empty_fig.data = []
            empty_fig.add_annotation(
                text="Could not retrieve cell sequences for paths.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#888"),
            )
            return empty_fig, {}

        all_cell_ids = sorted(set(
            cid for cids in path_cell_seqs.values() for cid in cids
        ))
        label_lines = []
        for ppid, cids in path_cell_seqs.items():
            slack_ps = next((s for p, s in phys_with_slack if p == ppid), None)
            label = f"Path {ppid}"
            if slack_ps is not None:
                label += f"  ({slack_ps:.1f} ps)"
            label += f"  [{len(cids)} cells]"
            label_lines.append(label)

        print(f"[update] Fetching cell trajectories...")
        traj_df = _get_cell_trajectories(gpl, all_cell_ids)
        if traj_df.empty:
            empty_fig.data = []
            empty_fig.add_annotation(
                text="No position data for path cells.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#888"),
            )
            return empty_fig, {}

        # ── Serialise for dcc.Store ──────────────────────────
        cached = {
            "path_cell_seqs": {str(k): v
                               for k, v in path_cell_seqs.items()},
            "phys_with_slack": [(int(pp), float(s))
                                for pp, s in phys_with_slack],
            "all_cell_ids": all_cell_ids,
            "label_lines": label_lines,
            "iter_lo": p["iter_lo"],
            "iter_hi": p["iter_hi"],
            "traj_records": traj_df.to_dict("records"),
        }

        fig = _render_figure_from_cache(cached, viz_iter,
                                         p["arrow_length"], p["toggles"],
                                         p["show_traj"])
        return fig, cached

    # ═════════════════════════════════════════════════════════════
    #  Callback  B  —  Snapshot slider  (instant from cache)
    # ═════════════════════════════════════════════════════════════

    @app.callback(
        Output("main-plot", "figure"),
        Input("viz-iter-slider", "value"),
        State("cached-data", "data"),
        State("show-trajectories-toggle", "value"),
        *_FORCE_STATES,
        prevent_initial_call=True,
    )
    def on_snapshot_slider(
        viz_iter_val, cached_data, show_trajectories,
        global_mult,
        t_wl, t_tim, t_dens, t_eff,
    ):
        if not cached_data:
            return go.Figure()
        viz_iter = int(viz_iter_val or 0)
        mult = float(global_mult or 1.0)
        arrow_length = _default_arrow_length * mult
        toggles = {
            "wl": bool(t_wl),
            "tim": bool(t_tim),
            "density": bool(t_dens),
            "effective": bool(t_eff),
        }
        return _render_figure_from_cache(
            cached_data, viz_iter, arrow_length, toggles,
            bool(show_trajectories),
        )

    # ═════════════════════════════════════════════════════════════
    #  Callbacks  C / D  —  Step ±1 buttons
    # ═════════════════════════════════════════════════════════════

    @app.callback(
        Output("viz-iter-slider", "value"),
        Input("step-minus-btn", "n_clicks"),
        State("viz-iter-slider", "value"),
        prevent_initial_call=True,
    )
    def _step_minus(_n, cur):
        return max(iter_min, (cur or iter_min) - 1)

    @app.callback(
        Output("viz-iter-slider", "value"),
        Input("step-plus-btn", "n_clicks"),
        State("viz-iter-slider", "value"),
        prevent_initial_call=True,
    )
    def _step_plus(_n, cur):
        return min(iter_max, (cur or iter_max) + 1)

    return app


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Dash GUI for GPL path tracing with force arrows"
    )
    parser.add_argument("--db", required=True,
                        help="Path to GPL SQLite database")
    parser.add_argument("--port", type=int, default=8050, help="Dash port")
    parser.add_argument("--read-only", action="store_true",
                        help="Skip preprocessing (fails if not preprocessed)")
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    print(f"Loading database: {db_path}")

    if args.read_only:
        print("  Opening read-only (derived tables must already exist)")
    else:
        print("  Preprocessing (if needed) — this may take several minutes …")

    app = make_app(db_path, read_only=args.read_only)
    print(f"Dash server: http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
