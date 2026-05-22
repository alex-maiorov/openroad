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
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
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
                 "xcol": "WlX", "ycol": "WlY", "default_scale": 5e5},
    "tim":      {"label": "Timing",     "color": "#d62728", "sym": "square",
                 "xcol": "TimX", "ycol": "TimY", "default_scale": 5e5},
    "density":  {"label": "Density",    "color": "#2ca02c", "sym": "diamond",
                 "xcol": "EstDensityForceX", "ycol": "EstDensityForceY",
                 "default_scale": 5e5},
    "effective":{"label": "Effective",  "color": "#9467bd", "sym": "star",
                 "xcol": "EffectiveX", "ycol": "EffectiveY",
                 "default_scale": 5e5},
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
        showlegend=True,
        hoverinfo="skip",
    )


def _build_snapshot_path_trace(cell_ids_in_path, traj_df, iteration,
                                path_label, path_color):
    """Build a Scatter trace showing path connectivity at *iteration*.

    Draws lines + markers connecting the cells in PathSeq order at the
    given iteration.
    """
    sub = traj_df[
        (traj_df["CellId"].isin(cell_ids_in_path)) &
        (traj_df["Iter"] == iteration)
    ].copy()
    if sub.empty:
        return None
    # Sort by cell index in the path sequence
    seq_order = {cid: i for i, cid in enumerate(cell_ids_in_path)}
    sub["_seq"] = sub["CellId"].map(seq_order)
    sub = sub.sort_values("_seq")

    return go.Scatter(
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
        name=f"{path_label} @ iter {iteration}",
        showlegend=True,
        hovertemplate=(
            f"<b>{path_label}</b><br>"
            f"Iter: {iteration}<br>"
            f"Cell: %{{customdata[0]}}<br>"
            f"Seq: %{{customdata[1]}}<br>"
            f"Pos: (%{{x:.0f}}, %{{y:.0f}})<extra></extra>"
        ),
        customdata=np.column_stack([
            sub["CellId"].values,
            sub["_seq"].values,
        ]),
    )


def _build_arrow_traces(force_df, fkey, scale, color, name, sym):
    """Build line + marker traces for one force type at the snapshot
    iteration."""
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
    fxn, fyn = fx[keep] * scale, fy[keep] * scale
    n = sum(keep)

    seg_x = np.full(n * 3, np.nan)
    seg_y = np.full(n * 3, np.nan)
    seg_x[0::3] = xn
    seg_y[0::3] = yn
    seg_x[1::3] = xn + fxn
    seg_y[1::3] = yn + fyn

    return [
        go.Scatter(
            x=seg_x.tolist(), y=seg_y.tolist(),
            mode="lines",
            line=dict(color=color, width=1.5),
            name=name, legendgroup=name, showlegend=True,
            hoverinfo="none",
        ),
        go.Scatter(
            x=(xn + fxn).tolist(), y=(yn + fyn).tolist(),
            mode="markers",
            marker=dict(symbol=sym, size=5, color=color,
                        line=dict(width=0.5, color="white")),
            legendgroup=name, showlegend=False,
            hovertemplate=(
                f"<b>{name}</b><br>Force: %{{customdata[0]:.4e}}"
                "<extra></extra>"
            ),
            customdata=np.column_stack([mag[keep].tolist()]),
        ),
    ]


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

    app = dash.Dash(__name__)
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
                html.Span("scale:", style={"fontSize": "11px",
                                           "color": "#888"}),
                dcc.Input(
                    id=f"force-{fkey}-scale", type="number",
                    value=cfg["default_scale"], min=0, step=1e4,
                    style={"width": "90px", "fontSize": "12px",
                           "padding": "2px 4px"},
                ),
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
                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Force arrows",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div("Check to show | scale factor",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "6px"}),
                    html.Div(children=[_force_row(k) for k in FORCE_ORDER]),
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
                ],
            ),

            # ── Plot area ─────────────────────────────────────
            html.Div(
                id="plot-area",
                style={"flex": "1", "padding": "10px",
                       "boxSizing": "border-box", "overflow": "hidden"},
                children=[
                    dcc.Loading(
                        id="loading", type="circle",
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
    #  Callback
    # ═════════════════════════════════════════════════════════════

    @app.callback(
        Output("main-plot", "figure"),
        Input("update-btn", "n_clicks"),
        State("iter-range-slider", "value"),
        State("slack-range-slider", "value"),
        State("min-sep-input", "value"),
        State("max-sim-input", "value"),
        State("top-n-input", "value"),
        State("viz-iter-slider", "value"),
        State("force-wl-scale", "value"),
        State("force-tim-scale", "value"),
        State("force-density-scale", "value"),
        State("force-effective-scale", "value"),
        State("force-wl-toggle", "value"),
        State("force-tim-toggle", "value"),
        State("force-density-toggle", "value"),
        State("force-effective-toggle", "value"),
        prevent_initial_call=False,
    )
    def update_plot(
        n_clicks,
        iter_range_val, slack_range_val,
        min_sep, max_sim, top_n,
        viz_iter,
        s_wl, s_tim, s_dens, s_eff,
        t_wl, t_tim, t_dens, t_eff,
    ):
        gpl = app._gpl

        # ── Parse inputs ─────────────────────────────────────
        iter_lo, iter_hi = (int(v) for v in (iter_range_val or [0, 0]))
        slo_ps, shi_ps = (float(v) for v in (slack_range_val or [0, 1]))
        min_sep = float(min_sep or 0)
        max_sim = float(max_sim or 1)
        top_n = int(top_n or 5)
        viz_iter = int(viz_iter or 0)

        scales = {
            "wl": float(s_wl or 5e5),
            "tim": float(s_tim or 5e5),
            "density": float(s_dens or 5e5),
            "effective": float(s_eff or 5e5),
        }
        toggles = {
            "wl": bool(t_wl),
            "tim": bool(t_tim),
            "density": bool(t_dens),
            "effective": bool(t_eff),
        }

        # ── Build figure skeleton ────────────────────────────
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

        # ── 1. Find worst paths ──────────────────────────────
        phys_with_slack = _get_worst_phys_ids(
            gpl, top_n, (iter_lo, iter_hi), min_sep, max_sim,
            slack_range_ps=(slo_ps, shi_ps),
        )
        phys_ids = [p[0] for p in phys_with_slack]
        if not phys_ids:
            fig.add_annotation(
                text="No paths match the current filters.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#888"),
            )
            return fig

        # ── 2. Get stable cell sequences for each path ───────
        path_cell_seqs = _get_path_cell_sequences(gpl, phys_ids)
        if not path_cell_seqs:
            fig.add_annotation(
                text="Could not retrieve cell sequences for paths.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#888"),
            )
            return fig

        # All unique cell IDs across all selected paths
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

        # ── 3. Fetch positions across ALL iterations ─────────
        # This is the key change: get every iteration's data for
        # the path cells, regardless of whether the path was
        # "active" (logged by STA) at any particular iteration.
        traj_df = _get_cell_trajectories(gpl, all_cell_ids)
        if traj_df.empty:
            fig.add_annotation(
                text="No position data for path cells.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#888"),
            )
            return fig

        # ── 4. Core region outline ───────────────────────────
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

        # ── 5. Trajectory traces (one per path) ──────────────
        for idx, (ppid, cids) in enumerate(path_cell_seqs.items()):
            color = _PATH_COLORS[idx % len(_PATH_COLORS)]
            label = f"Path {ppid}"
            tr = _build_trajectory_trace(
                cids, traj_df, label, color, iter_min, iter_max,
            )
            if tr is not None:
                fig.add_trace(tr)

        # ── 6. Snapshot path connectivity at viz_iter ─────────
        for idx, (ppid, cids) in enumerate(path_cell_seqs.items()):
            color = _PATH_COLORS[idx % len(_PATH_COLORS)]
            label = f"Path {ppid}"
            tr = _build_snapshot_path_trace(
                cids, traj_df, viz_iter, label, color,
            )
            if tr is not None:
                fig.add_trace(tr)

        # ── 7. Force arrows at viz_iter ──────────────────────
        force_df = _build_snapshot_force_df(gpl, all_cell_ids, viz_iter)
        if force_df is not None:
            for fkey in FORCE_ORDER:
                if not toggles.get(fkey):
                    continue
                cfg = FORCE_CONFIG[fkey]
                for tr in _build_arrow_traces(
                    force_df, fkey, scales[fkey],
                    cfg["color"], cfg["label"], cfg["sym"],
                ):
                    fig.add_trace(tr)

        # ── 8. Subtitle with path info ───────────────────────
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
    print(f"Dash server: http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
