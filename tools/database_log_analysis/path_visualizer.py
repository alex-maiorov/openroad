#!/usr/bin/env python3
"""Dash GUI for tracing worst-slacked timing paths in GPL placement.

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
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

# ── Ensure the parent package is importable ──────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

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
#  Data helpers (thin wrappers around existing GplDb methods)
# ══════════════════════════════════════════════════════════════════

def _get_iter_range(gpl):
    """Return (min_iter, max_iter) from the dense-gradients table."""
    df = gpl.query(
        "SELECT MIN(Iter) AS mi, MAX(Iter) AS ma FROM gpl_cell_dense_gradients"
    )
    return int(df["mi"].iloc[0]), int(df["ma"].iloc[0])


def _get_slack_range_ps(gpl):
    """Return (min_slack_ps, max_slack_ps) across all PhysicalPathIds.

    Queries the per-physical-path minimum slack to get a range that
    reflects actual worst-path slack values (often negative for
    timing-critical paths).
    """
    # Use path_signatures + path_slacks to get min slack per physical path
    if not gpl._exists("gpl_path_signatures"):
        return -500, 500  # fallback
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

    Parameters
    ----------
    slack_range_ps : (float, float) or None
        If set, only keep paths whose worst slack (in ps) falls within
        this (lo, hi) inclusive range.

    Returns
    -------
    list of (PhysicalPathId, min_slack_ps) tuples, sorted worst-first.
    """
    if not gpl._exists("gpl_path_slacks"):
        return []
    histories = gpl.worst_paths_history(
        top_n=top_n,
        iter_range=iter_range,
        min_separation=min_sep_ps * 1e-12,  # ps → seconds
        max_similarity=max_sim,
    )
    results = []
    for pdf in histories:
        pid = int(pdf["PhysicalPathId"].iloc[0])
        slack_ps = float(pdf["Slack"].min()) * 1e12
        # Filter by slack range if given
        if slack_range_ps:
            lo, hi = slack_range_ps
            if not (lo <= slack_ps <= hi):
                continue
        results.append((pid, slack_ps))

    # Deduplicate by PhysicalPathId (keep the one with worst slack)
    seen = set()
    deduped = []
    for pid, sps in results:
        if pid not in seen:
            seen.add(pid)
            deduped.append((pid, sps))
    return deduped


def _get_path_cells_with_pos(gpl, path_ids, iteration):
    """Get path cells at *iteration* with their (PosX, PosY) positions.

    Uses existing GplDb.path_cells() method and attaches positions.
    """
    cells = gpl.path_cells(iter_range=(iteration, iteration),
                           path_ids=path_ids)
    if cells.empty:
        return cells

    # Attach positions from dense gradients
    cell_ids = cells["CellId"].unique().tolist()
    pos = gpl.cell_dense_gradients(
        iter_range=(iteration, iteration), cell_ids=cell_ids
    )[["CellId", "PosX", "PosY"]]
    if not pos.empty:
        cells = cells.merge(pos, on="CellId", how="left")
    return cells


def _build_combined_force_df(gpl, cell_ids, iteration):
    """Return DataFrame with PosX, PosY and all 4 force components.

    Force columns: WlX, WlY, TimX, TimY,
                   EstDensityForceX, EstDensityForceY,
                   EffectiveX, EffectiveY.
    """
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

def _build_arrow_traces(force_df, fkey, scale, color, name, sym):
    """Build line + marker traces for one force type.

    Returns list of go.Scatter traces (empty if no nonzero forces).
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
    fxn, fyn = fx[keep] * scale, fy[keep] * scale

    # Interleaved: origin, tip, NaN, origin, tip, NaN, …
    n = sum(keep)
    seg_x = np.full(n * 3, np.nan)
    seg_y = np.full(n * 3, np.nan)
    seg_x[0::3] = xn
    seg_y[0::3] = yn
    seg_x[1::3] = xn + fxn
    seg_y[1::3] = yn + fyn

    tip_x = (xn + fxn).tolist()
    tip_y = (yn + fyn).tolist()
    tip_mag = mag[keep].tolist()

    return [
        go.Scatter(
            x=seg_x.tolist(), y=seg_y.tolist(),
            mode="lines",
            line=dict(color=color, width=1.5),
            name=name, legendgroup=name, showlegend=True,
            hoverinfo="none",
        ),
        go.Scatter(
            x=tip_x, y=tip_y,
            mode="markers",
            marker=dict(symbol=sym, size=5, color=color,
                        line=dict(width=0.5, color="white")),
            name=name, legendgroup=name, showlegend=False,
            hovertemplate=(
                f"<b>{name}</b><br>Force: %{{customdata[0]:.4e}}"
                "<extra></extra>"
            ),
            customdata=np.column_stack([tip_mag]),
        ),
    ]


def _build_path_traces(path_groups, iteration):
    """Build scatter+line traces for each path group.

    *path_groups*: dict {label: DataFrame with PosX, PosY, CellId, PathSeq}
    """
    traces = []
    for idx, (label, pdf) in enumerate(path_groups.items()):
        pdf = pdf.sort_values("PathSeq")
        color = _PATH_COLORS[idx % len(_PATH_COLORS)]
        traces.append(
            go.Scatter(
                x=pdf["PosX"].tolist(), y=pdf["PosY"].tolist(),
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=7, color=color, symbol="circle",
                            line=dict(width=1, color="white")),
                name=label, showlegend=True,
                hovertemplate=(
                    f"<b>{label}</b><br>Iter: {iteration}<br>"
                    f"Cell: %{{customdata[0]}}<br>"
                    f"Seq: %{{customdata[1]}}<br>"
                    f"Pos: (%{{x:.0f}}, %{{y:.0f}})<extra></extra>"
                ),
                customdata=np.column_stack([
                    pdf["CellId"].values, pdf["PathSeq"].values,
                ]),
            )
        )
    return traces


# ══════════════════════════════════════════════════════════════════
#  Dash Application
# ══════════════════════════════════════════════════════════════════

def make_app(db_path, read_only=False):
    """Create and return a Dash application instance.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    read_only : bool
        If True, use ``must_be_preprocessed=True`` (fails fast if derived
        tables are missing).  If False, auto-preprocess.
    """
    if read_only:
        gpl = GplDb(db_path, must_be_preprocessed=True)
    else:
        gpl = GplDb(db_path)

    # Data ranges (computed once)
    iter_min, iter_max = _get_iter_range(gpl)
    slack_lo, slack_hi = _get_slack_range_ps(gpl)
    # Round to nice multiples of 5 ps
    slack_lo = np.floor(slack_lo / 5) * 5
    slack_hi = np.ceil(slack_hi / 5) * 5
    if slack_lo >= slack_hi:
        slack_hi = slack_lo + 10

    app = dash.Dash(__name__)
    app.title = "GPL Path Visualizer"
    app._gpl = gpl

    # ── Helper: build a force-control row ─────────────────────
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

    # ── Layout ──────────────────────────────────────────────────
    app.layout = html.Div(
        style={"display": "flex", "height": "100vh", "margin": "0",
               "padding": "0", "fontFamily": "Segoe UI, Arial, sans-serif"},
        children=[
            # ── Sidebar ─────────────────────────────────────────
            html.Div(
                id="sidebar",
                style={
                    "width": "350px", "minWidth": "350px",
                    "padding": "16px 14px", "overflowY": "auto",
                    "backgroundColor": "#f8f9fa",
                    "borderRight": "1px solid #dee2e6",
                    "boxSizing": "border-box",
                },
                children=[
                    html.H3("GPL Path Visualizer",
                            style={"marginTop": "0", "marginBottom": "16px",
                                   "color": "#2c3e50"}),

                    # Iteration range
                    html.Label("Iteration range for worst-path search",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div(f"Iters: {iter_min}–{iter_max}",
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

                    # Slack range
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

                    # Path filtering params
                    html.Label("Path filtering",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div([
                        html.Div([
                            html.Label("Min slack separation (ps):",
                                       style={"fontSize": "12px"}),
                            dcc.Input(
                                id="min-sep-input", type="number",
                                value=0, step=1,
                                style={"width": "100%",
                                       "boxSizing": "border-box"}),
                        ], style={"marginBottom": "8px"}),
                        html.Div([
                            html.Label("Max path similarity (0–1):",
                                       style={"fontSize": "12px"}),
                            dcc.Input(
                                id="max-sim-input", type="number",
                                value=1.0, min=0.0, max=1.0, step=0.05,
                                style={"width": "100%",
                                       "boxSizing": "border-box"}),
                        ], style={"marginBottom": "8px"}),
                        html.Div([
                            html.Label("Top N paths:",
                                       style={"fontSize": "12px"}),
                            dcc.Input(
                                id="top-n-input", type="number",
                                value=5, min=1, max=50, step=1,
                                style={"width": "100%",
                                       "boxSizing": "border-box"}),
                        ]),
                    ]),
                    html.Hr(style={"margin": "14px 0"}),

                    # Viz iteration
                    html.Label("Visualisation iteration",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div("Iteration to display (path slacks at iters 150, 160, …, 440)",
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

                    # Force controls
                    html.Label("Force arrows",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div("Check to show | scale factor for arrow length",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "6px"}),
                    html.Div(
                        id="force-controls",
                        children=[_force_row(k) for k in FORCE_ORDER],
                    ),
                    html.Hr(style={"margin": "14px 0"}),

                    # Update button
                    html.Button(
                        "Update",
                        id="update-btn", n_clicks=0,
                        style={
                            "width": "100%", "padding": "10px 0",
                            "backgroundColor": "#2c3e50", "color": "white",
                            "border": "none", "borderRadius": "4px",
                            "fontSize": "15px", "fontWeight": "bold",
                            "cursor": "pointer",
                        },
                    ),
                ],
            ),

            # ── Main plot ───────────────────────────────────────
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

        # Parse
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

        fig = go.Figure()
        fig.update_layout(
            title=(f"Worst timing paths at iteration {viz_iter}"
                   f"  (iters {iter_lo}–{iter_hi} for ranking)"),
            xaxis_title="X position (nm)",
            yaxis_title="Y position (nm)",
            template="plotly_white",
            hovermode="closest",
            legend=dict(orientation="v", yanchor="top", y=1,
                        xanchor="left", x=1.02, font=dict(size=10)),
            margin=dict(l=40, r=140, t=50, b=40),
            dragmode="pan",
        )

        # ── 1. Find worst physical paths ────────────────────────
        phys_with_slack = _get_worst_phys_ids(
            gpl, top_n, (iter_lo, iter_hi), min_sep, max_sim,
            slack_range_ps=(slo_ps, shi_ps),
        )
        phys_ids = [p[0] for p in phys_with_slack]
        if not phys_ids:
            fig.add_annotation(
                text="No paths match the current filters.<br>"
                     "Try widening the slack or iteration range.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#888"),
            )
            return fig

        # ── 2. Map PhysicalPathId → PathId at this iteration ────
        sig = gpl.path_signatures(iter_range=(viz_iter, viz_iter))
        if sig.empty:
            # Show helpful info about which iterations these paths ARE in
            placeholders = ",".join("?" for _ in phys_ids)
            avail_iters = gpl.query(
                f"""
                SELECT DISTINCT Iter FROM gpl_path_signatures
                WHERE PhysicalPathId IN ({placeholders})
                ORDER BY Iter
                """,
                phys_ids,
            )
            iters_str = (", ".join(str(int(x)) for x in avail_iters["Iter"])
                         if not avail_iters.empty else "N/A")
            fig.add_annotation(
                text=(f"No paths at iteration {viz_iter}.<br>"
                      f"These worst paths have data at iters:<br>"
                      f"<b>{iters_str}</b>"),
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#888"),
            )
            return fig

        phys_to_path = {}
        for _, row in sig.iterrows():
            ppid = int(row["PhysicalPathId"])
            if ppid in phys_ids:
                phys_to_path[ppid] = int(row["PathId"])

        if not phys_to_path:
            placeholders = ",".join("?" for _ in phys_ids)
            avail_iters = gpl.query(
                f"""
                SELECT DISTINCT Iter FROM gpl_path_signatures
                WHERE PhysicalPathId IN ({placeholders})
                ORDER BY Iter
                """,
                phys_ids,
            )
            iters_str = (", ".join(str(int(x)) for x in avail_iters["Iter"])
                         if not avail_iters.empty else "N/A")
            fig.add_annotation(
                text=(f"Worst paths not active at iteration {viz_iter}.<br>"
                      f"They have data at iters: <b>{iters_str}</b><br>"
                      f"Move the 'Visualisation iteration' slider to one of those."),
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#888"),
            )
            return fig

        # ── 3. Get path cells with positions ─────────────────────
        path_ids = list(phys_to_path.values())
        cells_df = _get_path_cells_with_pos(gpl, path_ids, viz_iter)
        if cells_df.empty:
            fig.add_annotation(
                text="No path cells found for the selected iteration.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#888"),
            )
            return fig

        # Group by path
        path_groups = {}
        for ppid, pid in phys_to_path.items():
            sub = cells_df[cells_df["PathId"] == pid].copy()
            if not sub.empty:
                path_groups[f"Path {ppid}"] = sub

        # ── 4. Add path traces ──────────────────────────────────
        for tr in _build_path_traces(path_groups, viz_iter):
            fig.add_trace(tr)

        # ── 5. Add force arrow traces ───────────────────────────
        all_cids = list(set(cells_df["CellId"].tolist()))
        force_df = _build_combined_force_df(gpl, all_cids, viz_iter)
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

        # ── 6. Core region outline ──────────────────────────────
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
                x0=lx, y0=ly, x1=lx+bx*bcx, y1=ly+by*bcy,
                line=dict(color="#ccc", width=1, dash="dot"),
                fillcolor="rgba(200,200,200,0.03)",
                layer="below",
            )
        except (IndexError, TypeError, ValueError, KeyError):
            pass

        return fig

    return app


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Dash GUI for GPL path tracing with force arrows"
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to GPL SQLite database (default: auto-detect)",
    )
    parser.add_argument("--port", type=int, default=8050, help="Dash port")
    parser.add_argument("--read-only", action="store_true",
                        help="Skip preprocessing (fails if not preprocessed)")
    args = parser.parse_args()

    db_path = args.db
    if db_path is None:
        for cand in [
            "placement-visualization.sqlite",
            str(_HERE / ".." / ".." / "placement-visualization.sqlite"),
            str(_HERE / ".." / ".." / "tmp" / "test_gpl.sqlite"),
        ]:
            if os.path.isfile(cand):
                db_path = cand
                break
        if db_path is None:
            print("ERROR: Could not find a database. Use --db PATH")
            sys.exit(1)

    db_path = os.path.abspath(db_path)
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
