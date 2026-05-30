#!/usr/bin/env python3
"""Dash GUI for analysing the energy budget of timing-driven cell escapes.

Background
----------
In the PES framework, the WL + density forces define a potential energy
landscape φ(x,y) (the electrostatic potential).  Cells settle into local
minima ("basins") of this landscape.  Timing forces do *work* on cells,
and if this work exceeds the potential barrier around a cell's current
basin, the cell can be "kicked out" — escaping its local minimum and
causing a cascade of density readjustments.

This tool computes two quantities for cells involved in timing paths:

**Barrier height** (the "cost to escape")
    The difference between the minimum potential φ along a fixed-radius
    perimeter around the cell and the potential φ at the cell's position.
    A positive barrier means the cell is in a potential well.  A zero or
    negative barrier means the cell is already at or near a ridge.

**Available timing work** (the "budget")
    The distance the timing force can move the cell in one iteration:
    ``W_tim = |F_tim| × stepLength``.  This is effectively how much
    "energy" timing can deliver in one step.

The key scatter plot places every timing-affected cell at every
timing-pass iteration as one point:

    X = barrier height  (how hard to escape)
    Y = timing work      (how much energy available)

Cells *above* the y=x diagonal are energetically capable of escaping.
Coloured by whether they actually *did* change basin in subsequent
iterations (measured by a significant position change).

Usage
-----
    python -m tools.database_log_analysis.work_budget_analyzer \\
        --db path/to/placement-visualization.sqlite \\
        --port 8059
"""

import argparse
import sys
import os

import dash
from dash import dcc, html, Input, Output, State, DiskcacheManager
import diskcache
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

_TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from database_log_analysis import GplDb, make_metadata_panel


# ══════════════════════════════════════════════════════════════════
#  Barrier computation
# ══════════════════════════════════════════════════════════════════

def _compute_barriers(
    pot_df: pd.DataFrame,
    bin_cnt_x: int,
    bin_cnt_y: int,
    bin_size_x: float,
    bin_size_y: float,
    lx: float,
    ly: float,
    radius_nm: float = 5000.0,
    n_samples: int = 24,
) -> tuple:
    """Compute barrier height for every bin via perimeter scan.

    For each bin, samples φ at *n_samples* equally spaced points on a
    circle of radius *radius_nm*.  Barrier = min(φ_perimeter) − φ_center.

    Returns
    -------
    phi : np.ndarray [ny, nx]
        Potential surface.
    barriers : np.ndarray [ny, nx]
        Barrier height per bin (nan where no data).
    """
    # Build phi grid
    phi = np.full((bin_cnt_y, bin_cnt_x), np.nan, dtype=np.float64)
    for _, row in pot_df.iterrows():
        idx = int(row["BinIdx"])
        r = idx // bin_cnt_x
        c = idx % bin_cnt_x
        if 0 <= r < bin_cnt_y and 0 <= c < bin_cnt_x:
            phi[r, c] = float(row["ElectroPhi"])

    # Quick NaN check
    valid_mask = ~np.isnan(phi)
    if valid_mask.sum() < 2:
        return phi, np.full_like(phi, np.nan)

    # For interpolation: fill NaNs with nearest-neighbour fallback
    from scipy.ndimage import generic_filter
    # But scipy might not be available.  Use a simple iterative fill
    # or just skip NaN bins.
    #
    # We'll use a simple approach: for bins near the perimeter, bilinear
    # interpolation from the potential grid.  We can use numpy interp
    # but let's keep it simple with direct grid lookups + clamping.

    barriers = np.full_like(phi, np.nan, dtype=np.float64)

    # Precompute angles
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    dx_vals = radius_nm * np.cos(angles)  # in nm
    dy_vals = radius_nm * np.sin(angles)

    for r in range(bin_cnt_y):
        for c in range(bin_cnt_x):
            center_phi = phi[r, c]
            if np.isnan(center_phi):
                continue

            # Center of this bin in nm
            cx = lx + (c + 0.5) * bin_size_x
            cy = ly + (r + 0.5) * bin_size_y

            # Sample perimeter
            min_perim = np.inf
            any_valid = False
            for dx, dy in zip(dx_vals, dy_vals):
                px = cx + dx
                py = cy + dy
                # Map to bin indices
                bc = int((px - lx) / bin_size_x)
                br = int((py - ly) / bin_size_y)
                bc = max(0, min(bin_cnt_x - 1, bc))
                br = max(0, min(bin_cnt_y - 1, br))
                val = phi[br, bc]
                if not np.isnan(val):
                    min_perim = min(min_perim, val)
                    any_valid = True

            if any_valid:
                barriers[r, c] = min_perim - center_phi
            else:
                barriers[r, c] = 0.0

    return phi, barriers


def _cell_barrier(
    pos_x: float, pos_y: float,
    barriers: np.ndarray,
    lx: float, ly: float,
    bin_size_x: float, bin_size_y: float,
    bin_cnt_x: int, bin_cnt_y: int,
) -> float:
    """Look up barrier height at a cell's position via nearest-neighbour."""
    bc = int((pos_x - lx) / bin_size_x)
    br = int((pos_y - ly) / bin_size_y)
    bc = max(0, min(bin_cnt_x - 1, bc))
    br = max(0, min(bin_cnt_y - 1, br))
    val = barriers[br, bc]
    return float(val) if not np.isnan(val) else 0.0


# ══════════════════════════════════════════════════════════════════
#  App builder
# ══════════════════════════════════════════════════════════════════

def make_app(gpl: GplDb) -> dash.Dash:
    cache = diskcache.Cache("./tmp/dash_bg_cache")
    app = dash.Dash(
        __name__,
        background_callback_manager=DiskcacheManager(cache),
    )
    app.title = "Work Budget Analyzer"

    # ── Pre-fetch ranges ──────────────────────────────────────
    df = gpl.query(
        "SELECT MIN(Iter) AS mi, MAX(Iter) AS ma "
        "FROM gpl_cell_dense_gradients"
    )
    iter_min = int(df["mi"].iloc[0])
    iter_max = int(df["ma"].iloc[0])

    # Grid metadata
    try:
        meta = gpl.get_metadata()
        prefix = None
        for key in meta:
            if key.endswith("_binCntX"):
                prefix = key[: -len("binCntX")]
                break
        if prefix:
            _f = lambda k: float(meta[k][0])
            _i = lambda k: int(float(meta[k][0]))
            bin_cnt_x = _i(prefix + "binCntX")
            bin_cnt_y = _i(prefix + "binCntY")
            bin_size_x = _f(prefix + "binSizeX")
            bin_size_y = _f(prefix + "binSizeY")
            region_lx = _f(prefix + "lx")
            region_ly = _f(prefix + "ly")
            has_grid = True
        else:
            has_grid = False
    except (KeyError, IndexError):
        has_grid = False

    has_timing = gpl._exists("gpl_cell_timing_gradients")

    # ── Layout ─────────────────────────────────────────────────
    app.layout = html.Div(
        style={"display": "flex", "height": "100vh", "margin": "0",
               "padding": "0", "fontFamily": "Segoe UI, Arial, sans-serif"},
        children=[
            # ── Sidebar ───────────────────────────────────────
            html.Div(
                id="sidebar",
                style={"width": "300px", "minWidth": "300px",
                       "padding": "16px 14px", "overflowY": "auto",
                       "backgroundColor": "#f8f9fa",
                       "borderRight": "1px solid #dee2e6",
                       "boxSizing": "border-box"},
                children=[
                    html.H3("Work Budget Analyzer",
                            style={"marginTop": "0", "marginBottom": "6px",
                                   "color": "#2c3e50"}),
                    html.Div("Timing work vs. potential barrier — can timing "
                             "forces escape density basins?",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "14px"}),

                    html.Label("Iteration range",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    dcc.RangeSlider(
                        id="iter-range-slider",
                        min=iter_min, max=iter_max, step=1,
                        value=[iter_min, min(iter_min + 150, iter_max)],
                        marks={i: str(i) for i in
                               range(iter_min, iter_max + 1,
                                     max(1, (iter_max - iter_min) // 8))},
                        tooltip={"placement": "bottom",
                                 "always_visible": False},
                    ),

                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Perimeter radius (nm)",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div("Search radius for barrier detection",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "4px"}),
                    dcc.Slider(
                        id="radius-slider",
                        min=1000, max=30000, step=1000,
                        value=5000,
                        marks={1000: "1k", 5000: "5k", 10000: "10k",
                               20000: "20k", 30000: "30k"},
                    ),

                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Escape threshold",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div("Min displacement (nm) to count as 'escaped'",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "4px"}),
                    dcc.Slider(
                        id="escape-threshold-slider",
                        min=100, max=10000, step=100,
                        value=500,
                        marks={100: "0.1k", 500: "0.5k", 1000: "1k",
                               5000: "5k", 10000: "10k"},
                    ),

                    html.Hr(style={"margin": "14px 0"}),

                    html.Button(
                        "Analyze",
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
                                       "backgroundColor": "#dc3545",
                                       "color": "white",
                                       "border": "none", "borderRadius": "4px",
                                       "cursor": "pointer", "display": "none"}),
                    html.Div(id="status-msg",
                             style={"marginTop": "8px", "fontSize": "12px",
                                     "color": "#6c757d"}),
                    html.Div(id="summary-text",
                             style={"marginTop": "8px", "fontSize": "11px",
                                     "color": "#2c3e50",
                                     "lineHeight": "1.5"}),

                    make_metadata_panel(gpl),
                ],
            ),

            # ── Main plots ────────────────────────────────────
            html.Div(
                style={"flex": "1", "display": "flex",
                       "flexDirection": "column",
                       "padding": "10px", "gap": "8px",
                       "overflowY": "auto", "boxSizing": "border-box",
                       "min-height": "0"},
                children=[
                    # Top row: scatter + ratio histogram
                    html.Div(
                        style={"flex": "1.5", "min-height": "0",
                               "display": "flex", "gap": "8px"},
                        children=[
                            html.Div(
                                style={"flex": "2", "min-height": "0"},
                                children=[
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="plot-scatter",
                                            style={"height": "100%"},
                                            config={"scrollZoom": True},
                                        ),
                                        style={"height": "100%"},
                                        parent_style={"height": "100%"},
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"flex": "1.2", "min-height": "0"},
                                children=[
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="plot-ratio-hist",
                                            style={"height": "100%"},
                                        ),
                                        style={"height": "100%"},
                                        parent_style={"height": "100%"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Bottom row: time series of escape metrics
                    html.Div(
                        style={"flex": "1", "min-height": "0",
                               "display": "flex", "gap": "8px"},
                        children=[
                            html.Div(
                                style={"flex": "1", "min-height": "0"},
                                children=[
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="plot-timeseries",
                                            style={"height": "100%"},
                                        ),
                                        style={"height": "100%"},
                                        parent_style={"height": "100%"},
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"flex": "1", "min-height": "0"},
                                children=[
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="plot-escape-rate",
                                            style={"height": "100%"},
                                        ),
                                        style={"height": "100%"},
                                        parent_style={"height": "100%"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    # ═══════════════════════════════════════════════════════════
    #  Callback
    # ═══════════════════════════════════════════════════════════

    @app.callback(
        Output("plot-scatter", "figure"),
        Output("plot-ratio-hist", "figure"),
        Output("plot-timeseries", "figure"),
        Output("plot-escape-rate", "figure"),
        Output("summary-text", "children"),
        Output("status-msg", "children"),
        Input("update-btn", "n_clicks"),
        State("iter-range-slider", "value"),
        State("radius-slider", "value"),
        State("escape-threshold-slider", "value"),
        background=True,
        running=[
            (Output("update-btn", "disabled"), True, False),
            (Output("cancel-btn", "style"),
             {"width": "100%", "marginTop": "6px", "padding": "8px 0",
              "backgroundColor": "#dc3545", "color": "white",
              "border": "none", "borderRadius": "4px",
              "cursor": "pointer", "display": "block"},
             {"display": "none"}),
        ],
        cancel=[Input("cancel-btn", "n_clicks")],
        prevent_initial_call=False,
    )
    def update_plots(_n, iter_range, radius, escape_threshold):
        iter_lo, iter_hi = (int(v) for v in (iter_range or [0, 0]))
        radius = float(radius)
        escape_threshold = float(escape_threshold)

        print(f"[WorkBudget] Fetching data iters {iter_lo}–{iter_hi}...")

        if not has_grid or not has_timing:
            return (
                _empty("No grid data or no timing data"),
                _empty(""), _empty(""), _empty(""),
                "", "Missing data",
            )

        # ── 1. Get timing-pass iterations ─────────────────────
        timing_passes = gpl.query(
            "SELECT DISTINCT Iter FROM gpl_cell_timing_gradients "
            "WHERE Iter BETWEEN ? AND ? ORDER BY Iter",
            (iter_lo, iter_hi),
        )
        if timing_passes.empty:
            return (
                _empty("No timing passes in this range"),
                _empty(""), _empty(""), _empty(""),
                "", "No timing data",
            )
        tp_iters = sorted(timing_passes["Iter"].tolist())

        # ── 2. Fetch timing gradients and positions ────────────
        print(f"[WorkBudget] Fetching timing + position data for "
              f"{len(tp_iters)} timing passes...")
        dense = gpl.cell_dense_gradients(iter_range=(iter_lo, iter_hi))
        timing = gpl.cell_timing_gradients(iter_range=(iter_lo, iter_hi))

        if dense.empty or timing.empty:
            return (
                _empty("No data"), _empty(""), _empty(""), _empty(""),
                "", "No data",
            )

        # Filter to timing-pass iterations only
        dense_tp = dense[dense["Iter"].isin(set(tp_iters))].copy()
        timing_tp = timing[timing["Iter"].isin(set(tp_iters))].copy()

        if dense_tp.empty or timing_tp.empty:
            return (
                _empty("No intersecting data"), _empty(""),
                _empty(""), _empty(""),
                "", "No intersecting data",
            )

        # Merge: positions + timing forces
        merged = dense_tp.merge(
            timing_tp[["Iter", "CellId", "TimX", "TimY"]],
            on=["Iter", "CellId"], how="inner",
        )
        merged["TimMag"] = np.hypot(merged["TimX"], merged["TimY"])

        # ── 3. Step lengths ────────────────────────────────────
        scalars = gpl.iteration_scalars()
        scalars = scalars[scalars["Iter"].isin(set(tp_iters))]
        step_map = dict(zip(scalars["Iter"], scalars["StepLength"]))
        merged["StepLength"] = merged["Iter"].map(step_map).fillna(0.0)

        # ── 4. Barrier computation ─────────────────────────────
        # Compute barriers once per timing-pass iteration
        print(f"[WorkBudget] Computing barriers for {len(tp_iters)} "
              f"timing passes...")
        barrier_map = {}  # iter → np.ndarray [ny, nx]
        for tp_iter in tp_iters:
            pot = gpl.bin_potential(iter_range=(tp_iter, tp_iter))
            if pot.empty:
                continue
            _, barriers = _compute_barriers(
                pot, bin_cnt_x, bin_cnt_y,
                bin_size_x, bin_size_y,
                region_lx, region_ly,
                radius_nm=radius,
                n_samples=20,
            )
            barrier_map[tp_iter] = barriers

        # ── 5. Look up barrier + compute work ──────────────────
        records = []
        for _, row in merged.iterrows():
            it = int(row["Iter"])
            if it not in barrier_map:
                continue
            barr = _cell_barrier(
                float(row["PosX"]), float(row["PosY"]),
                barrier_map[it],
                region_lx, region_ly,
                bin_size_x, bin_size_y,
                bin_cnt_x, bin_cnt_y,
            )
            tim_work = float(row["TimMag"]) * float(row["StepLength"])
            records.append({
                "Iter": it,
                "CellId": int(row["CellId"]),
                "BarrierHeight": barr,
                "TimingWork": tim_work,
                "TimMag": float(row["TimMag"]),
            })

        if not records:
            return (
                _empty("No barriers computed"), _empty(""),
                _empty(""), _empty(""),
                "", "No data",
            )

        budget_df = pd.DataFrame(records)

        # ── 6. Determine actual escapes ────────────────────────
        # "Escaped" = cell moved > escape_threshold in the next K iters
        print(f"[WorkBudget] Computing escape events...")
        movements = gpl.cell_movements(
            iter_range=(iter_lo, iter_hi),
            cell_ids=sorted(budget_df["CellId"].unique().tolist()),
        )
        # For each (cell, iter) in budget_df, check distance in
        # the NEXT non-zero movement
        escape_map = {}
        if not movements.empty:
            for cid in budget_df["CellId"].unique():
                cm = movements[movements["CellId"] == cid].sort_values("Iter")
                for _, brow in budget_df[budget_df["CellId"] == cid].iterrows():
                    it = int(brow["Iter"])
                    # Find the row in cm for this iter
                    next_rows = cm[cm["Iter"] > it].sort_values("Iter")
                    if not next_rows.empty:
                        next_dist = float(next_rows.iloc[0]["Distance"])
                        escape_map[(cid, it)] = (
                            next_dist > escape_threshold
                        )
                    else:
                        escape_map[(cid, it)] = False

        budget_df["Escaped"] = budget_df.apply(
            lambda r: escape_map.get(
                (int(r["CellId"]), int(r["Iter"])), False,
            ),
            axis=1,
        )

        # ── 7. Budget ratio ────────────────────────────────────
        budget_df["BudgetRatio"] = np.where(
            budget_df["BarrierHeight"] > 1e-9,
            budget_df["TimingWork"] / budget_df["BarrierHeight"],
            np.inf,
        )
        # Cap for visualisation
        budget_df["BudgetRatioClipped"] = np.clip(
            budget_df["BudgetRatio"], 0.01, 100,
        )

        # ══════════════════════════════════════════════════════
        #  Panel 1: Scatter — Barrier vs Timing Work
        # ══════════════════════════════════════════════════════

        fig_scatter = go.Figure()

        not_escaped = budget_df[~budget_df["Escaped"]]
        escaped = budget_df[budget_df["Escaped"]]

        if not not_escaped.empty:
            sample_n = not_escaped if len(not_escaped) <= 5000 else (
                not_escaped.sample(5000, random_state=42)
            )
            fig_scatter.add_trace(go.Scatter(
                x=sample_n["BarrierHeight"],
                y=sample_n["TimingWork"],
                mode="markers",
                marker=dict(
                    size=5, color="#90caf9", opacity=0.6,
                    line=dict(width=0.5, color="#1976d2"),
                ),
                name="Stayed in basin",
                customdata=np.column_stack([
                    sample_n["CellId"], sample_n["Iter"],
                    sample_n["BudgetRatio"],
                ]),
                hovertemplate=(
                    "Cell %{customdata[0]} Iter %{customdata[1]}<br>"
                    "Barrier: %{x:.1f}<br>"
                    "Work: %{y:.1f}<br>"
                    "Ratio: %{customdata[2]:.2f}<br>"
                    "<b>Stayed</b><extra></extra>"
                ),
            ))

        if not escaped.empty:
            fig_scatter.add_trace(go.Scatter(
                x=escaped["BarrierHeight"],
                y=escaped["TimingWork"],
                mode="markers",
                marker=dict(
                    size=7, color="#d32f2f", opacity=0.8,
                    line=dict(width=0.5, color="white"),
                    symbol="x",
                ),
                name="Escaped basin",
                customdata=np.column_stack([
                    escaped["CellId"], escaped["Iter"],
                    escaped["BudgetRatio"],
                ]),
                hovertemplate=(
                    "Cell %{customdata[0]} Iter %{customdata[1]}<br>"
                    "Barrier: %{x:.1f}<br>"
                    "Work: %{y:.1f}<br>"
                    "Ratio: %{customdata[2]:.2f}<br>"
                    "<b>ESCAPED</b><extra></extra>"
                ),
            ))

        # Diagonal: y = x (work = barrier)
        max_val = max(
            budget_df["BarrierHeight"].max(),
            budget_df["TimingWork"].max(),
            1.0,
        )
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines",
            line=dict(color="#333", width=1, dash="dash"),
            name="Work = Barrier",
            hoverinfo="skip",
        ))

        frac_escape = budget_df["Escaped"].mean()

        fig_scatter.update_layout(
            title=(
                "Timing Work vs. Barrier Height<br>"
                f"<sup>{frac_escape*100:.1f}% of timing events cause escape "
                f"(threshold >{escape_threshold:.0f} nm)</sup>"
            ),
            template="plotly_white",
            xaxis_title="Barrier height (potential difference)",
            yaxis_title="Available timing work  (|F_tim| × stepLen)",
            xaxis=dict(type="log"),
            yaxis=dict(type="log"),
            legend=dict(
                orientation="h", yanchor="top", y=1.13,
                xanchor="left", x=0,
            ),
            hovermode="closest",
            margin=dict(l=60, r=20, t=70, b=60),
        )

        # ══════════════════════════════════════════════════════
        #  Panel 2: Ratio histogram
        # ══════════════════════════════════════════════════════

        fig_ratio = go.Figure()

        fig_ratio.add_trace(go.Histogram(
            x=np.log10(budget_df["BudgetRatioClipped"]),
            name="All timing events",
            nbinsx=40,
            marker_color="rgba(100, 100, 100, 0.5)",
            hovertemplate="log10(ratio) = %{x:.3f}<br>Count: %{y}<extra></extra>",
        ))

        if not escaped.empty:
            fig_ratio.add_trace(go.Histogram(
                x=np.log10(escaped["BudgetRatioClipped"]),
                name="Escaped",
                nbinsx=40,
                marker_color="rgba(211, 47, 47, 0.5)",
                hovertemplate="log10(ratio) = %{x:.3f}<br>Count: %{y}<extra></extra>",
            ))

        fig_ratio.add_vline(
            x=0, line_dash="dash", line_color="#333",
            line_width=2,
            annotation_text="W = B",
            annotation_position="top",
        )

        fig_ratio.update_layout(
            title="Budget Ratio Distribution  (Work / Barrier)",
            template="plotly_white",
            xaxis_title="log₁₀(Work / Barrier)",
            yaxis_title="Event count",
            barmode="overlay",
            legend=dict(orientation="h", yanchor="top", y=1.15,
                        xanchor="left", x=0),
            margin=dict(l=60, r=20, t=50, b=60),
        )

        # ══════════════════════════════════════════════════════
        #  Panel 3: Time series of mean barrier & mean work
        # ══════════════════════════════════════════════════════

        fig_ts = go.Figure()

        ts = budget_df.groupby("Iter").agg(
            mean_barrier=("BarrierHeight", "mean"),
            mean_work=("TimingWork", "mean"),
            escape_rate=("Escaped", "mean"),
        ).reset_index()

        fig_ts.add_trace(go.Scatter(
            x=ts["Iter"], y=ts["mean_barrier"],
            mode="lines+markers",
            name="Mean barrier height",
            line=dict(color="#1976d2", width=2),
            marker=dict(size=4),
        ))

        fig_ts.add_trace(go.Scatter(
            x=ts["Iter"], y=ts["mean_work"],
            mode="lines+markers",
            name="Mean timing work",
            line=dict(color="#d32f2f", width=2),
            marker=dict(size=4),
        ))

        fig_ts.update_layout(
            title="Mean Barrier & Work Over Time",
            template="plotly_white",
            xaxis_title="Iteration",
            yaxis_title="Value",
            legend=dict(orientation="h", yanchor="top", y=1.15,
                        xanchor="left", x=0),
            hovermode="x unified",
            margin=dict(l=60, r=20, t=50, b=60),
        )

        # ══════════════════════════════════════════════════════
        #  Panel 4: Escape rate over time
        # ══════════════════════════════════════════════════════

        fig_esc = go.Figure()

        fig_esc.add_trace(go.Scatter(
            x=ts["Iter"], y=ts["escape_rate"] * 100,
            mode="lines+markers",
            name="Escape rate",
            line=dict(color="#e65100", width=2),
            marker=dict(size=4),
            fill="tozeroy",
            fillcolor="rgba(230, 81, 0, 0.1)",
        ))

        # Also show the fraction of cells with W > B
        frac_gt = budget_df.groupby("Iter").apply(
            lambda g: (g["TimingWork"] > g["BarrierHeight"]).mean()
        ).reset_index(name="frac_work_gt_barrier")

        fig_esc.add_trace(go.Scatter(
            x=frac_gt["Iter"],
            y=frac_gt["frac_work_gt_barrier"] * 100,
            mode="lines",
            name="Fraction W > B",
            line=dict(color="#7b1fa2", width=1.5, dash="dash"),
        ))

        fig_esc.update_layout(
            title="Escape Rate & W>B Fraction",
            template="plotly_white",
            xaxis_title="Iteration",
            yaxis_title="% of timing events",
            legend=dict(orientation="h", yanchor="top", y=1.15,
                        xanchor="left", x=0),
            hovermode="x unified",
            margin=dict(l=60, r=20, t=50, b=60),
        )

        # ── Summary ───────────────────────────────────────────
        n_events = len(budget_df)
        n_cells = budget_df["CellId"].nunique()
        n_escapes = budget_df["Escaped"].sum()

        summary = [
            html.B(f"{n_events} timing events"),
            f" across {n_cells} cells, {len(tp_iters)} timing passes",
            html.Br(),
            html.Span(
                f"{n_escapes} escapes "
                f"({n_escapes / max(n_events, 1) * 100:.1f}%)",
                style={"color": "#e74c3c" if frac_escape > 0.1
                       else "#27ae60"},
            ),
            f"  ·  radius = {radius:.0f} nm",
        ]

        status = f"Iters {iter_lo}–{iter_hi}  ·  "
        status += f"{n_events} events  ·  {frac_escape*100:.1f}% escaped"

        return fig_scatter, fig_ratio, fig_ts, fig_esc, summary, status

    return app


def _empty(msg="No data"):
    return go.Figure(layout={
        "template": "plotly_white",
        "xaxis": {"visible": False},
        "yaxis": {"visible": False},
        "annotations": [{
            "text": msg, "showarrow": False,
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "font": {"size": 16, "color": "#888"},
        }],
    })


def main():
    parser = argparse.ArgumentParser(
        description="Work Budget Analyzer — timing work vs. barrier height"
    )
    parser.add_argument("--db", required=True,
                        help="Path to GPL SQLite database")
    parser.add_argument("--port", type=int, default=8059,
                        help="Dash port")
    parser.add_argument("--read-only", action="store_true",
                        help="Skip preprocessing")
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    print(f"Loading database: {db_path}")

    if args.read_only:
        print("  Opening read-only")
        gpl = GplDb(db_path, must_be_preprocessed=True)
    else:
        print("  Preprocessing (if needed) …")
        gpl = GplDb(db_path)

    app = make_app(gpl)
    print(f"Dash server: http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
