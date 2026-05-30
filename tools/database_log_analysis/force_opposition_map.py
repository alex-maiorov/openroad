#!/usr/bin/env python3
"""Dash GUI for visualising the spatial and temporal structure of force
opposition during global placement.

Background
----------
At every iteration, each cell experiences four forces: wirelength (WL),
density (electrostatic), timing, and routability.  The WL + density forces
define a "consensus" direction — the direction the placer *wants* to move
the cell to optimise area and wirelength.  Timing and routability are
"adversarial" forces that can pull cells in a different direction.

The **opposition** is cos(θ) between these two vector groups:
    +1  → adversarial force is *helping* (pushing the same way)
     0  → orthogonal (neither helping nor hurting)
    −1  → adversarial force is *fighting* (pushing the opposite way)

When many cells have cos(θ) ≪ 0, the optimisation is in conflict.  When
that conflict *persists* across iterations, cells can oscillate or bounce
between competing objectives, producing instability.

This tool shows:

1. **Spatial opposition heatmap** — cells binned onto a grid, mean opposition
   per bin (red = cells fighting, green = cells cooperating).  Scrub through
   iterations to see when/where conflict flares up.
2. **Opposition time series** — aggregate metrics over all iterations:
   fraction of cells with strong opposition (cos < −0.5), mean opposition.
3. **Opposition histogram** — distribution of opposition values at the
   selected iteration, with the adversarial-only distribution overlaid.

Usage
-----
    python -m tools.database_log_analysis.force_opposition_map \\
        --db path/to/placement-visualization.sqlite \\
        --port 8058
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
#  Helpers
# ══════════════════════════════════════════════════════════════════

def _opposition_vectors(
    dense: pd.DataFrame,
    timing: pd.DataFrame,
    routability: pd.DataFrame,
    density_forces: pd.DataFrame,
) -> pd.DataFrame:
    """Return a DataFrame with columns ``cos_theta`` and ``has_adversarial``
    appended to the *dense* index structure.
    """
    cons_x = dense["WlX"].values.copy()
    cons_y = dense["WlY"].values.copy()
    adv_x = np.zeros(len(dense), dtype=np.float64)
    adv_y = np.zeros(len(dense), dtype=np.float64)

    if not density_forces.empty:
        merged = dense[["Iter", "CellId"]].merge(
            density_forces[["Iter", "CellId", "EstDensityForceX",
                             "EstDensityForceY"]],
            on=["Iter", "CellId"], how="left",
        )
        cons_x += merged["EstDensityForceX"].fillna(0.0).values
        cons_y += merged["EstDensityForceY"].fillna(0.0).values

    if not timing.empty:
        merged = dense[["Iter", "CellId"]].merge(
            timing[["Iter", "CellId", "TimX", "TimY"]],
            on=["Iter", "CellId"], how="left",
        )
        adv_x += merged["TimX"].fillna(0.0).values
        adv_y += merged["TimY"].fillna(0.0).values

    if not routability.empty:
        merged = dense[["Iter", "CellId"]].merge(
            routability[["Iter", "CellId", "RtX", "RtY"]],
            on=["Iter", "CellId"], how="left",
        )
        adv_x += merged["RtX"].fillna(0.0).values
        adv_y += merged["RtY"].fillna(0.0).values

    mag_cons = np.hypot(cons_x, cons_y)
    mag_adv = np.hypot(adv_x, adv_y)

    cos_theta = np.zeros(len(dense), dtype=np.float64)
    mask = (mag_cons > 1e-12) & (mag_adv > 1e-12)
    cos_theta[mask] = (cons_x[mask] * adv_x[mask]
                       + cons_y[mask] * adv_y[mask]) / (
        mag_cons[mask] * mag_adv[mask]
    )
    cos_theta[(mag_cons < 1e-12) & (mag_adv > 1e-12)] = -1.0
    cos_theta[mag_adv < 1e-12] = 1.0  # no adversarial → aligned
    has_adv = (mag_adv > 1e-12)

    result = pd.DataFrame({
        "Iter": dense["Iter"].values,
        "CellId": dense["CellId"].values,
        "PosX": dense["PosX"].values,
        "PosY": dense["PosY"].values,
        "cos_theta": cos_theta,
        "has_adversarial": has_adv,
        "mag_cons": mag_cons,
        "mag_adv": mag_adv,
    })
    return result


def _build_opposition_heatmap(
    opp_df: pd.DataFrame,
    bin_cnt: int,
    lx: float, ly: float,
    ux: float, uy: float,
) -> np.ndarray:
    """Bin cells spatially and compute mean opposition per bin."""
    grid = np.full((bin_cnt, bin_cnt), np.nan, dtype=np.float64)
    counts = np.zeros((bin_cnt, bin_cnt), dtype=np.int32)

    bin_w = (ux - lx) / bin_cnt
    bin_h = (uy - ly) / bin_cnt

    cols = np.clip(
        ((opp_df["PosX"].values - lx) / bin_w).astype(np.int32),
        0, bin_cnt - 1,
    )
    rows = np.clip(
        ((opp_df["PosY"].values - ly) / bin_h).astype(np.int32),
        0, bin_cnt - 1,
    )

    for i in range(len(opp_df)):
        r, c = rows[i], cols[i]
        if np.isnan(grid[r, c]):
            grid[r, c] = float(opp_df["cos_theta"].iloc[i])
        else:
            grid[r, c] += float(opp_df["cos_theta"].iloc[i])
        counts[r, c] += 1

    mask = counts > 0
    grid[mask] /= counts[mask]
    return grid


# ══════════════════════════════════════════════════════════════════
#  App builder
# ══════════════════════════════════════════════════════════════════

def make_app(gpl: GplDb) -> dash.Dash:
    cache = diskcache.Cache("./tmp/dash_bg_cache")
    app = dash.Dash(
        __name__,
        background_callback_manager=DiskcacheManager(cache),
    )
    app.title = "Force Opposition Map"

    # ── Pre-fetch ranges ──────────────────────────────────────
    df = gpl.query(
        "SELECT MIN(Iter) AS mi, MAX(Iter) AS ma "
        "FROM gpl_cell_dense_gradients"
    )
    iter_min = int(df["mi"].iloc[0])
    iter_max = int(df["ma"].iloc[0])

    # Region bounds from metadata
    try:
        meta = gpl.get_metadata()
        prefix = None
        for key in meta:
            if key.endswith("_binCntX"):
                prefix = key[: -len("binCntX")]
                break
        if prefix:
            _f = lambda k: float(meta[k][0])
            region_lx = _f(prefix + "lx")
            region_ly = _f(prefix + "ly")
            bin_cnt_x = int(float(meta[prefix + "binCntX"][0]))
            bin_cnt_y = int(float(meta[prefix + "binCntY"][0]))
            bin_size_x = _f(prefix + "binSizeX")
            bin_size_y = _f(prefix + "binSizeY")
            region_ux = region_lx + bin_cnt_x * bin_size_x
            region_uy = region_ly + bin_cnt_y * bin_size_y
            has_region = True
        else:
            has_region = False
    except (KeyError, IndexError):
        has_region = False
        region_lx = region_ly = 0.0
        region_ux = region_uy = 1.0

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
                    html.H3("Force Opposition Map",
                            style={"marginTop": "0", "marginBottom": "6px",
                                   "color": "#2c3e50"}),
                    html.Div("Spatial & temporal structure of adversarial "
                             "force conflict",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "14px"}),

                    html.Label("Iteration range",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    dcc.RangeSlider(
                        id="iter-range-slider",
                        min=iter_min, max=iter_max, step=1,
                        value=[iter_min, min(iter_min + 100, iter_max)],
                        marks={i: str(i) for i in
                               range(iter_min, iter_max + 1,
                                     max(1, (iter_max - iter_min) // 8))},
                        tooltip={"placement": "bottom",
                                 "always_visible": False},
                    ),

                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Spatial grid size",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    dcc.Slider(
                        id="spatial-grid-slider",
                        min=20, max=120, step=10,
                        value=60,
                        marks={20: "20", 40: "40", 60: "60",
                               80: "80", 120: "120"},
                    ),

                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Opposition threshold",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div("Cells with cos(θ) < threshold are 'in conflict'",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "4px"}),
                    dcc.Slider(
                        id="threshold-slider",
                        min=-1.0, max=0.5, step=0.1,
                        value=-0.5,
                        marks={-1: "−1", -0.5: "−0.5", 0: "0", 0.5: "0.5"},
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
                    # Top: spatial heatmap + histogram side by side
                    html.Div(
                        style={"flex": "1.5", "min-height": "0",
                               "display": "flex", "gap": "8px"},
                        children=[
                            html.Div(
                                style={"flex": "2.5", "min-height": "0"},
                                children=[
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="plot-spatial",
                                            style={"height": "100%"},
                                            config={"scrollZoom": True},
                                        ),
                                        style={"height": "100%"},
                                        parent_style={"height": "100%"},
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"flex": "1.5", "min-height": "0"},
                                children=[
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="plot-histogram",
                                            style={"height": "100%"},
                                        ),
                                        style={"height": "100%"},
                                        parent_style={"height": "100%"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Bottom: time series
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
                ],
            ),
        ],
    )

    # ═══════════════════════════════════════════════════════════
    #  Callback
    # ═══════════════════════════════════════════════════════════

    @app.callback(
        Output("plot-spatial", "figure"),
        Output("plot-histogram", "figure"),
        Output("plot-timeseries", "figure"),
        Output("status-msg", "children"),
        Input("update-btn", "n_clicks"),
        State("iter-range-slider", "value"),
        State("spatial-grid-slider", "value"),
        State("threshold-slider", "value"),
        background=True,
        running=[
            (Output("update-btn", "disabled"), True, False),
            (Output("cancel-btn", "style"),
             {"width": "100%", "marginTop": "6px", "padding": "8px 0",
              "backgroundColor": "#dc3545", "color": "white",
              "border": "none", "borderRadius": "4px",
              "cursor": "pointer", "display": "block"},
             {"display": "none"}),
            (Output("status-msg", "children"), "Computing...", ""),
        ],
        cancel=[Input("cancel-btn", "n_clicks")],
        prevent_initial_call=False,
    )
    def update_plots(_n, iter_range, spatial_grid, threshold):
        iter_lo, iter_hi = (int(v) for v in (iter_range or [0, 0]))
        grid_size = int(spatial_grid)
        threshold = float(threshold)

        print(f"[OppMap] Fetching data iters {iter_lo}–{iter_hi}...")

        # ── Fetch all needed data ──────────────────────────────
        dense = gpl.cell_dense_gradients(iter_range=(iter_lo, iter_hi))
        if dense.empty:
            return (_empty("No data"), _empty(""), _empty(""), "")

        timing = gpl.cell_timing_gradients(iter_range=(iter_lo, iter_hi))
        routability = gpl.cell_routability_gradients(
            iter_range=(iter_lo, iter_hi),
        )
        density_forces = gpl.cell_density_forces(
            iter_range=(iter_lo, iter_hi),
        )

        opp = _opposition_vectors(dense, timing, routability,
                                  density_forces)

        # ── 1. Spatial heatmap (mean opposition per iteration) ──
        # Use mid-point iteration for the spatial view
        mid_iter = (iter_lo + iter_hi) // 2
        opp_mid = opp[opp["Iter"] == mid_iter]
        if opp_mid.empty:
            opp_mid = opp[opp["Iter"] == opp["Iter"].iloc[0]]

        fig_spatial = go.Figure()

        if has_region and not opp_mid.empty:
            grid = _build_opposition_heatmap(
                opp_mid, grid_size,
                region_lx, region_ly, region_ux, region_uy,
            )

            fig_spatial.add_trace(go.Heatmap(
                z=grid.T,  # Transpose for image-like orientation
                x0=region_lx, dx=(region_ux - region_lx) / grid_size,
                y0=region_ly, dy=(region_uy - region_ly) / grid_size,
                zmin=-1, zmax=1,
                colorscale=[
                    [0.0, "#d32f2f"],      # −1: strong opposition
                    [0.3, "#ff9800"],
                    [0.5, "#f5f5f5"],       # 0: neutral
                    [0.7, "#90caf9"],
                    [1.0, "#2e7d32"],       # +1: aligned
                ],
                colorbar=dict(
                    title="Mean cos(θ)",
                    titleside="right",
                ),
                name="Opposition",
                hovertemplate=(
                    "cos(θ) = %{z:.3f}<extra></extra>"
                ),
            ))

        fig_spatial.update_layout(
            title=f"Spatial Opposition — Iteration {mid_iter}",
            template="plotly_white",
            xaxis_title="X (nm)",
            yaxis_title="Y (nm)",
            xaxis=dict(constrain="domain"),
            yaxis=dict(constrain="domain", scaleanchor="x"),
            margin=dict(l=60, r=60, t=50, b=60),
        )

        # ── 2. Histogram ───────────────────────────────────────
        fig_hist = go.Figure()

        all_vals = opp["cos_theta"].values
        adv_vals = opp.loc[opp["has_adversarial"], "cos_theta"].values

        if len(all_vals) > 0:
            fig_hist.add_trace(go.Histogram(
                x=all_vals,
                name="All cells",
                nbinsx=50,
                marker_color="rgba(100, 100, 100, 0.6)",
                hovertemplate="cos(θ) range: %{x}<br>Count: %{y}<extra></extra>",
            ))
        if len(adv_vals) > 0:
            fig_hist.add_trace(go.Histogram(
                x=adv_vals,
                name="Cells with adversarial force",
                nbinsx=50,
                marker_color="rgba(211, 47, 47, 0.5)",
                hovertemplate="cos(θ) range: %{x}<br>Count: %{y}<extra></extra>",
            ))

        # Threshold marker
        fig_hist.add_vline(
            x=threshold, line_dash="dash", line_color="#e74c3c",
            line_width=2,
            annotation_text=f"conflict threshold ({threshold})",
            annotation_position="top",
        )

        frac_conflict = (
            np.mean(all_vals < threshold)
            if len(all_vals) > 0 else 0
        )

        fig_hist.update_layout(
            title=(
                f"Opposition Distribution at Iter {mid_iter}<br>"
                f"<sup>{frac_conflict*100:.1f}% of cells in conflict "
                f"(cos < {threshold})</sup>"
            ),
            template="plotly_white",
            xaxis_title="cos(θ)  (−1 = opposed, +1 = aligned)",
            yaxis_title="Cell count",
            barmode="overlay",
            legend=dict(orientation="h", yanchor="top", y=1.15,
                        xanchor="left", x=0),
            margin=dict(l=60, r=20, t=70, b=60),
        )

        # ── 3. Time series ─────────────────────────────────────
        fig_ts = go.Figure()

        ts_agg = opp.groupby("Iter").agg(
            frac_conflict=("cos_theta", lambda x: np.mean(x < threshold)),
            mean_opposition=("cos_theta", "mean"),
            median_opposition=("cos_theta", "median"),
            mean_adv_mag=("mag_adv", "mean"),
        ).reset_index()

        fig_ts.add_trace(go.Scatter(
            x=ts_agg["Iter"],
            y=ts_agg["frac_conflict"],
            mode="lines+markers",
            name=f"Fraction in conflict (cos < {threshold})",
            line=dict(color="#d32f2f", width=2),
            marker=dict(size=4),
            yaxis="y1",
        ))

        fig_ts.add_trace(go.Scatter(
            x=ts_agg["Iter"],
            y=ts_agg["mean_opposition"],
            mode="lines",
            name="Mean cos(θ)",
            line=dict(color="#1976d2", width=1.5, dash="dash"),
            yaxis="y2",
        ))

        # Timing pass markers (where adversarial force is non-zero)
        timing_iters = sorted(opp.loc[opp["has_adversarial"],
                                      "Iter"].unique())
        for ti in timing_iters:
            fig_ts.add_vline(
                x=ti, line_dash="dot", line_color="rgba(0,0,0,0.15)",
                line_width=1,
            )

        fig_ts.update_layout(
            title="Opposition Over Time",
            template="plotly_white",
            xaxis_title="Iteration",
            yaxis=dict(
                title="Fraction in conflict",
                titlefont=dict(color="#d32f2f"),
                tickfont=dict(color="#d32f2f"),
                range=[0, max(0.05, ts_agg["frac_conflict"].max() * 1.1)],
            ),
            yaxis2=dict(
                title="Mean cos(θ)",
                titlefont=dict(color="#1976d2"),
                tickfont=dict(color="#1976d2"),
                overlaying="y",
                side="right",
                range=[-1.1, 1.1],
            ),
            legend=dict(orientation="h", yanchor="top", y=1.15,
                        xanchor="left", x=0),
            hovermode="x unified",
            margin=dict(l=60, r=60, t=50, b=60),
        )

        status = (
            f"Iters {iter_lo}–{iter_hi}  |  "
            f"{frac_conflict*100:.1f}% in conflict at iter {mid_iter}"
        )
        return fig_spatial, fig_hist, fig_ts, status

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
        description="Force Opposition Map — spatial+time conflict analysis"
    )
    parser.add_argument("--db", required=True,
                        help="Path to GPL SQLite database")
    parser.add_argument("--port", type=int, default=8058,
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
