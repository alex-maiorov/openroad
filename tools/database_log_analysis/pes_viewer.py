#!/usr/bin/env python3
"""Dash GUI for exploring the electrostatic Potential Energy Surface (PES).

Background
----------
The density force in ePlace-based global placement is the negative gradient
of an electrostatic potential φ:  **F_dens = −∇φ**.  The potential φ is
computed via FFT every Nesterov iteration and now logged to the
``gpl_bin_grid`` table (column ``ElectroPhi``).

Together with the wirelength force, the potential defines a "consensus
landscape" — cells descend into local minima (basins) guided by WL + density.
Timing and routability forces are *adversarial*: they can push cells out of
their current basin, causing instability when they persistently oppose the
consensus direction.

This tool plots:

1. **Potential heatmap** — φ(x,y) as a 2D scalar field (blue = deep basins,
   yellow = high-potential barriers).
2. **Cell overlay** — cell positions at the selected iteration, coloured by
   **force opposition**: cos(θ) between (F_wl+F_dens) and (F_tim+F_rt).
   Red = timing/routability fighting the consensus; green = helping.
3. **Force arrows** — timing force vectors for cells with significant
   opposition, showing *which cells are being pulled away from their basin*.

Usage
-----
    python -m tools.database_log_analysis.pes_viewer \\
        --db path/to/placement-visualization.sqlite \\
        --port 8057
"""

import argparse
import sys
import os

import dash
from dash import dcc, html, Input, Output, State, DiskcacheManager
import diskcache
import plotly.graph_objects as go
import numpy as np
import pandas as pd

_TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from database_log_analysis import GplDb, make_metadata_panel


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

def _get_grid_meta(gpl: GplDb):
    """Return (bin_cnt_x, bin_cnt_y, bin_size_x, bin_size_y, lx, ly)."""
    meta = gpl.get_metadata()
    # Try "region_core_*" first, then fall back to any "region_*" keys
    prefix = None
    for key in meta:
        if key.endswith("_binCntX"):
            prefix = key[: -len("binCntX")]
            break
    if prefix is None:
        raise RuntimeError("No bin grid metadata found")
    _f = lambda k: float(meta[k][0])
    _i = lambda k: int(float(meta[k][0]))
    return (
        _i(prefix + "binCntX"),
        _i(prefix + "binCntY"),
        _f(prefix + "binSizeX"),
        _f(prefix + "binSizeY"),
        _f(prefix + "lx"),
        _f(prefix + "ly"),
    )


def _opposition_score(
    dense: pd.DataFrame,
    timing: pd.DataFrame,
    routability: pd.DataFrame,
    density_forces: pd.DataFrame,
) -> pd.Series:
    """Compute cos(angle) between conservative (WL+dens) and adversarial
    (tim+rout) forces.

    Returns a Series of cos(θ) values indexed the same as *dense*.
    +1 = adversarial *helps* consensus; −1 = adversarial fights it.
    """
    cons_x = dense["WlX"].values.copy()
    cons_y = dense["WlY"].values.copy()
    adv_x = np.zeros(len(dense), dtype=np.float64)
    adv_y = np.zeros(len(dense), dtype=np.float64)

    # Add density to conservative
    if not density_forces.empty:
        merged = dense[["Iter", "CellId"]].merge(
            density_forces[["Iter", "CellId", "EstDensityForceX",
                             "EstDensityForceY"]],
            on=["Iter", "CellId"], how="left",
        )
        cons_x += merged["EstDensityForceX"].fillna(0.0).values
        cons_y += merged["EstDensityForceY"].fillna(0.0).values

    # Timing → adversarial
    if not timing.empty:
        merged = dense[["Iter", "CellId"]].merge(
            timing[["Iter", "CellId", "TimX", "TimY"]],
            on=["Iter", "CellId"], how="left",
        )
        adv_x += merged["TimX"].fillna(0.0).values
        adv_y += merged["TimY"].fillna(0.0).values

    # Routability → adversarial
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
    # If no adversarial force, cos=1 (no conflict)
    cos_theta[mag_adv < 1e-12] = 1.0
    return pd.Series(cos_theta, index=dense.index, dtype=np.float64)


def _build_potential_surface(pot_df: pd.DataFrame, bin_cnt_x: int,
                              bin_cnt_y: int) -> np.ndarray:
    """Reshape bin potential rows into a 2D numpy array [ny, nx]."""
    phi = np.full((bin_cnt_y, bin_cnt_x), np.nan, dtype=np.float32)
    if pot_df.empty:
        return phi
    for _, row in pot_df.iterrows():
        idx = int(row["BinIdx"])
        row_idx = idx // bin_cnt_x
        col_idx = idx % bin_cnt_x
        if 0 <= row_idx < bin_cnt_y and 0 <= col_idx < bin_cnt_x:
            phi[row_idx, col_idx] = float(row["ElectroPhi"])
    return phi


# ══════════════════════════════════════════════════════════════════
#  App builder
# ══════════════════════════════════════════════════════════════════

def make_app(gpl: GplDb) -> dash.Dash:
    cache = diskcache.Cache("./tmp/dash_bg_cache")
    app = dash.Dash(
        __name__,
        background_callback_manager=DiskcacheManager(cache),
    )
    app.title = "PES Viewer"

    # ── Pre-fetch iteration range & grid metadata ──────────────
    df = gpl.query(
        "SELECT MIN(Iter) AS mi, MAX(Iter) AS ma "
        "FROM gpl_cell_dense_gradients"
    )
    iter_min = int(df["mi"].iloc[0])
    iter_max = int(df["ma"].iloc[0])

    try:
        grid_meta = _get_grid_meta(gpl)
        has_grid = True
    except (RuntimeError, KeyError, IndexError):
        grid_meta = (1, 1, 1.0, 1.0, 0.0, 0.0)
        has_grid = False

    bin_cnt_x, bin_cnt_y, bin_size_x, bin_size_y, region_lx, region_ly = grid_meta

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
                    html.H3("PES Viewer",
                            style={"marginTop": "0", "marginBottom": "6px",
                                   "color": "#2c3e50"}),
                    html.Div("Electrostatic potential surface + force overlay",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "14px"}),

                    html.Label("Iteration",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    dcc.Slider(
                        id="iter-slider",
                        min=iter_min, max=iter_max, step=1,
                        value=iter_min,
                        marks={i: str(i) for i in
                               range(iter_min, iter_max + 1, max(1, (iter_max - iter_min) // 10))},
                        tooltip={"placement": "bottom",
                                 "always_visible": False},
                    ),

                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Cells to highlight",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div("Cells with strongest opposition (red = fighting)",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "4px"}),
                    dcc.Slider(
                        id="top-k-slider",
                        min=0, max=500, step=10,
                        value=200,
                        marks={0: "0", 100: "100", 200: "200",
                               300: "300", 500: "500"},
                    ),

                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Force arrows",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div("Show timing force direction on cells",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "4px"}),
                    dcc.Checklist(
                        id="show-arrows-toggle",
                        options=[{"label": " Show timing force arrows",
                                  "value": "yes"}],
                        value=["yes"],
                        style={"fontSize": "12px"},
                    ),
                    html.Div([
                        html.Label("Arrow scale:", style={"fontSize": "11px"}),
                        dcc.Slider(
                            id="arrow-scale-slider",
                            min=1, max=20, step=1, value=5,
                            marks={1: "1×", 5: "5×", 10: "10×", 20: "20×"},
                        ),
                    ], style={"marginTop": "4px"}),

                    html.Hr(style={"margin": "14px 0"}),

                    dcc.Checklist(
                        id="show-potential-toggle",
                        options=[{"label": " Show potential heatmap",
                                  "value": "yes"}],
                        value=["yes"],
                        style={"fontSize": "12px"},
                    ),
                    dcc.Checklist(
                        id="log-scale-toggle",
                        options=[{"label": " Log-scale potential",
                                  "value": "yes"}],
                        value=[],
                        style={"fontSize": "12px", "marginTop": "4px"},
                    ),

                    html.Hr(style={"margin": "14px 0"}),

                    html.Button(
                        "Update Plot",
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

            # ── Main plot ─────────────────────────────────────
            html.Div(
                style={"flex": "1", "display": "flex",
                       "flexDirection": "column",
                       "padding": "10px", "boxSizing": "border-box",
                       "min-height": "0"},
                children=[
                    dcc.Loading(
                        dcc.Graph(
                            id="plot-pes",
                            style={"height": "100%"},
                            config={"scrollZoom": True},
                        ),
                        style={"height": "100%"},
                        parent_style={"height": "100%"},
                    ),
                ],
            ),
        ],
    )

    # ═══════════════════════════════════════════════════════════
    #  Callback
    # ═══════════════════════════════════════════════════════════

    @app.callback(
        Output("plot-pes", "figure"),
        Output("status-msg", "children"),
        Input("update-btn", "n_clicks"),
        State("iter-slider", "value"),
        State("top-k-slider", "value"),
        State("show-arrows-toggle", "value"),
        State("arrow-scale-slider", "value"),
        State("show-potential-toggle", "value"),
        State("log-scale-toggle", "value"),
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
    def update_plot(_n, iteration, top_k, show_arrows, arrow_scale,
                    show_potential, log_scale):
        show_arrows = bool(show_arrows)
        show_potential = bool(show_potential)
        log_scale = bool(log_scale)
        arrow_scale = float(arrow_scale)
        iter_val = int(iteration)
        top_k = int(top_k)

        fig = go.Figure()

        # ── 1. Potential heatmap ──────────────────────────────
        pot_data = None
        if show_potential and has_grid:
            print(f"[PES] Fetching potential for iter {iter_val}...")
            pot = gpl.bin_potential(iter_range=(iter_val, iter_val))
            if not pot.empty:
                phi = _build_potential_surface(pot, bin_cnt_x, bin_cnt_y)
                if log_scale:
                    phi = np.log1p(np.maximum(phi, 0))
                x_edges = np.linspace(
                    region_lx, region_lx + bin_cnt_x * bin_size_x,
                    bin_cnt_x + 1,
                )
                y_edges = np.linspace(
                    region_ly, region_ly + bin_cnt_y * bin_size_y,
                    bin_cnt_y + 1,
                )

                # Determine colour range — clip outliers for visual clarity
                valid = phi[~np.isnan(phi)]
                if len(valid) > 0:
                    vmin = np.percentile(valid, 5)
                    vmax = np.percentile(valid, 95)

                    fig.add_trace(go.Heatmap(
                        x=x_edges,
                        y=y_edges,
                        z=phi,
                        zmin=vmin,
                        zmax=vmax,
                        colorscale="Viridis",
                        colorbar=dict(
                            title="Potential φ" + (" (log)" if log_scale else ""),
                            titleside="right",
                        ),
                        name="Potential φ",
                        hovertemplate="φ = %{z:.4e}<extra></extra>",
                    ))
                    pot_data = phi

        # ── 2. Cell data: positions + forces ──────────────────
        print(f"[PES] Fetching cell data for iter {iter_val}...")
        dense = gpl.cell_dense_gradients(iter_range=(iter_val, iter_val))
        if dense.empty:
            if pot_data is not None:
                fig.update_layout(
                    title=f"Potential Surface — Iteration {iter_val}",
                    template="plotly_white",
                    xaxis_title="X (nm)",
                    yaxis_title="Y (nm)",
                )
            return fig, "No cell data for this iteration."

        timing = gpl.cell_timing_gradients(iter_range=(iter_val, iter_val))
        routability = gpl.cell_routability_gradients(
            iter_range=(iter_val, iter_val),
        )
        density_forces = gpl.cell_density_forces(
            iter_range=(iter_val, iter_val),
        )

        # Compute opposition
        opposition = _opposition_score(dense, timing, routability,
                                       density_forces)

        # ── 3. Cell scatter, coloured by opposition ───────────
        # Build combined DataFrame
        cells = dense[["CellId", "PosX", "PosY"]].copy()
        cells["Opposition"] = opposition.values  # cos(θ), +1=aligned, −1=opposed

        # Get timing force per cell for arrows
        if not timing.empty:
            tim_cols = timing[["CellId", "TimX", "TimY"]].copy()
            cells = cells.merge(tim_cols, on="CellId", how="left")
        else:
            cells["TimX"] = 0.0
            cells["TimY"] = 0.0

        cells["TimX"] = cells["TimX"].fillna(0.0)
        cells["TimY"] = cells["TimY"].fillna(0.0)
        cells["TimMag"] = np.hypot(cells["TimX"], cells["TimY"])

        # Plot low-opposition cells faintly (background)
        low_opp = cells[(cells["Opposition"] > 0.5)]
        mid_opp = cells[(cells["Opposition"] <= 0.5)
                         & (cells["Opposition"] > -0.3)]
        high_opp = cells[(cells["Opposition"] <= -0.3)]

        # Low opposition: faded grey
        if not low_opp.empty:
            sample = low_opp if len(low_opp) <= 5000 else low_opp.sample(
                5000, random_state=42,
            )
            fig.add_trace(go.Scatter(
                x=sample["PosX"], y=sample["PosY"],
                mode="markers",
                marker=dict(size=2, color="#cccccc", opacity=0.3,
                            symbol="circle"),
                name="Aligned (cos>0.5)",
                hoverinfo="skip",
            ))

        # Mid opposition: subtle
        if not mid_opp.empty:
            sample = mid_opp if len(mid_opp) <= 5000 else mid_opp.sample(
                5000, random_state=42,
            )
            fig.add_trace(go.Scatter(
                x=sample["PosX"], y=sample["PosY"],
                mode="markers",
                marker=dict(size=3, color="#90caf9", opacity=0.5,
                            symbol="circle"),
                name="Mild opposition",
                hoverinfo="skip",
            ))

        # High opposition: prominent, coloured by severity
        if len(high_opp) > 0:
            # Limit for visibility
            if len(high_opp) > top_k and top_k > 0:
                plot_high = high_opp.nsmallest(top_k, "Opposition")
            else:
                plot_high = high_opp

            fig.add_trace(go.Scatter(
                x=plot_high["PosX"], y=plot_high["PosY"],
                mode="markers",
                marker=dict(
                    size=6,
                    color=plot_high["Opposition"],
                    colorscale=[
                        [0.0, "#d32f2f"],    # −1: strong opposition (red)
                        [0.5, "#ff9800"],     # −0.3: moderate
                        [1.0, "#ffeb3b"],     # mild
                    ],
                    cmin=-1, cmax=-0.2,
                    colorbar=dict(
                        title="Opposition<br>cos(θ)",
                        titleside="right",
                        tickvals=[-1, -0.5, -0.2],
                        ticktext=["−1", "−0.5", "−0.2"],
                    ),
                    line=dict(width=0.5, color="white"),
                ),
                name="Opposed (cos<−0.3)",
                customdata=np.column_stack([
                    plot_high["CellId"],
                    plot_high["Opposition"],
                    plot_high["TimMag"],
                ]),
                hovertemplate=(
                    "<b>Cell %{customdata[0]}</b><br>"
                    "Opposition: %{customdata[1]:.3f}<br>"
                    "Tim |F|: %{customdata[2]:.1f}<extra></extra>"
                ),
            ))

        # ── 4. Timing force arrows ─────────────────────────────
        if show_arrows and len(high_opp) > 0 and high_opp["TimMag"].max() > 0:
            arrow_cells = high_opp.nsmallest(
                min(top_k, len(high_opp)), "Opposition",
            )
            arrow_cells = arrow_cells[arrow_cells["TimMag"] > 1e-9]

            max_mag = arrow_cells["TimMag"].max()
            for _, row in arrow_cells.iterrows():
                dx = row["TimX"] / max_mag * 2000 * arrow_scale
                dy = row["TimY"] / max_mag * 2000 * arrow_scale
                fig.add_annotation(
                    x=row["PosX"] + dx,
                    y=row["PosY"] + dy,
                    ax=row["PosX"],
                    ay=row["PosY"],
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="rgba(255, 50, 50, 0.7)",
                    text="",
                )

        # ── 5. Layout ──────────────────────────────────────────
        title_parts = [f"Potential Energy Surface — Iteration {iter_val}"]
        if not show_potential:
            title_parts = [f"Cell Force Opposition — Iteration {iter_val}"]

        fig.update_layout(
            title="  ".join(title_parts),
            template="plotly_white",
            xaxis_title="X (nm)",
            yaxis_title="Y (nm)",
            xaxis=dict(
                scaleanchor=None if not has_grid else "y",
                constrain="domain",
            ),
            yaxis=dict(constrain="domain"),
            hovermode="closest",
            legend=dict(
                orientation="v", yanchor="top", y=1,
                xanchor="left", x=1.02, font=dict(size=9),
            ),
            margin=dict(l=60, r=160, t=50, b=60),
        )

        status = (
            f"Iter {iter_val} — "
            f"{len(high_opp)} cells with cos(θ) < −0.3"
        )
        return fig, status

    return app


def main():
    parser = argparse.ArgumentParser(
        description="PES Viewer — electrostatic potential + force opposition"
    )
    parser.add_argument("--db", required=True,
                        help="Path to GPL SQLite database")
    parser.add_argument("--port", type=int, default=8057,
                        help="Dash port")
    parser.add_argument("--read-only", action="store_true",
                        help="Skip preprocessing")
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    print(f"Loading database: {db_path}")

    if args.read_only:
        print("  Opening read-only (derived tables must already exist)")
        gpl = GplDb(db_path, must_be_preprocessed=True)
    else:
        print("  Preprocessing (if needed) …")
        gpl = GplDb(db_path)

    app = make_app(gpl)
    print(f"Dash server: http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
