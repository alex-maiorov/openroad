#!/usr/bin/env python3
"""Dash GUI for viewing position evolution of top timing-critical cells.

Fetches the full cell history (positions + forces) once into the frontend,
then lets you scrub through iterations instantly with a snapshot slider.

Usage
-----
    python -m tools.database_log_analysis.cell_trajectory_viewer \
        --db path/to/placement-visualization.sqlite \
        --port 8053
"""

import argparse
import sys
import os
import time
import io

import dash
from dash import dcc, html, Input, Output, State, DiskcacheManager
import diskcache
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import pandas as pd
import json

_TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from database_log_analysis import GplDb

# ── Force config (matches path_visualizer) ──────────────────────
FORCE_CONFIG = {
    "wl":       {"label": "Wirelength", "color": "#1f77b4",
                 "xcol": "WlX", "ycol": "WlY", "default_scale": 5e5},
    "tim":      {"label": "Timing",     "color": "#d62728",
                 "xcol": "TimX", "ycol": "TimY", "default_scale": 5e5},
    "density":  {"label": "Density",    "color": "#2ca02c",
                 "xcol": "EstDensityForceX", "ycol": "EstDensityForceY",
                 "default_scale": 5e5},
    "effective":{"label": "Effective",  "color": "#9467bd",
                 "xcol": "EffectiveX", "ycol": "EffectiveY",
                 "default_scale": 5e5},
}
FORCE_ORDER = ["wl", "tim", "density", "effective"]


def _build_arrow_trace(force_df, fkey, scale, color):
    """Build a quiver trace for one force type."""
    x = force_df["PosX"].values
    y = force_df["PosY"].values
    cfg = FORCE_CONFIG[fkey]
    fx = force_df[cfg["xcol"]].values
    fy = force_df[cfg["ycol"]].values

    mag = np.hypot(fx, fy)
    keep = mag > 1e-30
    if not keep.any():
        return None

    xn, yn = x[keep], y[keep]
    fxn, fyn = fx[keep], fy[keep]

    fig_q = ff.create_quiver(
        x=xn, y=yn, u=fxn, v=fyn,
        scale=scale, arrow_scale=0.15, angle=np.pi / 7,
        name=f"{cfg['label']} force",
        line_color=color, line_width=1.5,
    )
    trace = fig_q.data[0]
    trace.showlegend = True
    trace.hoverinfo = "none"
    return trace


def make_app(gpl: GplDb) -> dash.Dash:
    cache = diskcache.Cache("./tmp/dash_bg_cache")
    app = dash.Dash(
        __name__,
        background_callback_manager=DiskcacheManager(cache),
    )
    app.title = "Cell Trajectory Viewer"
    app._gpl = gpl

    cur = gpl.conn.execute(
        "SELECT MIN(Iter), MAX(Iter) FROM gpl_cell_dense_gradients"
    )
    row = cur.fetchone()
    iter_min, iter_max = (
        (int(row[0]), int(row[1])) if row and row[0] is not None else (0, 0)
    )

    # ── Layout ──────────────────────────────────────────────────
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
                    "fontWeight": "bold", "color": cfg["color"],
                    "minWidth": "80px", "fontSize": "13px",
                }),
                html.Span("scale:", style={"fontSize": "11px", "color": "#888"}),
                dcc.Input(
                    id=f"force-{fkey}-scale", type="number",
                    value=cfg["default_scale"], min=0, step=1e4,
                    style={"width": "90px", "fontSize": "12px", "padding": "2px 4px"},
                ),
            ],
        )

    app.layout = html.Div(
        style={"display": "flex", "flexDirection": "row", "height": "100vh",
               "fontFamily": "sans-serif"},
        children=[
            # Left sidebar
            html.Div(
                style={"width": "300px", "padding": "20px",
                       "borderRight": "1px solid #ccc",
                       "backgroundColor": "#f8f9fa", "overflowY": "auto",
                       "display": "flex", "flexDirection": "column", "gap": "15px"},
                children=[
                    html.H2("Cell Trajectory Viewer",
                            style={"margin": "0 0 10px 0", "fontSize": "20px"}),

                    html.Div([
                        html.Label("Top K Cells", style={"fontWeight": "bold"}),
                        html.Div("Ranked by aggregate timing force",
                                 style={"fontSize": "11px", "color": "#6c757d"}),
                        dcc.Slider(id="top-k-slider", min=1, max=50, step=1,
                                   marks={1: "1", 10: "10", 25: "25", 50: "50"},
                                   value=5),
                    ]),
                    html.Div([
                        html.Label("Iteration Range (data fetch)",
                                   style={"fontWeight": "bold"}),
                        dcc.RangeSlider(
                            id="iter-range-slider",
                            min=iter_min, max=iter_max, step=1,
                            marks={iter_min: str(iter_min),
                                   iter_max: str(iter_max)},
                            value=[iter_min, iter_max],
                            tooltip={"placement": "bottom",
                                     "always_visible": True},
                        ),
                    ]),
                    html.Button(
                        "Fetch Data", id="fetch-btn", n_clicks=0,
                        style={"padding": "10px",
                               "backgroundColor": "#007bff", "color": "white",
                               "border": "none", "borderRadius": "4px",
                               "cursor": "pointer"},
                    ),
                    html.Button(
                        "Cancel", id="cancel-btn",
                        style={"padding": "8px", "backgroundColor": "#dc3545",
                               "color": "white", "border": "none",
                               "borderRadius": "4px", "cursor": "pointer",
                               "display": "none"},
                    ),
                    html.Div(id="fetch-status",
                             style={"fontSize": "12px", "color": "#6c757d"}),

                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Snapshot iteration",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div("Cell positions + force arrows at this iter",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "4px"}),
                    dcc.Slider(
                        id="snapshot-iter-slider",
                        min=iter_min, max=iter_max, step=1,
                        value=iter_min,
                        marks={iter_min: str(iter_min),
                               iter_max: str(iter_max)},
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
                ]
            ),

            # Main plot area
            html.Div(
                style={"flex": "1", "padding": "10px",
                       "boxSizing": "border-box", "overflow": "hidden"},
                children=[
                    dcc.Loading(
                        id="loading", type="circle",
                        children=[dcc.Graph(id="main-plot",
                                            style={"height": "100vh"})],
                    ),
                ],
            ),

            # Hidden store for the full trajectory data
            dcc.Store(id="trajectory-store", storage_type="memory"),
        ]
    )

    # ═════════════════════════════════════════════════════════════
    #  Callback 1: Fetch all trajectory + force data → Store
    # ═════════════════════════════════════════════════════════════
    @app.callback(
        Output("trajectory-store", "data"),
        Output("fetch-status", "children"),
        Output("snapshot-iter-slider", "max"),
        Output("snapshot-iter-slider", "marks"),
        Output("snapshot-iter-slider", "value"),
        Input("fetch-btn", "n_clicks"),
        State("top-k-slider", "value"),
        State("iter-range-slider", "value"),
        background=True,
        running=[
            (Output("fetch-btn", "disabled"), True, False),
            (Output("cancel-btn", "style"),
             {"display": "block", "padding": "8px",
              "backgroundColor": "#dc3545", "color": "white",
              "border": "none", "borderRadius": "4px",
              "cursor": "pointer"},
             {"display": "none"}),
            (Output("fetch-status", "children"), "Fetching data...", ""),
        ],
        cancel=[Input("cancel-btn", "n_clicks")],
        prevent_initial_call=False,
    )
    def fetch_data(n_clicks, top_k, iter_range):
        gpl = app._gpl
        i_lo, i_hi = int(iter_range[0]), int(iter_range[1])

        # 1. Top cells
        print(f"Fetching top {top_k} timing cells...")
        top = gpl.top_timing_cells(top_n=top_k, iter_range=(i_lo, i_hi))
        if top.empty:
            return None, "No data.", i_hi, {}, i_lo
        cell_ids = top["CellId"].tolist()
        print(f"  Cells: {cell_ids}")

        # 2. Dense gradients (WlX, WlY, PosX, PosY)
        print(f"  Fetching dense gradients for {len(cell_ids)} cells...")
        dense = gpl.cell_dense_gradients(
            iter_range=(i_lo, i_hi), cell_ids=cell_ids
        )
        if dense.empty:
            return None, "No trajectory data.", i_hi, {}, i_lo
        df = dense[["Iter", "CellId", "PosX", "PosY", "WlX", "WlY"]].copy()

        # 3. Timing gradients
        print(f"  Fetching timing gradients...")
        timing = gpl.cell_timing_gradients(
            iter_range=(i_lo, i_hi), cell_ids=cell_ids
        )
        if timing.empty:
            df["TimX"] = df["TimY"] = 0.0
        else:
            df = pd.merge(df, timing[["Iter", "CellId", "TimX", "TimY"]],
                          on=["Iter", "CellId"], how="left")
            df[["TimX", "TimY"]] = df[["TimX", "TimY"]].fillna(0.0)

        # 4. Density forces
        print(f"  Fetching density forces...")
        dens = gpl.cell_density_forces(
            iter_range=(i_lo, i_hi), cell_ids=cell_ids
        )
        if dens.empty:
            df["EstDensityForceX"] = df["EstDensityForceY"] = 0.0
        else:
            df = pd.merge(
                df,
                dens[["Iter", "CellId", "EstDensityForceX",
                       "EstDensityForceY"]],
                on=["Iter", "CellId"], how="left",
            )
            df[["EstDensityForceX", "EstDensityForceY"]] = (
                df[["EstDensityForceX", "EstDensityForceY"]].fillna(0.0)
            )

        # 5. Compute effective force
        df["EffectiveX"] = df["WlX"] + df["TimX"] + df["EstDensityForceX"]
        df["EffectiveY"] = df["WlY"] + df["TimY"] + df["EstDensityForceY"]

        # Store as JSON
        print(f"  Serializing {len(df)} rows to JSON...")
        t0 = time.time()
        out = df.to_json(orient="records", double_precision=2)
        print(f"  Done ({time.time()-t0:.2f}s).  Ready for instant scrubbing.")

        marks = {i_lo: str(i_lo), i_hi: str(i_hi)}
        return out, f"Loaded {len(cell_ids)} cells × {i_hi-i_lo+1} iters.", \
            i_hi, marks, i_lo

    # ═════════════════════════════════════════════════════════════
    #  Callback 2: Render plot from cached Store + snapshot slider
    # ═════════════════════════════════════════════════════════════
    @app.callback(
        Output("main-plot", "figure"),
        Input("snapshot-iter-slider", "value"),
        State("trajectory-store", "data"),
        State("force-wl-toggle", "value"),
        State("force-tim-toggle", "value"),
        State("force-density-toggle", "value"),
        State("force-effective-toggle", "value"),
        State("force-wl-scale", "value"),
        State("force-tim-scale", "value"),
        State("force-density-scale", "value"),
        State("force-effective-scale", "value"),
        prevent_initial_call=False,
    )
    def render_plot(
        viz_iter, store_json,
        tgl_wl, tgl_tim, tgl_dens, tgl_eff,
        scl_wl, scl_tim, scl_dens, scl_eff,
    ):
        if store_json is None:
            return go.Figure(layout={"title": "Click 'Fetch Data' to load cells."})

        # Deserialize
        df = pd.read_json(io.StringIO(store_json), orient="records")

        # Filter to snapshot iteration
        snap = df[df["Iter"] == viz_iter]
        if snap.empty:
            viz_iter = int(df["Iter"].iloc[0])
            snap = df[df["Iter"] == viz_iter]

        cell_ids = sorted(snap["CellId"].unique())
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set1

        toggles = {
            "wl": bool(tgl_wl), "tim": bool(tgl_tim),
            "density": bool(tgl_dens), "effective": bool(tgl_eff),
        }
        scales = {
            "wl": float(scl_wl or 5e5), "tim": float(scl_tim or 5e5),
            "density": float(scl_dens or 5e5),
            "effective": float(scl_eff or 5e5),
        }

        fig = go.Figure()
        fig.update_layout(
            title=f"Cell positions at iteration {viz_iter}",
            xaxis_title="X (nm)", yaxis_title="Y (nm)",
            template="plotly_white", hovermode="closest",
            legend=dict(orientation="v", yanchor="top", y=1,
                        xanchor="left", x=1.02, font=dict(size=9)),
            margin=dict(l=40, r=160, t=50, b=40),
            dragmode="pan",
        )

        for ci, cell_id in enumerate(cell_ids):
            cdata = snap[snap["CellId"] == cell_id]
            color = colors[ci % len(colors)]
            label = f"Cell {cell_id}"

            # Position marker
            fig.add_trace(go.Scatter(
                x=cdata["PosX"], y=cdata["PosY"],
                mode="markers",
                marker=dict(size=8, color=color,
                            line=dict(width=1, color="white")),
                name=label,
                legendgroup=label,
                hovertemplate=f"Cell {cell_id}<br>"
                              f"X=%{{x:.1f}}  Y=%{{y:.1f}}<extra></extra>",
            ))

            # Force arrows
            for fkey in FORCE_ORDER:
                if not toggles.get(fkey):
                    continue
                cfg = FORCE_CONFIG[fkey]
                tr = _build_arrow_trace(cdata, fkey, scales[fkey], cfg["color"])
                if tr is not None:
                    tr.legendgroup = label
                    tr.showlegend = False
                    fig.add_trace(tr)

        return fig

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Dash GUI for cell trajectory scrubbing"
    )
    parser.add_argument("--db", required=True,
                        help="Path to GPL SQLite database")
    parser.add_argument("--port", type=int, default=8053, help="Dash port")
    parser.add_argument("--read-only", action="store_true",
                        help="Open database read-only (must be preprocessed)")
    args = parser.parse_args()

    if args.read_only:
        gpl = GplDb(args.db, must_be_preprocessed=True)
    else:
        gpl = GplDb(args.db)
    app = make_app(gpl)
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
