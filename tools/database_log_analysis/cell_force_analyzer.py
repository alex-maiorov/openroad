#!/usr/bin/env python3
"""Dash GUI for analyzing the statistical properties of cells affected by timing forces.

Usage
-----
    python -m tools.database_log_analysis.cell_force_analyzer \
        --db path/to/placement-visualization.sqlite \
        --port 8051
"""

import argparse
import sys
import os

import dash
from dash import dcc, html, Input, Output, State, DiskcacheManager
import diskcache
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

_TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from database_log_analysis import GplDb

def make_app(gpl: GplDb) -> dash.Dash:
    cache = diskcache.Cache("./tmp/dash_bg_cache")
    app = dash.Dash(
        __name__,
        background_callback_manager=DiskcacheManager(cache),
    )
    app.title = "Cell Force Analyzer"
    
    # Pre-fetch iter min/max
    cur = gpl.conn.execute("SELECT MIN(Iter), MAX(Iter) FROM gpl_cell_dense_gradients")
    row = cur.fetchone()
    iter_min, iter_max = (int(row[0]), int(row[1])) if row and row[0] is not None else (0, 0)
    
    app.layout = html.Div(
        style={"display": "flex", "flexDirection": "row", "height": "100vh", "fontFamily": "sans-serif"},
        children=[
            # Left sidebar
            html.Div(
                style={"width": "300px", "padding": "20px", "borderRight": "1px solid #ccc",
                       "backgroundColor": "#f8f9fa", "overflowY": "auto", "display": "flex",
                       "flexDirection": "column", "gap": "15px"},
                children=[
                    html.H2("Cell Force Analyzer", style={"margin": "0 0 10px 0", "fontSize": "20px"}),
                    
                    html.Div([
                        html.Label("Top K Cells", style={"fontWeight": "bold"}),
                        html.Div("Ranked by aggregate timing force magnitude across all iterations",
                                 style={"fontSize": "11px", "color": "#6c757d", "marginBottom": "5px"}),
                        dcc.Slider(
                            id="top-k-slider",
                            min=1, max=50, step=1,
                            marks={1: "1", 10: "10", 25: "25", 50: "50"},
                            value=5,
                        ),
                    ]),
                    
                    html.Div([
                        html.Label("Iteration Range", style={"fontWeight": "bold"}),
                        dcc.RangeSlider(
                            id="iter-range-slider",
                            min=iter_min, max=iter_max, step=1,
                            marks={iter_min: str(iter_min), iter_max: str(iter_max)},
                            value=[iter_min, iter_max],
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ]),
                    
                    html.Button("Update Plots", id="update-btn", n_clicks=0,
                                style={"marginTop": "10px", "padding": "10px",
                                       "backgroundColor": "#007bff", "color": "white",
                                       "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
                    html.Button("Cancel", id="cancel-btn",
                                style={"marginTop": "5px", "padding": "8px",
                                       "backgroundColor": "#dc3545", "color": "white",
                                       "border": "none", "borderRadius": "4px",
                                       "cursor": "pointer", "display": "none"}),
                    html.Div(id="status-msg", style={"marginTop": "10px", "fontSize": "13px",
                                                      "color": "#6c757d"}),
                ]
            ),
            
            # Main content
            html.Div(
                style={"flex": "1", "padding": "20px", "display": "flex",
                       "flexDirection": "column", "gap": "20px", "overflowY": "auto"},
                children=[
                    dcc.Graph(id="plot-path-count", style={"flex": "1"}),
                    dcc.Graph(id="plot-force-alignment", style={"flex": "1"}),
                    dcc.Graph(id="plot-force-magnitudes", style={"flex": "1"}),
                ]
            )
        ]
    )
    
    @app.callback(
        Output("plot-path-count", "figure"),
        Output("plot-force-alignment", "figure"),
        Output("plot-force-magnitudes", "figure"),
        Input("update-btn", "n_clicks"),
        State("top-k-slider", "value"),
        State("iter-range-slider", "value"),
        background=True,
        running=[
            (Output("update-btn", "disabled"), True, False),
            (Output("cancel-btn", "style"),
             {"display": "block", "marginTop": "5px", "padding": "8px",
              "backgroundColor": "#dc3545", "color": "white",
              "border": "none", "borderRadius": "4px", "cursor": "pointer"},
             {"display": "none"}),
            (Output("status-msg", "children"), "Computing...", ""),
        ],
        cancel=[Input("cancel-btn", "n_clicks")],
        prevent_initial_call=False,
    )
    def update_plots(n_clicks, top_k, iter_range):
        print(f"Fetching top {top_k} timing cells...")
        # 1. Get Top K cells by total timing force within iter_range
        top_cells_df = gpl.top_timing_cells(top_n=top_k, iter_range=iter_range)
        if top_cells_df.empty:
            return go.Figure(layout={"title": "No data"}), go.Figure(layout={"title": "No data"})
            
        cell_ids = top_cells_df["CellId"].tolist()
        
        # 2. Get number of violating paths per cell per iteration
        # Since paths are logged sparsely (e.g. every 10 iterations), we need to forward-fill
        # the path counts to all iterations in the range.
        print(f"Fetching path counts for {len(cell_ids)} cells...")
        path_counts = gpl.cell_path_counts(cell_ids=cell_ids)
        if not path_counts.empty:
            # Create a full grid of (all_iters x cell_ids) to forward-fill properly
            all_iters = list(range(iter_range[0], iter_range[1] + 1))
            grid = pd.MultiIndex.from_product([all_iters, cell_ids], names=["Iter", "CellId"]).to_frame(index=False)
            
            # Merge with our sparse counts
            path_counts = pd.merge(grid, path_counts, on=["Iter", "CellId"], how="left")
            
            # Forward fill per cell
            path_counts["PathCount"] = path_counts.groupby("CellId")["PathCount"].ffill().fillna(0)
        else:
            path_counts = pd.DataFrame()
            
        # 3. Fetch full force data (positions + all force components)
        print(f"Fetching force data for {len(cell_ids)} cells...")
        dense = gpl.cell_dense_gradients(iter_range=iter_range, cell_ids=cell_ids)
        if dense.empty:
            empty_fig = go.Figure(layout={"title": "No force data"})
            return fig1, empty_fig, empty_fig
        
        force_df = dense[["Iter", "CellId", "WlX", "WlY"]].copy()
        
        timing = gpl.cell_timing_gradients(iter_range=iter_range, cell_ids=cell_ids)
        if timing.empty:
            force_df["TimX"] = force_df["TimY"] = 0.0
        else:
            force_df = pd.merge(
                force_df,
                timing[["Iter", "CellId", "TimX", "TimY"]],
                on=["Iter", "CellId"], how="left",
            )
            force_df[["TimX", "TimY"]] = force_df[["TimX", "TimY"]].fillna(0.0)
        
        density = gpl.cell_density_forces(iter_range=iter_range, cell_ids=cell_ids)
        if density.empty:
            force_df["DensX"] = force_df["DensY"] = 0.0
        else:
            force_df = pd.merge(
                force_df,
                density[["Iter", "CellId", "EstDensityForceX", "EstDensityForceY"]],
                on=["Iter", "CellId"], how="left",
            )
            force_df[["EstDensityForceX", "EstDensityForceY"]] = (
                force_df[["EstDensityForceX", "EstDensityForceY"]].fillna(0.0)
            )
            force_df.rename(
                columns={"EstDensityForceX": "DensX", "EstDensityForceY": "DensY"},
                inplace=True,
            )
        
        # Compute derived columns
        force_df["EffectiveX"] = force_df["WlX"] + force_df["TimX"] + force_df["DensX"]
        force_df["EffectiveY"] = force_df["WlY"] + force_df["TimY"] + force_df["DensY"]
        force_df["Alignment"] = (
            force_df["EffectiveX"] * force_df["TimX"]
            + force_df["EffectiveY"] * force_df["TimY"]
        )
        force_df["mag_wl"]   = np.sqrt(force_df["WlX"]**2   + force_df["WlY"]**2)
        force_df["mag_tim"]  = np.sqrt(force_df["TimX"]**2  + force_df["TimY"]**2)
        force_df["mag_dens"] = np.sqrt(force_df["DensX"]**2 + force_df["DensY"]**2)
        force_df["mag_eff"]  = np.sqrt(force_df["EffectiveX"]**2 + force_df["EffectiveY"]**2)
        
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set1
        
        # Create Plot 1: Number of violating paths
        fig1 = go.Figure()
        fig1.update_layout(
            title="Number of Violating Paths Over Time",
            xaxis_title="Iteration",
            yaxis_title="# of Paths",
            hovermode="x unified",
            template="plotly_white",
        )
        
        # Create Plot 2: Force Alignment
        fig2 = go.Figure()
        fig2.update_layout(
            title="Force Alignment (Total Force · Timing Force) Over Time",
            xaxis_title="Iteration",
            yaxis_title="Dot Product",
            hovermode="x unified",
            template="plotly_white",
        )
        
        # Create Plot 3: Force Magnitudes (timing vs other forces)
        fig3 = go.Figure()
        fig3.update_layout(
            title="Force Magnitude Over Time",
            xaxis_title="Iteration",
            yaxis_title="|Force|",
            hovermode="x unified",
            template="plotly_white",
        )
        
        for ci, cell_id in enumerate(cell_ids):
            color = colors[ci % len(colors)]
            label = f"Cell {cell_id}"
            
            # Paths plot
            if not path_counts.empty:
                c_paths = path_counts[path_counts["CellId"] == cell_id].sort_values("Iter")
                if not c_paths.empty:
                    fig1.add_trace(go.Scatter(
                        x=c_paths["Iter"], y=c_paths["PathCount"],
                        mode="lines+markers",
                        name=f"Cell {cell_id}",
                    ))
            
            # Alignment plot
            c_f = force_df[force_df["CellId"] == cell_id].sort_values("Iter")
            if not c_f.empty:
                fig2.add_trace(go.Scatter(
                    x=c_f["Iter"], y=c_f["Alignment"],
                    mode="lines+markers",
                    name=label,
                ))
                
                # Magnitudes plot: timing force as solid line, WL + density as faint
                fig3.add_trace(go.Scatter(
                    x=c_f["Iter"], y=c_f["mag_tim"],
                    mode="lines",
                    name=f"{label} |Tim|",
                    line=dict(color=color, width=2),
                ))
                fig3.add_trace(go.Scatter(
                    x=c_f["Iter"], y=c_f["mag_wl"],
                    mode="lines",
                    name=f"{label} |WL|",
                    line=dict(color=color, width=1, dash="dot"),
                    showlegend=False,
                ))
                fig3.add_trace(go.Scatter(
                    x=c_f["Iter"], y=c_f["mag_dens"],
                    mode="lines",
                    name=f"{label} |Dens|",
                    line=dict(color=color, width=1, dash="dash"),
                    showlegend=False,
                ))
                    
        return fig1, fig2, fig3

    return app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to GPL SQLite database")
    parser.add_argument("--port", type=int, default=8051, help="Dash port")
    args = parser.parse_args()
    
    gpl = GplDb(args.db, must_be_preprocessed=False)
    app = make_app(gpl)
    print(f"Dash server: http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)

if __name__ == "__main__":
    main()
