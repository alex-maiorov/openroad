#!/usr/bin/env python3
"""Dash GUI for analyzing path similarity of top critical cells.

Usage
-----
    python -m tools.database_log_analysis.path_similarity_analyzer \
        --db path/to/placement-visualization.sqlite \
        --port 8052
"""

import argparse
import sys
import os
import time

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

from database_log_analysis import GplDb, make_metadata_panel

def lcs_length(a, b):
    """Calculate the longest common subsequence length of two sequences."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(m):
        prev = 0
        for j in range(n):
            temp = dp[j+1]
            if a[i] == b[j]:
                dp[j+1] = prev + 1
            else:
                dp[j+1] = dp[j+1] if dp[j+1] >= dp[j] else dp[j]
            prev = temp
    return dp[n]

def connected_components(nodes, edges):
    """Find disjoint sets from a list of nodes and pairwise edges."""
    adj = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    visited = set()
    comps = []
    for n in nodes:
        if n not in visited:
            comp = []
            stack = [n]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    comp.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            comps.append(comp)
    return comps

def make_app(gpl: GplDb) -> dash.Dash:
    cache = diskcache.Cache("./tmp/dash_bg_cache")
    app = dash.Dash(
        __name__,
        background_callback_manager=DiskcacheManager(cache),
    )
    app.title = "Path Similarity Analyzer"
    
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
                    html.H2("Path Similarity Analyzer", style={"margin": "0 0 10px 0", "fontSize": "20px"}),
                    
                    html.Div([
                        html.Label("Top K Cells", style={"fontWeight": "bold"}),
                        html.Div("Ranked by aggregate timing force magnitude",
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
                    
                    html.Div([
                        html.Label("Similarity Threshold", style={"fontWeight": "bold"}),
                        html.Div("Paths with similarity >= this are considered in the same category",
                                 style={"fontSize": "11px", "color": "#6c757d", "marginBottom": "5px"}),
                        dcc.Slider(
                            id="sim-threshold-slider",
                            min=0.0, max=1.0, step=0.05,
                            marks={0: "0", 0.5: "0.5", 1: "1.0"},
                            value=0.5,
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
                    make_metadata_panel(gpl),
                ]
            ),
            
            # Main content
            html.Div(
                style={"flex": "1", "padding": "20px", "display": "flex",
                       "flexDirection": "column", "gap": "20px", "overflowY": "auto"},
                children=[
                    dcc.Graph(id="plot-categories", style={"flex": "1"}),
                    dcc.Graph(id="plot-largest-pct", style={"flex": "1"}),
                    dcc.Graph(id="plot-mean-sim", style={"flex": "1"}),
                ]
            )
        ]
    )
    
    @app.callback(
        Output("plot-categories", "figure"),
        Output("plot-largest-pct", "figure"),
        Output("plot-mean-sim", "figure"),
        Input("update-btn", "n_clicks"),
        State("top-k-slider", "value"),
        State("iter-range-slider", "value"),
        State("sim-threshold-slider", "value"),
        background=True,
        running=[
            (Output("update-btn", "disabled"), True, False),
            (Output("cancel-btn", "style"),
             {"display": "block", "marginTop": "5px", "padding": "8px",
              "backgroundColor": "#dc3545", "color": "white",
              "border": "none", "borderRadius": "4px", "cursor": "pointer"},
             {"display": "none"}),
            (Output("status-msg", "children"), "Computing similarity graph...", ""),
        ],
        cancel=[Input("cancel-btn", "n_clicks")],
        prevent_initial_call=False,
    )
    def update_plots(n_clicks, top_k, iter_range, threshold):
        # 1. Get Top K cells
        print(f"Fetching top {top_k} timing cells...")
        top_cells_df = gpl.top_timing_cells(top_n=top_k, iter_range=iter_range)
        if top_cells_df.empty:
            empty_fig = go.Figure(layout={"title": "No data"})
            return empty_fig, empty_fig, empty_fig
        
        cell_ids = top_cells_df["CellId"].tolist()
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set1
        
        # Build results DataFrames per cell
        all_results: list = []  # [(CellId, df of Iter, NumCategories, LargestPct, MeanSimilarity)]
        all_iters = list(range(iter_range[0], iter_range[1] + 1))
        
        for ci, cell_id in enumerate(cell_ids):
            color = colors[ci % len(colors)]
            label = f"Cell {cell_id}"
            
            # 2. Get active paths for THIS cell only
            active_df, seqs_df = gpl.similarity_path_data(
                cell_ids=[cell_id], iter_range=iter_range
            )
            if active_df.empty or seqs_df.empty:
                print(f"  [{label}] No paths — skipping.")
                continue
            
            # Parse sequences
            path_seqs_dict = {}
            path_set_dict = {}
            for _, row in seqs_df.iterrows():
                seq = [int(c) for c in row["CellSequence"].split(",")]
                pid = row["PhysicalPathId"]
                path_seqs_dict[pid] = seq
                path_set_dict[pid] = set(seq)
            
            unique_ids = list(path_seqs_dict.keys())
            n_unique = len(unique_ids)
            n_pairs = n_unique * (n_unique - 1) // 2
            print(f"  [{label}] {n_unique} unique paths ({n_pairs} pairs) — "
                  f"precomputing similarities...")
            
            # Precompute pairwise similarities per cell
            t0 = time.time()
            sim_cache = {}
            lcs_calls = 0
            skipped = 0
            for i in range(n_unique):
                p1 = unique_ids[i]
                s1 = path_seqs_dict[p1]; l1 = len(s1); set1 = path_set_dict[p1]
                for j in range(i + 1, n_unique):
                    p2 = unique_ids[j]
                    l2 = len(path_seqs_dict[p2]); ml = max(l1, l2)
                    inter = len(set1 & path_set_dict[p2])
                    if ml > 0 and inter / ml < threshold:
                        skipped += 1
                        sim_cache[(p1,p2)] = sim_cache[(p2,p1)] = 0.0
                        continue
                    lcs_calls += 1
                    sim = lcs_length(s1, path_seqs_dict[p2]) / ml if ml > 0 else 0
                    sim_cache[(p1,p2)] = sim_cache[(p2,p1)] = sim
            print(f"    {lcs_calls} LCS calls, {skipped} skipped "
                  f"({time.time()-t0:.2f}s)")
            
            # Iterate over snapshot iterations
            results_per_cell = []
            for it, group in active_df.groupby("Iter"):
                active_ids = group["PhysicalPathId"].tolist()
                na = len(active_ids)
                
                if na == 0:
                    continue
                elif na == 1:
                    results_per_cell.append({
                        "Iter": it, "NumCategories": 1,
                        "LargestPct": 100.0, "MeanSimilarity": 1.0,
                    })
                    continue
                
                edges = []
                for i in range(na):
                    for j in range(i + 1, na):
                        if sim_cache[(active_ids[i], active_ids[j])] >= threshold:
                            edges.append((active_ids[i], active_ids[j]))
                
                comps = connected_components(active_ids, edges)
                largest = max(comps, key=len) if comps else []
                largest_sz = len(largest)
                
                # Mean similarity within largest component
                if largest_sz > 1:
                    group_sims = []
                    for i in range(largest_sz):
                        for j in range(i + 1, largest_sz):
                            group_sims.append(
                                sim_cache[(largest[i], largest[j])]
                            )
                    mean_sim = float(np.mean(group_sims))
                else:
                    mean_sim = 1.0
                
                results_per_cell.append({
                    "Iter": it,
                    "NumCategories": len(comps),
                    "LargestPct": (largest_sz / na) * 100.0 if na > 0 else 0,
                    "MeanSimilarity": mean_sim,
                })
            
            if not results_per_cell:
                print(f"  [{label}] No snapshot data — skipping.")
                continue
            
            res_df = pd.DataFrame(results_per_cell)
            # Forward/backward fill across full iteration range
            grid = pd.DataFrame({"Iter": all_iters})
            merged = pd.merge(grid, res_df, on="Iter", how="left")
            for col in ["NumCategories", "LargestPct", "MeanSimilarity"]:
                merged[col] = merged[col].ffill().bfill()
            merged["CellId"] = cell_id
            merged["Label"] = label
            merged["Color"] = color
            all_results.append((cell_id, label, color, merged))
        
        print(f"Path similarity computation finished for {len(all_results)}/{len(cell_ids)} cells.")
        
        if not all_results:
            empty_fig = go.Figure(layout={"title": "No paths to analyze"})
            return empty_fig, empty_fig, empty_fig
        
        # Build plots — one trace per cell
        fig1 = go.Figure()
        fig2 = go.Figure()
        fig3 = go.Figure()
        
        for cell_id, label, color, merged in all_results:
            fig1.add_trace(go.Scatter(
                x=merged["Iter"], y=merged["NumCategories"],
                mode="lines", line_shape="hv", name=label,
                line=dict(color=color),
            ))
            fig2.add_trace(go.Scatter(
                x=merged["Iter"], y=merged["LargestPct"],
                mode="lines", line_shape="hv", name=label,
                line=dict(color=color),
            ))
            fig3.add_trace(go.Scatter(
                x=merged["Iter"], y=merged["MeanSimilarity"],
                mode="lines", line_shape="hv", name=label,
                line=dict(color=color),
            ))
        
        fig1.update_layout(
            title=f"Number of Disjoint Path Categories (threshold >= {threshold})",
            xaxis_title="Iteration", yaxis_title="Count", template="plotly_white",
        )
        fig2.update_layout(
            title="Percentage of Paths in the Largest Category",
            xaxis_title="Iteration", yaxis_title="Percentage (%)",
            template="plotly_white", yaxis_range=[0, 105],
        )
        fig3.update_layout(
            title="Mean Pairwise Path Similarity within Largest Category",
            xaxis_title="Iteration", yaxis_title="Similarity (0 to 1)",
            template="plotly_white", yaxis_range=[0, 1.05],
        )
        
        return fig1, fig2, fig3
        
        return fig1, fig2, fig3

    return app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to GPL SQLite database")
    parser.add_argument("--port", type=int, default=8052, help="Dash port")
    parser.add_argument("--read-only", action="store_true",
                        help="Open database read-only (must be preprocessed)")
    args = parser.parse_args()

    if args.read_only:
        print("  Opening read-only (derived tables must already exist)")
        gpl = GplDb(args.db, must_be_preprocessed=True)
    else:
        gpl = GplDb(args.db)
    app = make_app(gpl)
    print(f"Dash server: http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)

if __name__ == "__main__":
    main()
