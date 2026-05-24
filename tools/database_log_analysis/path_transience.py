#!/usr/bin/env python3
"""Dash GUI for exploring path transience in GPL timing-driven placement.

Why do 87.7 % of physical paths appear at only one iteration?
How does path churn relate to placement convergence?
Which cells are persistently on timing-critical paths?

Three linked plots:
  1. Path lifecycle heatmap  —  which iterations does each physical
     path appear at, colored by its slack.
  2. Path turnover stacked bars  —  new / continuing / returning
     paths per iteration.
  3. Cell–path scatter  —  how many distinct paths each cell
     participates in vs how many iterations it is active.

Usage
-----
    python -m tools.database_log_analysis.path_transience \\
        --db path/to/placement-visualization.sqlite \\
        --port 8055

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
import numpy as np
import pandas as pd

_TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from database_log_analysis import GplDb, make_metadata_panel


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

def _parse_iter_range(val, default):
    if val is None:
        return default
    return int(val[0]), int(val[1])


# ══════════════════════════════════════════════════════════════════
#  Application builder
# ══════════════════════════════════════════════════════════════════

def make_app(gpl: GplDb) -> dash.Dash:
    cache = diskcache.Cache("./tmp/dash_bg_cache")
    app = dash.Dash(
        __name__,
        background_callback_manager=DiskcacheManager(cache),
    )
    app.title = "Path Transience Explorer"
    app._gpl = gpl

    # ── Pre-fetch iteration range ─────────────────────────────
    df = gpl.query(
        "SELECT MIN(Iter) AS mi, MAX(Iter) AS ma "
        "FROM gpl_path_slacks"
    )
    iter_min = int(df["mi"].iloc[0])
    iter_max = int(df["ma"].iloc[0])
    all_iters = list(range(iter_min, iter_max + 1, 10))

    has_timing = gpl._exists("gpl_path_slacks")

    # ── Layout ────────────────────────────────────────────────
    app.layout = html.Div(
        style={"display": "flex", "height": "100vh", "margin": "0",
               "padding": "0", "fontFamily": "Segoe UI, Arial, sans-serif"},
        children=[
            # ── Sidebar ───────────────────────────────────────
            html.Div(
                id="sidebar",
                style={"width": "320px", "minWidth": "320px",
                       "padding": "16px 14px", "overflowY": "auto",
                       "backgroundColor": "#f8f9fa",
                       "borderRight": "1px solid #dee2e6",
                       "boxSizing": "border-box"},
                children=[
                    html.H3("Path Transience Explorer",
                            style={"marginTop": "0", "marginBottom": "14px",
                                   "color": "#2c3e50"}),

                    html.Label("Iteration range",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    dcc.RangeSlider(
                        id="iter-range-slider",
                        min=iter_min, max=iter_max, step=10,
                        value=[iter_min, iter_max],
                        marks={i: str(i) for i in
                               range(iter_min, iter_max + 1, 100)},
                        tooltip={"placement": "bottom",
                                 "always_visible": False},
                    ),
                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Heatmap rows",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div("Number of most-persistent paths to show",
                             style={"fontSize": "11px", "color": "#6c757d"}),
                    dcc.Slider(
                        id="heatmap-rows-slider",
                        min=20, max=200, step=10,
                        value=80,
                        marks={20: "20", 50: "50", 100: "100",
                               150: "150", 200: "200"},
                    ),
                    html.Div([
                        html.Label("Min appearances:",
                                   style={"fontSize": "11px",
                                          "marginRight": "4px"}),
                        html.Span(id="min-appear-display", children="2",
                                  style={"fontWeight": "bold"}),
                    ]),
                    dcc.Slider(
                        id="min-appear-slider",
                        min=1, max=20, step=1,
                        value=2,
                        marks={1: "1", 5: "5", 10: "10", 15: "15", 20: "20"},
                    ),
                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Slack coloring",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div("Slack range for heatmap color scale (ps)",
                             style={"fontSize": "11px", "color": "#6c757d"}),
                    dcc.RangeSlider(
                        id="slack-color-range",
                        min=-50000, max=10000, step=500,
                        value=[-30000, 5000],
                        marks={-50000: "-50k", -25000: "-25k",
                               0: "0", 5000: "5k", 10000: "10k"},
                        tooltip={"placement": "bottom",
                                 "always_visible": True},
                    ),
                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Cell scatter filtering",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div("Min distinct paths per cell to include",
                             style={"fontSize": "11px", "color": "#6c757d"}),
                    dcc.Slider(
                        id="min-cell-paths-slider",
                        min=1, max=20, step=1,
                        value=1,
                        marks={1: "1", 5: "5", 10: "10", 15: "15", 20: "20"},
                    ),
                    html.Hr(style={"margin": "14px 0"}),

                    dcc.Store(id="transience-cache", storage_type="memory"),

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
                                       "backgroundColor": "#dc3545",
                                       "color": "white",
                                       "border": "none", "borderRadius": "4px",
                                       "cursor": "pointer", "display": "none"}),
                    html.Div(id="status-msg",
                             style={"marginTop": "8px", "fontSize": "12px",
                                    "color": "#6c757d"}),
                    html.Div(id="summary-stats",
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
                    # Top half: heatmap (left) + slack-by-category (right)
                    html.Div(
                        style={"flex": "1", "min-height": "0",
                               "display": "flex", "gap": "8px"},
                        children=[
                            html.Div(
                                style={"flex": "1.4", "min-height": "0"},
                                children=[
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="plot-heatmap",
                                            style={"height": "100%"},
                                            config={"scrollZoom": True},
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
                                            id="plot-slack-by-category",
                                            style={"height": "100%"},
                                        ),
                                        style={"height": "100%"},
                                        parent_style={"height": "100%"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Bottom half: Turnover + Cell scatter (side by side)
                    html.Div(
                        style={"flex": "1", "min-height": "0",
                               "display": "flex", "gap": "8px"},
                        children=[
                            html.Div(
                                style={"flex": "1", "min-height": "0"},
                                children=[
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="plot-turnover",
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
                                            id="plot-cell-scatter",
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

    # ═════════════════════════════════════════════════════════════
    #  Data-fetch helpers (inside make_app for closure access)
    # ═════════════════════════════════════════════════════════════

    def _fetch_all(iter_lo, iter_hi):
        """Fetch and return a dict with all data needed for the 3 plots.

        Returns None if no path data is available.
        """
        if not has_timing:
            return None

        print(f"[transience] Fetching path×iter matrix "
              f"(iters {iter_lo}–{iter_hi})...")

        # ── Path×Iter matrix ──────────────────────────────────
        path_iter = gpl.query(f"""
            SELECT sig.PhysicalPathId, sig.Iter, ps.Slack
            FROM gpl_path_signatures sig
            JOIN gpl_path_slacks ps
              ON sig.PathId = ps.PathId AND sig.Iter = ps.Iter
            WHERE sig.Iter BETWEEN ? AND ?
            ORDER BY sig.PhysicalPathId, sig.Iter
        """, [iter_lo, iter_hi])

        if path_iter.empty:
            return None

        # Slack in picoseconds
        path_iter["Slack_ps"] = path_iter["Slack"] * 1e12

        # ── Classify each (PhysId, Iter) as new/continuing/returning ─
        print("[transience] Classifying path categories...")
        pi_sorted = path_iter.sort_values(["PhysicalPathId", "Iter"])
        categories = []
        # Build per-phys-id set of iters for fast lookup
        phys_iter_sets = {}
        for pid, grp in pi_sorted.groupby("PhysicalPathId"):
            phys_iter_sets[pid] = set(grp["Iter"].values)

        for _, row in pi_sorted.iterrows():
            pid = int(row["PhysicalPathId"])
            it = int(row["Iter"])
            iters = phys_iter_sets[pid]
            min_it = min(iters)
            if it == min_it:
                categories.append("new")
            elif (it - 10) in iters:
                categories.append("continuing")
            else:
                categories.append("returning")
        path_iter["category"] = categories

        # ── Per-path summary ──────────────────────────────────
        path_summary = path_iter.groupby("PhysicalPathId").agg(
            n_appearances=("Iter", "count"),
            min_slack_ps=("Slack_ps", "min"),
            mean_slack_ps=("Slack_ps", "mean"),
            first_iter=("Iter", "min"),
            last_iter=("Iter", "max"),
        ).reset_index()

        # Iter span (how many path-query intervals the path spans)
        path_summary["iter_span"] = (
            (path_summary["last_iter"] - path_summary["first_iter"]) / 10
        ).astype(int)

        # ── Cell count per path ────────────────────────────────
        # Compute directly from the cell_path data we already fetch
        # below, avoiding a second SQL query with too many params.
        print("[transience] Computing cell counts per path...")
        # Defer: we'll compute from cell_path below and merge back

        # ── Cell×Path summary ──────────────────────────────────
        print("[transience] Fetching cell–path membership...")
        cell_path = gpl.query(f"""
            SELECT pc.CellId, sig.PhysicalPathId, sig.Iter
            FROM gpl_path_cells pc
            JOIN gpl_path_signatures sig
              ON pc.PathId = sig.PathId AND pc.Iter = sig.Iter
            WHERE sig.Iter BETWEEN ? AND ?
        """, [iter_lo, iter_hi])

        # ── Cell count per path (computed from cell_path) ─────
        cc = cell_path.groupby("PhysicalPathId")["CellId"].nunique()
        path_summary["cell_count"] = (
            path_summary["PhysicalPathId"]
            .map(cc)
            .fillna(0)
            .astype(int)
        )

        cell_summary = cell_path.groupby("CellId").agg(
            distinct_paths=("PhysicalPathId", "nunique"),
            distinct_iters=("Iter", "nunique"),
            total_appearances=("PhysicalPathId", "count"),
        ).reset_index()

        # Average slack of paths containing this cell
        cell_slack = pd.merge(
            cell_path,
            path_iter[["PhysicalPathId", "Iter", "Slack_ps"]],
            on=["PhysicalPathId", "Iter"],
        )
        cell_avg_slack = cell_slack.groupby("CellId")["Slack_ps"].mean()
        cell_summary["avg_slack_ps"] = cell_summary["CellId"].map(
            cell_avg_slack
        )

        # ── Turnover analysis ─────────────────────────────────
        # For each iteration, classify each PhysicalPathId:
        #   "new"       = first appearance at this iter
        #   "continuing"= seen at this iter AND previous iter
        #   "returning" = seen before, not at prev iter, seen at this iter
        print("[transience] Computing path turnover...")
        path_iter_sorted = path_iter.sort_values(
            ["PhysicalPathId", "Iter"]
        )

        # Build a list of all iters present, per phys_id
        phys_iters = {}
        for _, row in path_iter_sorted.iterrows():
            pid = int(row["PhysicalPathId"])
            it = int(row["Iter"])
            phys_iters.setdefault(pid, []).append(it)

        turnover_rows = []
        unique_iters = sorted(path_iter["Iter"].unique())
        for it in unique_iters:
            new_cnt = 0
            continuing_cnt = 0
            returning_cnt = 0
            for pid, iters in phys_iters.items():
                if it not in iters:
                    continue
                if iters[0] == it:
                    new_cnt += 1
                elif it - 10 in iters:  # appeared at previous query slot
                    continuing_cnt += 1
                else:
                    returning_cnt += 1
            turnover_rows.append({
                "Iter": it,
                "new": new_cnt,
                "continuing": continuing_cnt,
                "returning": returning_cnt,
            })
        turnover = pd.DataFrame(turnover_rows)
        turnover["total"] = (turnover["new"] + turnover["continuing"]
                             + turnover["returning"])

        print(f"[transience] Done. {len(path_summary)} paths, "
              f"{len(cell_summary)} cells.")

        return {
            "path_iter_records": path_iter.to_dict("records"),
            "path_summary_records": path_summary.to_dict("records"),
            "cell_summary_records": cell_summary.to_dict("records"),
            "turnover_records": turnover.to_dict("records"),
            "iter_lo": iter_lo,
            "iter_hi": iter_hi,
            "unique_iters": unique_iters,
            "n_total_paths": len(path_summary),
            "n_total_cells": len(cell_summary),
        }

    # ═════════════════════════════════════════════════════════════
    #  Callback  A  —  Data fetch + cache  (Update button)
    # ═════════════════════════════════════════════════════════════

    @app.callback(
        Output("transience-cache", "data"),
        Input("update-btn", "n_clicks"),
        State("iter-range-slider", "value"),
        background=True,
        running=[
            (Output("update-btn", "disabled"), True, False),
            (Output("cancel-btn", "style"),
             {"width": "100%", "marginTop": "6px", "padding": "8px 0",
              "backgroundColor": "#dc3545", "color": "white",
              "border": "none", "borderRadius": "4px", "cursor": "pointer",
              "display": "block"},
             {"display": "none"}),
            (Output("status-msg", "children"),
             "Fetching path transience data...", ""),
        ],
        cancel=[Input("cancel-btn", "n_clicks")],
        prevent_initial_call=False,
    )
    def fetch_data(_n, iter_range):
        iter_lo, iter_hi = _parse_iter_range(
            iter_range, (iter_min, iter_max)
        )
        data = _fetch_all(iter_lo, iter_hi)
        return data if data else {}

    # ═════════════════════════════════════════════════════════════
    #  Callback  B  —  Render all three plots from cache
    # ═════════════════════════════════════════════════════════════

    @app.callback(
        Output("plot-heatmap", "figure"),
        Output("plot-slack-by-category", "figure"),
        Output("plot-turnover", "figure"),
        Output("plot-cell-scatter", "figure"),
        Output("summary-stats", "children"),
        Output("min-appear-display", "children"),
        Input("transience-cache", "data"),
        State("heatmap-rows-slider", "value"),
        State("min-appear-slider", "value"),
        State("slack-color-range", "value"),
        State("min-cell-paths-slider", "value"),
        prevent_initial_call=True,
    )
    def render_plots(data, heatmap_rows, min_appear,
                     slack_color_range, min_cell_paths):
        if not data or not data.get("path_iter_records"):
            empty = _empty_fig("No data — click Update")
            return empty, empty, empty, empty, "No data", str(min_appear or 2)

        # ── Restore dataframes ────────────────────────────────
        pi = pd.DataFrame.from_records(data["path_iter_records"])
        ps = pd.DataFrame.from_records(data["path_summary_records"])
        cs = pd.DataFrame.from_records(data["cell_summary_records"])
        to = pd.DataFrame.from_records(data["turnover_records"])
        iter_lo = data["iter_lo"]
        iter_hi = data["iter_hi"]
        unique_iters = sorted(data["unique_iters"])

        heatmap_rows = int(heatmap_rows or 80)
        min_appear = int(min_appear or 2)
        slack_lo, slack_hi = (float(v) for v in
                              (slack_color_range or (-30000, 5000)))
        min_cell_paths = int(min_cell_paths or 1)

        # ── Summary stats ────────────────────────────────────
        n_total = data["n_total_paths"]
        n_singleton = int((ps["n_appearances"] == 1).sum())
        n_persistent = int((ps["n_appearances"] > 1).sum())
        max_appear = int(ps["n_appearances"].max())

        summary = [
            html.B(f"{n_total:,} unique physical paths"),
            f" over {len(unique_iters)} STA queries (iters "
            f"{iter_lo}–{iter_hi})",
            html.Br(),
            html.Span(f"{n_singleton:,} appear once ("
                      f"{100*n_singleton/max(n_total,1):.0f}%)",
                      style={"color": "#e74c3c"}),
            "  •  ",
            html.Span(f"{n_persistent:,} appear ≥2× ("
                      f"{100*n_persistent/max(n_total,1):.0f}%)",
                      style={"color": "#27ae60"}),
            "  •  ",
            f"max appearances: {max_appear}",
        ]

        # ════════════════════════════════════════════════════════
        #  Plot 1: Path Lifecycle Heatmap
        # ════════════════════════════════════════════════════════

        # Filter to paths meeting min_appear, take top N by persistence
        eligible = ps[ps["n_appearances"] >= min_appear].nlargest(
            heatmap_rows, "n_appearances"
        )
        top_ids = set(eligible["PhysicalPathId"])

        # Build pivot: rows=PhysicalPathId, cols=Iter, values=Slack_ps
        sub = pi[pi["PhysicalPathId"].isin(top_ids)]
        pivot = sub.pivot_table(
            index="PhysicalPathId", columns="Iter",
            values="Slack_ps", aggfunc="first",
        )

        # Sort rows by n_appearances (most persistent at top)
        row_order = (ps[ps["PhysicalPathId"].isin(top_ids)]
                     .set_index("PhysicalPathId")
                     .loc[pivot.index]
                     .sort_values("n_appearances", ascending=False)
                     .index)
        pivot = pivot.loc[row_order]
        pivot = pivot.reindex(columns=sorted(unique_iters))

        # Build hover text matrix
        hover_text = []
        for phys_id in pivot.index:
            row_text = []
            for it in pivot.columns:
                val = pivot.loc[phys_id, it]
                if pd.isna(val):
                    row_text.append(f"PhysId {phys_id}<br>Iter {it}<br>"
                                    "(absent)")
                else:
                    row_text.append(
                        f"PhysId {phys_id}<br>Iter {it}<br>"
                        f"Slack: {val:.1f} ps"
                    )
            hover_text.append(row_text)

        # Y-axis labels: PhysId + cell_count + n_appearances
        y_labels = []
        info = ps.set_index("PhysicalPathId")
        for phys_id in pivot.index:
            row = info.loc[phys_id]
            y_labels.append(
                f"P{phys_id}  "
                f"[{int(row['cell_count'])}c]  "
                f"×{int(row['n_appearances'])}"
            )

        fig1 = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.astype(str),
            y=y_labels,
            colorscale="RdBu",
            zmin=slack_lo,
            zmax=slack_hi,
            zmid=0,
            text=hover_text,
            hoverinfo="text",
            hoverongaps=False,
            colorbar=dict(
                title="Slack (ps)",
                titleside="right",
                tickprefix="",
            ),
        ))
        fig1.update_layout(
            title=(f"Path lifecycle heatmap — top {len(pivot)} "
                   f"most persistent paths "
                   f"(min {min_appear} appearances)"),
            xaxis_title="Iteration (STA query slot)",
            yaxis_title="PhysicalPathId  [cells]  ×appearances",
            template="plotly_white",
            height=None,
            margin=dict(l=20, r=20, t=50, b=40),
            yaxis=dict(tickfont=dict(size=8)),
            xaxis=dict(tickfont=dict(size=9)),
        )

        # ════════════════════════════════════════════════════════
        #  Plot 2: Path Turnover Stacked Bars
        # ════════════════════════════════════════════════════════

        fig2 = go.Figure()
        colors_turnover = {
            "new": "#e74c3c",
            "continuing": "#2ecc71",
            "returning": "#f39c12",
        }

        # Stacked bars: "continuing" at bottom, "returning" middle,
        # "new" at top
        fig2.add_trace(go.Bar(
            x=to["Iter"], y=to["continuing"],
            name="Continuing (prev iter)",
            marker_color=colors_turnover["continuing"],
            hovertemplate="Continuing: %{y}<extra></extra>",
        ))
        fig2.add_trace(go.Bar(
            x=to["Iter"], y=to["returning"],
            name="Returning (gap, then back)",
            marker_color=colors_turnover["returning"],
            hovertemplate="Returning: %{y}<extra></extra>",
        ))
        fig2.add_trace(go.Bar(
            x=to["Iter"], y=to["new"],
            name="New (first ever)",
            marker_color=colors_turnover["new"],
            hovertemplate="New: %{y}<extra></extra>",
        ))

        # Overlay line: total
        fig2.add_trace(go.Scatter(
            x=to["Iter"], y=to["total"],
            name="Total paths",
            mode="lines+markers",
            line=dict(color="#2c3e50", width=2),
            marker=dict(size=4),
            yaxis="y2",
            hovertemplate="Total: %{y}<extra></extra>",
        ))

        fig2.update_layout(
            title="Path turnover per iteration",
            xaxis_title="Iteration",
            yaxis_title="Number of paths",
            template="plotly_white",
            barmode="stack",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="top", y=1.12,
                        xanchor="center", x=0.5, font=dict(size=9)),
            margin=dict(l=40, r=40, t=30, b=40),
            yaxis2=dict(
                overlaying="y", side="right",
                showgrid=False, title="Total",
            ),
        )

        # ════════════════════════════════════════════════════════
        #  Plot 3: Cell–Path Scatter
        # ════════════════════════════════════════════════════════

        cs_filt = cs[cs["distinct_paths"] >= min_cell_paths].copy()
        n_cells_total = len(cs)
        n_cells_shown = len(cs_filt)

        fig3 = go.Figure()

        if not cs_filt.empty:
            sizes = np.log1p(cs_filt["total_appearances"].values)
            sizes = np.clip(sizes / max(np.median(sizes), 1e-9) * 8, 4, 30)

            # Color by avg slack
            avg_slack = cs_filt["avg_slack_ps"].fillna(0)

            fig3.add_trace(go.Scatter(
                x=cs_filt["distinct_paths"],
                y=cs_filt["distinct_iters"],
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=avg_slack,
                    colorscale="RdBu",
                    cmin=slack_lo,
                    cmax=slack_hi,
                    cmid=0,
                    colorbar=dict(title="Avg slack (ps)",
                                  titleside="right"),
                    line=dict(width=0.5, color="white"),
                ),
                customdata=np.column_stack([
                    cs_filt["CellId"],
                    cs_filt["distinct_paths"],
                    cs_filt["distinct_iters"],
                    cs_filt["total_appearances"],
                    avg_slack,
                ]),
                hovertemplate=(
                    "<b>Cell %{customdata[0]}</b><br>"
                    "Distinct paths: %{customdata[1]}<br>"
                    "Distinct iters: %{customdata[2]}<br>"
                    "Total appearances: %{customdata[3]}<br>"
                    "Avg slack: %{customdata[4]:.1f} ps<extra></extra>"
                ),
                name="Cells",
            ))

            # Add jittered identity line for reference
            max_val = max(cs_filt["distinct_paths"].max(),
                          cs_filt["distinct_iters"].max())
            fig3.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines",
                line=dict(color="#95a5a6", width=1, dash="dash"),
                name="y=x (1 path/iter)",
                hoverinfo="skip",
            ))

        fig3.update_layout(
            title=(f"Cell–path membership scatter "
                   f"({n_cells_shown:,} of {n_cells_total:,} cells, "
                   f"≥{min_cell_paths} distinct paths)"),
            xaxis_title="Distinct PhysicalPathIds",
            yaxis_title="Distinct iterations active",
            template="plotly_white",
            hovermode="closest",
            margin=dict(l=40, r=40, t=30, b=40),
        )

        # ════════════════════════════════════════════════════════
        #  Plot 4: Slack Distribution by Path Category
        # ════════════════════════════════════════════════════════
        # Box plots grouped by iteration, split by new/continuing/returning.
        # This shows whether newly-appearing paths have worse slack than
        # continuing ones, and how returning paths compare.

        cat_colors = {"new": "#e74c3c", "continuing": "#2ecc71",
                      "returning": "#f39c12"}
        cat_order = ["continuing", "returning", "new"]

        fig4 = go.Figure()
        for cat in cat_order:
            sub_cat = pi[pi["category"] == cat]
            if sub_cat.empty:
                continue
            # Group by iteration, collect slack values
            for it in unique_iters:
                vals = sub_cat[sub_cat["Iter"] == it]["Slack_ps"]
                if len(vals) == 0:
                    continue
                fig4.add_trace(go.Box(
                    y=vals,
                    x=[str(it)] * len(vals),
                    name=cat,
                    legendgroup=cat,
                    marker_color=cat_colors[cat],
                    line_width=1,
                    fillcolor=cat_colors[cat],
                    opacity=0.5,
                    showlegend=(it == unique_iters[0]),
                    boxpoints="outliers",
                    jitter=0.3,
                    pointpos=0,
                    marker=dict(size=2, opacity=0.3),
                    hoverinfo="skip",
                ))

        fig4.update_layout(
            title="Path slack distribution by category per iteration",
            xaxis_title="Iteration",
            yaxis_title="Slack (ps)",
            template="plotly_white",
            boxmode="group",
            boxgroupgap=0.3,
            boxgap=0.1,
            legend=dict(orientation="h", yanchor="top", y=1.08,
                        xanchor="center", x=0.5, font=dict(size=9)),
            margin=dict(l=50, r=20, t=30, b=40),
            xaxis=dict(tickfont=dict(size=8), tickangle=-45),
        )
        # Add a horizontal line at slack=0
        fig4.add_hline(y=0, line_dash="dash", line_color="#95a5a6",
                       line_width=1, opacity=0.6)

        return (fig1, fig4, fig2, fig3, summary,
                str(min_appear))

    return app


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

def _empty_fig(msg):
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


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Dash GUI for path transience exploration"
    )
    parser.add_argument("--db", required=True,
                        help="Path to GPL SQLite database")
    parser.add_argument("--port", type=int, default=8055,
                        help="Dash port")
    parser.add_argument("--read-only", action="store_true",
                        help="Skip preprocessing")
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    print(f"Loading database: {db_path}")

    if args.read_only:
        gpl = GplDb(db_path, must_be_preprocessed=True)
    else:
        print("  Preprocessing (if needed) …")
        gpl = GplDb(db_path)

    app = make_app(gpl)
    print(f"Dash server: http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
