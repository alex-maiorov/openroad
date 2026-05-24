#!/usr/bin/env python3
"""Dash GUI for exploring cell-level statistics in relation to timing paths.

Shows how force composition, opposition, and movement vary across cells
grouped by how many violating timing paths they belong to.  Designed for
open-ended exploration: pick axes, filter cell types, compare path cells
against non-path cells.

Usage
-----
    python -m tools.database_log_analysis.cell_path_stats \\
        --db path/to/placement-visualization.sqlite \\
        --port 8054

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

from database_log_analysis import GplDb

# ══════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════

FORCE_METRICS = {
    "mag_wl":    {"label": "|WL force|",          "col": "mag_wl"},
    "mag_tim":   {"label": "|Timing force|",      "col": "mag_tim"},
    "mag_dens":  {"label": "|Density force|",     "col": "mag_dens"},
    "mag_eff":   {"label": "|Effective force|",   "col": "mag_eff"},
    "opposition": {"label": "Opposition (WL vs Tim)",
                   "col": "opposition"},
    "alignment": {"label": "Alignment (Eff · Tim)",
                  "col": "Alignment"},
}

DIST_METRICS = {
    "total_dist":  {"label": "Total displacement (nm)", "col": "total_dist"},
    "rms_dist":    {"label": "RMS displacement / iter", "col": "rms_dist"},
    "max_dist":    {"label": "Max single-step (nm)",     "col": "max_dist"},
}

PATH_BUCKETS = [
    (0, 0, "#95a5a6", "0 paths"),
    (1, 1, "#3498db", "1 path"),
    (2, 3, "#e67e22", "2-3 paths"),
    (4, 9999, "#e74c3c", "4+ paths"),
]


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

def _bucket_path_count(n):
    for lo, hi, color, label in PATH_BUCKETS:
        if lo <= n <= hi:
            return label
    return "4+ paths"


def _bucket_color(n):
    for lo, hi, color, label in PATH_BUCKETS:
        if lo <= n <= hi:
            return color
    return "#e74c3c"


# ══════════════════════════════════════════════════════════════════
#  Application builder
# ══════════════════════════════════════════════════════════════════

def make_app(gpl: GplDb) -> dash.Dash:
    cache = diskcache.Cache("./tmp/dash_bg_cache")
    app = dash.Dash(
        __name__,
        background_callback_manager=DiskcacheManager(cache),
    )
    app.title = "Cell-Path Statistics"
    app._gpl = gpl

    # ── Pre-fetch iteration range ─────────────────────────────
    df = gpl.query(
        "SELECT MIN(Iter) AS mi, MAX(Iter) AS ma "
        "FROM gpl_cell_dense_gradients"
    )
    iter_min = int(df["mi"].iloc[0])
    iter_max = int(df["ma"].iloc[0])

    # Check for timing data
    has_timing = gpl._exists("gpl_path_slacks")

    # ── Layout ────────────────────────────────────────────────
    app.layout = html.Div(
        style={"display": "flex", "height": "100vh", "margin": "0",
               "padding": "0", "fontFamily": "Segoe UI, Arial, sans-serif"},
        children=[
            # ── Sidebar ───────────────────────────────────────
            html.Div(
                id="sidebar",
                style={"width": "330px", "minWidth": "330px",
                       "padding": "16px 14px", "overflowY": "auto",
                       "backgroundColor": "#f8f9fa",
                       "borderRight": "1px solid #dee2e6",
                       "boxSizing": "border-box"},
                children=[
                    html.H3("Cell–Path Statistics",
                            style={"marginTop": "0", "marginBottom": "14px",
                                   "color": "#2c3e50"}),

                    html.Label("Iteration range (path ranking window)",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div(f"Iter {iter_min}–{iter_max}",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "4px"}),
                    dcc.RangeSlider(
                        id="iter-range-slider",
                        min=iter_min, max=iter_max, step=10,
                        value=[max(iter_min, iter_max - 200), iter_max],
                        marks={i: str(i) for i in
                               range(iter_min, iter_max + 1, 50)},
                        tooltip={"placement": "bottom",
                                 "always_visible": False},
                    ),
                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Snapshot iteration (force data)",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    dcc.Slider(
                        id="snapshot-iter-slider",
                        min=iter_min, max=iter_max, step=1,
                        value=min(iter_max, iter_max),
                        marks={i: str(i) for i in
                               range(iter_min, iter_max + 1, 50)},
                        tooltip={"placement": "bottom",
                                 "always_visible": True},
                    ),
                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Path selection",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div([
                        html.Label("Top N worst paths:",
                                   style={"fontSize": "12px"}),
                        dcc.Input(id="top-n-input", type="number",
                                  value=20, min=1, max=100, step=1,
                                  style={"width": "100%",
                                         "boxSizing": "border-box"}),
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        dcc.Checklist(
                            id="include-nonpath-toggle",
                            options=[{"label": " Include non-path cells "
                                              "for comparison",
                                      "value": "yes"}],
                            value=["yes"],
                            style={"fontSize": "12px"},
                        ),
                    ]),
                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Cell type filter",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    dcc.RadioItems(
                        id="cell-type-radio",
                        options=[
                            {"label": "All cells", "value": "all"},
                            {"label": "Standard cells only", "value": "std"},
                            {"label": "Macros only", "value": "macro"},
                        ],
                        value="all",
                        style={"fontSize": "12px"},
                    ),
                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Plot 1 — Force scatter axes",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div([
                        html.Label("X:", style={"fontSize": "11px",
                                                 "marginRight": "4px"}),
                        dcc.Dropdown(
                            id="scatter-x-dropdown",
                            options=[{"label": v["label"], "value": k}
                                     for k, v in FORCE_METRICS.items()],
                            value="mag_tim",
                            clearable=False,
                            style={"fontSize": "12px", "marginBottom": "4px"},
                        ),
                        html.Label("Y:", style={"fontSize": "11px",
                                                 "marginRight": "4px"}),
                        dcc.Dropdown(
                            id="scatter-y-dropdown",
                            options=[{"label": v["label"], "value": k}
                                     for k, v in FORCE_METRICS.items()],
                            value="mag_wl",
                            clearable=False,
                            style={"fontSize": "12px"},
                        ),
                    ]),
                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Plot 2 — Distribution metric",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    dcc.Dropdown(
                        id="dist-metric-dropdown",
                        options=[{"label": v["label"], "value": k}
                                 for k, v in FORCE_METRICS.items()],
                        value="opposition",
                        clearable=False,
                        style={"fontSize": "12px"},
                    ),
                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Plot 3 — Movement metric",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    dcc.Dropdown(
                        id="movement-metric-dropdown",
                        options=[{"label": v["label"], "value": k}
                                 for k, v in DIST_METRICS.items()],
                        value="total_dist",
                        clearable=False,
                        style={"fontSize": "12px"},
                    ),
                    html.Hr(style={"margin": "14px 0"}),

                    # Dual-store architecture:
                    #  path-cache = expensive path lookup (recomputed on
                    #               Update button only)
                    #  stats-cached-data = full dataset with forces
                    #                      (recomputed on slider change or
                    #                       path-cache change)
                    dcc.Store(id="path-cache", storage_type="memory"),
                    dcc.Store(id="stats-cached-data", storage_type="memory"),

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
                ],
            ),

            # ── Main content ─────────────────────────────────
            html.Div(
                style={"flex": "1", "display": "flex",
                       "flexDirection": "column",
                       "padding": "10px", "gap": "8px",
                       "overflowY": "auto", "boxSizing": "border-box"},
                children=[
                    # Summary stats row
                    html.Div(
                        id="summary-stats",
                        style={"padding": "8px 12px",
                               "backgroundColor": "#eaf2f8",
                               "borderRadius": "6px",
                               "fontSize": "13px",
                               "minHeight": "40px",
                               "lineHeight": "1.6"},
                    ),
                    # Three plots, each filling available space
                    html.Div(
                        style={"flex": "1", "min-height": "0",
                               "display": "flex", "flexDirection": "column",
                               "gap": "8px"},
                        children=[
                            html.Div(
                                style={"flex": "1", "min-height": "0"},
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
                                style={"flex": "1", "min-height": "0",
                                       "display": "flex", "gap": "8px"},
                                children=[
                                    html.Div(
                                        style={"flex": "1", "min-height": "0"},
                                        children=[
                                            dcc.Loading(
                                                dcc.Graph(
                                                    id="plot-distribution",
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
                                                    id="plot-movement",
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
            ),
        ],
    )

    # ═════════════════════════════════════════════════════════════
    #  Data-fetch helpers (used inside the callbacks)
    # ═════════════════════════════════════════════════════════════

    def _fetch_path_data(iter_range, top_n, include_nonpath, cell_type):
        """Return a cacheable dict with path cell IDs, counts,
        static info, and movement data (no force data)."""
        iter_lo, iter_hi = (int(v) for v in (iter_range or [0, 0]))

        path_cell_ids = set()
        path_cell_counter = {}

        if has_timing:
            histories = gpl.worst_paths_history(
                top_n=top_n, iter_range=(iter_lo, iter_hi),
            )
            for pdf in histories:
                ppid = int(pdf["PhysicalPathId"].iloc[0])
                cids = _get_cell_sequence_for_physical_path(gpl, ppid)
                for cid in cids:
                    path_cell_ids.add(cid)
                    path_cell_counter[cid] = (
                        path_cell_counter.get(cid, 0) + 1
                    )

        all_cell_ids = set(path_cell_ids)
        if include_nonpath and path_cell_ids:
            placeholders = ",".join("?" for _ in path_cell_ids)
            df_non = gpl.query(
                f"SELECT DISTINCT CellId FROM gpl_cell_dense_gradients "
                f"WHERE CellId NOT IN ({placeholders}) "
                f"ORDER BY RANDOM() LIMIT ?",
                list(path_cell_ids)
                + [max(500, len(path_cell_ids) * 2)],
            )
            for cid in df_non["CellId"].tolist():
                all_cell_ids.add(int(cid))

        all_cell_ids = sorted(all_cell_ids)
        if not all_cell_ids:
            return None

        # Static info + type filter
        static = gpl.cell_static_info()
        static = static[static["CellId"].isin(all_cell_ids)].copy()
        if cell_type == "std":
            static = static[~static["IsMacro"].astype(bool)]
        elif cell_type == "macro":
            static = static[static["IsMacro"].astype(bool)]
        final_cell_ids = sorted(static["CellId"].tolist())
        if not final_cell_ids:
            return None

        # Movement data
        mov = gpl.cell_movements(
            iter_range=(iter_lo, iter_hi),
            cell_ids=final_cell_ids,
        )
        mov_records = {}
        if not mov.empty:
            for cid, grp in mov.groupby("CellId"):
                dists = grp["Distance"].values
                mov_records[int(cid)] = {
                    "total_dist": float(dists.sum()),
                    "rms_dist": float(np.sqrt((dists ** 2).mean())),
                    "max_dist": float(dists.max()),
                }
        else:
            for cid in final_cell_ids:
                mov_records[cid] = {"total_dist": 0.0, "rms_dist": 0.0,
                                    "max_dist": 0.0}

        return {
            "cell_ids": final_cell_ids,
            "path_counter": {str(k): v
                             for k, v in path_cell_counter.items()},
            "static_records": static.to_dict("records"),
            "mov_records": mov_records,
            "iter_lo": iter_lo,
            "iter_hi": iter_hi,
            "n_path_cells": len(path_cell_ids),
        }

    def _merge_forces(path_cache, snap_iter):
        """Fetch forces at *snap_iter* and merge with cached path data.

        Returns a dict ready for ``stats-cached-data``, or ``None``
        if the merge produces no rows.
        """
        if not path_cache:
            return None
        cell_ids = path_cache["cell_ids"]
        path_counter = {int(k): v
                        for k, v in path_cache["path_counter"].items()}

        # Forces at snapshot iteration
        dense = gpl.cell_dense_gradients(
            iter_range=(snap_iter, snap_iter), cell_ids=cell_ids,
        )
        if dense.empty:
            return None
        force_df = dense[["CellId", "PosX", "PosY",
                          "WlX", "WlY"]].copy()

        timing = gpl.cell_timing_gradients(
            iter_range=(snap_iter, snap_iter), cell_ids=cell_ids,
        )
        if timing.empty:
            force_df["TimX"] = 0.0
            force_df["TimY"] = 0.0
        else:
            force_df = pd.merge(force_df,
                                timing[["CellId", "TimX", "TimY"]],
                                on="CellId", how="left")
            force_df[["TimX", "TimY"]] = (
                force_df[["TimX", "TimY"]].fillna(0.0))

        density = gpl.cell_density_forces(
            iter_range=(snap_iter, snap_iter), cell_ids=cell_ids,
        )
        if density.empty:
            force_df["DensX"] = 0.0
            force_df["DensY"] = 0.0
        else:
            force_df = pd.merge(
                force_df,
                density[["CellId", "EstDensityForceX",
                         "EstDensityForceY"]],
                on="CellId", how="left",
            )
            force_df.rename(
                columns={"EstDensityForceX": "DensX",
                         "EstDensityForceY": "DensY"},
                inplace=True,
            )
            force_df[["DensX", "DensY"]] = (
                force_df[["DensX", "DensY"]].fillna(0.0))

        # Derived columns
        force_df["EffectiveX"] = (force_df["WlX"] + force_df["TimX"]
                                  + force_df["DensX"])
        force_df["EffectiveY"] = (force_df["WlY"] + force_df["TimY"]
                                  + force_df["DensY"])
        force_df["mag_wl"]   = np.hypot(force_df["WlX"],  force_df["WlY"])
        force_df["mag_tim"]  = np.hypot(force_df["TimX"], force_df["TimY"])
        force_df["mag_dens"] = np.hypot(force_df["DensX"], force_df["DensY"])
        force_df["mag_eff"]  = np.hypot(force_df["EffectiveX"],
                                        force_df["EffectiveY"])
        dot_wl_tim = (force_df["WlX"] * force_df["TimX"]
                      + force_df["WlY"] * force_df["TimY"])
        mag_prod = force_df["mag_wl"] * force_df["mag_tim"]
        force_df["opposition"] = np.where(
            mag_prod > 1e-30, -dot_wl_tim / mag_prod, 0.0,
        )
        force_df["Alignment"] = (
            force_df["EffectiveX"] * force_df["TimX"]
            + force_df["EffectiveY"] * force_df["TimY"]
        )

        # Merge static info
        static_df = pd.DataFrame.from_records(
            path_cache["static_records"]
        )
        out = pd.merge(force_df, static_df, on="CellId", how="left")
        out["CellArea"] = (out["Width"].fillna(1)
                           * out["Height"].fillna(1))
        out["IsMacro"] = out["IsMacro"].fillna(0).astype(int)

        # Merge movement
        mov = path_cache.get("mov_records", {})
        for col in ["total_dist", "rms_dist", "max_dist"]:
            out[col] = out["CellId"].map(
                lambda cid: mov.get(cid, {}).get(col, 0.0)
            )

        # Path count per cell
        out["PathCount"] = out["CellId"].map(
            lambda cid: path_counter.get(cid, 0)
        )
        out["PathBucket"] = out["PathCount"].apply(_bucket_path_count)
        out["PathColor"] = out["PathCount"].apply(_bucket_color)

        return {
            "records": out.to_dict("records"),
            "snap_iter": snap_iter,
            "iter_lo": path_cache["iter_lo"],
            "iter_hi": path_cache["iter_hi"],
            "n_path_cells": path_cache["n_path_cells"],
            "n_total": len(out),
        }

    # ═════════════════════════════════════════════════════════════
    #  Callback  A  —  Data assembly  (Update button or slider)
    # ═════════════════════════════════════════════════════════════
    #
    #  Two triggers share one callback:
    #    • update-btn  → full path lookup + forces
    #    • snapshot-iter-slider  → re-reads path-cache,
    #      re-fetches only force data for the new iteration
    #
    #  This way the snapshot slider responds instantly (single
    #  lightweight DB query) while the Update button handles the
    #  expensive path-discovery work.

    @app.callback(
        Output("path-cache", "data"),
        Output("stats-cached-data", "data"),
        Input("update-btn", "n_clicks"),
        Input("snapshot-iter-slider", "value"),
        State("iter-range-slider", "value"),
        State("top-n-input", "value"),
        State("include-nonpath-toggle", "value"),
        State("cell-type-radio", "value"),
        State("path-cache", "data"),
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
             "Computing...", ""),
        ],
        cancel=[Input("cancel-btn", "n_clicks")],
        prevent_initial_call=False,
    )
    def assemble_data(update_clicks, snap_iter, iter_range, top_n,
                      include_nonpath, cell_type, existing_path_cache):
        ctx = dash.callback_context
        triggered_id = ctx.triggered_id
        snap_iter = int(snap_iter or 0)
        top_n = int(top_n or 20)
        include_nonpath = bool(include_nonpath)

        # Determine whether to do a lightweight (slider) or
        # heavyweight (update button / initial load) fetch.
        is_lightweight = (
            triggered_id == "snapshot-iter-slider"
            and existing_path_cache is not None
        )

        if is_lightweight:
            print(f"[cell-path-stats] Slider → iter {snap_iter} "
                  f"(forces only)")
            stats = _merge_forces(existing_path_cache, snap_iter)
            return (existing_path_cache,
                    stats if stats else {})
        else:
            print(f"[cell-path-stats] Update → top {top_n} paths, "
                  f"iter {snap_iter}")
            path_cache = _fetch_path_data(
                iter_range, top_n, include_nonpath, cell_type,
            )
            if path_cache is None:
                return {}, {}
            stats = _merge_forces(path_cache, snap_iter)
            return path_cache, stats if stats else {}

    # ═════════════════════════════════════════════════════════════
    #  Callback  B  —  Render plots  (triggered by cache update)
    # ═════════════════════════════════════════════════════════════

    @app.callback(
        Output("plot-scatter", "figure"),
        Output("plot-distribution", "figure"),
        Output("plot-movement", "figure"),
        Output("summary-stats", "children"),
        Input("stats-cached-data", "data"),
        State("scatter-x-dropdown", "value"),
        State("scatter-y-dropdown", "value"),
        State("dist-metric-dropdown", "value"),
        State("movement-metric-dropdown", "value"),
        prevent_initial_call=True,
    )
    def render_plots(cached, x_key, y_key, dist_key, mov_key):
        if not cached or not cached.get("records"):
            return (_empty("No data"), _empty("No data"),
                    _empty("No data"), "No data")

        df = pd.DataFrame.from_records(cached["records"])
        snap_iter = cached["snap_iter"]
        iter_lo = cached["iter_lo"]
        iter_hi = cached["iter_hi"]
        n_total = cached["n_total"]
        n_path_cells = int((df["PathCount"] > 0).sum())

        # ── Summary stats ────────────────────────────────────
        summary_children = [
            html.B(f"{n_total} cells"),
            f" at iter {snap_iter} ",
            f"(paths ranked over {iter_lo}–{iter_hi})  •  ",
            html.B(f"{n_path_cells} on ≥1 path"),
            f" ({100 * n_path_cells / max(n_total, 1):.0f}%)",
        ]
        # Per-bucket counts
        bucket_info = []
        for lo, hi, color, label in PATH_BUCKETS:
            n = int(((df["PathCount"] >= lo) & (df["PathCount"] <= hi)).sum())
            if n:
                bucket_info.append(
                    html.Span(f"{label}: {n}", style={"color": color,
                                                       "marginLeft": "8px"})
                )
        summary = html.Div(summary_children + bucket_info)

        # ── Plot 1: Force scatter ────────────────────────────
        x_meta = FORCE_METRICS.get(x_key, FORCE_METRICS["mag_tim"])
        y_meta = FORCE_METRICS.get(y_key, FORCE_METRICS["mag_wl"])

        fig1 = go.Figure()
        fig1.update_layout(
            title=(f"Force scatter — {x_meta['label']} vs "
                   f"{y_meta['label']} at iter {snap_iter}"),
            xaxis_title=x_meta["label"],
            yaxis_title=y_meta["label"],
            template="plotly_white",
            hovermode="closest",
            legend=dict(orientation="v", yanchor="top", y=1,
                        xanchor="left", x=1.02, font=dict(size=9)),
            margin=dict(l=50, r=160, t=40, b=50),
        )

        mask_zero = ((df[x_meta["col"]] < 1e-20)
                     & (df[y_meta["col"]] < 1e-20))
        if mask_zero.any():
            sub = df[mask_zero]
            fig1.add_trace(go.Scatter(
                x=sub[x_meta["col"]], y=sub[y_meta["col"]],
                mode="markers",
                marker=dict(size=3, color="#e0e0e0", opacity=0.3),
                name="No force",
                hoverinfo="skip",
            ))

        for lo, hi, color, label in PATH_BUCKETS:
            sub = df[(df["PathCount"] >= lo)
                     & (df["PathCount"] <= hi) & ~mask_zero]
            if sub.empty:
                continue
            sizes = np.sqrt(sub["CellArea"].values)
            sizes = np.clip(sizes / max(np.median(sizes), 1e-9) * 6,
                            3, 25)

            fig1.add_trace(go.Scatter(
                x=sub[x_meta["col"]], y=sub[y_meta["col"]],
                mode="markers",
                marker=dict(size=sizes, color=color, opacity=0.7,
                            line=dict(width=0.5, color="white")),
                name=label,
                customdata=np.column_stack([
                    sub["CellId"], sub["PathCount"],
                    sub["mag_wl"], sub["mag_tim"],
                    sub["mag_dens"], sub["mag_eff"],
                    sub["opposition"], sub["Alignment"],
                    np.where(sub["IsMacro"] > 0, "Macro", "StdCell"),
                ]),
                hovertemplate=(
                    "<b>Cell %{customdata[0]}</b> (%{customdata[8]})<br>"
                    "Paths: %{customdata[1]}<br>"
                    f"{x_meta['label']}: " "%{x:.4f}<br>"
                    f"{y_meta['label']}: " "%{y:.4f}<br>"
                    "|WL|=%{customdata[2]:.4f}  "
                    "|Tim|=%{customdata[3]:.4f}<br>"
                    "|Dens|=%{customdata[4]:.4f}  "
                    "|Eff|=%{customdata[5]:.4f}<br>"
                    "Opp.=%{customdata[6]:.3f}  "
                    "Align.=%{customdata[7]:.3f}<extra></extra>"
                ),
            ))

        # ── Plot 2: Distribution violin ─────────────────────
        dist_meta = FORCE_METRICS.get(dist_key,
                                       FORCE_METRICS["opposition"])
        fig2 = go.Figure()
        fig2.update_layout(
            title=(f"Distribution of {dist_meta['label']} "
                   "by path count"),
            yaxis_title=dist_meta["label"],
            template="plotly_white",
            violinmode="overlay",
            margin=dict(l=50, r=20, t=40, b=50),
        )
        for lo, hi, color, label in PATH_BUCKETS:
            sub = df[(df["PathCount"] >= lo)
                     & (df["PathCount"] <= hi)]
            if sub.empty:
                continue
            vals = sub[dist_meta["col"]].dropna()
            if len(vals) < 2:
                continue
            fig2.add_trace(go.Violin(
                y=vals, name=label,
                line_color=color, fillcolor=color,
                opacity=0.35, points=False,
                spanmode="soft", meanline_visible=True,
                side="positive", width=2.5,
            ))

        # ── Plot 3: Movement vs path load ────────────────────
        mov_meta = DIST_METRICS.get(mov_key,
                                     DIST_METRICS["total_dist"])
        fig3 = go.Figure()
        fig3.update_layout(
            title=(f"Cell movement vs path involvement "
                   f"(iters {iter_lo}–{iter_hi})"),
            xaxis_title="Number of paths",
            yaxis_title=mov_meta["label"],
            template="plotly_white",
            hovermode="closest",
            margin=dict(l=50, r=20, t=40, b=50),
        )
        jitter = np.random.default_rng(42).uniform(
            -0.15, 0.15, size=len(df),
        )
        sizes = np.sqrt(df["CellArea"].values)
        sizes = np.clip(sizes / max(np.median(sizes), 1e-9) * 8, 3, 22)

        fig3.add_trace(go.Scatter(
            x=df["PathCount"].values + jitter,
            y=df[mov_meta["col"]],
            mode="markers",
            marker=dict(size=sizes, color=df["PathColor"],
                        opacity=0.7,
                        line=dict(width=0.5, color="white")),
            name="Cells",
            customdata=np.column_stack([
                df["CellId"], df["PathCount"],
                df["mag_tim"], df["mag_wl"],
                df["total_dist"],
                np.where(df["IsMacro"] > 0, "Macro", "StdCell"),
            ]),
            hovertemplate=(
                "<b>Cell %{customdata[0]}</b> (%{customdata[5]})<br>"
                "Paths: %{customdata[1]}<br>"
                "|Tim|=%{customdata[2]:.4f}  "
                "|WL|=%{customdata[3]:.4f}<br>"
                "Total dist: %{customdata[4]:.1f} nm<extra></extra>"
            ),
        ))

        return fig1, fig2, fig3, summary

    return app


# ══════════════════════════════════════════════════════════════════
#  Helper: get cell sequence for a PhysicalPathId
# ══════════════════════════════════════════════════════════════════

def _get_cell_sequence_for_physical_path(gpl, ppid):
    """Return the ordered [CellId, ...] sequence for a physical path."""
    df = gpl.query(
        "SELECT DISTINCT pc.CellId "
        "FROM gpl_path_cells pc "
        "JOIN gpl_path_signatures sig "
        "  ON pc.PathId = sig.PathId AND pc.Iter = sig.Iter "
        "WHERE sig.PhysicalPathId = ? "
        "ORDER BY pc.PathSeq",
        [ppid],
    )
    if df.empty:
        return []
    return df["CellId"].astype(int).tolist()


def _empty(msg):
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
        description="Dash GUI for cell-path statistical exploration"
    )
    parser.add_argument("--db", required=True,
                        help="Path to GPL SQLite database")
    parser.add_argument("--port", type=int, default=8054,
                        help="Dash port")
    parser.add_argument("--read-only", action="store_true",
                        help="Skip preprocessing (fails if not preprocessed)")
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    print(f"Loading database: {db_path}")

    if args.read_only:
        print("  Opening read-only (derived tables must already exist)")
        gpl = GplDb(db_path, must_be_preprocessed=True)
    else:
        print("  Preprocessing (if needed) — this may take several minutes ...")
        gpl = GplDb(db_path)

    app = make_app(gpl)
    print(f"Dash server: http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
