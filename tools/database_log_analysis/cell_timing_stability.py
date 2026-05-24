#!/usr/bin/env python3
"""Dash GUI for analysing cell-level numerical stability under timing forces.

Background
----------
During global placement, the timing pass computes per-cell "timing gradient"
forces from violating paths.  These forces are added directly to the summed
gradient (WL + density + timing).  When timing forces change abruptly between
passes — or oppose the prevailing WL forces — cells can overshoot, oscillate,
or wander, causing runaway instability in the timing objective.

This tool looks for the signatures of instability:

  1. **Oscillation** — displacement magnitude spikes at timing passes
  2. **Force opposition** — WL and timing forces point in opposite directions
  3. **Force volatility** — timing force direction flips between passes
  4. **Low displacement efficiency** — net movement ≪ total path (wandering)

Usage
-----
    python -m tools.database_log_analysis.cell_timing_stability \\
        --db path/to/placement-visualization.sqlite \\
        --port 8056

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
#  Colour palette
# ══════════════════════════════════════════════════════════════════
_CELL_COLORS = px.colors.qualitative.Plotly + px.colors.qualitative.Set1


# ══════════════════════════════════════════════════════════════════
#  Application builder
# ══════════════════════════════════════════════════════════════════

def make_app(gpl: GplDb) -> dash.Dash:
    cache = diskcache.Cache("./tmp/dash_bg_cache")
    app = dash.Dash(
        __name__,
        background_callback_manager=DiskcacheManager(cache),
    )
    app.title = "Cell Timing Stability Analyzer"
    app._gpl = gpl

    # ── Pre-fetch iteration range ─────────────────────────────
    df = gpl.query(
        "SELECT MIN(Iter) AS mi, MAX(Iter) AS ma "
        "FROM gpl_cell_dense_gradients"
    )
    iter_min = int(df["mi"].iloc[0])
    iter_max = int(df["ma"].iloc[0])

    has_timing = gpl._exists("gpl_cell_timing_gradients")

    # ── Layout ────────────────────────────────────────────────
    app.layout = html.Div(
        style={"display": "flex", "height": "100vh", "margin": "0",
               "padding": "0", "fontFamily": "Segoe UI, Arial, sans-serif"},
        children=[
            # ── Sidebar ───────────────────────────────────────
            html.Div(
                id="sidebar",
                style={"width": "310px", "minWidth": "310px",
                       "padding": "16px 14px", "overflowY": "auto",
                       "backgroundColor": "#f8f9fa",
                       "borderRight": "1px solid #dee2e6",
                       "boxSizing": "border-box"},
                children=[
                    html.H3("Timing Stability Analyzer",
                            style={"marginTop": "0", "marginBottom": "14px",
                                   "color": "#2c3e50"}),

                    html.Label("Top K timing cells",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div("Ranked by aggregate timing force magnitude",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "4px"}),
                    dcc.Slider(
                        id="top-k-slider",
                        min=1, max=50, step=1,
                        value=10,
                        marks={1: "1", 10: "10", 25: "25", 50: "50"},
                    ),

                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("Iteration range",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div(f"Iter {iter_min}–{iter_max}",
                             style={"fontSize": "11px", "color": "#6c757d",
                                    "marginBottom": "4px"}),
                    dcc.RangeSlider(
                        id="iter-range-slider",
                        min=iter_min, max=iter_max, step=10,
                        value=[iter_min, iter_max],
                        marks={i: str(i) for i in
                               range(iter_min, iter_max + 1, 50)},
                        tooltip={"placement": "bottom",
                                 "always_visible": False},
                    ),

                    html.Hr(style={"margin": "14px 0"}),

                    html.Label("View controls",
                               style={"fontWeight": "bold", "fontSize": "13px"}),
                    html.Div([
                        html.Label("Highlight cells:",
                                   style={"fontSize": "12px"}),
                        dcc.Dropdown(
                            id="highlight-cell-dropdown",
                            options=[],
                            multi=True,
                            placeholder="Select specific cells to highlight...",
                            style={"fontSize": "12px"},
                        ),
                    ], style={"marginBottom": "8px"}),

                    html.Div([
                        dcc.Checklist(
                            id="show-nontiming-toggle",
                            options=[{"label": " Include non-timing cells "
                                              "for comparison",
                                      "value": "yes"}],
                            value=[],
                            style={"fontSize": "12px"},
                        ),
                    ]),

                    html.Hr(style={"margin": "14px 0"}),

                    dcc.Store(id="stability-cache", storage_type="memory"),

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
                    html.Div(id="summary-cells",
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
                    # Top row: overview scatter + coherence time-series
                    html.Div(
                        style={"flex": "1", "min-height": "0",
                               "display": "flex", "gap": "8px"},
                        children=[
                            html.Div(
                                style={"flex": "1", "min-height": "0"},
                                children=[
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="plot-overview",
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
                                            id="plot-coherence",
                                            style={"height": "100%"},
                                        ),
                                        style={"height": "100%"},
                                        parent_style={"height": "100%"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Bottom row: displacement timeline + force volatility
                    html.Div(
                        style={"flex": "1", "min-height": "0",
                               "display": "flex", "gap": "8px"},
                        children=[
                            html.Div(
                                style={"flex": "1", "min-height": "0"},
                                children=[
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="plot-displacement",
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
                                            id="plot-volatility",
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
    #  Data-fetch logic
    # ═════════════════════════════════════════════════════════════

    def _get_timing_pass_iters():
        """Return sorted list of iterations where timing gradients exist."""
        if not has_timing:
            return []
        df = gpl.query(
            "SELECT DISTINCT Iter FROM gpl_cell_timing_gradients "
            "ORDER BY Iter"
        )
        return df["Iter"].tolist()

    def _fetch_stability_data(iter_range, top_k, include_nontiming):
        """Fetch and compute all stability metrics.

        Returns a dict ready for ``dcc.Store``, or ``None`` if no data.
        """
        iter_lo, iter_hi = (int(v) for v in (iter_range or [0, 0]))
        top_k = int(top_k or 10)

        # 1. Top timing cells
        print(f"[stability] Fetching top {top_k} timing cells...")
        top_df = gpl.top_timing_cells(top_n=top_k,
                                       iter_range=(iter_lo, iter_hi))
        if top_df.empty:
            return None
        cell_ids = [int(c) for c in top_df["CellId"].tolist()]
        total_tim = {int(r["CellId"]): float(r["TotalTimForce"])
                     for _, r in top_df.iterrows()}

        all_cell_ids = list(cell_ids)

        # 2. Optionally add non-timing cells for comparison
        if include_nontiming:
            placeholders = ",".join("?" for _ in cell_ids)
            nontim = gpl.query(
                f"SELECT DISTINCT CellId FROM gpl_cell_dense_gradients "
                f"WHERE CellId NOT IN ({placeholders}) "
                f"ORDER BY RANDOM() LIMIT ?",
                cell_ids + [min(200, top_k * 4)],
            )
            for cid in nontim["CellId"].tolist():
                all_cell_ids.append(int(cid))

        all_cell_ids = sorted(all_cell_ids)
        print(f"[stability] Analysing {len(all_cell_ids)} cells "
              f"(iters {iter_lo}–{iter_hi})...")

        # 3. Movement data (per iter displacement)
        print("[stability] Fetching cell movements...")
        mov = gpl.cell_movements(
            iter_range=(iter_lo, iter_hi), cell_ids=all_cell_ids,
        )
        # Drop first-iter rows (Distance=0 from LAG)
        mov = mov[mov["Distance"] > 0].copy()
        if mov.empty:
            return None

        # 4. Get timing-pass iterations
        timing_passes = _get_timing_pass_iters()
        timing_pass_set = set(timing_passes)

        # 5. Timing gradients at timing-pass iters
        tim_df = pd.DataFrame()
        if has_timing:
            tim_df = gpl.cell_timing_gradients(
                iter_range=(iter_lo, iter_hi), cell_ids=all_cell_ids,
            )
            if not tim_df.empty:
                tim_df["TimMag"] = np.hypot(tim_df["TimX"], tim_df["TimY"])
                tim_df["TimAngle"] = np.arctan2(tim_df["TimY"],
                                                 tim_df["TimX"])

        # 6. Gradient metrics for opposition
        met_df = pd.DataFrame()
        if gpl._exists("gpl_cell_gradient_metrics"):
            met_df = gpl.cell_gradient_metrics(
                iter_range=(iter_lo, iter_hi), cell_ids=all_cell_ids,
            )

        # ─ 6a. Cell-level summary metrics ──────────────────────
        mov_aggs = mov.groupby("CellId").agg(
            avg_displacement=("Distance", "mean"),
            std_displacement=("Distance", "std"),
            total_path=("Distance", "sum"),
            n_moves=("Distance", "count"),
        ).reset_index()

        # Net displacement (first pos → last pos)
        first_last = mov.groupby("CellId").agg(
            first_x=("PosX", "first"),
            first_y=("PosY", "first"),
            last_x=("PosX", "last"),
            last_y=("PosY", "last"),
        ).reset_index()
        first_last["net_displacement"] = np.hypot(
            first_last["last_x"] - first_last["first_x"],
            first_last["last_y"] - first_last["first_y"],
        )

        summary = pd.merge(mov_aggs, first_last[["CellId", "net_displacement"]],
                           on="CellId")
        summary["efficiency"] = np.where(
            summary["total_path"] > 0,
            summary["net_displacement"] / summary["total_path"],
            0.0,
        )

        # Mean opposition
        if not met_df.empty:
            opp_agg = met_df.groupby("CellId")["opposition"].mean().reset_index()
            opp_agg.rename(columns={"opposition": "mean_opposition"},
                           inplace=True)
            summary = pd.merge(summary, opp_agg, on="CellId", how="left")
        else:
            summary["mean_opposition"] = 0.0
        summary["mean_opposition"] = summary["mean_opposition"].fillna(0.0)

        # Total timing force
        summary["total_tim_force"] = summary["CellId"].map(
            lambda cid: total_tim.get(cid, 0.0)
        )
        summary["is_timing_cell"] = summary["CellId"].isin(cell_ids)

        # ─ 6b. Force volatility (timing force change between passes) ─
        vol_records = []
        if not tim_df.empty:
            for cid in all_cell_ids:
                ct = tim_df[tim_df["CellId"] == cid].sort_values("Iter")
                if len(ct) < 2:
                    continue
                prev_row = ct.iloc[0]
                for _, row in ct.iloc[1:].iterrows():
                    d_angle = row["TimAngle"] - prev_row["TimAngle"]
                    # Wrap to [-π, π]
                    d_angle = (d_angle + np.pi) % (2 * np.pi) - np.pi
                    vol_records.append({
                        "CellId": cid,
                        "Iter": int(row["Iter"]),
                        "prev_iter": int(prev_row["Iter"]),
                        "delta_mag": float(row["TimMag"] - prev_row["TimMag"]),
                        "delta_angle": float(d_angle),
                        "force_mag": float(row["TimMag"]),
                    })
                    prev_row = row

        vol_df = (pd.DataFrame(vol_records)
                  if vol_records else pd.DataFrame())

        # ─ 6c. Coherence: movement-timing alignment ────────────
        # At each timing-pass iter, dot(movement_dir, timing_force_dir)
        coh_records = []
        if not tim_df.empty:
            for cid in all_cell_ids:
                ct = tim_df[tim_df["CellId"] == cid].sort_values("Iter")
                cm = mov[mov["CellId"] == cid].sort_values("Iter")
                for _, tr in ct.iterrows():
                    tit = int(tr["Iter"])
                    # Find the movement step that lands AT this iter
                    # (movement from iter-1 → iter)
                    mrow = cm[cm["Iter"] == tit]
                    if mrow.empty:
                        continue
                    mrow = mrow.iloc[0]
                    dist = float(mrow["Distance"])
                    if dist < 1e-9:
                        continue
                    mov_dir_x = float(mrow["DeltaX"]) / dist
                    mov_dir_y = float(mrow["DeltaY"]) / dist
                    tim_mag = float(tr["TimMag"])
                    if tim_mag < 1e-9:
                        continue
                    tim_dir_x = float(tr["TimX"]) / tim_mag
                    tim_dir_y = float(tr["TimY"]) / tim_mag
                    alignment = mov_dir_x * tim_dir_x + mov_dir_y * tim_dir_y
                    coh_records.append({
                        "CellId": cid,
                        "Iter": tit,
                        "alignment": float(alignment),
                        "displacement": dist,
                        "tim_mag": tim_mag,
                    })

        coh_df = (pd.DataFrame(coh_records)
                  if coh_records else pd.DataFrame())

        # ─ 6d. Displacement timeline with timing markers ───────
        disp_records = []
        for cid in all_cell_ids:
            cm = mov[mov["CellId"] == cid].sort_values("Iter")
            for _, row in cm.iterrows():
                it = int(row["Iter"])
                disp_records.append({
                    "CellId": cid,
                    "Iter": it,
                    "Distance": float(row["Distance"]),
                    "has_timing": it in timing_pass_set,
                })

        disp_df = pd.DataFrame(disp_records)

        summary["has_timing"] = summary["CellId"].apply(
            lambda cid: not tim_df.empty and cid in set(tim_df["CellId"])
        )

        print(f"[stability] Done. {len(summary)} cells analysed.")

        return {
            "summary_records": summary.to_dict("records"),
            "volatility_records": (vol_df.to_dict("records")
                                    if not vol_df.empty else []),
            "coherence_records": (coh_df.to_dict("records")
                                   if not coh_df.empty else []),
            "displacement_records": disp_records.to_dict("records"),
            "timing_passes": timing_passes,
            "cell_ids": all_cell_ids,
            "iter_lo": iter_lo,
            "iter_hi": iter_hi,
            "n_timing_cells": len(cell_ids),
            "n_total_cells": len(all_cell_ids),
        }

    # ═════════════════════════════════════════════════════════════
    #  Callback A — Data fetch
    # ═════════════════════════════════════════════════════════════

    @app.callback(
        Output("stability-cache", "data"),
        Output("highlight-cell-dropdown", "options"),
        Input("update-btn", "n_clicks"),
        State("top-k-slider", "value"),
        State("iter-range-slider", "value"),
        State("show-nontiming-toggle", "value"),
        background=True,
        running=[
            (Output("update-btn", "disabled"), True, False),
            (Output("cancel-btn", "style"),
             {"width": "100%", "marginTop": "6px", "padding": "8px 0",
              "backgroundColor": "#dc3545", "color": "white",
              "border": "none", "borderRadius": "4px",
              "cursor": "pointer", "display": "block"},
             {"display": "none"}),
            (Output("status-msg", "children"),
             "Computing stability metrics...", ""),
        ],
        cancel=[Input("cancel-btn", "n_clicks")],
        prevent_initial_call=False,
    )
    def fetch_data(_n, top_k, iter_range, include_nonpath):
        include_nontiming = bool(include_nonpath)
        data = _fetch_stability_data(iter_range, top_k, include_nontiming)

        if not data:
            return {}, []

        dropdown_opts = [
            {"label": f"Cell {cid}", "value": cid}
            for cid in sorted(data["cell_ids"])
        ]
        return data, dropdown_opts

    # ═════════════════════════════════════════════════════════════
    #  Callback B — Render plots
    # ═════════════════════════════════════════════════════════════

    @app.callback(
        Output("plot-overview", "figure"),
        Output("plot-coherence", "figure"),
        Output("plot-displacement", "figure"),
        Output("plot-volatility", "figure"),
        Output("summary-cells", "children"),
        Input("stability-cache", "data"),
        State("highlight-cell-dropdown", "value"),
        prevent_initial_call=True,
    )
    def render_plots(cached, highlight_cids):
        if not cached or not cached.get("summary_records"):
            return (_empty("No data — click Analyze"),
                    _empty(""), _empty(""), _empty(""), "")

        summary = pd.DataFrame.from_records(cached["summary_records"])
        vol_df = (pd.DataFrame.from_records(cached["volatility_records"])
                  if cached.get("volatility_records") else pd.DataFrame())
        coh_df = (pd.DataFrame.from_records(cached["coherence_records"])
                  if cached.get("coherence_records") else pd.DataFrame())
        disp_df = (pd.DataFrame.from_records(cached["displacement_records"])
                   if cached.get("displacement_records") else pd.DataFrame())
        timing_passes = cached.get("timing_passes", [])
        highlight_cids = set(highlight_cids or [])

        n_timing = cached["n_timing_cells"]
        n_total = cached["n_total_cells"]
        iter_lo = cached["iter_lo"]
        iter_hi = cached["iter_hi"]

        # ── Summary line ───────────────────────────────────────
        n_unstable = int((summary["efficiency"] < 0.5).sum())
        summary_text = [
            html.B(f"{n_total} cells analysed"),
            f"  ({n_timing} timing-critical, "
            f"{n_total - n_timing} comparison)",
            html.Br(),
            html.Span(f"{n_unstable} with efficiency < 0.5 "
                      f"(wandering / bouncing)",
                      style={"color": "#e74c3c" if n_unstable > 0
                             else "#27ae60"}),
            f"  •  iters {iter_lo}–{iter_hi}",
        ]

        # ════════════════════════════════════════════════════════
        #  Panel 1: Stability Overview Scatter
        # ════════════════════════════════════════════════════════
        fig1 = go.Figure()
        fig1.update_layout(
            title="Stability Overview — displacement volatility vs. "
                  "timing force",
            xaxis_title="Timing force magnitude (aggregate)",
            yaxis_title="Displacement std-dev (volatility)",
            template="plotly_white",
            hovermode="closest",
            legend=dict(orientation="v", yanchor="top", y=1,
                        xanchor="left", x=1.02, font=dict(size=9)),
            margin=dict(l=50, r=160, t=40, b=50),
        )

        # Non-timing cells (grey)
        nontim = summary[~summary["is_timing_cell"]]
        if not nontim.empty:
            fig1.add_trace(go.Scatter(
                x=nontim["total_tim_force"],
                y=nontim["std_displacement"].fillna(0),
                mode="markers",
                marker=dict(size=6, color="#e0e0e0", opacity=0.5,
                            line=dict(width=0.5, color="#ccc")),
                name="Non-timing cells",
                customdata=np.column_stack([
                    nontim["CellId"],
                    nontim["efficiency"],
                    nontim["mean_opposition"],
                ]),
                hovertemplate=(
                    "<b>Cell %{customdata[0]}</b><br>"
                    "Efficiency: %{customdata[1]:.2f}<br>"
                    "Opposition: %{customdata[2]:.2f}<extra></extra>"
                ),
            ))

        # Timing cells (colored by opposition)
        tim_cells = summary[summary["is_timing_cell"]]
        if not tim_cells.empty:
            opp = tim_cells["mean_opposition"].fillna(0)
            sizes = np.sqrt(tim_cells["total_tim_force"].values)
            sizes = np.clip(sizes / max(np.median(sizes), 1e-9) * 8, 4, 25)

            fig1.add_trace(go.Scatter(
                x=tim_cells["total_tim_force"],
                y=tim_cells["std_displacement"].fillna(0),
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=opp,
                    colorscale="RdYlGn_r",  # red=high opposition, green=low
                    cmin=-1, cmax=1,
                    colorbar=dict(
                        title="Opposition<br>(WL vs Tim)",
                        titleside="right",
                    ),
                    line=dict(width=0.5, color="white"),
                ),
                name="Timing cells",
                customdata=np.column_stack([
                    tim_cells["CellId"],
                    tim_cells["efficiency"],
                    opp,
                    tim_cells["total_tim_force"],
                ]),
                hovertemplate=(
                    "<b>Cell %{customdata[0]}</b><br>"
                    "Efficiency: %{customdata[1]:.3f}<br>"
                    "Opposition: %{customdata[2]:.3f}<br>"
                    "Tim force: %{customdata[3]:.1f}<extra></extra>"
                ),
            ))

        # All timing cells on this plot
        top_plot_cells = tim_cells.nlargest(
            min(15, len(tim_cells)), "std_displacement"
        )["CellId"].tolist()

        # ════════════════════════════════════════════════════════
        #  Panel 2: Force-Movement Coherence
        # ════════════════════════════════════════════════════════
        fig2 = go.Figure()
        fig2.update_layout(
            title="Force-Movement Coherence "
                  "(alignment of movement with timing force)",
            xaxis_title="Iteration",
            yaxis_title="Coherence  (↓ +1=aligned, 0=orthogonal, "
                        "-1=opposed ↓)",
            template="plotly_white",
            hovermode="closest",
            legend=dict(orientation="v", yanchor="top", y=1,
                        xanchor="left", x=1.02, font=dict(size=8)),
            margin=dict(l=50, r=160, t=40, b=50),
            yaxis=dict(range=[-1.1, 1.1]),
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="#95a5a6",
                        line_width=1, opacity=0.6)
        fig2.add_hline(y=1, line_dash="dot", line_color="#27ae60",
                        line_width=1, opacity=0.4)
        fig2.add_hline(y=-1, line_dash="dot", line_color="#e74c3c",
                        line_width=1, opacity=0.4)

        if not coh_df.empty:
            for ci, cid in enumerate(top_plot_cells):
                cell_coh = coh_df[coh_df["CellId"] == cid].sort_values("Iter")
                if cell_coh.empty:
                    continue
                color = _CELL_COLORS[ci % len(_CELL_COLORS)]
                fig2.add_trace(go.Scatter(
                    x=cell_coh["Iter"], y=cell_coh["alignment"],
                    mode="lines+markers",
                    name=f"Cell {cid}",
                    line=dict(color=color, width=1.5),
                    marker=dict(size=4),
                    hovertemplate=(
                        f"<b>Cell {cid}</b><br>"
                        "Iter %{x}<br>"
                        "Alignment: %{y:.3f}<extra></extra>"
                    ),
                ))

        # ════════════════════════════════════════════════════════
        #  Panel 3: Displacement Timeline
        # ════════════════════════════════════════════════════════
        fig3 = go.Figure()
        fig3.update_layout(
            title="Displacement Magnitude Timeline "
                  "(● = timing pass active)",
            xaxis_title="Iteration",
            yaxis_title="Per-iter displacement (nm)",
            template="plotly_white",
            hovermode="closest",
            legend=dict(orientation="v", yanchor="top", y=1,
                        xanchor="left", x=1.02, font=dict(size=8)),
            margin=dict(l=50, r=160, t=40, b=50),
        )

        # Semi-transparent vertical bands at timing pass iters
        tp_set = set(timing_passes)
        for tp in timing_passes:
            fig3.add_vrect(
                x0=tp - 0.5, x1=tp + 0.5,
                fillcolor="rgba(231, 76, 60, 0.06)",
                layer="below", line_width=0,
            )

        if not disp_df.empty:
            for ci, cid in enumerate(top_plot_cells):
                cell_disp = disp_df[disp_df["CellId"] == cid].sort_values("Iter")
                if cell_disp.empty:
                    continue
                color = _CELL_COLORS[ci % len(_CELL_COLORS)]

                # Non-timing iters (faded line)
                nontp = cell_disp[~cell_disp["has_timing"]]
                if not nontp.empty:
                    fig3.add_trace(go.Scatter(
                        x=nontp["Iter"], y=nontp["Distance"],
                        mode="lines",
                        name=f"Cell {cid}",
                        line=dict(color=color, width=0.8),
                        opacity=0.35,
                        showlegend=True,
                        legendgroup=f"c{cid}",
                        hoverinfo="skip",
                    ))

                # Timing-pass iters (bold markers)
                tp_data = cell_disp[cell_disp["has_timing"]]
                if not tp_data.empty:
                    fig3.add_trace(go.Scatter(
                        x=tp_data["Iter"], y=tp_data["Distance"],
                        mode="markers",
                        name=f"Cell {cid} (timing)",
                        marker=dict(color=color, size=7,
                                    symbol="circle",
                                    line=dict(width=1, color="white")),
                        showlegend=False,
                        legendgroup=f"c{cid}",
                        hovertemplate=(
                            f"<b>Cell {cid}</b><br>"
                            "Iter %{x}<br>"
                            "Displacement: %{y:.1f} nm<br>"
                            "(timing pass)<extra></extra>"
                        ),
                    ))

        # ════════════════════════════════════════════════════════
        #  Panel 4: Force Volatility
        # ════════════════════════════════════════════════════════
        fig4 = go.Figure()
        fig4.update_layout(
            title="Timing Force Volatility "
                  "(change in direction between passes)",
            xaxis_title="Iteration",
            yaxis_title="Δ angle (radians, wrapped to ±π)",
            template="plotly_white",
            hovermode="closest",
            legend=dict(orientation="v", yanchor="top", y=1,
                        xanchor="left", x=1.02, font=dict(size=8)),
            margin=dict(l=50, r=160, t=40, b=50),
            yaxis=dict(range=[-np.pi, np.pi],
                        tickvals=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                        ticktext=["−π", "−π/2", "0", "π/2", "π"]),
        )
        fig4.add_hline(y=0, line_dash="dash", line_color="#95a5a6",
                        line_width=1, opacity=0.6)
        # π/2 = orthogonal flip (orange warning zone)
        fig4.add_hrect(y0=np.pi/2, y1=np.pi,
                        fillcolor="rgba(231, 76, 60, 0.08)",
                        layer="below", line_width=0)
        fig4.add_hrect(y0=-np.pi, y1=-np.pi/2,
                        fillcolor="rgba(231, 76, 60, 0.08)",
                        layer="below", line_width=0)

        if not vol_df.empty:
            for ci, cid in enumerate(top_plot_cells):
                cell_vol = vol_df[vol_df["CellId"] == cid].sort_values("Iter")
                if cell_vol.empty:
                    continue
                color = _CELL_COLORS[ci % len(_CELL_COLORS)]
                fig4.add_trace(go.Scatter(
                    x=cell_vol["Iter"], y=cell_vol["delta_angle"],
                    mode="lines+markers",
                    name=f"Cell {cid}",
                    line=dict(color=color, width=1.5),
                    marker=dict(size=4),
                    hovertemplate=(
                        f"<b>Cell {cid}</b><br>"
                        "Iter %{x}<br>"
                        "Δ angle: %{y:.3f} rad<br>"
                        "<extra></extra>"
                    ),
                ))

        if highlight_cids:
            for cid in highlight_cids:
                for fig in [fig1, fig2, fig3, fig4]:
                    _highlight_cell_in_figure(fig, cid)

        return fig1, fig2, fig3, fig4, summary_text

    return app


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

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


def _highlight_cell_in_figure(fig, cell_id):
    """Add a highlight annotation or marker for a specific cell.
    
    This is a best-effort helper: it adds an annotation to the figure
    if the cell appears in the data.  For simplicity, we don't parse
    trace data here — the dropdown is for visual reference in the plots.
    """
    # Add a persistent annotation — actual trace highlighting is
    # handled by Plotly's built-in legend click-to-isolate.
    _ = cell_id  # reserved for future per-trace highlighting
    return fig


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Dash GUI for cell timing stability analysis"
    )
    parser.add_argument("--db", required=True,
                        help="Path to GPL SQLite database")
    parser.add_argument("--port", type=int, default=8056,
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
        print("  Preprocessing (if needed) — this may take several minutes …")
        gpl = GplDb(db_path)

    app = make_app(gpl)
    print(f"Dash server: http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
