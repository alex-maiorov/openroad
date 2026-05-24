import dash.html

from .core import DbConnection
from .gpl_db import GplDb
from .exa_db import ExaDb

__all__ = ["DbConnection", "GplDb", "ExaDb", "make_metadata_panel"]


def make_metadata_panel(db: DbConnection) -> dash.html.Div:
    """Return a scrollable metadata panel for the sidebar of a Dash GUI.

    Reads ``db.get_metadata()`` and renders key-value rows in a compact,
    scrollable box suitable for sitting at the bottom of the controls
    sidebar without taking up excessive space.

    Parameters
    ----------
    db : DbConnection
        An open database connection (e.g. ``GplDb`` instance) whose
        metadata table will be read.

    Returns
    -------
    dash.html.Div
        A Dash HTML Div containing the scrollable metadata display.
    """
    meta = db.get_metadata()
    if not meta:
        return dash.html.Div(
            "No metadata.",
            style={"fontSize": "11px", "color": "#888",
                   "padding": "8px"},
        )

    # Build compact rows: one row per (key, value) pair
    rows = []
    for key in sorted(meta):
        for val in meta[key]:
            # Truncate/round floating-point values for display
            val_str = _fmt_metadata_val(val)
            rows.append(
                dash.html.Tr([
                    dash.html.Td(key, style={
                        "fontSize": "10px", "color": "#495057",
                        "fontFamily": "monospace",
                        "padding": "1px 4px",
                        "borderBottom": "1px solid #e9ecef",
                        "whiteSpace": "nowrap",
                        "maxWidth": "160px",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                    }),
                    dash.html.Td(val_str, style={
                        "fontSize": "10px", "color": "#212529",
                        "fontFamily": "monospace",
                        "padding": "1px 4px",
                        "borderBottom": "1px solid #e9ecef",
                        "wordBreak": "break-all",
                    }),
                ])
            )

    table = dash.html.Table(
        rows,
        style={"width": "100%", "borderCollapse": "collapse"},
    )

    return dash.html.Div(
        style={
            "marginTop": "auto",
            "borderTop": "2px solid #dee2e6",
            "paddingTop": "8px",
        },
        children=[
            dash.html.Div(
                "Database Metadata",
                style={
                    "fontSize": "11px",
                    "fontWeight": "bold",
                    "color": "#6c757d",
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                    "marginBottom": "4px",
                },
            ),
            dash.html.Div(
                table,
                style={
                    "maxHeight": "180px",
                    "overflowY": "auto",
                    "border": "1px solid #dee2e6",
                    "borderRadius": "4px",
                    "backgroundColor": "#fff",
                },
            ),
        ],
    )


def _fmt_metadata_val(val: str) -> str:
    """Format a metadata string value for compact display."""
    try:
        f = float(val)
        abs_f = abs(f)
        if abs_f == 0.0:
            return "0"
        if abs_f < 0.001 or abs_f > 1e6:
            # Use scientific notation
            return f"{f:.4e}"
        if abs_f >= 100:
            return f"{f:.2f}"
        return f"{f:.6g}"
    except (ValueError, TypeError):
        # Not a float — return as-is, truncated if very long
        s = str(val)
        if len(s) > 80:
            return s[:77] + "..."
        return s
