#!/usr/bin/env python3
"""Standalone CLI to preprocess a GPL SQLite database.

Thin wrapper over ``GplDb`` — delegates all preprocessing logic to
the library, which is the ultimate source of truth.

Usage
-----
    python -m tools.database_log_analysis.preprocess_db \\
        --db path/to/placement-visualization.sqlite

    # Custom batch size (smaller = less RAM):
    python -m tools.database_log_analysis.preprocess_db \\
        --db path/to/placement-visualization.sqlite --batch-size 10

    # Verify preprocessing is complete (read-only check):
    python -m tools.database_log_analysis.preprocess_db \\
        --db path/to/placement-visualization.sqlite --read-only

After preprocessing, launch any analysis tool in read-only mode:

    python -m tools.database_log_analysis.path_visualizer \\
        --db path/to/db.sqlite --read-only --port 8050
"""

import argparse
import os
import sys
import time

_TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from database_log_analysis import GplDb


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a GPL SQLite database (thin wrapper over GplDb)"
    )
    parser.add_argument(
        "--db", required=True,
        help="Path to GPL SQLite database",
    )
    parser.add_argument(
        "--batch-size", type=int, default=25,
        help="Iterations per batch (default 25)",
    )
    parser.add_argument(
        "--read-only", action="store_true",
        help="Open read-only.  Fails if preprocessing is incomplete.",
    )
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    print(f"Database: {db_path}")

    if args.read_only:
        GplDb(db_path, must_be_preprocessed=True)
        print("All derived tables present — ready for read-only analysis.")
        return

    t0 = time.time()
    GplDb(db_path, batch_size=args.batch_size)
    elapsed = time.time() - t0

    print(f"Preprocessing complete ({elapsed:.1f}s).")
    print(f"Launch analysis tools with --read-only, e.g.:")
    print(f"  python -m tools.database_log_analysis.path_visualizer \\")
    print(f"      --db {db_path} --read-only --port 8050")


if __name__ == "__main__":
    main()
