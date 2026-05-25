#!/usr/bin/env bash
# run_analysis_suite.sh — Preprocess a GPL database and launch all 7 analysis
# Dash GUIs in read-only mode, each on its dedicated port.
#
# Usage:
#     ./run_analysis_suite.sh <path/to/database.sqlite>
#
# What it does:
#   1. Activates the .venv (created by setup_venv.sh).
#   2. Preprocesses the database (idempotent — skips if already done).
#   3. Launches all 7 Dash servers in the background, each with --read-only.
#   4. Prints the URLs and waits.  Ctrl+C kills all servers.
#
# Pre-requisites:
#   - Run setup_venv.sh first (creates .venv and installs dependencies).
#   - The database must be a valid GPL SQLite database.

set -euo pipefail

# ──────────────────────────────────────────────────────────────────
#  Resolve paths
# ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
TOOLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"   # tools/

if [ $# -lt 1 ]; then
    echo "USAGE: $0 <path/to/database.sqlite>" >&2
    echo "" >&2
    echo "Preprocesses the database, then launches all 7 Dash analysis GUIs" >&2
    echo "in read-only mode on ports 8050–8056." >&2
    exit 1
fi

DB_PATH="$(realpath "$1")"

if [ ! -f "${DB_PATH}" ]; then
    echo "ERROR: database file not found: ${DB_PATH}" >&2
    exit 1
fi

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "ERROR: virtual environment not found at ${VENV_DIR}" >&2
    echo "Run setup_venv.sh first." >&2
    exit 1
fi

# ──────────────────────────────────────────────────────────────────
#  Activate the virtual environment
# ──────────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Make sure we can import the package
if ! python -c "from database_log_analysis import GplDb" 2>/dev/null; then
    # Patch sys.path so the tools/ parent is importable
    export PYTHONPATH="${TOOLS_DIR}:${PYTHONPATH:-}"
fi

# ──────────────────────────────────────────────────────────────────
#  Step 1 — Preprocess the database
# ──────────────────────────────────────────────────────────────────
echo "============================================================"
echo " Step 1: Preprocessing database"
echo "   DB:   ${DB_PATH}"
echo "============================================================"
echo ""

python -m tools.database_log_analysis.preprocess_db --db "${DB_PATH}"

echo ""
echo "============================================================"
echo " Step 2: Launching analysis tools (read-only mode)"
echo "============================================================"
echo ""

# ──────────────────────────────────────────────────────────────────
#  Tool definitions:  name | port
# ──────────────────────────────────────────────────────────────────
TOOLS=(
    "path_visualizer         8050"
    "cell_force_analyzer      8051"
    "path_similarity_analyzer 8052"
    "cell_trajectory_viewer   8053"
    "cell_path_stats          8054"
    "path_transience          8055"
    "cell_timing_stability     8056"
)

# Collect PIDs so we can kill them on exit
PIDS=()

cleanup() {
    echo ""
    echo "Shutting down all Dash servers …"
    for pid in "${PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "All servers stopped."
}
trap cleanup EXIT INT TERM

# ──────────────────────────────────────────────────────────────────
#  Launch each tool in the background
# ──────────────────────────────────────────────────────────────────
for entry in "${TOOLS[@]}"; do
    read -r tool_name port <<< "${entry}"

    echo "  Starting ${tool_name} on port ${port} (read-only) …"

    python -m "tools.database_log_analysis.${tool_name}" \
        --db "${DB_PATH}" \
        --read-only \
        --port "${port}" \
        &
    PIDS+=($!)
done

echo ""
echo "============================================================"
echo " All 7 Dash servers are running.  Open in your browser:"
echo ""
echo "   http://localhost:8050  →  Path Visualizer"
echo "   http://localhost:8051  →  Cell Force Analyzer"
echo "   http://localhost:8052  →  Path Similarity Analyzer"
echo "   http://localhost:8053  →  Cell Trajectory Viewer"
echo "   http://localhost:8054  →  Cell Path Statistics"
echo "   http://localhost:8055  →  Path Transience"
echo "   http://localhost:8056  →  Cell Timing Stability"
echo ""
echo " Press Ctrl+C to stop all servers."
echo "============================================================"
echo ""

# Wait for all background processes
wait
