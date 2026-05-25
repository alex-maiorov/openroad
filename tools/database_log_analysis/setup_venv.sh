#!/usr/bin/env bash
# setup_venv.sh — Create a Python virtual environment and install dependencies
# for the OpenROAD Database Log Analysis suite.
#
# Usage:
#     cd tools/database_log_analysis && ./setup_venv.sh
#
# This script *must* be run from its own directory (tools/database_log_analysis/).
# It creates .venv/ if absent, then pip-installs the pinned requirements.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

if [ ! -f "${REQ_FILE}" ]; then
    echo "ERROR: requirements.txt not found at ${REQ_FILE}" >&2
    echo "This script must be run from tools/database_log_analysis/" >&2
    exit 1
fi

if [ -d "${VENV_DIR}" ]; then
    echo "Virtual environment already exists at ${VENV_DIR}"
else
    echo "Creating virtual environment at ${VENV_DIR} ..."
    python3 -m venv "${VENV_DIR}"
    echo "Virtual environment created."
fi

# Activate the venv (this only affects the current shell context within this
# script; callers should source activate separately if desired).
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Installing/upgrading pip …"
python -m pip install --upgrade pip --quiet

echo "Installing requirements from ${REQ_FILE} …"
pip install -r "${REQ_FILE}"

echo ""
echo "============================================================"
echo " Virtual environment ready."
echo " Activate with:"
echo "   source ${VENV_DIR}/bin/activate"
echo "============================================================"
