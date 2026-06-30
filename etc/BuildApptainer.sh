#!/usr/bin/env bash
# BuildApptainer.sh — Incremental Apptainer build for OpenROAD
#
# Usage:
#   ./etc/BuildApptainer.sh              # full build (skips pre if sif exists)
#   ./etc/BuildApptainer.sh --force      # full build from scratch
#   ./etc/BuildApptainer.sh -pre         # only (re)build openroad_pre.sif
#   ./etc/BuildApptainer.sh -build       # only compile OpenROAD using pre.sif
#   ./etc/BuildApptainer.sh -final       # only package openroad.sif from pre-built binary
#
# Requires: apptainer, bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

FORCE=0
BUILD_PRE=1
BUILD_OR=1
BUILD_FINAL=1

for arg in "$@"; do
  case "$arg" in
    -pre)   BUILD_OR=0; BUILD_FINAL=0 ;;
    -build) BUILD_PRE=0; BUILD_FINAL=0 ;;
    -final) BUILD_PRE=0; BUILD_OR=0 ;;
    --force) FORCE=1 ;;
  esac
done

cd "$ROOT_DIR"

# ── Phase 1: pre-requisite image ─────────────────────────────────────
if [ "$BUILD_PRE" -eq 1 ]; then
  if [ "$FORCE" -eq 0 ] && [ -f openroad_pre.sif ]; then
    echo "=== openroad_pre.sif exists, skipping (use --force to rebuild) ==="
  else
    echo "=== Building openroad_pre.sif (dependencies) ==="
    [ "$FORCE" -eq 1 ] && rm -f openroad_pre.sif
    apptainer build openroad_pre.sif openroad_pre.def
  fi
fi

# ── Phase 2: build OpenROAD using the pre image ───────────────────────
if [ "$BUILD_OR" -eq 1 ]; then
  echo "=== Building OpenROAD ==="
  apptainer exec \
    --bind "$ROOT_DIR:/OpenROAD" \
    openroad_pre.sif \
    bash -c "
      DEPS_ARGS=\"\"
      if [ -f /opt/openroad_deps_prefixes.txt ]; then
        DEPS_ARGS=\$(cat /opt/openroad_deps_prefixes.txt)
      fi
      cd /OpenROAD
      cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DUSE_ADAPTIVECPP=ON \$DEPS_ARGS
      cmake --build build -- -j \$(nproc)
    "
fi

# ── Phase 3: final packaging image ───────────────────────────────────
if [ "$BUILD_FINAL" -eq 1 ]; then
  echo "=== Building openroad.sif (final image) ==="
  rm -f openroad.sif
  apptainer build openroad.sif openroad.def
fi

echo "=== Done ==="
ls -lh openroad_pre.sif openroad.sif 2>/dev/null || true
