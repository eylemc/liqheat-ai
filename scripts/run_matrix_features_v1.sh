#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/liqheat-ai"
source .venv/bin/activate

mkdir -p \
  data/matrix \
  data/forecast_v3 \
  reports/matrix \
  logs

echo "============================================================"
echo "MATRIX FEATURE ENGINE STARTED"
echo "Started: $(date --iso-8601=seconds)"
echo "============================================================"

python -u scripts/build_matrix_features_v1.py

echo
echo "============================================================"
echo "MATRIX FEATURE ENGINE FINISHED"
echo "Finished: $(date --iso-8601=seconds)"
echo "============================================================"
