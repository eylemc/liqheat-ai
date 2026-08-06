#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/liqheat-ai"
source .venv/bin/activate

mkdir -p \
  models/forecast_v3_ablation \
  reports/forecast_v3_ablation \
  logs

echo "============================================================"
echo "FORECAST V3 MATRIX ABLATION STARTED"
echo "Started: $(date --iso-8601=seconds)"
echo "============================================================"

python -u scripts/train_matrix_ablation_v3.py

echo
echo "============================================================"
echo "FORECAST V3 MATRIX ABLATION FINISHED"
echo "Finished: $(date --iso-8601=seconds)"
echo "============================================================"
