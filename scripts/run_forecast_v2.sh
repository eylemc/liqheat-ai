#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/liqheat-ai"
source .venv/bin/activate

mkdir -p \
  models/forecast_v2 \
  reports/forecast_v2 \
  logs

echo "============================================================"
echo "LIQHEAT FORECAST V2 STARTED"
echo "Started: $(date --iso-8601=seconds)"
echo "============================================================"

python -u scripts/train_forecast_v2.py

echo
echo "============================================================"
echo "LIQHEAT FORECAST V2 FINISHED"
echo "Finished: $(date --iso-8601=seconds)"
echo "============================================================"
