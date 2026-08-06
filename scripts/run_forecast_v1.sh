#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$HOME/liqheat-ai"
LOG_FILE="$PROJECT_DIR/logs/forecast_v1.log"

cd "$PROJECT_DIR"
source .venv/bin/activate

mkdir -p \
  logs \
  data/forecast_v1 \
  models/forecast_v1 \
  reports/forecast_v1

{
  echo "============================================================"
  echo "LIQHEAT FORECAST V1 STARTED"
  echo "Started: $(date --iso-8601=seconds)"
  echo "============================================================"

  python scripts/build_forecast_v1_dataset.py

  python scripts/train_forecast_v1.py

  echo
  echo "============================================================"
  echo "LIQHEAT FORECAST V1 FINISHED"
  echo "Finished: $(date --iso-8601=seconds)"
  echo "============================================================"
} 2>&1 | tee "$LOG_FILE"
