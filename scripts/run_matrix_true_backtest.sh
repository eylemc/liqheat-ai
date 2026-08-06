#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/liqheat-ai"
source .venv/bin/activate

mkdir -p \
  reports/matrix_true_backtest \
  logs

echo "============================================================"
echo "TRUE MATRIX BACKTEST STARTED"
echo "Started: $(date --iso-8601=seconds)"
echo "============================================================"

python -u \
  scripts/matrix_true_backtest.py

echo
echo "============================================================"
echo "TRUE MATRIX BACKTEST FINISHED"
echo "Finished: $(date --iso-8601=seconds)"
echo "============================================================"
