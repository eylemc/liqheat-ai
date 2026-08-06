#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/liqheat-ai"
source .venv/bin/activate

mkdir -p \
  reports/matrix_alignment \
  logs

echo "============================================================"
echo "MATRIX ALIGNMENT STUDY STARTED"
echo "Started: $(date --iso-8601=seconds)"
echo "============================================================"

python -u \
  scripts/analyze_matrix_alignment.py

echo
echo "============================================================"
echo "MATRIX ALIGNMENT STUDY FINISHED"
echo "Finished: $(date --iso-8601=seconds)"
echo "============================================================"
