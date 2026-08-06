#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/liqheat-ai"
source .venv/bin/activate

mkdir -p \
  reports/matrix_event_study_v3 \
  logs

echo "============================================================"
echo "MATRIX EVENT STUDY V3 STARTED"
echo "Started: $(date --iso-8601=seconds)"
echo "============================================================"

python -u \
  scripts/matrix_event_study_v3.py

echo
echo "============================================================"
echo "MATRIX EVENT STUDY V3 FINISHED"
echo "Finished: $(date --iso-8601=seconds)"
echo "============================================================"
