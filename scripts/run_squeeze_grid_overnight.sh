#!/usr/bin/env bash

set -u
set -o pipefail

cd "$(dirname "$0")/.."

mkdir -p logs
mkdir -p data/research/topology_v2_squeeze_grid

MASTER_LOG="logs/squeeze_grid_master.log"
STATUS_FILE="logs/squeeze_grid_status.txt"

{
    echo "============================================================"
    echo "LIQHEAT SQUEEZE GRID START"
    echo "Started: $(date -Is)"
    echo "Host   : $(hostname)"
    echo "PWD    : $(pwd)"
    echo "============================================================"
} | tee -a "$MASTER_LOG"

echo "RUNNING" > "$STATUS_FILE"

.venv/bin/python \
    scripts/build_and_run_squeeze_grid.py \
    2>&1 | tee -a "$MASTER_LOG"

GRID_EXIT=${PIPESTATUS[0]}

{
    echo
    echo "Grid runner exit code: $GRID_EXIT"
    echo "Building aggregate summary..."
    echo
} | tee -a "$MASTER_LOG"

.venv/bin/python \
    scripts/summarize_squeeze_grid.py \
    2>&1 | tee -a "$MASTER_LOG"

SUMMARY_EXIT=${PIPESTATUS[0]}

{
    echo
    echo "============================================================"
    echo "LIQHEAT SQUEEZE GRID FINISHED"
    echo "Finished: $(date -Is)"
    echo "Grid exit code   : $GRID_EXIT"
    echo "Summary exit code: $SUMMARY_EXIT"
    echo "============================================================"
} | tee -a "$MASTER_LOG"

if [[ "$SUMMARY_EXIT" -eq 0 ]]; then
    echo "COMPLETE" > "$STATUS_FILE"
else
    echo "SUMMARY_FAILED" > "$STATUS_FILE"
fi

exit "$SUMMARY_EXIT"
