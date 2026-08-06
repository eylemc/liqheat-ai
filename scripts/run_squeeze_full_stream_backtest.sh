#!/usr/bin/env bash

set -u
set -o pipefail

cd "$(dirname "$0")/.."

mkdir -p logs
mkdir -p \
  data/backtests/topology_v2_squeeze_fixed_50_25_cost_grid_cost_grid

LOG="logs/squeeze_full_stream_backtest.log"
STATUS="logs/squeeze_full_stream_backtest_status.txt"

{
    echo "============================================================"
    echo "LIQHEAT TRUE FULL-STREAM BACKTEST START"
    echo "Started: $(date -Is)"
    echo "Host   : $(hostname)"
    echo "PWD    : $(pwd)"
    echo "============================================================"
} | tee "$LOG"

echo "RUNNING" > "$STATUS"

.venv/bin/python \
    scripts/backtest_topology_v2_squeeze_full_stream.py \
    2>&1 | tee -a "$LOG"

EXIT_CODE=${PIPESTATUS[0]}

{
    echo
    echo "============================================================"
    echo "LIQHEAT TRUE FULL-STREAM BACKTEST FINISHED"
    echo "Finished : $(date -Is)"
    echo "Exit code: $EXIT_CODE"
    echo "============================================================"
} | tee -a "$LOG"

if [[ "$EXIT_CODE" -eq 0 ]]; then
    echo "COMPLETE" > "$STATUS"
else
    echo "FAILED" > "$STATUS"
fi

exit "$EXIT_CODE"
