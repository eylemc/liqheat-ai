#!/usr/bin/env bash

cd "$(dirname "$0")/.."

PID_FILE="logs/squeeze_grid.pid"
STATUS_FILE="logs/squeeze_grid_status.txt"
MASTER_LOG="logs/squeeze_grid_master.log"

echo "============================================================"
echo "SQUEEZE GRID STATUS"
echo "============================================================"

if [[ -f "$STATUS_FILE" ]]; then
    echo "Status: $(cat "$STATUS_FILE")"
else
    echo "Status: UNKNOWN"
fi

if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE")"
    echo "PID   : $PID"

    if kill -0 "$PID" 2>/dev/null; then
        echo "Process: RUNNING"
        ps -fp "$PID"
    else
        echo "Process: NOT RUNNING"
    fi
else
    echo "PID file not found."
fi

echo
echo "Completed reports:"
find \
    data/research/topology_v2_squeeze_grid \
    -mindepth 2 \
    -maxdepth 2 \
    -name report.json \
    2>/dev/null \
    | wc -l

echo
echo "Latest master log:"
tail -n 30 "$MASTER_LOG" 2>/dev/null || true
