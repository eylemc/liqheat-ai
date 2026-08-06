#!/usr/bin/env bash

cd "$(dirname "$0")/.."

STATUS_FILE="logs/squeeze_full_stream_backtest_status.txt"
PID_FILE="logs/squeeze_full_stream_backtest.pid"
LOG_FILE="logs/squeeze_full_stream_backtest.log"

echo "============================================================"
echo "TRUE FULL-STREAM BACKTEST STATUS"
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
fi

echo
echo "Latest log:"
tail -n 40 "$LOG_FILE" 2>/dev/null || true
