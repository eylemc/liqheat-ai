#!/usr/bin/env bash

cd "$(dirname "$0")/.."

echo "============================================================"
echo "SQUEEZE BACKTEST STATUS"
echo "============================================================"

if [[ -f logs/squeeze_backtest_status.txt ]]; then
    echo "Status: $(cat logs/squeeze_backtest_status.txt)"
else
    echo "Status: UNKNOWN"
fi

echo

if [[ -f logs/squeeze_backtest.pid ]]; then
    PID="$(cat logs/squeeze_backtest.pid)"
    echo "PID: $PID"

    if kill -0 "$PID" 2>/dev/null; then
        echo "Process: RUNNING"
        ps -fp "$PID"
    else
        echo "Process: NOT RUNNING"
    fi
fi

echo
echo "Latest log:"
tail -n 35 logs/squeeze_backtest.log 2>/dev/null || true
