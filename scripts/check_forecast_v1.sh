#!/usr/bin/env bash
set -u

cd "$HOME/liqheat-ai"

echo "============================================================"
echo "FORECAST V1 STATUS"
echo "============================================================"

if tmux has-session -t forecast_v1 2>/dev/null; then
  echo "Session: RUNNING"
else
  echo "Session: NOT RUNNING"
fi

echo

if [[ -f logs/forecast_v1.log ]]; then
  tail -60 logs/forecast_v1.log
else
  echo "No log yet."
fi
