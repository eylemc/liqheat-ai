#!/usr/bin/env bash
set -u

cd "$HOME/liqheat-ai"

echo "============================================================"
echo "FORECAST V2 STATUS"
echo "============================================================"

if tmux has-session -t forecast_v2 2>/dev/null; then
  echo "Session: RUNNING"
else
  echo "Session: NOT RUNNING"
fi

echo

ps aux \
  | grep "train_forecast_v2.py" \
  | grep -v grep \
  || true

echo

if [[ -f logs/forecast_v2.log ]]; then
  tail -80 logs/forecast_v2.log
else
  echo "No log yet."
fi
