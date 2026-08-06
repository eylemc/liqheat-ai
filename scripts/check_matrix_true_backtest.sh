#!/usr/bin/env bash
set -u

cd "$HOME/liqheat-ai"

echo "============================================================"
echo "TRUE MATRIX BACKTEST STATUS"
echo "============================================================"

if tmux has-session \
  -t matrix_true_test \
  2>/dev/null
then
  echo "Session: RUNNING"
else
  echo "Session: NOT RUNNING"
fi

echo

ps aux \
  | grep "matrix_true_backtest.py" \
  | grep -v grep \
  || true

echo

if [[ -f \
  logs/matrix_true_backtest.log
]]
then
  tail -100 \
    logs/matrix_true_backtest.log
else
  echo "No log yet."
fi
