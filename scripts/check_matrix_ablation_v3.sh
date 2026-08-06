#!/usr/bin/env bash
set -u

cd "$HOME/liqheat-ai"

echo "============================================================"
echo "FORECAST V3 ABLATION STATUS"
echo "============================================================"

if tmux has-session \
  -t matrix_ablation \
  2>/dev/null
then
  echo "Session: RUNNING"
else
  echo "Session: NOT RUNNING"
fi

echo

ps aux \
  | grep \
    "train_matrix_ablation_v3.py" \
  | grep -v grep \
  || true

echo

if [[ -f \
  logs/matrix_ablation_v3.log
]]
then
  tail -100 \
    logs/matrix_ablation_v3.log
else
  echo "No log yet."
fi
