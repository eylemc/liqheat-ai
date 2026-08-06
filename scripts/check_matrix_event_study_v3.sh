#!/usr/bin/env bash
set -u

cd "$HOME/liqheat-ai"

echo "============================================================"
echo "MATRIX EVENT STUDY V3 STATUS"
echo "============================================================"

if tmux has-session \
  -t matrix_event_v3 \
  2>/dev/null
then
  echo "Session: RUNNING"
else
  echo "Session: NOT RUNNING"
fi

echo

ps aux \
  | grep \
    "matrix_event_study_v3.py" \
  | grep -v grep \
  || true

echo

if [[ -f \
  logs/matrix_event_study_v3.log
]]
then
  tail -120 \
    logs/matrix_event_study_v3.log
else
  echo "No log yet."
fi
