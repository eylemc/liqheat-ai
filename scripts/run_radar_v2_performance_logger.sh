#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/liqheat-ai"
source .venv/bin/activate

mkdir -p logs data/monitoring reports/radar_v2_performance

SESSION="radar_v2_logger"
LOG_FILE="logs/radar_v2_performance.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Radar V2 performance logger is already running."
  exit 0
fi

: > "$LOG_FILE"

tmux new-session -d \
  -s "$SESSION" \
  "cd ~/liqheat-ai && source .venv/bin/activate && python -u scripts/radar_v2_performance_logger.py 2>&1 | tee logs/radar_v2_performance.log"

echo "============================================================"
echo "RADAR V2 PERFORMANCE LOGGER STARTED"
echo "============================================================"
echo "Live log:"
echo "  tail -n 100 -f ~/liqheat-ai/logs/radar_v2_performance.log"
echo "Status:"
echo "  ~/liqheat-ai/scripts/check_radar_v2_performance_logger.sh"
echo "Attach:"
echo "  tmux attach -t radar_v2_logger"

sleep 3
tail -80 "$LOG_FILE"
