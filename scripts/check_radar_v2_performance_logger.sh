#!/usr/bin/env bash
set -u

cd "$HOME/liqheat-ai"

DB="data/monitoring/radar_v2_performance.sqlite"
STATUS_JSON="reports/radar_v2_performance/radar_v2_performance_status.json"

printf '%s\n' "============================================================"
printf '%s\n' "RADAR V2 PERFORMANCE LOGGER STATUS"
printf '%s\n' "============================================================"

if tmux has-session -t radar_v2_logger 2>/dev/null; then
  echo "Session: RUNNING"
else
  echo "Session: NOT RUNNING"
fi

echo
ps aux | grep "radar_v2_performance_logger.py" | grep -v grep || true

echo
if [[ -f "$STATUS_JSON" ]]; then
  echo "Status report:"
  python -m json.tool "$STATUS_JSON"
else
  echo "No status report yet."
fi

echo
if [[ -f "$DB" ]]; then
  echo "Database: $DB"
  du -h "$DB"

  if command -v sqlite3 >/dev/null 2>&1; then
    echo
    echo "Signal counts by opportunity:"
    sqlite3 -header -column "$DB" \
      "SELECT opportunity, COUNT(*) AS signals FROM radar_signals GROUP BY opportunity ORDER BY signals DESC;"

    echo
    echo "Mature outcomes by horizon:"
    sqlite3 -header -column "$DB" \
      "SELECT horizon_minutes, COUNT(*) AS outcomes FROM radar_outcomes GROUP BY horizon_minutes ORDER BY horizon_minutes;"
  fi
else
  echo "Database not created yet."
fi

echo
if [[ -f logs/radar_v2_performance.log ]]; then
  echo "Latest log:"
  tail -80 logs/radar_v2_performance.log
fi
