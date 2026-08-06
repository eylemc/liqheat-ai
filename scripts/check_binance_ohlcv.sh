#!/usr/bin/env bash
set -u

cd "$HOME/liqheat-ai"

echo "============================================================"
echo "BINANCE OHLCV STATUS"
echo "============================================================"

if tmux has-session -t ohlcv_download 2>/dev/null; then
  echo "Session: RUNNING"
else
  echo "Session: NOT RUNNING"
fi

echo
ps aux \
  | grep "download_binance_futures_ohlcv.py" \
  | grep -v grep \
  || true

echo
find data/market/binance-futures-um \
  -type f \
  -name '*.parquet' \
  -printf '%p %k KB\n' \
  2>/dev/null \
  | sort

echo
if [[ -f logs/binance_ohlcv.log ]]; then
  tail -80 logs/binance_ohlcv.log
else
  echo "No log yet."
fi
