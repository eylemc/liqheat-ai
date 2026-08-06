#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/liqheat-ai"
source .venv/bin/activate

mkdir -p \
  data/market/binance-futures-um \
  reports/market_data \
  logs

echo "============================================================"
echo "BINANCE OHLCV DOWNLOAD STARTED"
echo "Started: $(date --iso-8601=seconds)"
echo "============================================================"

python -u scripts/download_binance_futures_ohlcv.py \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --start 2026-03-30T00:00:00Z

echo
echo "============================================================"
echo "BINANCE OHLCV DOWNLOAD FINISHED"
echo "Finished: $(date --iso-8601=seconds)"
echo "============================================================"
