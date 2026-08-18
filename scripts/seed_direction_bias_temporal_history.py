from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import textara_api as api
from src.direction_bias_temporal import record_direction_bias

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSDT", "XAGUSDT"]
TIMEFRAME = "24h"
WINDOW_MINUTES = 130
LIMIT = 180


def main() -> int:
    since = (datetime.now(timezone.utc) - timedelta(minutes=WINDOW_MINUTES)).isoformat()
    total = 0
    print("=" * 88)
    print("SEED TEMPORAL DIRECTION BIAS HISTORY")
    print("=" * 88)
    print("Since:", since)

    for symbol in SYMBOLS:
        response = (
            api.supabase.table("liq_logging")
            .select("id,logged_at,symbol,timeframe,current_price,liquidation_count,price_min,price_max,payload")
            .eq("symbol", symbol)
            .eq("timeframe", TIMEFRAME)
            .gte("logged_at", since)
            .order("logged_at", desc=False)
            .limit(LIMIT)
            .execute()
        )
        rows = list(response.data or [])
        built = []
        ids = []
        for row in rows:
            feature = api.topology_feature_from_live_row(row)
            if feature is None:
                continue
            built.append(feature)
            ids.append(str(feature["id"]))
        if not built:
            print(f"{symbol}: no usable rows")
            continue

        frame = api.add_ml_features(built)
        count = 0
        for i in range(len(frame)):
            one = frame.iloc[[i]]
            try:
                dm = api.topology_direction_engine.score_latest(one, symbol)
            except Exception:
                continue
            record_direction_bias(symbol, ids[i], dm)
            count += 1
        total += count
        print(f"{symbol}: fetched={len(rows)} seeded={count}")

    print("Total seeded:", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
