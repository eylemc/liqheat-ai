from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

SYMBOL = os.getenv("LIQHEAT_SYMBOL", "BTCUSDT")
TIMEFRAME = os.getenv("LIQHEAT_TIMEFRAME", "1h")
DAYS = int(os.getenv("LIQHEAT_DAYS", "7"))
PAGE_SIZE = 1000
FUTURE_HOURS = 4

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SECRET_KEY")

if not url or not key:
    raise RuntimeError(
        "SUPABASE_URL veya SUPABASE_SECRET_KEY .env içinde bulunamadı"
    )

sb = create_client(url, key)

# PostgREST JSON path aliases:
# payload'ın dev dataPoints alanını indirmeden yalnızca aggregated değerleri çeker.
select_columns = ",".join(
    [
        "id",
        "symbol",
        "timeframe",
        "logged_at",
        "current_price",
        "liquidation_count",
        "price_min",
        "price_max",
        "rows",
        "cols",
        "total_longs:payload->aggregated->>totalLongs",
        "total_shorts:payload->aggregated->>totalShorts",
        "max_volume:payload->aggregated->>maxVolume",
    ]
)

cutoff = (
    pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=DAYS)
).isoformat()

print("=== EXPORT SETTINGS ===")
print("Symbol:", SYMBOL)
print("Timeframe:", TIMEFRAME)
print("Cutoff:", cutoff)
print("Payload dataPoints indirilmeyecek.")
print()

all_rows: list[dict] = []
offset = 0

while True:
    query = (
        sb.table("liq_logging")
        .select(select_columns)
        .eq("symbol", SYMBOL)
        .eq("timeframe", TIMEFRAME)
        .gte("logged_at", cutoff)
        .order("logged_at", desc=False)
        .range(offset, offset + PAGE_SIZE - 1)
    )

    result = query.execute()
    batch = result.data or []

    if not batch:
        break

    all_rows.extend(batch)
    offset += len(batch)

    print(f"\rÇekilen kayıt: {len(all_rows):,}", end="", flush=True)

    if len(batch) < PAGE_SIZE:
        break

    time.sleep(0.05)

print()

if not all_rows:
    raise RuntimeError("Supabase sorgusu kayıt döndürmedi")

df = pd.DataFrame(all_rows)

numeric_columns = [
    "current_price",
    "liquidation_count",
    "price_min",
    "price_max",
    "rows",
    "cols",
    "total_longs",
    "total_shorts",
    "max_volume",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True)
df = df.sort_values("logged_at").drop_duplicates(
    subset=["symbol", "timeframe", "logged_at"],
    keep="last",
)

# Temel snapshot özellikleri
epsilon = 1e-9

df["total_liquidations"] = (
    df["total_longs"].fillna(0) + df["total_shorts"].fillna(0)
)

df["long_short_ratio"] = (
    (df["total_longs"].fillna(0) + epsilon)
    / (df["total_shorts"].fillna(0) + epsilon)
)

df["long_share"] = (
    df["total_longs"].fillna(0)
    / (df["total_liquidations"] + epsilon)
)

df["short_share"] = (
    df["total_shorts"].fillna(0)
    / (df["total_liquidations"] + epsilon)
)

df["liq_imbalance"] = (
    (df["total_longs"].fillna(0) - df["total_shorts"].fillna(0))
    / (df["total_liquidations"] + epsilon)
)

df["price_range"] = df["price_max"] - df["price_min"]

df["price_range_pct"] = (
    df["price_range"]
    / df["current_price"].replace(0, np.nan)
)

df["price_position_in_range"] = (
    (df["current_price"] - df["price_min"])
    / df["price_range"].replace(0, np.nan)
)

df["max_volume_share"] = (
    df["max_volume"]
    / (df["total_liquidations"] + epsilon)
)

# Geçmişe bakan momentum özellikleri. Gelecek bilgisi kullanılmaz.
df["return_5m"] = df["current_price"].pct_change(5)
df["return_15m"] = df["current_price"].pct_change(15)
df["return_60m"] = df["current_price"].pct_change(60)

df["longs_change_5m"] = df["total_longs"].pct_change(5)
df["shorts_change_5m"] = df["total_shorts"].pct_change(5)

df["imbalance_change_5m"] = df["liq_imbalance"].diff(5)
df["imbalance_change_15m"] = df["liq_imbalance"].diff(15)

df["price_volatility_15m"] = (
    df["current_price"]
    .pct_change()
    .rolling(15, min_periods=10)
    .std()
)

df["price_volatility_60m"] = (
    df["current_price"]
    .pct_change()
    .rolling(60, min_periods=30)
    .std()
)

# 4 saat sonraki fiyatı satır kaydırarak değil, gerçek timestamp ile eşleştir.
future = df[["logged_at", "current_price"]].copy()
future["target_time"] = future["logged_at"] - pd.Timedelta(
    hours=FUTURE_HOURS
)
future = future.rename(
    columns={
        "logged_at": "future_observed_at",
        "current_price": "future_price_4h",
    }
).sort_values("target_time")

df = pd.merge_asof(
    df.sort_values("logged_at"),
    future,
    left_on="logged_at",
    right_on="target_time",
    direction="nearest",
    tolerance=pd.Timedelta(minutes=3),
)

df["future_return_4h"] = (
    df["future_price_4h"] / df["current_price"] - 1.0
)

df["future_direction_4h"] = np.select(
    [
        df["future_return_4h"] > 0.001,
        df["future_return_4h"] < -0.001,
    ],
    [1, -1],
    default=0,
)

# Merge yardımcı alanını temizle.
df = df.drop(columns=["target_time"], errors="ignore")

output_path = (
    OUTPUT_DIR
    / f"{SYMBOL.lower()}_{TIMEFRAME}_{DAYS}d_features_v1.parquet"
)

df.to_parquet(output_path, index=False, compression="zstd")

summary = {
    "symbol": SYMBOL,
    "timeframe": TIMEFRAME,
    "days_requested": DAYS,
    "rows": int(len(df)),
    "first_timestamp": str(df["logged_at"].min()),
    "last_timestamp": str(df["logged_at"].max()),
    "labeled_rows_4h": int(df["future_return_4h"].notna().sum()),
    "output": str(output_path),
    "file_size_mb": round(output_path.stat().st_size / 1024 / 1024, 3),
}

summary_path = OUTPUT_DIR / "export_summary_v1.json"
summary_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n=== EXPORT COMPLETE ===")
for key_name, value in summary.items():
    print(f"{key_name}: {value}")
