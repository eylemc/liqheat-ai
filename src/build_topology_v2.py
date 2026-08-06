from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

SYMBOL = os.getenv("LIQHEAT_SYMBOL", "BTCUSDT")
TIMEFRAME = os.getenv("LIQHEAT_TIMEFRAME", "1h")
DAYS = int(os.getenv("LIQHEAT_DAYS", "14"))

SAMPLE_MINUTES = int(os.getenv("LIQHEAT_SAMPLE_MINUTES", "5"))
FUTURE_HOURS = int(os.getenv("LIQHEAT_FUTURE_HOURS", "4"))

# Hedef havuz için fiyatın en fazla yüzde kaç uzağına bakacağız?
MAX_POOL_DISTANCE_PCT = float(
    os.getenv("LIQHEAT_MAX_POOL_DISTANCE_PCT", "0.03")
)

PAGE_SIZE = int(os.getenv("LIQHEAT_PAGE_SIZE", "100"))
MAX_RETRIES = 5

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SECRET_KEY")

if not url or not key:
    raise RuntimeError(
        "SUPABASE_URL veya SUPABASE_SECRET_KEY bulunamadı"
    )

sb = create_client(url, key)

cutoff = (
    pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=DAYS)
).isoformat()

select_columns = ",".join(
    [
        "id",
        "logged_at",
        "current_price",
        "liquidation_count",
        "price_min",
        "price_max",
        "payload",
    ]
)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def fetch_page(offset: int) -> list[dict[str, Any]]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = (
                sb.table("liq_logging")
                .select(select_columns)
                .eq("symbol", SYMBOL)
                .eq("timeframe", TIMEFRAME)
                .gte("logged_at", cutoff)
                .order("logged_at", desc=False)
                .range(offset, offset + PAGE_SIZE - 1)
                .execute()
            )
            return result.data or []

        except Exception as exc:
            last_error = exc
            wait_seconds = min(30, 2 ** attempt)
            print(
                f"\nSayfa hatası; deneme {attempt}/{MAX_RETRIES}: "
                f"{exc}"
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Supabase sayfası alınamadı: {last_error}"
    )


def aggregate_price_levels(
    data_points: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    dataPoints:
      [timestamp_ms, price_level, long_volume, short_volume]

    Aynı fiyat seviyesindeki bütün zaman hücrelerini toplar.
    """
    if not isinstance(data_points, list) or not data_points:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    levels: dict[float, list[float]] = {}

    for point in data_points:
        if not isinstance(point, list) or len(point) < 4:
            continue

        price = safe_float(point[1], np.nan)
        long_volume = max(0.0, safe_float(point[2]))
        short_volume = max(0.0, safe_float(point[3]))

        if not math.isfinite(price) or price <= 0:
            continue

        # Floating point fiyatları kararlı anahtara dönüştür.
        rounded_price = round(price, 8)

        if rounded_price not in levels:
            levels[rounded_price] = [0.0, 0.0]

        levels[rounded_price][0] += long_volume
        levels[rounded_price][1] += short_volume

    if not levels:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    ordered = sorted(levels.items())

    prices = np.array(
        [item[0] for item in ordered],
        dtype=np.float64,
    )
    longs = np.array(
        [item[1][0] for item in ordered],
        dtype=np.float64,
    )
    shorts = np.array(
        [item[1][1] for item in ordered],
        dtype=np.float64,
    )

    return prices, longs, shorts


def weighted_average(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    total = float(weights.sum())

    if total <= 0 or len(values) == 0:
        return np.nan

    return float(np.average(values, weights=weights))


def strongest_pool(
    prices: np.ndarray,
    volumes: np.ndarray,
    current_price: float,
    side: str,
) -> tuple[float, float, float]:
    """
    Dönüş:
      hedef fiyat,
      hedef hacim,
      hedef uzaklığı (oransal)
    """
    if side == "upper":
        mask = (
            (prices > current_price)
            & (
                prices
                <= current_price * (1.0 + MAX_POOL_DISTANCE_PCT)
            )
        )
    elif side == "lower":
        mask = (
            (prices < current_price)
            & (
                prices
                >= current_price * (1.0 - MAX_POOL_DISTANCE_PCT)
            )
        )
    else:
        raise ValueError(f"Bilinmeyen side: {side}")

    if not np.any(mask):
        return np.nan, 0.0, np.nan

    selected_prices = prices[mask]
    selected_volumes = volumes[mask]

    if selected_volumes.size == 0:
        return np.nan, 0.0, np.nan

    index = int(np.argmax(selected_volumes))
    pool_price = float(selected_prices[index])
    pool_volume = float(selected_volumes[index])

    distance = abs(pool_price / current_price - 1.0)

    return pool_price, pool_volume, distance


def band_sum(
    prices: np.ndarray,
    volumes: np.ndarray,
    current_price: float,
    low_pct: float,
    high_pct: float,
    side: str,
) -> float:
    distances = prices / current_price - 1.0

    if side == "upper":
        mask = (
            (distances > low_pct)
            & (distances <= high_pct)
        )
    else:
        absolute_down = -distances
        mask = (
            (absolute_down > low_pct)
            & (absolute_down <= high_pct)
        )

    return float(volumes[mask].sum())


def extract_features(row: dict[str, Any]) -> dict[str, Any] | None:
    current_price = safe_float(row.get("current_price"), np.nan)

    if not math.isfinite(current_price) or current_price <= 0:
        return None

    payload = row.get("payload")

    if not isinstance(payload, dict):
        return None

    aggregated = payload.get("aggregated") or {}
    data_points = payload.get("dataPoints")

    prices, longs, shorts = aggregate_price_levels(data_points)

    if len(prices) == 0:
        return None

    total_by_level = longs + shorts

    above_mask = prices > current_price
    below_mask = prices < current_price

    upper_volume = float(total_by_level[above_mask].sum())
    lower_volume = float(total_by_level[below_mask].sum())
    overall_volume = upper_volume + lower_volume
    epsilon = 1e-9

    upper_price, upper_pool_volume, upper_distance = (
        strongest_pool(
            prices,
            total_by_level,
            current_price,
            "upper",
        )
    )

    lower_price, lower_pool_volume, lower_distance = (
        strongest_pool(
            prices,
            total_by_level,
            current_price,
            "lower",
        )
    )

    relative_distance = prices / current_price - 1.0

    center_all = weighted_average(
        relative_distance,
        total_by_level,
    )

    center_above = weighted_average(
        relative_distance[above_mask],
        total_by_level[above_mask],
    )

    center_below = weighted_average(
        relative_distance[below_mask],
        total_by_level[below_mask],
    )

    nonzero_levels = int(np.count_nonzero(total_by_level > 0))

    sorted_volumes = np.sort(total_by_level)[::-1]
    top1 = float(sorted_volumes[0]) if len(sorted_volumes) else 0.0
    top3 = float(sorted_volumes[:3].sum())
    top5 = float(sorted_volumes[:5].sum())

    total_longs = safe_float(
        aggregated.get("totalLongs")
    )
    total_shorts = safe_float(
        aggregated.get("totalShorts")
    )
    max_volume = safe_float(
        aggregated.get("maxVolume")
    )

    feature = {
        "id": row.get("id"),
        "logged_at": row.get("logged_at"),
        "current_price": current_price,
        "liquidation_count": safe_float(
            row.get("liquidation_count")
        ),
        "price_min": safe_float(row.get("price_min"), np.nan),
        "price_max": safe_float(row.get("price_max"), np.nan),

        "total_longs": total_longs,
        "total_shorts": total_shorts,
        "total_aggregated": total_longs + total_shorts,
        "aggregated_imbalance": (
            (total_longs - total_shorts)
            / (total_longs + total_shorts + epsilon)
        ),
        "max_volume": max_volume,

        "upper_volume": upper_volume,
        "lower_volume": lower_volume,
        "upper_lower_ratio": (
            (upper_volume + epsilon)
            / (lower_volume + epsilon)
        ),
        "topology_imbalance": (
            (upper_volume - lower_volume)
            / (overall_volume + epsilon)
        ),

        "upper_pool_price": upper_price,
        "upper_pool_volume": upper_pool_volume,
        "upper_pool_distance": upper_distance,

        "lower_pool_price": lower_price,
        "lower_pool_volume": lower_pool_volume,
        "lower_pool_distance": lower_distance,

        "pool_volume_ratio": (
            (upper_pool_volume + epsilon)
            / (lower_pool_volume + epsilon)
        ),
        "pool_distance_ratio": (
            (upper_distance + epsilon)
            / (lower_distance + epsilon)
            if math.isfinite(upper_distance)
            and math.isfinite(lower_distance)
            else np.nan
        ),

        "upper_0_0_5": band_sum(
            prices, total_by_level, current_price,
            0.000, 0.005, "upper"
        ),
        "upper_0_5_1": band_sum(
            prices, total_by_level, current_price,
            0.005, 0.010, "upper"
        ),
        "upper_1_2": band_sum(
            prices, total_by_level, current_price,
            0.010, 0.020, "upper"
        ),
        "upper_2_3": band_sum(
            prices, total_by_level, current_price,
            0.020, 0.030, "upper"
        ),

        "lower_0_0_5": band_sum(
            prices, total_by_level, current_price,
            0.000, 0.005, "lower"
        ),
        "lower_0_5_1": band_sum(
            prices, total_by_level, current_price,
            0.005, 0.010, "lower"
        ),
        "lower_1_2": band_sum(
            prices, total_by_level, current_price,
            0.010, 0.020, "lower"
        ),
        "lower_2_3": band_sum(
            prices, total_by_level, current_price,
            0.020, 0.030, "lower"
        ),

        "weighted_center_all": center_all,
        "weighted_center_above": center_above,
        "weighted_center_below": center_below,

        "nonzero_price_levels": nonzero_levels,
        "top1_level_share": top1 / (overall_volume + epsilon),
        "top3_level_share": top3 / (overall_volume + epsilon),
        "top5_level_share": top5 / (overall_volume + epsilon),
    }

    feature["near_topology_imbalance"] = (
        (
            feature["upper_0_0_5"]
            + feature["upper_0_5_1"]
            - feature["lower_0_0_5"]
            - feature["lower_0_5_1"]
        )
        /
        (
            feature["upper_0_0_5"]
            + feature["upper_0_5_1"]
            + feature["lower_0_0_5"]
            + feature["lower_0_5_1"]
            + epsilon
        )
    )

    return feature


print("=== V2 TOPOLOGY EXPORT ===")
print("Symbol:", SYMBOL)
print("Timeframe:", TIMEFRAME)
print("Days:", DAYS)
print("Sampling:", f"{SAMPLE_MINUTES} dakika")
print("Future horizon:", f"{FUTURE_HOURS} saat")
print("Pool maximum distance:", MAX_POOL_DISTANCE_PCT)
print("Cutoff:", cutoff)
print()

raw_price_rows: list[dict[str, Any]] = []
feature_rows: list[dict[str, Any]] = []

offset = 0
downloaded = 0
processed = 0
skipped_sampling = 0
last_sample_bucket: pd.Timestamp | None = None

while True:
    batch = fetch_page(offset)

    if not batch:
        break

    for row in batch:
        downloaded += 1

        timestamp = pd.to_datetime(
            row.get("logged_at"),
            utc=True,
            errors="coerce",
        )

        price = safe_float(
            row.get("current_price"),
            np.nan,
        )

        if pd.notna(timestamp) and math.isfinite(price):
            raw_price_rows.append(
                {
                    "logged_at": timestamp,
                    "current_price": price,
                }
            )

        if pd.isna(timestamp):
            continue

        bucket_time = timestamp.floor(
            f"{SAMPLE_MINUTES}min"
        )

        # Her 5 dakikalık dilimde ilk snapshot'ı kullan.
        if last_sample_bucket is not None and (
            bucket_time == last_sample_bucket
        ):
            skipped_sampling += 1
            continue

        last_sample_bucket = bucket_time

        feature = extract_features(row)

        if feature is not None:
            feature_rows.append(feature)
            processed += 1

    offset += len(batch)

    print(
        f"\rHam: {downloaded:,} | "
        f"Feature: {processed:,} | "
        f"Sampling skip: {skipped_sampling:,}",
        end="",
        flush=True,
    )

    del batch
    gc.collect()

    if offset % 1000 == 0:
        time.sleep(0.05)

print()

if not feature_rows:
    raise RuntimeError("Hiç topology feature üretilemedi")

features = pd.DataFrame(feature_rows)
features["logged_at"] = pd.to_datetime(
    features["logged_at"],
    utc=True,
)
features = (
    features
    .sort_values("logged_at")
    .drop_duplicates("logged_at", keep="last")
    .reset_index(drop=True)
)

prices = pd.DataFrame(raw_price_rows)
prices = (
    prices
    .sort_values("logged_at")
    .drop_duplicates("logged_at", keep="last")
    .reset_index(drop=True)
)

# Geçmiş momentum bilgileri; yalnızca o ana kadarki değerler.
features["return_5m"] = features["current_price"].pct_change(1)
features["return_15m"] = features["current_price"].pct_change(3)
features["return_30m"] = features["current_price"].pct_change(6)
features["return_60m"] = features["current_price"].pct_change(12)

features["topology_change_5m"] = (
    features["topology_imbalance"].diff(1)
)
features["topology_change_15m"] = (
    features["topology_imbalance"].diff(3)
)
features["near_topology_change_15m"] = (
    features["near_topology_imbalance"].diff(3)
)

features["realized_volatility_30m"] = (
    features["current_price"]
    .pct_change()
    .rolling(6, min_periods=4)
    .std()
)
features["realized_volatility_60m"] = (
    features["current_price"]
    .pct_change()
    .rolling(12, min_periods=8)
    .std()
)


def create_target(row: pd.Series) -> dict[str, Any]:
    start = row["logged_at"]
    end = start + pd.Timedelta(hours=FUTURE_HOURS)

    future_path = prices[
        (prices["logged_at"] > start)
        & (prices["logged_at"] <= end)
    ]

    upper = row["upper_pool_price"]
    lower = row["lower_pool_price"]

    if (
        future_path.empty
        or not math.isfinite(upper)
        or not math.isfinite(lower)
    ):
        return {
            "target_class": np.nan,
            "target_name": None,
            "upper_hit_at": pd.NaT,
            "lower_hit_at": pd.NaT,
            "future_max_price": np.nan,
            "future_min_price": np.nan,
            "future_close_price": np.nan,
            "future_return": np.nan,
        }

    upper_hits = future_path[
        future_path["current_price"] >= upper
    ]
    lower_hits = future_path[
        future_path["current_price"] <= lower
    ]

    upper_hit_at = (
        upper_hits.iloc[0]["logged_at"]
        if not upper_hits.empty
        else pd.NaT
    )
    lower_hit_at = (
        lower_hits.iloc[0]["logged_at"]
        if not lower_hits.empty
        else pd.NaT
    )

    if pd.notna(upper_hit_at) and pd.notna(lower_hit_at):
        if upper_hit_at < lower_hit_at:
            target_class = 2
            target_name = "UPPER"
        elif lower_hit_at < upper_hit_at:
            target_class = 0
            target_name = "LOWER"
        else:
            target_class = 1
            target_name = "NEUTRAL"
    elif pd.notna(upper_hit_at):
        target_class = 2
        target_name = "UPPER"
    elif pd.notna(lower_hit_at):
        target_class = 0
        target_name = "LOWER"
    else:
        target_class = 1
        target_name = "NEUTRAL"

    future_close = float(
        future_path.iloc[-1]["current_price"]
    )

    return {
        "target_class": target_class,
        "target_name": target_name,
        "upper_hit_at": upper_hit_at,
        "lower_hit_at": lower_hit_at,
        "future_max_price": float(
            future_path["current_price"].max()
        ),
        "future_min_price": float(
            future_path["current_price"].min()
        ),
        "future_close_price": future_close,
        "future_return": (
            future_close / row["current_price"] - 1.0
        ),
    }


print("4 saatlik havuz ziyaret etiketleri oluşturuluyor...")

target_rows = [
    create_target(row)
    for _, row in features.iterrows()
]

targets = pd.DataFrame(target_rows)
features = pd.concat([features, targets], axis=1)

output_path = (
    OUTPUT_DIR
    / (
        f"{SYMBOL.lower()}_{TIMEFRAME}_{DAYS}d_"
        f"topology_v2.parquet"
    )
)

features.to_parquet(
    output_path,
    index=False,
    compression="zstd",
)

class_counts = (
    features["target_name"]
    .value_counts(dropna=False)
    .to_dict()
)

summary = {
    "symbol": SYMBOL,
    "timeframe": TIMEFRAME,
    "days": DAYS,
    "downloaded_raw_rows": downloaded,
    "sampled_feature_rows": len(features),
    "first_timestamp": str(features["logged_at"].min()),
    "last_timestamp": str(features["logged_at"].max()),
    "class_counts": class_counts,
    "output": str(output_path),
    "output_mb": round(
        output_path.stat().st_size / 1024 / 1024,
        3,
    ),
}

Path("reports/topology_v2_export_summary.json").write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
        default=str,
    ),
    encoding="utf-8",
)

print("\n=== TOPOLOGY EXPORT COMPLETE ===")
print(json.dumps(summary, indent=2, ensure_ascii=False))
