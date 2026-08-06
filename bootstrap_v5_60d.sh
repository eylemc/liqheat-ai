#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/liqheat-ai"
source .venv/bin/activate

mkdir -p src data/processed reports logs

python -m pip install --quiet \
  python-dotenv \
  supabase \
  numpy \
  pandas \
  pyarrow \
  scikit-learn

cat > src/validate_nearest_pool_v5.py <<'PY'
from __future__ import annotations

import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
)
from supabase import create_client

SYMBOL = os.getenv("LIQHEAT_SYMBOL", "BTCUSDT")
TIMEFRAME = os.getenv("LIQHEAT_TIMEFRAME", "1h")
DAYS = int(os.getenv("LIQHEAT_DAYS", "60"))
FUTURE_HOURS = int(os.getenv("LIQHEAT_FUTURE_HOURS", "4"))
MAX_POOL_DISTANCE_PCT = float(
    os.getenv("LIQHEAT_MAX_POOL_DISTANCE_PCT", "0.03")
)

# Aynı anda çok saldırgan gitmeyelim.
WORKERS = int(os.getenv("LIQHEAT_V5_WORKERS", "6"))
PAGE_SIZE = 1000
MAX_RETRIES = 5

OUTPUT_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL veya SUPABASE_SECRET_KEY bulunamadı"
    )

cutoff = (
    pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=DAYS)
).floor("h")

end_time = pd.Timestamp.now(tz="UTC").ceil("h")

print("==========================================")
print("LiqHeat V5 — 60 Günlük Nearest-Pool Testi")
print("==========================================")
print("Symbol:", SYMBOL)
print("Timeframe:", TIMEFRAME)
print("Başlangıç:", cutoff)
print("Bitiş:", end_time)
print("Future horizon:", f"{FUTURE_HOURS} saat")
print("Max pool distance:", f"%{MAX_POOL_DISTANCE_PCT * 100:.1f}")
print("Workers:", WORKERS)
print()

main_client = create_client(SUPABASE_URL, SUPABASE_KEY)


def safe_float(
    value: Any,
    default: float = np.nan,
) -> float:
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass

    return default


def fetch_price_series() -> pd.DataFrame:
    """
    Bütün dakika kayıtlarından yalnızca timestamp ve fiyatı çeker.
    Payload çekilmediği için veri transferi görece küçüktür.
    """
    print("1/3 Hafif fiyat serisi indiriliyor...")

    rows: list[dict[str, Any]] = []
    offset = 0

    while True:
        last_error: Exception | None = None
        batch: list[dict[str, Any]] = []

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = (
                    main_client.table("liq_logging")
                    .select("logged_at,current_price")
                    .eq("symbol", SYMBOL)
                    .eq("timeframe", TIMEFRAME)
                    .gte("logged_at", cutoff.isoformat())
                    .order("logged_at", desc=False)
                    .range(offset, offset + PAGE_SIZE - 1)
                    .execute()
                )

                batch = result.data or []
                last_error = None
                break

            except Exception as exc:
                last_error = exc
                wait = min(30, 2 ** attempt)
                print(
                    f"\nFiyat sayfası hatası "
                    f"{attempt}/{MAX_RETRIES}: {exc}"
                )
                time.sleep(wait)

        if last_error is not None:
            raise RuntimeError(
                f"Fiyat serisi alınamadı: {last_error}"
            )

        if not batch:
            break

        rows.extend(batch)
        offset += len(batch)

        print(
            f"\rFiyat kayıtları: {len(rows):,}",
            end="",
            flush=True,
        )

        if len(batch) < PAGE_SIZE:
            break

    print()

    if not rows:
        raise RuntimeError("Fiyat serisi boş döndü")

    frame = pd.DataFrame(rows)
    frame["logged_at"] = pd.to_datetime(
        frame["logged_at"],
        utc=True,
        errors="coerce",
    )
    frame["current_price"] = pd.to_numeric(
        frame["current_price"],
        errors="coerce",
    )

    frame = (
        frame.dropna()
        .sort_values("logged_at")
        .drop_duplicates("logged_at", keep="last")
        .reset_index(drop=True)
    )

    print("Fiyat serisi tamamlandı:", len(frame))
    print(
        "Aralık:",
        frame["logged_at"].min(),
        "→",
        frame["logged_at"].max(),
    )

    return frame


thread_local = threading.local()


def get_thread_client():
    client = getattr(thread_local, "client", None)

    if client is None:
        client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
        )
        thread_local.client = client

    return client


def fetch_hour_snapshot(
    hour_start: pd.Timestamp,
) -> dict[str, Any] | None:
    """
    Belirtilen saatin ilk heatmap snapshot'ını getirir.
    """
    hour_end = hour_start + pd.Timedelta(hours=1)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = get_thread_client()

            result = (
                client.table("liq_logging")
                .select(
                    "id,logged_at,current_price,"
                    "liquidation_count,price_min,"
                    "price_max,payload"
                )
                .eq("symbol", SYMBOL)
                .eq("timeframe", TIMEFRAME)
                .gte("logged_at", hour_start.isoformat())
                .lt("logged_at", hour_end.isoformat())
                .order("logged_at", desc=False)
                .limit(1)
                .execute()
            )

            data = result.data or []

            if not data:
                return None

            return data[0]

        except Exception as exc:
            last_error = exc
            time.sleep(min(20, 2 ** attempt))

    print(
        f"\nSaat alınamadı: {hour_start} — {last_error}"
    )
    return None


def aggregate_levels(
    data_points: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aynı fiyat seviyesindeki bütün zaman hücrelerini toplar.

    Çıktı:
      prices
      total_liquidation_volume
    """
    if not isinstance(data_points, list):
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    levels: dict[float, float] = {}

    for point in data_points:
        if not isinstance(point, list) or len(point) < 4:
            continue

        price = safe_float(point[1])
        long_volume = safe_float(point[2], 0.0)
        short_volume = safe_float(point[3], 0.0)

        if not math.isfinite(price) or price <= 0:
            continue

        volume = max(0.0, long_volume) + max(
            0.0,
            short_volume,
        )

        rounded = round(price, 8)
        levels[rounded] = levels.get(rounded, 0.0) + volume

    if not levels:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    ordered = sorted(levels.items())

    return (
        np.array(
            [item[0] for item in ordered],
            dtype=np.float64,
        ),
        np.array(
            [item[1] for item in ordered],
            dtype=np.float64,
        ),
    )


def strongest_pool(
    prices: np.ndarray,
    volumes: np.ndarray,
    current_price: float,
    side: str,
) -> tuple[float, float, float]:
    distances = prices / current_price - 1.0

    if side == "upper":
        mask = (
            (distances > 0)
            & (distances <= MAX_POOL_DISTANCE_PCT)
        )
    else:
        mask = (
            (distances < 0)
            & (distances >= -MAX_POOL_DISTANCE_PCT)
        )

    if not np.any(mask):
        return np.nan, 0.0, np.nan

    selected_prices = prices[mask]
    selected_volumes = volumes[mask]

    index = int(np.argmax(selected_volumes))

    pool_price = float(selected_prices[index])
    pool_volume = float(selected_volumes[index])
    pool_distance = abs(
        pool_price / current_price - 1.0
    )

    return pool_price, pool_volume, pool_distance


def extract_snapshot(
    row: dict[str, Any],
) -> dict[str, Any] | None:
    current_price = safe_float(row.get("current_price"))

    if (
        not math.isfinite(current_price)
        or current_price <= 0
    ):
        return None

    payload = row.get("payload")

    if not isinstance(payload, dict):
        return None

    prices, volumes = aggregate_levels(
        payload.get("dataPoints")
    )

    if len(prices) == 0:
        return None

    (
        upper_price,
        upper_volume,
        upper_distance,
    ) = strongest_pool(
        prices,
        volumes,
        current_price,
        "upper",
    )

    (
        lower_price,
        lower_volume,
        lower_distance,
    ) = strongest_pool(
        prices,
        volumes,
        current_price,
        "lower",
    )

    if not (
        math.isfinite(upper_price)
        and math.isfinite(lower_price)
        and math.isfinite(upper_distance)
        and math.isfinite(lower_distance)
    ):
        return None

    nearest_prediction = (
        1 if upper_distance < lower_distance else 0
    )

    return {
        "id": row.get("id"),
        "logged_at": pd.to_datetime(
            row.get("logged_at"),
            utc=True,
        ),
        "current_price": current_price,
        "liquidation_count": safe_float(
            row.get("liquidation_count"),
            0.0,
        ),
        "upper_pool_price": upper_price,
        "upper_pool_volume": upper_volume,
        "upper_pool_distance": upper_distance,
        "lower_pool_price": lower_price,
        "lower_pool_volume": lower_volume,
        "lower_pool_distance": lower_distance,
        "nearest_prediction": nearest_prediction,
        "nearest_name": (
            "UPPER"
            if nearest_prediction == 1
            else "LOWER"
        ),
        "distance_advantage": abs(
            upper_distance - lower_distance
        ),
    }


def fetch_hourly_snapshots() -> pd.DataFrame:
    print("\n2/3 Saatlik heatmap snapshot'ları indiriliyor...")

    hours = list(
        pd.date_range(
            start=cutoff,
            end=end_time,
            freq="1h",
            inclusive="left",
        )
    )

    print("İstenen saat sayısı:", len(hours))

    extracted: list[dict[str, Any]] = []
    completed = 0

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:
        futures = {
            executor.submit(
                fetch_hour_snapshot,
                hour,
            ): hour
            for hour in hours
        }

        for future in as_completed(futures):
            completed += 1

            try:
                raw_row = future.result()

                if raw_row:
                    feature = extract_snapshot(raw_row)

                    if feature:
                        extracted.append(feature)

            except Exception as exc:
                hour = futures[future]
                print(
                    f"\nSaat işleme hatası: {hour}: {exc}"
                )

            print(
                f"\rSaat: {completed:,}/{len(hours):,} | "
                f"Uygun snapshot: {len(extracted):,}",
                end="",
                flush=True,
            )

    print()

    if not extracted:
        raise RuntimeError(
            "Hiç saatlik snapshot üretilemedi"
        )

    frame = pd.DataFrame(extracted)
    frame = (
        frame.sort_values("logged_at")
        .drop_duplicates("logged_at", keep="last")
        .reset_index(drop=True)
    )

    print("Saatlik snapshot tamamlandı:", len(frame))

    return frame


def label_snapshot(
    row: pd.Series,
    prices: pd.DataFrame,
) -> dict[str, Any]:
    start = row["logged_at"]
    end = start + pd.Timedelta(hours=FUTURE_HOURS)

    path = prices[
        (prices["logged_at"] > start)
        & (prices["logged_at"] <= end)
    ]

    if path.empty:
        return {
            "actual_direction": np.nan,
            "actual_name": None,
            "upper_hit_at": pd.NaT,
            "lower_hit_at": pd.NaT,
            "future_max_price": np.nan,
            "future_min_price": np.nan,
        }

    upper_hits = path[
        path["current_price"]
        >= row["upper_pool_price"]
    ]
    lower_hits = path[
        path["current_price"]
        <= row["lower_pool_price"]
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
            actual_direction = 1
            actual_name = "UPPER"
        elif lower_hit_at < upper_hit_at:
            actual_direction = 0
            actual_name = "LOWER"
        else:
            actual_direction = np.nan
            actual_name = "SIMULTANEOUS"

    elif pd.notna(upper_hit_at):
        actual_direction = 1
        actual_name = "UPPER"

    elif pd.notna(lower_hit_at):
        actual_direction = 0
        actual_name = "LOWER"

    else:
        actual_direction = np.nan
        actual_name = "NEUTRAL"

    return {
        "actual_direction": actual_direction,
        "actual_name": actual_name,
        "upper_hit_at": upper_hit_at,
        "lower_hit_at": lower_hit_at,
        "future_max_price": float(
            path["current_price"].max()
        ),
        "future_min_price": float(
            path["current_price"].min()
        ),
    }


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan

    probability = successes / total
    denominator = 1 + z * z / total

    center = (
        probability + z * z / (2 * total)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            probability * (1 - probability) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )

    return center - margin, center + margin


def calculate_metrics(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    clean = frame.dropna(
        subset=[
            "actual_direction",
            "nearest_prediction",
        ]
    ).copy()

    if clean.empty:
        return {
            "rows": 0,
            "accuracy": None,
        }

    actual = clean["actual_direction"].astype(int)
    predicted = clean["nearest_prediction"].astype(int)

    accuracy = accuracy_score(actual, predicted)
    correct = int((actual == predicted).sum())
    low, high = wilson_interval(
        correct,
        len(clean),
    )

    return {
        "rows": int(len(clean)),
        "correct": correct,
        "accuracy": float(accuracy),
        "accuracy_ci95_low": float(low),
        "accuracy_ci95_high": float(high),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                actual,
                predicted,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                actual,
                predicted,
            )
        ),
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=[0, 1],
        ).tolist(),
        "actual_lower": int((actual == 0).sum()),
        "actual_upper": int((actual == 1).sum()),
    }


price_series = fetch_price_series()
snapshots = fetch_hourly_snapshots()

print("\n3/3 Dört saatlik hedefler oluşturuluyor...")

labels = [
    label_snapshot(row, price_series)
    for _, row in snapshots.iterrows()
]

dataset = pd.concat(
    [
        snapshots,
        pd.DataFrame(labels),
    ],
    axis=1,
)

dataset["correct"] = np.where(
    dataset["actual_direction"].notna(),
    (
        dataset["actual_direction"]
        == dataset["nearest_prediction"]
    ),
    np.nan,
)

dataset["month"] = (
    dataset["logged_at"]
    .dt.to_period("M")
    .astype(str)
)

dataset["week"] = (
    dataset["logged_at"]
    .dt.to_period("W-MON")
    .astype(str)
)

dataset_path = (
    OUTPUT_DIR
    / f"{SYMBOL.lower()}_{DAYS}d_nearest_pool_v5.parquet"
)

dataset.to_parquet(
    dataset_path,
    index=False,
    compression="zstd",
)

overall = calculate_metrics(dataset)

monthly: dict[str, Any] = {}
for month, group in dataset.groupby("month"):
    monthly[str(month)] = calculate_metrics(group)

weekly: dict[str, Any] = {}
for week, group in dataset.groupby("week"):
    weekly[str(week)] = calculate_metrics(group)

# 4 saat aralıklı, hedef pencereleri üst üste binmeyen alt test.
non_overlapping = (
    dataset.sort_values("logged_at")
    .iloc[::FUTURE_HOURS]
    .copy()
)

non_overlapping_metrics = calculate_metrics(
    non_overlapping
)

# Daha güçlü mesafe avantajı olan örnekler.
confidence_results: list[dict[str, Any]] = []

for threshold in [
    0.0000,
    0.0005,
    0.0010,
    0.0020,
    0.0030,
    0.0050,
    0.0100,
]:
    subset = dataset[
        dataset["distance_advantage"] >= threshold
    ]

    metrics = calculate_metrics(subset)

    confidence_results.append({
        "minimum_distance_advantage": threshold,
        **metrics,
    })

neutral_count = int(
    (dataset["actual_name"] == "NEUTRAL").sum()
)

simultaneous_count = int(
    (dataset["actual_name"] == "SIMULTANEOUS").sum()
)

report = {
    "settings": {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "days": DAYS,
        "future_hours": FUTURE_HOURS,
        "max_pool_distance_pct":
            MAX_POOL_DISTANCE_PCT,
    },
    "dataset": {
        "price_rows": int(len(price_series)),
        "hourly_snapshots": int(len(snapshots)),
        "labeled_rows": int(
            dataset["actual_direction"]
            .notna()
            .sum()
        ),
        "neutral_rows": neutral_count,
        "simultaneous_rows": simultaneous_count,
        "start": str(dataset["logged_at"].min()),
        "end": str(dataset["logged_at"].max()),
        "parquet": str(dataset_path),
    },
    "overall": overall,
    "non_overlapping_4h": non_overlapping_metrics,
    "monthly": monthly,
    "weekly": weekly,
    "distance_advantage_analysis":
        confidence_results,
}

report_path = (
    REPORT_DIR
    / f"{SYMBOL.lower()}_{DAYS}d_nearest_pool_v5.json"
)

report_path.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

monthly_rows = []

for month, metrics in monthly.items():
    monthly_rows.append({
        "month": month,
        **metrics,
    })

pd.DataFrame(monthly_rows).to_csv(
    REPORT_DIR
    / f"{SYMBOL.lower()}_{DAYS}d_monthly_v5.csv",
    index=False,
)

weekly_rows = []

for week, metrics in weekly.items():
    weekly_rows.append({
        "week": week,
        **metrics,
    })

pd.DataFrame(weekly_rows).to_csv(
    REPORT_DIR
    / f"{SYMBOL.lower()}_{DAYS}d_weekly_v5.csv",
    index=False,
)

print("\n==========================================")
print("V5 — 60 GÜNLÜK SONUÇ")
print("==========================================")

print("\nOVERALL")
for key, value in overall.items():
    print(f"{key}: {value}")

print("\nNON-OVERLAPPING 4H")
for key, value in non_overlapping_metrics.items():
    print(f"{key}: {value}")

print("\nAYLIK")
for month, metrics in monthly.items():
    accuracy = metrics.get("accuracy")
    rows = metrics.get("rows")

    if accuracy is None:
        print(month, "veri yok")
    else:
        print(
            f"{month}: "
            f"rows={rows}, "
            f"accuracy={accuracy:.4f}, "
            f"balanced="
            f"{metrics['balanced_accuracy']:.4f}, "
            f"MCC={metrics['mcc']:.4f}, "
            f"CI95="
            f"[{metrics['accuracy_ci95_low']:.4f}, "
            f"{metrics['accuracy_ci95_high']:.4f}]"
        )

print("\nMESAFE AVANTAJI")
for row in confidence_results:
    accuracy = row.get("accuracy")

    if accuracy is None:
        continue

    print(
        f"advantage>="
        f"{row['minimum_distance_advantage']:.4f}: "
        f"rows={row['rows']}, "
        f"accuracy={accuracy:.4f}, "
        f"balanced="
        f"{row['balanced_accuracy']:.4f}"
    )

print("\nDosyalar:")
print(dataset_path)
print(report_path)
PY

echo
echo "=========================================="
echo "LiqHeat V5 başlıyor"
echo "=========================================="

python src/validate_nearest_pool_v5.py \
  2>&1 | tee logs/validate_nearest_pool_v5.log

echo
echo "=========================================="
echo "V5 tamamlandı"
echo "=========================================="

python - <<'PY'
import json
from pathlib import Path

path = Path(
    "reports/btcusdt_60d_nearest_pool_v5.json"
)

report = json.loads(path.read_text())

overall = report["overall"]
independent = report["non_overlapping_4h"]

print("\nFINAL SCOREBOARD")

print(
    "Overall:",
    "rows=",
    overall["rows"],
    "accuracy=",
    round(overall["accuracy"], 4),
    "balanced=",
    round(overall["balanced_accuracy"], 4),
    "MCC=",
    round(overall["mcc"], 4),
)

print(
    "Non-overlapping 4h:",
    "rows=",
    independent["rows"],
    "accuracy=",
    round(independent["accuracy"], 4),
    "balanced=",
    round(
        independent["balanced_accuracy"],
        4,
    ),
    "MCC=",
    round(independent["mcc"], 4),
)
PY
