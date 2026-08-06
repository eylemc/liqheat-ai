#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/liqheat-ai"
source .venv/bin/activate

mkdir -p src reports logs

cat > src/analyze_edge_matrix_v6.py <<'PY'
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SYMBOL = os.getenv("LIQHEAT_SYMBOL", "BTCUSDT")
DAYS = int(os.getenv("LIQHEAT_DAYS", "60"))

DATA_PATH = Path(
    f"data/processed/"
    f"{SYMBOL.lower()}_{DAYS}d_nearest_pool_v5.parquet"
)

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"V5 dataset bulunamadı: {DATA_PATH}"
    )

df = pd.read_parquet(DATA_PATH)

df["logged_at"] = pd.to_datetime(
    df["logged_at"],
    utc=True,
    errors="coerce",
)

numeric_columns = [
    "actual_direction",
    "nearest_prediction",
    "upper_pool_distance",
    "lower_pool_distance",
    "upper_pool_volume",
    "lower_pool_volume",
    "distance_advantage",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

df = df.dropna(
    subset=[
        "logged_at",
        "actual_direction",
        "nearest_prediction",
        "upper_pool_distance",
        "lower_pool_distance",
        "upper_pool_volume",
        "lower_pool_volume",
    ]
).copy()

df["actual_direction"] = (
    df["actual_direction"].astype(int)
)
df["nearest_prediction"] = (
    df["nearest_prediction"].astype(int)
)

df["correct"] = (
    df["actual_direction"]
    == df["nearest_prediction"]
).astype(int)

upper_is_nearest = (
    df["nearest_prediction"] == 1
)

df["nearest_distance"] = np.where(
    upper_is_nearest,
    df["upper_pool_distance"],
    df["lower_pool_distance"],
)

df["farther_distance"] = np.where(
    upper_is_nearest,
    df["lower_pool_distance"],
    df["upper_pool_distance"],
)

df["nearest_volume"] = np.where(
    upper_is_nearest,
    df["upper_pool_volume"],
    df["lower_pool_volume"],
)

df["farther_volume"] = np.where(
    upper_is_nearest,
    df["lower_pool_volume"],
    df["upper_pool_volume"],
)

epsilon = 1e-12

# Mutlak mesafe farkı.
df["distance_edge"] = (
    df["farther_distance"]
    - df["nearest_distance"]
)

# En yakın havuz diğerine göre yüzde kaç daha yakın?
df["distance_ratio"] = (
    df["nearest_distance"] + epsilon
) / (
    df["farther_distance"] + epsilon
)

# 1'e yaklaştıkça mesafe avantajı zayıf,
# 0'a yaklaştıkça nearest avantajı güçlü.
df["distance_strength"] = (
    1.0 - df["distance_ratio"]
)

# En yakın havuzun hacim üstünlüğü.
df["volume_ratio"] = (
    df["nearest_volume"] + epsilon
) / (
    df["farther_volume"] + epsilon
)

df["log_volume_ratio"] = np.log10(
    df["volume_ratio"].clip(
        lower=1e-6,
        upper=1e6,
    )
)

# Uzaklık başına hacim basıncı.
df["nearest_pressure"] = (
    df["nearest_volume"] + epsilon
) / (
    df["nearest_distance"] + epsilon
)

df["farther_pressure"] = (
    df["farther_volume"] + epsilon
) / (
    df["farther_distance"] + epsilon
)

df["pressure_ratio"] = (
    df["nearest_pressure"] + epsilon
) / (
    df["farther_pressure"] + epsilon
)

df["log_pressure_ratio"] = np.log10(
    df["pressure_ratio"].clip(
        lower=1e-6,
        upper=1e6,
    )
)

df["hour_utc"] = df["logged_at"].dt.hour
df["day_of_week"] = df["logged_at"].dt.day_name()

df["session"] = pd.cut(
    df["hour_utc"],
    bins=[-1, 6, 12, 17, 23],
    labels=[
        "Asia",
        "Europe Morning",
        "US Session",
        "Late US",
    ],
)

# Üst sinyal / alt sinyal ayrımı.
df["signal_direction"] = np.where(
    df["nearest_prediction"] == 1,
    "UPPER",
    "LOWER",
)


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan

    p = successes / total
    denominator = 1.0 + z * z / total

    center = (
        p + z * z / (2.0 * total)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            p * (1.0 - p) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )

    return center - margin, center + margin


def summarize(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    count = int(len(frame))

    if count == 0:
        return {
            "rows": 0,
            "correct": 0,
            "accuracy": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    correct = int(frame["correct"].sum())
    accuracy = correct / count
    low, high = wilson_interval(correct, count)

    actual_upper = int(
        (frame["actual_direction"] == 1).sum()
    )
    predicted_upper = int(
        (frame["nearest_prediction"] == 1).sum()
    )

    return {
        "rows": count,
        "correct": correct,
        "accuracy": float(accuracy),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "actual_upper_rate": float(
            actual_upper / count
        ),
        "predicted_upper_rate": float(
            predicted_upper / count
        ),
        "median_distance_edge": float(
            frame["distance_edge"].median()
        ),
        "median_volume_ratio": float(
            frame["volume_ratio"].median()
        ),
        "median_pressure_ratio": float(
            frame["pressure_ratio"].median()
        ),
    }


overall = summarize(df)

# Eşik testleri.
distance_thresholds = [
    0.0000,
    0.0005,
    0.0010,
    0.0020,
    0.0030,
    0.0050,
    0.0100,
]

volume_thresholds = [
    0.0,
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
    3.00,
    5.00,
    10.00,
]

pressure_thresholds = [
    0.0,
    0.50,
    1.00,
    1.50,
    2.00,
    3.00,
    5.00,
    10.00,
]

distance_results = []

for threshold in distance_thresholds:
    subset = df[
        df["distance_edge"] >= threshold
    ]

    distance_results.append({
        "minimum_distance_edge": threshold,
        **summarize(subset),
    })

volume_results = []

for threshold in volume_thresholds:
    subset = df[
        df["volume_ratio"] >= threshold
    ]

    volume_results.append({
        "minimum_nearest_volume_ratio": threshold,
        **summarize(subset),
    })

pressure_results = []

for threshold in pressure_thresholds:
    subset = df[
        df["pressure_ratio"] >= threshold
    ]

    pressure_results.append({
        "minimum_nearest_pressure_ratio": threshold,
        **summarize(subset),
    })

# Distance × volume matrisi.
distance_bins = [
    -np.inf,
    0.0005,
    0.0010,
    0.0020,
    0.0030,
    0.0050,
    np.inf,
]

distance_labels = [
    "<0.05%",
    "0.05–0.10%",
    "0.10–0.20%",
    "0.20–0.30%",
    "0.30–0.50%",
    ">=0.50%",
]

volume_bins = [
    -np.inf,
    0.50,
    1.00,
    2.00,
    5.00,
    np.inf,
]

volume_labels = [
    "<0.5x",
    "0.5–1x",
    "1–2x",
    "2–5x",
    ">=5x",
]

df["distance_band"] = pd.cut(
    df["distance_edge"],
    bins=distance_bins,
    labels=distance_labels,
    right=False,
)

df["volume_band"] = pd.cut(
    df["volume_ratio"],
    bins=volume_bins,
    labels=volume_labels,
    right=False,
)

matrix_rows = []

for distance_band in distance_labels:
    for volume_band in volume_labels:
        subset = df[
            (df["distance_band"] == distance_band)
            & (df["volume_band"] == volume_band)
        ]

        matrix_rows.append({
            "distance_band": distance_band,
            "volume_band": volume_band,
            **summarize(subset),
        })

matrix_df = pd.DataFrame(matrix_rows)

# Seans analizi.
session_results = {}

for session, group in df.groupby(
    "session",
    observed=True,
):
    session_results[str(session)] = summarize(group)

# Yön bazlı analiz.
direction_results = {}

for direction, group in df.groupby(
    "signal_direction"
):
    direction_results[str(direction)] = summarize(group)

# Saat bazlı analiz.
hour_results = {}

for hour, group in df.groupby("hour_utc"):
    hour_results[str(int(hour))] = summarize(group)

# Gün bazlı analiz.
day_results = {}

for day, group in df.groupby("day_of_week"):
    day_results[str(day)] = summarize(group)

# Basit high-confidence kuralları.
confidence_rules = [
    {
        "name": "distance_0_10",
        "distance": 0.0010,
        "volume": None,
        "pressure": None,
    },
    {
        "name": "distance_0_20",
        "distance": 0.0020,
        "volume": None,
        "pressure": None,
    },
    {
        "name": "distance_0_30",
        "distance": 0.0030,
        "volume": None,
        "pressure": None,
    },
    {
        "name": "distance_0_10_volume_1x",
        "distance": 0.0010,
        "volume": 1.0,
        "pressure": None,
    },
    {
        "name": "distance_0_20_volume_1x",
        "distance": 0.0020,
        "volume": 1.0,
        "pressure": None,
    },
    {
        "name": "distance_0_30_volume_1x",
        "distance": 0.0030,
        "volume": 1.0,
        "pressure": None,
    },
    {
        "name": "distance_0_10_pressure_1x",
        "distance": 0.0010,
        "volume": None,
        "pressure": 1.0,
    },
    {
        "name": "distance_0_20_pressure_1x",
        "distance": 0.0020,
        "volume": None,
        "pressure": 1.0,
    },
    {
        "name": "distance_0_30_pressure_1x",
        "distance": 0.0030,
        "volume": None,
        "pressure": 1.0,
    },
    {
        "name": "distance_0_20_pressure_2x",
        "distance": 0.0020,
        "volume": None,
        "pressure": 2.0,
    },
    {
        "name": "distance_0_30_pressure_2x",
        "distance": 0.0030,
        "volume": None,
        "pressure": 2.0,
    },
]

confidence_results = []

for rule in confidence_rules:
    mask = pd.Series(
        True,
        index=df.index,
    )

    if rule["distance"] is not None:
        mask &= (
            df["distance_edge"]
            >= rule["distance"]
        )

    if rule["volume"] is not None:
        mask &= (
            df["volume_ratio"]
            >= rule["volume"]
        )

    if rule["pressure"] is not None:
        mask &= (
            df["pressure_ratio"]
            >= rule["pressure"]
        )

    subset = df[mask]

    confidence_results.append({
        **rule,
        "coverage": float(
            len(subset) / len(df)
        ) if len(df) else 0.0,
        **summarize(subset),
    })

# 4 saat aralıklı bağımsız alt örneklemde
# aynı confidence kurallarını tekrar ölç.
independent = (
    df.sort_values("logged_at")
    .iloc[::4]
    .copy()
)

independent_results = []

for rule in confidence_rules:
    mask = pd.Series(
        True,
        index=independent.index,
    )

    if rule["distance"] is not None:
        mask &= (
            independent["distance_edge"]
            >= rule["distance"]
        )

    if rule["volume"] is not None:
        mask &= (
            independent["volume_ratio"]
            >= rule["volume"]
        )

    if rule["pressure"] is not None:
        mask &= (
            independent["pressure_ratio"]
            >= rule["pressure"]
        )

    subset = independent[mask]

    independent_results.append({
        **rule,
        "coverage": float(
            len(subset) / len(independent)
        ) if len(independent) else 0.0,
        **summarize(subset),
    })

report = {
    "settings": {
        "symbol": SYMBOL,
        "days": DAYS,
        "source": str(DATA_PATH),
    },
    "overall": overall,
    "independent_4h_overall": summarize(
        independent
    ),
    "distance_thresholds": distance_results,
    "volume_thresholds": volume_results,
    "pressure_thresholds": pressure_results,
    "confidence_rules": confidence_results,
    "independent_4h_confidence_rules":
        independent_results,
    "sessions": session_results,
    "directions": direction_results,
    "hours_utc": hour_results,
    "days": day_results,
}

report_path = (
    REPORT_DIR
    / f"{SYMBOL.lower()}_{DAYS}d_edge_matrix_v6.json"
)

report_path.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

matrix_path = (
    REPORT_DIR
    / f"{SYMBOL.lower()}_{DAYS}d_edge_matrix_v6.csv"
)

matrix_df.to_csv(
    matrix_path,
    index=False,
)

confidence_path = (
    REPORT_DIR
    / f"{SYMBOL.lower()}_{DAYS}d_confidence_rules_v6.csv"
)

pd.DataFrame(
    confidence_results
).to_csv(
    confidence_path,
    index=False,
)

independent_confidence_path = (
    REPORT_DIR
    / (
        f"{SYMBOL.lower()}_{DAYS}d_"
        f"independent_confidence_v6.csv"
    )
)

pd.DataFrame(
    independent_results
).to_csv(
    independent_confidence_path,
    index=False,
)

print("==========================================")
print("LiqHeat V6 — Distance × Volume Matrix")
print("==========================================")

print("\nOVERALL")
for key, value in overall.items():
    print(f"{key}: {value}")

print("\nINDEPENDENT 4H OVERALL")
for key, value in summarize(independent).items():
    print(f"{key}: {value}")

print("\nDISTANCE EDGE")
for row in distance_results:
    if row["accuracy"] is None:
        continue

    print(
        f">={row['minimum_distance_edge']:.4f}  "
        f"rows={row['rows']:>4}  "
        f"accuracy={row['accuracy']:.4f}  "
        f"CI95=[{row['ci95_low']:.4f}, "
        f"{row['ci95_high']:.4f}]"
    )

print("\nVOLUME RATIO")
for row in volume_results:
    if row["accuracy"] is None:
        continue

    print(
        f">={row['minimum_nearest_volume_ratio']:.2f}x  "
        f"rows={row['rows']:>4}  "
        f"accuracy={row['accuracy']:.4f}"
    )

print("\nPRESSURE RATIO")
for row in pressure_results:
    if row["accuracy"] is None:
        continue

    print(
        f">={row['minimum_nearest_pressure_ratio']:.2f}x  "
        f"rows={row['rows']:>4}  "
        f"accuracy={row['accuracy']:.4f}"
    )

print("\nCONFIDENCE RULES — ALL HOURLY")
for row in confidence_results:
    if row["accuracy"] is None:
        continue

    print(
        f"{row['name']:<30} "
        f"rows={row['rows']:>4}  "
        f"coverage={row['coverage']:.3f}  "
        f"accuracy={row['accuracy']:.4f}  "
        f"CI95=[{row['ci95_low']:.4f}, "
        f"{row['ci95_high']:.4f}]"
    )

print("\nCONFIDENCE RULES — INDEPENDENT 4H")
for row in independent_results:
    if row["accuracy"] is None:
        continue

    print(
        f"{row['name']:<30} "
        f"rows={row['rows']:>4}  "
        f"coverage={row['coverage']:.3f}  "
        f"accuracy={row['accuracy']:.4f}  "
        f"CI95=[{row['ci95_low']:.4f}, "
        f"{row['ci95_high']:.4f}]"
    )

print("\nSESSIONS")
for session, metrics in session_results.items():
    print(
        f"{session:<18} "
        f"rows={metrics['rows']:>4}  "
        f"accuracy={metrics['accuracy']:.4f}"
    )

print("\nSIGNAL DIRECTION")
for direction, metrics in direction_results.items():
    print(
        f"{direction:<8} "
        f"rows={metrics['rows']:>4}  "
        f"accuracy={metrics['accuracy']:.4f}"
    )

print("\nOUTPUTS")
print(report_path)
print(matrix_path)
print(confidence_path)
print(independent_confidence_path)
PY

echo
echo "=========================================="
echo "LiqHeat V6 başlıyor"
echo "=========================================="

python src/analyze_edge_matrix_v6.py \
  2>&1 | tee logs/analyze_edge_matrix_v6.log

echo
echo "=========================================="
echo "V6 tamamlandı"
echo "=========================================="
