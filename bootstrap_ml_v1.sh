#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/liqheat-ai"
source .venv/bin/activate

mkdir -p src data/processed models reports logs

python -m pip install --quiet \
  python-dotenv \
  supabase \
  pandas \
  pyarrow \
  scikit-learn \
  xgboost \
  joblib

cat > src/export_features_v1.py <<'PY'
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
PY

cat > src/train_xgboost_v1.py <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

SYMBOL = os.getenv("LIQHEAT_SYMBOL", "BTCUSDT")
TIMEFRAME = os.getenv("LIQHEAT_TIMEFRAME", "1h")
DAYS = int(os.getenv("LIQHEAT_DAYS", "7"))

DATA_PATH = Path(
    f"data/processed/"
    f"{SYMBOL.lower()}_{TIMEFRAME}_{DAYS}d_features_v1.parquet"
)

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "current_price",
    "liquidation_count",
    "price_range_pct",
    "price_position_in_range",
    "total_longs",
    "total_shorts",
    "total_liquidations",
    "long_short_ratio",
    "long_share",
    "short_share",
    "liq_imbalance",
    "max_volume",
    "max_volume_share",
    "return_5m",
    "return_15m",
    "return_60m",
    "longs_change_5m",
    "shorts_change_5m",
    "imbalance_change_5m",
    "imbalance_change_15m",
    "price_volatility_15m",
    "price_volatility_60m",
]

TARGET = "future_return_4h"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset bulunamadı: {DATA_PATH}")

df = pd.read_parquet(DATA_PATH)
df = df.sort_values("logged_at")

model_df = df[["logged_at", TARGET] + FEATURES].copy()

# Sonsuz değerleri ve aşırı pct_change patlamalarını temizle.
model_df = model_df.replace([np.inf, -np.inf], np.nan)

for column in [
    "longs_change_5m",
    "shorts_change_5m",
]:
    model_df[column] = model_df[column].clip(-10, 10)

model_df = model_df.dropna(subset=[TARGET])
model_df = model_df.dropna(subset=FEATURES)

if len(model_df) < 1000:
    raise RuntimeError(
        f"Eğitim için yetersiz temiz satır: {len(model_df)}"
    )

# Finansal zaman serisinde rastgele shuffle yok.
# İlk %70 train, sonraki %15 validation, son %15 test.
train_end = int(len(model_df) * 0.70)
validation_end = int(len(model_df) * 0.85)

train = model_df.iloc[:train_end]
validation = model_df.iloc[train_end:validation_end]
test = model_df.iloc[validation_end:]

X_train = train[FEATURES]
y_train = train[TARGET]

X_validation = validation[FEATURES]
y_validation = validation[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

print("=== DATA SPLIT ===")
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))
print("Train period:", train["logged_at"].min(), "→", train["logged_at"].max())
print("Test period:", test["logged_at"].min(), "→", test["logged_at"].max())

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1200,
    max_depth=7,
    learning_rate=0.025,
    min_child_weight=8,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=1.5,
    tree_method="hist",
    device="cuda",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=75,
)

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_train, y_train),
        (X_validation, y_validation),
    ],
    verbose=100,
)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

actual_direction = np.sign(y_test.to_numpy())
predicted_direction = np.sign(predictions)

directional_accuracy = float(
    np.mean(actual_direction == predicted_direction)
)

# Çok küçük hareketleri nötr kabul eden daha gerçekçi yön metriği.
threshold = 0.001

actual_threshold_direction = np.where(
    y_test.to_numpy() > threshold,
    1,
    np.where(y_test.to_numpy() < -threshold, -1, 0),
)

predicted_threshold_direction = np.where(
    predictions > threshold,
    1,
    np.where(predictions < -threshold, -1, 0),
)

threshold_directional_accuracy = float(
    np.mean(
        actual_threshold_direction
        == predicted_threshold_direction
    )
)

baseline_zero = np.zeros_like(y_test.to_numpy())
baseline_mae = mean_absolute_error(y_test, baseline_zero)
baseline_rmse = mean_squared_error(y_test, baseline_zero) ** 0.5

importance = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }
).sort_values("importance", ascending=False)

predictions_df = pd.DataFrame(
    {
        "logged_at": test["logged_at"].to_numpy(),
        "actual_return_4h": y_test.to_numpy(),
        "predicted_return_4h": predictions,
        "actual_direction": actual_threshold_direction,
        "predicted_direction": predicted_threshold_direction,
    }
)

model_path = MODEL_DIR / "xgboost_btc_1h_4h_v1.json"
joblib_path = MODEL_DIR / "xgboost_btc_1h_4h_v1.joblib"
report_path = REPORT_DIR / "xgboost_btc_1h_4h_v1_metrics.json"
importance_path = REPORT_DIR / "xgboost_btc_1h_4h_v1_importance.csv"
predictions_path = REPORT_DIR / "xgboost_btc_1h_4h_v1_predictions.parquet"

model.save_model(model_path)
joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "threshold": threshold,
    },
    joblib_path,
)

importance.to_csv(importance_path, index=False)
predictions_df.to_parquet(
    predictions_path,
    index=False,
    compression="zstd",
)

metrics = {
    "rows_total": int(len(model_df)),
    "train_rows": int(len(train)),
    "validation_rows": int(len(validation)),
    "test_rows": int(len(test)),
    "best_iteration": int(model.best_iteration),
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "directional_accuracy": directional_accuracy,
    "threshold_directional_accuracy": threshold_directional_accuracy,
    "baseline_zero_mae": float(baseline_mae),
    "baseline_zero_rmse": float(baseline_rmse),
    "beats_baseline_mae": bool(mae < baseline_mae),
    "beats_baseline_rmse": bool(rmse < baseline_rmse),
    "test_start": str(test["logged_at"].min()),
    "test_end": str(test["logged_at"].max()),
}

report_path.write_text(
    json.dumps(metrics, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n=== TEST METRICS ===")
for key_name, value in metrics.items():
    print(f"{key_name}: {value}")

print("\n=== TOP FEATURES ===")
print(importance.head(15).to_string(index=False))

print("\n=== OUTPUT FILES ===")
print(model_path)
print(joblib_path)
print(report_path)
print(importance_path)
print(predictions_path)
PY

echo
echo "=========================================="
echo "1/2 Feature dataset hazırlanıyor"
echo "=========================================="
python src/export_features_v1.py 2>&1 | tee logs/export_features_v1.log

echo
echo "=========================================="
echo "2/2 XGBoost modeli eğitiliyor"
echo "=========================================="
python src/train_xgboost_v1.py 2>&1 | tee logs/train_xgboost_v1.log

echo
echo "=========================================="
echo "LiqHeat ML v1 tamamlandı"
echo "=========================================="
echo
echo "Metrikler:"
cat reports/xgboost_btc_1h_4h_v1_metrics.json

echo
echo "En önemli feature'lar:"
head -16 reports/xgboost_btc_1h_4h_v1_importance.csv
