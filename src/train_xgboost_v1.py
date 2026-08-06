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
