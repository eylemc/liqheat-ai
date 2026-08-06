from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

SYMBOL = os.getenv("LIQHEAT_SYMBOL", "BTCUSDT")
TIMEFRAME = os.getenv("LIQHEAT_TIMEFRAME", "1h")
DAYS = int(os.getenv("LIQHEAT_DAYS", "14"))
FUTURE_HOURS = int(os.getenv("LIQHEAT_FUTURE_HOURS", "4"))

DATA_PATH = Path(
    "data/processed"
) / (
    f"{SYMBOL.lower()}_{TIMEFRAME}_{DAYS}d_"
    f"topology_v2.parquet"
)

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "liquidation_count",

    "total_longs",
    "total_shorts",
    "total_aggregated",
    "aggregated_imbalance",
    "max_volume",

    "upper_volume",
    "lower_volume",
    "upper_lower_ratio",
    "topology_imbalance",

    "upper_pool_volume",
    "upper_pool_distance",
    "lower_pool_volume",
    "lower_pool_distance",
    "pool_volume_ratio",
    "pool_distance_ratio",

    "upper_0_0_5",
    "upper_0_5_1",
    "upper_1_2",
    "upper_2_3",

    "lower_0_0_5",
    "lower_0_5_1",
    "lower_1_2",
    "lower_2_3",

    "near_topology_imbalance",
    "weighted_center_all",
    "weighted_center_above",
    "weighted_center_below",

    "nonzero_price_levels",
    "top1_level_share",
    "top3_level_share",
    "top5_level_share",

    "return_5m",
    "return_15m",
    "return_30m",
    "return_60m",

    "topology_change_5m",
    "topology_change_15m",
    "near_topology_change_15m",

    "realized_volatility_30m",
    "realized_volatility_60m",
]

TARGET = "target_class"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset bulunamadı: {DATA_PATH}")

df = pd.read_parquet(DATA_PATH)
df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True)
df = df.sort_values("logged_at")
df = df.replace([np.inf, -np.inf], np.nan)

model_df = df[
    ["logged_at", TARGET, "target_name"] + FEATURES
].copy()

model_df = model_df.dropna(subset=[TARGET])
model_df = model_df.dropna(subset=FEATURES)
model_df[TARGET] = model_df[TARGET].astype(int)

if len(model_df) < 1500:
    raise RuntimeError(
        f"Temiz eğitim verisi yetersiz: {len(model_df)}"
    )

# Kronolojik ayrım:
# %70 train, %15 validation, %15 test.
n = len(model_df)
train_boundary = int(n * 0.70)
validation_boundary = int(n * 0.85)

train_end_time = model_df.iloc[
    train_boundary - 1
]["logged_at"]

validation_start_time = (
    train_end_time + pd.Timedelta(hours=FUTURE_HOURS)
)

validation_nominal_end = model_df.iloc[
    validation_boundary - 1
]["logged_at"]

test_start_time = (
    validation_nominal_end
    + pd.Timedelta(hours=FUTURE_HOURS)
)

train = model_df[
    model_df["logged_at"] <= train_end_time
].copy()

validation = model_df[
    (model_df["logged_at"] >= validation_start_time)
    & (
        model_df["logged_at"]
        <= validation_nominal_end
    )
].copy()

test = model_df[
    model_df["logged_at"] >= test_start_time
].copy()

if min(len(train), len(validation), len(test)) < 100:
    raise RuntimeError(
        "Embargo sonrası train/validation/test bölümlerinden "
        "biri çok küçük kaldı"
    )

X_train = train[FEATURES]
y_train = train[TARGET]

X_validation = validation[FEATURES]
y_validation = validation[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

counts = y_train.value_counts()
class_weights = {
    class_id: len(y_train) / (3 * count)
    for class_id, count in counts.items()
}

sample_weight = y_train.map(class_weights).to_numpy()

print("=== V2 DATA SPLIT ===")
print("Train:", len(train), train["logged_at"].min(), "→", train["logged_at"].max())
print("Validation:", len(validation), validation["logged_at"].min(), "→", validation["logged_at"].max())
print("Test:", len(test), test["logged_at"].min(), "→", test["logged_at"].max())
print()
print("Train classes:")
print(y_train.value_counts().sort_index())
print("Validation classes:")
print(y_validation.value_counts().sort_index())
print("Test classes:")
print(y_test.value_counts().sort_index())

model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_estimators=1200,
    max_depth=6,
    learning_rate=0.025,
    min_child_weight=10,
    subsample=0.80,
    colsample_bytree=0.80,
    reg_alpha=0.10,
    reg_lambda=2.0,
    tree_method="hist",
    device="cuda",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=75,
)

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weight,
    eval_set=[
        (X_train, y_train),
        (X_validation, y_validation),
    ],
    verbose=100,
)

# DMatrix kullanarak cihaz uyumsuzluğu uyarısını önle.
test_matrix = xgb.DMatrix(
    X_test,
    feature_names=FEATURES,
)

probabilities = model.get_booster().predict(test_matrix)
predictions = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions,
)
macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
)

majority_class = int(y_train.mode().iloc[0])
baseline_predictions = np.full(
    len(y_test),
    majority_class,
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions,
)
baseline_balanced_accuracy = balanced_accuracy_score(
    y_test,
    baseline_predictions,
)
baseline_macro_f1 = f1_score(
    y_test,
    baseline_predictions,
    average="macro",
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1, 2],
)

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1, 2],
    target_names=["LOWER", "NEUTRAL", "UPPER"],
    output_dict=True,
    zero_division=0,
)

importance = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }
).sort_values(
    "importance",
    ascending=False,
)

predictions_df = pd.DataFrame(
    {
        "logged_at": test["logged_at"].to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": predictions,
        "prob_lower": probabilities[:, 0],
        "prob_neutral": probabilities[:, 1],
        "prob_upper": probabilities[:, 2],
    }
)

metrics = {
    "rows_total": int(len(model_df)),
    "train_rows": int(len(train)),
    "validation_rows": int(len(validation)),
    "test_rows": int(len(test)),
    "best_iteration": int(model.best_iteration),

    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "macro_f1": float(macro_f1),

    "baseline_majority_class": majority_class,
    "baseline_accuracy": float(baseline_accuracy),
    "baseline_balanced_accuracy": float(
        baseline_balanced_accuracy
    ),
    "baseline_macro_f1": float(baseline_macro_f1),

    "beats_baseline_accuracy": bool(
        accuracy > baseline_accuracy
    ),
    "beats_baseline_balanced_accuracy": bool(
        balanced_accuracy
        > baseline_balanced_accuracy
    ),
    "beats_baseline_macro_f1": bool(
        macro_f1 > baseline_macro_f1
    ),

    "confusion_matrix": matrix.tolist(),
    "classification_report": report,

    "test_start": str(test["logged_at"].min()),
    "test_end": str(test["logged_at"].max()),
}

model_path = MODEL_DIR / "xgboost_topology_v2.json"
joblib_path = MODEL_DIR / "xgboost_topology_v2.joblib"
metrics_path = REPORT_DIR / "topology_v2_metrics.json"
importance_path = REPORT_DIR / "topology_v2_importance.csv"
predictions_path = REPORT_DIR / "topology_v2_predictions.parquet"

model.save_model(model_path)

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "class_names": {
            0: "LOWER",
            1: "NEUTRAL",
            2: "UPPER",
        },
    },
    joblib_path,
)

metrics_path.write_text(
    json.dumps(
        metrics,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

importance.to_csv(importance_path, index=False)

predictions_df.to_parquet(
    predictions_path,
    index=False,
    compression="zstd",
)

print("\n=== V2 TEST METRICS ===")
print("Accuracy:", round(accuracy, 4))
print("Balanced accuracy:", round(balanced_accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print()
print("Baseline accuracy:", round(baseline_accuracy, 4))
print(
    "Baseline balanced accuracy:",
    round(baseline_balanced_accuracy, 4),
)
print("Baseline macro F1:", round(baseline_macro_f1, 4))

print("\n=== CONFUSION MATRIX ===")
print("Rows=actual, columns=predicted")
print("        LOWER NEUTRAL UPPER")
for name, row in zip(
    ["LOWER  ", "NEUTRAL", "UPPER  "],
    matrix,
):
    print(name, row)

print("\n=== CLASSIFICATION REPORT ===")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=["LOWER", "NEUTRAL", "UPPER"],
        zero_division=0,
    )
)

print("\n=== TOP FEATURES ===")
print(importance.head(20).to_string(index=False))

print("\n=== OUTPUT FILES ===")
print(model_path)
print(metrics_path)
print(importance_path)
print(predictions_path)
