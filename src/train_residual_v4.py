from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

SYMBOL = os.getenv("LIQHEAT_SYMBOL", "BTCUSDT")
TIMEFRAME = os.getenv("LIQHEAT_TIMEFRAME", "1h")
DAYS = int(os.getenv("LIQHEAT_DAYS", "14"))
FUTURE_HOURS = int(os.getenv("LIQHEAT_FUTURE_HOURS", "4"))

# Bağımsızlığı artırmak için her saat bir karar noktası.
DECISION_INTERVAL = os.getenv(
    "LIQHEAT_V4_DECISION_INTERVAL",
    "1h",
)

DATA_PATH = Path(
    f"data/processed/"
    f"{SYMBOL.lower()}_{TIMEFRAME}_{DAYS}d_topology_v2.parquet"
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

    "topology_change_5m",
    "topology_change_15m",
    "near_topology_change_15m",

    # Baseline kararının karakteri
    "nearest_distance",
    "farther_distance",
    "distance_advantage",
    "nearest_pool_volume",
    "farther_pool_volume",
    "nearest_volume_ratio",
]

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"V2 dataset bulunamadı: {DATA_PATH}"
    )

df = pd.read_parquet(DATA_PATH)
df["logged_at"] = pd.to_datetime(
    df["logged_at"],
    utc=True,
)
df = df.sort_values("logged_at")
df = df.replace([np.inf, -np.inf], np.nan)

# NEUTRAL kaldırılır:
# 0 = LOWER, 2 = UPPER
df = df[df["target_class"].isin([0, 2])].copy()
df["actual_direction"] = (
    df["target_class"].astype(int) == 2
).astype(int)

# Baseline:
# Üst havuz daha yakınsa UPPER, alt daha yakınsa LOWER.
df["nearest_prediction"] = np.where(
    df["upper_pool_distance"]
    < df["lower_pool_distance"],
    1,
    0,
)

# Residual hedef:
# 1 = nearest-pool kuralı yanıldı
# 0 = nearest-pool kuralı doğru
df["exception_target"] = (
    df["nearest_prediction"]
    != df["actual_direction"]
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

df["distance_advantage"] = (
    df["farther_distance"]
    - df["nearest_distance"]
)

df["nearest_pool_volume"] = np.where(
    upper_is_nearest,
    df["upper_pool_volume"],
    df["lower_pool_volume"],
)

df["farther_pool_volume"] = np.where(
    upper_is_nearest,
    df["lower_pool_volume"],
    df["upper_pool_volume"],
)

epsilon = 1e-9

df["nearest_volume_ratio"] = (
    df["nearest_pool_volume"] + epsilon
) / (
    df["farther_pool_volume"] + epsilon
)

# Her saat içindeki ilk uygun snapshot.
df["decision_bucket"] = df["logged_at"].dt.floor(
    DECISION_INTERVAL
)

df = (
    df.sort_values("logged_at")
    .drop_duplicates(
        subset=["decision_bucket"],
        keep="first",
    )
    .reset_index(drop=True)
)

required = (
    ["logged_at", "actual_direction",
     "nearest_prediction", "exception_target"]
    + FEATURES
)

model_df = df[required].dropna().copy()

if len(model_df) < 200:
    raise RuntimeError(
        f"Saatlik karar noktası yetersiz: {len(model_df)}"
    )

# Kronolojik %70 / %15 / %15 ve 4 saat embargo.
n = len(model_df)

train_boundary = int(n * 0.70)
validation_boundary = int(n * 0.85)

train_end = model_df.iloc[
    train_boundary - 1
]["logged_at"]

validation_nominal_end = model_df.iloc[
    validation_boundary - 1
]["logged_at"]

validation_start = (
    train_end + pd.Timedelta(hours=FUTURE_HOURS)
)

test_start = (
    validation_nominal_end
    + pd.Timedelta(hours=FUTURE_HOURS)
)

train = model_df[
    model_df["logged_at"] <= train_end
].copy()

validation = model_df[
    (model_df["logged_at"] >= validation_start)
    & (
        model_df["logged_at"]
        <= validation_nominal_end
    )
].copy()

test = model_df[
    model_df["logged_at"] >= test_start
].copy()

if min(len(train), len(validation), len(test)) < 25:
    raise RuntimeError(
        "Embargo sonrasında bölümler çok küçük kaldı"
    )

print("=== V4 RESIDUAL DATA SPLIT ===")
print(
    "Train:",
    len(train),
    train["logged_at"].min(),
    "→",
    train["logged_at"].max(),
)
print(
    "Validation:",
    len(validation),
    validation["logged_at"].min(),
    "→",
    validation["logged_at"].max(),
)
print(
    "Test:",
    len(test),
    test["logged_at"].min(),
    "→",
    test["logged_at"].max(),
)

print("\nNearest-rule error rates:")
print(
    "Train:",
    round(train["exception_target"].mean(), 4),
)
print(
    "Validation:",
    round(validation["exception_target"].mean(), 4),
)
print(
    "Test:",
    round(test["exception_target"].mean(), 4),
)


def direction_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
        "f1_upper": float(
            f1_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(actual, predicted)
        ),
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=[0, 1],
        ).tolist(),
    }


X_train = train[FEATURES]
y_train = train["exception_target"].to_numpy()

X_validation = validation[FEATURES]
y_validation_exception = validation[
    "exception_target"
].to_numpy()

X_test = test[FEATURES]
y_test_exception = test[
    "exception_target"
].to_numpy()

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_pos_weight = (
    negative_count / positive_count
    if positive_count > 0
    else 1.0
)

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.025,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.20,
    reg_lambda=3.0,
    gamma=0.02,
    tree_method="hist",
    device="cuda",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=80,
)

print("\n=== TRAINING RESIDUAL MODEL ===")

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_train, y_train),
        (X_validation, y_validation_exception),
    ],
    verbose=50,
)

validation_matrix = xgb.DMatrix(
    X_validation,
    feature_names=FEATURES,
)

test_matrix = xgb.DMatrix(
    X_test,
    feature_names=FEATURES,
)

validation_exception_probability = (
    model.get_booster().predict(
        validation_matrix
    )
)

test_exception_probability = (
    model.get_booster().predict(
        test_matrix
    )
)

validation_actual_direction = validation[
    "actual_direction"
].to_numpy()

validation_nearest = validation[
    "nearest_prediction"
].to_numpy()

test_actual_direction = test[
    "actual_direction"
].to_numpy()

test_nearest = test[
    "nearest_prediction"
].to_numpy()


def hybrid_prediction(
    nearest: np.ndarray,
    exception_probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    should_flip = (
        exception_probability >= threshold
    )

    return np.where(
        should_flip,
        1 - nearest,
        nearest,
    )


# Threshold yalnızca validation verisiyle seçilir.
best_threshold = 0.50
best_validation_score = (-1.0, -1.0, -1.0)
best_validation_metrics: dict[str, Any] | None = None

for threshold in np.arange(0.30, 0.901, 0.01):
    predictions = hybrid_prediction(
        validation_nearest,
        validation_exception_probability,
        float(threshold),
    )

    metrics = direction_metrics(
        validation_actual_direction,
        predictions,
    )

    flip_rate = float(
        np.mean(
            validation_exception_probability
            >= threshold
        )
    )

    # Öncelik balanced accuracy, sonra MCC,
    # sonra daha az flip ile aynı performans.
    score = (
        metrics["balanced_accuracy"],
        metrics["mcc"],
        -flip_rate,
    )

    if score > best_validation_score:
        best_validation_score = score
        best_threshold = float(threshold)
        best_validation_metrics = metrics

test_hybrid = hybrid_prediction(
    test_nearest,
    test_exception_probability,
    best_threshold,
)

baseline_metrics = direction_metrics(
    test_actual_direction,
    test_nearest,
)

hybrid_metrics = direction_metrics(
    test_actual_direction,
    test_hybrid,
)

exception_predictions = (
    test_exception_probability >= 0.50
).astype(int)

exception_metrics = {
    "accuracy": float(
        accuracy_score(
            y_test_exception,
            exception_predictions,
        )
    ),
    "balanced_accuracy": float(
        balanced_accuracy_score(
            y_test_exception,
            exception_predictions,
        )
    ),
    "precision": float(
        precision_score(
            y_test_exception,
            exception_predictions,
            zero_division=0,
        )
    ),
    "recall": float(
        recall_score(
            y_test_exception,
            exception_predictions,
            zero_division=0,
        )
    ),
    "mcc": float(
        matthews_corrcoef(
            y_test_exception,
            exception_predictions,
        )
    ),
}

try:
    exception_metrics["roc_auc"] = float(
        roc_auc_score(
            y_test_exception,
            test_exception_probability,
        )
    )
except ValueError:
    exception_metrics["roc_auc"] = None

test_flip_mask = (
    test_exception_probability
    >= best_threshold
)

flip_count = int(test_flip_mask.sum())
flip_rate = float(test_flip_mask.mean())

if flip_count:
    flipped_accuracy = float(
        accuracy_score(
            test_actual_direction[test_flip_mask],
            test_hybrid[test_flip_mask],
        )
    )
else:
    flipped_accuracy = None

non_flip_mask = ~test_flip_mask

if non_flip_mask.any():
    non_flipped_accuracy = float(
        accuracy_score(
            test_actual_direction[non_flip_mask],
            test_hybrid[non_flip_mask],
        )
    )
else:
    non_flipped_accuracy = None

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
}).sort_values(
    "importance",
    ascending=False,
)

prediction_output = pd.DataFrame({
    "logged_at": test["logged_at"].to_numpy(),
    "actual_direction": test_actual_direction,
    "nearest_prediction": test_nearest,
    "exception_probability":
        test_exception_probability,
    "hybrid_prediction": test_hybrid,
    "flipped": test_flip_mask,
})

model_path = (
    MODEL_DIR / "xgboost_residual_v4.json"
)
joblib_path = (
    MODEL_DIR / "xgboost_residual_v4.joblib"
)
report_path = (
    REPORT_DIR / "residual_v4_report.json"
)
importance_path = (
    REPORT_DIR / "residual_v4_importance.csv"
)
predictions_path = (
    REPORT_DIR / "residual_v4_predictions.parquet"
)

model.save_model(model_path)

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "exception_threshold": best_threshold,
        "rule": (
            "Nearest pool is default; flip direction "
            "when exception probability exceeds threshold."
        ),
    },
    joblib_path,
)

importance.to_csv(
    importance_path,
    index=False,
)

prediction_output.to_parquet(
    predictions_path,
    index=False,
    compression="zstd",
)

report = {
    "dataset": {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "days": DAYS,
        "decision_interval": DECISION_INTERVAL,
        "rows": int(len(model_df)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "test_start": str(
            test["logged_at"].min()
        ),
        "test_end": str(
            test["logged_at"].max()
        ),
    },
    "model": {
        "best_iteration": int(
            model.best_iteration
        ),
        "exception_threshold": best_threshold,
        "validation_metrics":
            best_validation_metrics,
        "exception_detection_test":
            exception_metrics,
    },
    "direction_results": {
        "nearest_pool_baseline":
            baseline_metrics,
        "residual_hybrid":
            hybrid_metrics,
        "accuracy_improvement": float(
            hybrid_metrics["accuracy"]
            - baseline_metrics["accuracy"]
        ),
        "balanced_accuracy_improvement": float(
            hybrid_metrics["balanced_accuracy"]
            - baseline_metrics["balanced_accuracy"]
        ),
        "mcc_improvement": float(
            hybrid_metrics["mcc"]
            - baseline_metrics["mcc"]
        ),
    },
    "flip_analysis": {
        "test_flip_count": flip_count,
        "test_flip_rate": flip_rate,
        "flipped_cases_accuracy":
            flipped_accuracy,
        "non_flipped_cases_accuracy":
            non_flipped_accuracy,
    },
    "top_features": (
        importance.head(20)
        .to_dict(orient="records")
    ),
}

report_path.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("\n==========================================")
print("V4 FINAL RESULTS")
print("==========================================")

print("\nNearest-pool baseline:")
for key, value in baseline_metrics.items():
    print(key, value)

print("\nResidual hybrid:")
for key, value in hybrid_metrics.items():
    print(key, value)

print("\nImprovements:")
print(
    "Accuracy:",
    round(
        report["direction_results"][
            "accuracy_improvement"
        ],
        4,
    ),
)
print(
    "Balanced accuracy:",
    round(
        report["direction_results"][
            "balanced_accuracy_improvement"
        ],
        4,
    ),
)
print(
    "MCC:",
    round(
        report["direction_results"][
            "mcc_improvement"
        ],
        4,
    ),
)

print("\nException model:")
for key, value in exception_metrics.items():
    print(key, value)

print("\nThreshold / flips:")
print("Threshold:", best_threshold)
print("Test flip count:", flip_count)
print("Test flip rate:", round(flip_rate, 4))
print("Flipped accuracy:", flipped_accuracy)
print(
    "Non-flipped accuracy:",
    non_flipped_accuracy,
)

print("\n=== TOP FEATURES ===")
print(
    importance.head(20).to_string(
        index=False
    )
)

print("\nOutputs:")
print(model_path)
print(report_path)
print(importance_path)
print(predictions_path)
