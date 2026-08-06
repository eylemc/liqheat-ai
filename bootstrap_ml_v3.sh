#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/liqheat-ai"
source .venv/bin/activate

mkdir -p src models reports logs

cat > src/train_binary_v3.py <<'PY'
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
    classification_report,
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

DATA_PATH = Path(
    f"data/processed/"
    f"{SYMBOL.lower()}_{TIMEFRAME}_{DAYS}d_topology_v2.parquet"
)

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Ana model: heatmap topolojisi + geçmiş fiyat davranışı.
FULL_FEATURES = [
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

# İkinci model: fiyat momentumu olmadan yalnızca LiqHeat topolojisi.
TOPOLOGY_FEATURES = [
    feature for feature in FULL_FEATURES
    if feature not in {
        "return_5m",
        "return_15m",
        "return_30m",
        "return_60m",
        "realized_volatility_30m",
        "realized_volatility_60m",
    }
]

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"V2 dataset bulunamadı: {DATA_PATH}"
    )

df = pd.read_parquet(DATA_PATH)
df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True)
df = df.sort_values("logged_at")
df = df.replace([np.inf, -np.inf], np.nan)

# V2 sınıfları:
# 0 = LOWER
# 1 = NEUTRAL
# 2 = UPPER
#
# V3:
# 0 = LOWER
# 1 = UPPER
df = df[df["target_class"].isin([0, 2])].copy()
df["binary_target"] = (
    df["target_class"].astype(int) == 2
).astype(int)

required = sorted(set(FULL_FEATURES + [
    "logged_at",
    "binary_target",
    "upper_pool_distance",
    "lower_pool_distance",
    "upper_pool_volume",
    "lower_pool_volume",
]))

model_df = df[required].dropna().copy()

if len(model_df) < 2000:
    raise RuntimeError(
        f"Binary eğitim verisi yetersiz: {len(model_df)}"
    )

# Kronolojik split ve 4 saatlik embargo.
n = len(model_df)
train_index = int(n * 0.70)
validation_index = int(n * 0.85)

train_end = model_df.iloc[train_index - 1]["logged_at"]
validation_nominal_end = model_df.iloc[
    validation_index - 1
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
    & (model_df["logged_at"] <= validation_nominal_end)
].copy()

test = model_df[
    model_df["logged_at"] >= test_start
].copy()

if min(len(train), len(validation), len(test)) < 200:
    raise RuntimeError(
        "Embargo sonrası veri bölümlerinden biri çok küçük"
    )

print("=== V3 BINARY DATA SPLIT ===")
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

print("\nTrain classes:")
print(train["binary_target"].value_counts().sort_index())
print("Validation classes:")
print(validation["binary_target"].value_counts().sort_index())
print("Test classes:")
print(test["binary_target"].value_counts().sort_index())


def find_best_threshold(
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.50
    best_metrics: dict[str, float] = {
        "balanced_accuracy": -1.0,
        "f1": -1.0,
        "accuracy": -1.0,
    }

    for threshold in np.arange(0.25, 0.751, 0.005):
        predicted = (
            probabilities >= threshold
        ).astype(int)

        balanced = balanced_accuracy_score(
            actual,
            predicted,
        )
        f1 = f1_score(
            actual,
            predicted,
            zero_division=0,
        )
        accuracy = accuracy_score(
            actual,
            predicted,
        )

        candidate = (
            balanced,
            f1,
            accuracy,
        )
        current = (
            best_metrics["balanced_accuracy"],
            best_metrics["f1"],
            best_metrics["accuracy"],
        )

        if candidate > current:
            best_threshold = float(threshold)
            best_metrics = {
                "balanced_accuracy": float(balanced),
                "f1": float(f1),
                "accuracy": float(accuracy),
            }

    return best_threshold, best_metrics


def evaluate_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
        "f1": float(
            f1_score(actual, predicted, zero_division=0)
        ),
        "precision_upper": float(
            precision_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        "recall_upper": float(
            recall_score(
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

    if probabilities is not None:
        try:
            result["roc_auc"] = float(
                roc_auc_score(actual, probabilities)
            )
        except ValueError:
            result["roc_auc"] = None

    return result


def confidence_analysis(
    actual: np.ndarray,
    probabilities: np.ndarray,
    decision_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    predicted = (
        probabilities >= decision_threshold
    ).astype(int)

    # Karar sınırından uzaklık.
    confidence = np.abs(
        probabilities - decision_threshold
    )

    # 0.05 mesafe yaklaşık p>=threshold+0.05
    # veya p<=threshold-0.05 anlamına gelir.
    for margin in [0.00, 0.05, 0.10, 0.15, 0.20]:
        mask = confidence >= margin
        count = int(mask.sum())

        if count == 0:
            continue

        rows.append({
            "minimum_threshold_margin": margin,
            "signals": count,
            "coverage": float(count / len(actual)),
            "accuracy": float(
                accuracy_score(
                    actual[mask],
                    predicted[mask],
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    actual[mask],
                    predicted[mask],
                )
            ),
        })

    return rows


def train_model(
    name: str,
    features: list[str],
) -> dict[str, Any]:
    X_train = train[features]
    y_train = train["binary_target"].to_numpy()

    X_validation = validation[features]
    y_validation = validation[
        "binary_target"
    ].to_numpy()

    X_test = test[features]
    y_test = test["binary_target"].to_numpy()

    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())

    # Hafif sınıf dengesizliğini eğitim ağırlığıyla dengele.
    positive_weight = (
        negatives / positives
        if positives > 0
        else 1.0
    )

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=1500,
        max_depth=5,
        learning_rate=0.02,
        min_child_weight=12,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_alpha=0.15,
        reg_lambda=2.5,
        gamma=0.01,
        tree_method="hist",
        device="cuda",
        scale_pos_weight=positive_weight,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=100,
    )

    print(f"\n=== TRAINING: {name} ===")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_validation, y_validation),
        ],
        verbose=100,
    )

    validation_matrix = xgb.DMatrix(
        X_validation,
        feature_names=features,
    )
    test_matrix = xgb.DMatrix(
        X_test,
        feature_names=features,
    )

    validation_probabilities = (
        model.get_booster().predict(
            validation_matrix
        )
    )
    test_probabilities = (
        model.get_booster().predict(
            test_matrix
        )
    )

    threshold, validation_threshold_metrics = (
        find_best_threshold(
            y_validation,
            validation_probabilities,
        )
    )

    test_predictions = (
        test_probabilities >= threshold
    ).astype(int)

    metrics = evaluate_predictions(
        y_test,
        test_predictions,
        test_probabilities,
    )

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    model_path = MODEL_DIR / f"xgboost_{name}_v3.json"
    joblib_path = MODEL_DIR / f"xgboost_{name}_v3.joblib"
    importance_path = (
        REPORT_DIR / f"{name}_v3_importance.csv"
    )
    predictions_path = (
        REPORT_DIR / f"{name}_v3_predictions.parquet"
    )

    model.save_model(model_path)

    joblib.dump(
        {
            "model": model,
            "features": features,
            "decision_threshold": threshold,
            "class_names": {
                0: "LOWER",
                1: "UPPER",
            },
        },
        joblib_path,
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    pd.DataFrame({
        "logged_at": test["logged_at"].to_numpy(),
        "actual": y_test,
        "predicted": test_predictions,
        "probability_upper": test_probabilities,
    }).to_parquet(
        predictions_path,
        index=False,
        compression="zstd",
    )

    result = {
        "name": name,
        "features": len(features),
        "best_iteration": int(model.best_iteration),
        "decision_threshold_from_validation": threshold,
        "validation_threshold_metrics":
            validation_threshold_metrics,
        "test_metrics": metrics,
        "confidence_analysis": confidence_analysis(
            y_test,
            test_probabilities,
            threshold,
        ),
        "top_features": (
            importance.head(15)
            .to_dict(orient="records")
        ),
        "model_path": str(model_path),
    }

    print(
        f"\n{name} threshold:",
        round(threshold, 3),
    )
    print(
        f"{name} test accuracy:",
        round(metrics["accuracy"], 4),
    )
    print(
        f"{name} balanced accuracy:",
        round(metrics["balanced_accuracy"], 4),
    )
    print(
        f"{name} ROC AUC:",
        round(metrics["roc_auc"], 4)
        if metrics["roc_auc"] is not None
        else None,
    )
    print(
        f"{name} MCC:",
        round(metrics["mcc"], 4),
    )

    return result


y_test = test["binary_target"].to_numpy()

# Baseline 1: Train çoğunluk sınıfını sürekli tahmin et.
majority_class = int(
    train["binary_target"].mode().iloc[0]
)
majority_predictions = np.full(
    len(test),
    majority_class,
)
majority_metrics = evaluate_predictions(
    y_test,
    majority_predictions,
)

# Baseline 2: En yakın havuz önce vurulur.
nearest_pool_predictions = np.where(
    test["upper_pool_distance"].to_numpy()
    <
    test["lower_pool_distance"].to_numpy(),
    1,
    0,
)
nearest_pool_metrics = evaluate_predictions(
    y_test,
    nearest_pool_predictions,
)

# Baseline 3:
# Uzaklık başına likidasyon gücü daha yüksek olan taraf.
epsilon = 1e-9

upper_pressure = (
    test["upper_pool_volume"].to_numpy()
    /
    (
        test["upper_pool_distance"].to_numpy()
        + epsilon
    )
)
lower_pressure = (
    test["lower_pool_volume"].to_numpy()
    /
    (
        test["lower_pool_distance"].to_numpy()
        + epsilon
    )
)

pressure_predictions = np.where(
    upper_pressure > lower_pressure,
    1,
    0,
)
pressure_metrics = evaluate_predictions(
    y_test,
    pressure_predictions,
)

print("\n=== V3 BASELINES ===")
print("Majority:", majority_metrics)
print("Nearest pool:", nearest_pool_metrics)
print("Volume/distance pressure:", pressure_metrics)

topology_result = train_model(
    "binary_topology_only",
    TOPOLOGY_FEATURES,
)

full_result = train_model(
    "binary_full",
    FULL_FEATURES,
)

best_model = max(
    [topology_result, full_result],
    key=lambda item: (
        item["test_metrics"]["balanced_accuracy"],
        item["test_metrics"]["mcc"],
        item["test_metrics"]["accuracy"],
    ),
)

final_report = {
    "dataset": {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "days": DAYS,
        "rows_after_neutral_removal": int(
            len(model_df)
        ),
        "neutral_rows_removed": int(
            (df.get("target_class", pd.Series()) == 1).sum()
        ) if "target_class" in df else 0,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "test_start": str(test["logged_at"].min()),
        "test_end": str(test["logged_at"].max()),
    },
    "baselines": {
        "majority": majority_metrics,
        "nearest_pool": nearest_pool_metrics,
        "volume_distance_pressure": pressure_metrics,
    },
    "models": {
        "topology_only": topology_result,
        "full": full_result,
    },
    "best_model": best_model["name"],
}

report_path = REPORT_DIR / "binary_v3_report.json"
report_path.write_text(
    json.dumps(
        final_report,
        indent=2,
        ensure_ascii=False,
        default=float,
    ),
    encoding="utf-8",
)

print("\n==========================================")
print("V3 FINAL COMPARISON")
print("==========================================")

comparison = [
    (
        "Majority baseline",
        majority_metrics,
    ),
    (
        "Nearest-pool baseline",
        nearest_pool_metrics,
    ),
    (
        "Volume/distance baseline",
        pressure_metrics,
    ),
    (
        "Topology-only XGBoost",
        topology_result["test_metrics"],
    ),
    (
        "Full XGBoost",
        full_result["test_metrics"],
    ),
]

for name, metrics in comparison:
    print(
        f"{name:28} "
        f"accuracy={metrics['accuracy']:.4f}  "
        f"balanced={metrics['balanced_accuracy']:.4f}  "
        f"MCC={metrics['mcc']:.4f}"
    )

print("\nBest model:", best_model["name"])

print("\n=== BEST MODEL CONFIDENCE ANALYSIS ===")
for row in best_model["confidence_analysis"]:
    print(
        f"margin>={row['minimum_threshold_margin']:.2f}  "
        f"signals={row['signals']:>4}  "
        f"coverage={row['coverage']:.3f}  "
        f"accuracy={row['accuracy']:.4f}  "
        f"balanced={row['balanced_accuracy']:.4f}"
    )

print("\n=== BEST MODEL TOP FEATURES ===")
for row in best_model["top_features"]:
    print(
        f"{row['feature']:<32} "
        f"{row['importance']:.6f}"
    )

print("\nReport:", report_path)
PY

echo
echo "=========================================="
echo "LiqHeat ML V3 — Binary UPPER/LOWER"
echo "=========================================="

python src/train_binary_v3.py \
  2>&1 | tee logs/train_binary_v3.log

echo
echo "=========================================="
echo "V3 tamamlandı"
echo "=========================================="

echo
echo "Kısa rapor:"
python - <<'PY'
import json
from pathlib import Path

report = json.loads(
    Path("reports/binary_v3_report.json").read_text()
)

print("Best model:", report["best_model"])

print("\nBaselines:")
for name, metrics in report["baselines"].items():
    print(
        name,
        "accuracy=",
        round(metrics["accuracy"], 4),
        "balanced=",
        round(metrics["balanced_accuracy"], 4),
        "MCC=",
        round(metrics["mcc"], 4),
    )

print("\nModels:")
for name, result in report["models"].items():
    metrics = result["test_metrics"]
    print(
        name,
        "accuracy=",
        round(metrics["accuracy"], 4),
        "balanced=",
        round(metrics["balanced_accuracy"], 4),
        "AUC=",
        round(metrics["roc_auc"], 4),
        "MCC=",
        round(metrics["mcc"], 4),
    )
PY
