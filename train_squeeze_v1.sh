#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source .venv/bin/activate

python <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool

MODEL_DIR = Path("models/squeeze_v1")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = Path(
    "data/research/topology_v2_squeeze_events/"
    "squeeze_event_dataset.parquet"
)

CATEGORICAL_FEATURES = [
    "symbol",
    "nearest_side",
]

NUMERIC_FEATURES = [
    "current_price",
    "has_upper_level",
    "has_lower_level",
    "has_topology",
    "nearest_side_code",
    "upper_distance_pct",
    "lower_distance_pct",
    "distance_advantage",
    "signed_distance_edge",
    "log1p_upper_distance_pct",
    "log1p_lower_distance_pct",
    "log1p_distance_advantage",
    "upper_pool_volume",
    "lower_pool_volume",
    "nearest_pool_volume",
    "farther_pool_volume",
    "upper_total_volume",
    "lower_total_volume",
    "upper_active_levels",
    "lower_active_levels",
    "log1p_upper_pool_volume",
    "log1p_lower_pool_volume",
    "log1p_nearest_pool_volume",
    "log1p_farther_pool_volume",
    "log1p_upper_total_volume",
    "log1p_lower_total_volume",
    "log1p_upper_active_levels",
    "log1p_lower_active_levels",
    "pool_volume_ratio",
    "log1p_pool_volume_ratio",
    "distance_pressure_ratio",
    "log1p_distance_pressure_ratio",
    "topology_imbalance",
    "total_volume_imbalance_check",
    "active_level_difference",
    "active_level_total",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend_utc",
]

FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)

print("=" * 80)
print("Loading dataset...")
print("=" * 80)

df = pd.read_parquet(DATASET_PATH)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

X = df[FEATURE_COLUMNS].copy()
y = df["target_event"]

for col in CATEGORICAL_FEATURES:
    X[col] = (
        X[col]
        .astype("string")
        .fillna("<MISSING>")
        .astype(str)
    )

for col in NUMERIC_FEATURES:
    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

cat_indices = [
    FEATURE_COLUMNS.index(c)
    for c in CATEGORICAL_FEATURES
]

pool = Pool(
    X,
    label=y,
    cat_features=cat_indices,
    feature_names=FEATURE_COLUMNS,
)

print("=" * 80)
print("Training model...")
print("=" * 80)

model = CatBoostClassifier(
    iterations=1200,
    depth=7,
    learning_rate=0.04,
    loss_function="MultiClass",
    eval_metric="TotalF1:average=Macro",
    auto_class_weights="Balanced",
    random_seed=42,
    random_strength=1.0,
    l2_leaf_reg=7.0,
    bootstrap_type="Bayesian",
    bagging_temperature=1.0,
    verbose=100,
    thread_count=-1,
)

model.fit(pool)

print("=" * 80)
print("Saving model...")
print("=" * 80)

model.save_model(
    str(MODEL_DIR / "model.cbm")
)

with open(
    MODEL_DIR / "features.json",
    "w"
) as f:
    json.dump(
        FEATURE_COLUMNS,
        f,
        indent=2
    )

metadata = {
    "version": "squeeze-v1",
    "rows": int(len(df)),
    "feature_count": len(FEATURE_COLUMNS),
    "classes": [-1, 0, 1],
}

with open(
    MODEL_DIR / "metadata.json",
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=2
    )

print("=" * 80)
print("Calculating thresholds...")
print("=" * 80)

proba = model.predict_proba(pool)

classes = list(model.classes_)

event_cols = [
    i
    for i, cls in enumerate(classes)
    if cls != 0
]

event_probability = (
    proba[:, event_cols]
    .sum(axis=1)
)

thresholds = {
    "watch": float(np.quantile(event_probability, 0.95)),
    "alert": float(np.quantile(event_probability, 0.99)),
    "critical": float(np.quantile(event_probability, 0.999)),
}

with open(
    MODEL_DIR / "thresholds.json",
    "w"
) as f:
    json.dump(
        thresholds,
        f,
        indent=2
    )

print()
print("Thresholds:")
print(json.dumps(
    thresholds,
    indent=2
))

print()
print("Artifacts:")
for path in sorted(MODEL_DIR.glob("*")):
    print(" -", path)

print()
print("DONE")
PY
