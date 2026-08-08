#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "liq_topology_v2_ml_features.parquet"
)

LABELS_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "liq_topology_v2_sweep_labels.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "topology_direction_bias_ml_v1"
)

NUMERIC_FEATURES = [
    "upper_distance_pct",
    "lower_distance_pct",
    "distance_advantage",
    "signed_distance_edge",
    "upper_pool_volume",
    "lower_pool_volume",
    "nearest_pool_volume",
    "farther_pool_volume",
    "pool_volume_ratio",
    "distance_pressure_ratio",
    "topology_imbalance",
    "active_level_difference",
    "active_level_total",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

CATEGORICAL_FEATURES = [
    "symbol",
    "timeframe",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro")
        ),
        "log_loss": float(
            log_loss(y_true, y_prob)
        ),
        "rows": int(len(y_true)),
    }


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading features...")
    feat = pd.read_parquet(
        FEATURES_FILE,
        columns=[
            "id",
            "logged_at",
            *ALL_FEATURES,
        ],
    )

    print("Loading labels...")
    lbl = pd.read_parquet(
        LABELS_FILE,
        columns=[
            "id",
            "sweep_valid_1h",
            "sweep_code_1h",
        ],
    )

    print("Joining...")
    df = feat.merge(lbl, on="id", how="inner")

    df = df[df["sweep_valid_1h"] == 1]
    df = df[df["sweep_code_1h"].isin([-1, 1])]

    df["target"] = (
        df["sweep_code_1h"] == 1
    ).astype(int)

    df["logged_at"] = pd.to_datetime(
        df["logged_at"],
        utc=True,
        errors="coerce",
    )

    df = df.sort_values("logged_at")

    before = len(df)

    df = df.dropna(
        subset=NUMERIC_FEATURES
    )

    after = len(df)

    print(
        f"Rows after cleanup: {after:,} "
        f"(dropped {before-after:,})"
    )

    n = len(df)

    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test = df.iloc[valid_end:].copy()

    X_train = train[ALL_FEATURES]
    y_train = train["target"]

    X_valid = valid[ALL_FEATURES]
    y_valid = valid["target"]

    X_test = test[ALL_FEATURES]
    y_test = test["target"]

    cat_idx = [
        ALL_FEATURES.index(c)
        for c in CATEGORICAL_FEATURES
    ]

    print("Training CatBoost...")

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="BalancedAccuracy",
        random_seed=42,
        verbose=100,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_idx,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )

    MODEL_DIR = (
        PROJECT_ROOT
        / "data"
        / "models"
        / "topology_direction_v1"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(
        MODEL_DIR / "model.cbm"
    )

    with open(
        MODEL_DIR / "features.json",
        "w",
    ) as f:
        json.dump(
            {
                "features": ALL_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "target": "sweep_code_1h",
                "positive_class": "UPPER_FIRST",
                "negative_class": "LOWER_FIRST",
            },
            f,
            indent=2,
        )

    print()
    print("Saved model:")
    print(MODEL_DIR / "model.cbm")

    results = {}

    for name, X, y in [
        ("train", X_train, y_train),
        ("validation", X_valid, y_valid),
        ("test", X_test, y_test),
    ]:

        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= 0.5).astype(int)

        results[name] = metrics(
            y,
            pred,
            prob,
        )

    importance = pd.DataFrame(
        {
            "feature": ALL_FEATURES,
            "importance": model.get_feature_importance(),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("===== RESULTS =====")
    print(
        json.dumps(
            results,
            indent=2,
        )
    )

    print()
    print("===== TOP FEATURES =====")
    print(
        importance.head(30).to_string(
            index=False
        )
    )

    print()
    print(f"Reports: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
