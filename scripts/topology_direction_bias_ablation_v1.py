#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

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
    / "topology_direction_bias_ablation_v1"
)

DISTANCE = [
    "upper_distance_pct",
    "lower_distance_pct",
    "distance_advantage",
    "signed_distance_edge",
]

VOLUME = [
    "upper_pool_volume",
    "lower_pool_volume",
    "nearest_pool_volume",
    "farther_pool_volume",
    "pool_volume_ratio",
    "distance_pressure_ratio",
]

TOPOLOGY = [
    "topology_imbalance",
    "active_level_difference",
    "active_level_total",
]

CALENDAR = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

CATEGORICAL = [
    "symbol",
    "timeframe",
]

FULL = (
    DISTANCE
    + VOLUME
    + TOPOLOGY
    + CALENDAR
    + CATEGORICAL
)

MODELS = {
    "distance_only": DISTANCE + CATEGORICAL,
    "volume_only": VOLUME + CATEGORICAL,
    "topology_only": TOPOLOGY + CATEGORICAL,
    "calendar_only": CALENDAR + CATEGORICAL,
    "full": FULL,
}


def calc_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
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


def train_model(
    train_df,
    valid_df,
    test_df,
    features,
):

    cat_features = [
        i
        for i, col in enumerate(features)
        if col in CATEGORICAL
    ]

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="BalancedAccuracy",
        random_seed=42,
        verbose=False,
    )

    model.fit(
        train_df[features],
        train_df["target"],
        cat_features=cat_features,
        eval_set=(
            valid_df[features],
            valid_df["target"],
        ),
        use_best_model=True,
    )

    prob = model.predict_proba(
        test_df[features]
    )[:, 1]

    pred = (prob >= 0.5).astype(int)

    metrics = calc_metrics(
        test_df["target"],
        pred,
        prob,
    )

    return model, metrics


def symbol_breakdown(
    model,
    test_df,
    features,
):

    rows = []

    for symbol, part in test_df.groupby("symbol"):

        prob = model.predict_proba(
            part[features]
        )[:, 1]

        pred = (prob >= 0.5).astype(int)

        rows.append(
            {
                "symbol": symbol,
                "rows": len(part),
                "balanced_accuracy": balanced_accuracy_score(
                    part["target"],
                    pred,
                ),
                "accuracy": accuracy_score(
                    part["target"],
                    pred,
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "balanced_accuracy",
        ascending=False,
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading features...")

    feature_cols = list(
        {
            "id",
            "logged_at",
            *FULL,
        }
    )

    feat = pd.read_parquet(
        FEATURES_FILE,
        columns=feature_cols,
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

    df = feat.merge(
        lbl,
        on="id",
        how="inner",
    )

    df = df[
        df["sweep_valid_1h"] == 1
    ]

    df = df[
        df["sweep_code_1h"].isin([-1, 1])
    ]

    df["target"] = (
        df["sweep_code_1h"] == 1
    ).astype(int)

    df["logged_at"] = pd.to_datetime(
        df["logged_at"],
        utc=True,
    )

    df = df.sort_values("logged_at")

    before = len(df)

    df = df.dropna(
        subset=[
            c
            for c in FULL
            if c not in CATEGORICAL
        ]
    )

    print(
        f"Rows after cleanup: {len(df):,} "
        f"(dropped {before-len(df):,})"
    )

    n = len(df)

    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    valid_df = df.iloc[train_end:valid_end]
    test_df = df.iloc[valid_end:]

    results = {}

    for name, features in MODELS.items():

        print()
        print("=" * 60)
        print(name)
        print("=" * 60)

        model, metrics = train_model(
            train_df,
            valid_df,
            test_df,
            features,
        )

        results[name] = metrics

        print(
            json.dumps(
                metrics,
                indent=2,
            )
        )

        if name == "full":

            imp = pd.DataFrame(
                {
                    "feature": features,
                    "importance": model.get_feature_importance(),
                }
            ).sort_values(
                "importance",
                ascending=False,
            )

            imp.to_csv(
                OUTPUT_DIR
                / "full_feature_importance.csv",
                index=False,
            )

            symbols = symbol_breakdown(
                model,
                test_df,
                features,
            )

            symbols.to_csv(
                OUTPUT_DIR
                / "symbol_breakdown.csv",
                index=False,
            )

            print()
            print("SYMBOL BREAKDOWN")
            print(
                symbols.to_string(
                    index=False
                )
            )

    with open(
        OUTPUT_DIR / "results.json",
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)

    summary = pd.DataFrame(results).T

    print(
        summary[
            [
                "balanced_accuracy",
                "accuracy",
                "macro_f1",
            ]
        ]
        .sort_values(
            "balanced_accuracy",
            ascending=False,
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
