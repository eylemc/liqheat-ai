#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import pandas as pd

from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_FILE = (
    PROJECT_ROOT
    / "data/features/liq_topology_v2_ml_features.parquet"
)

LABELS_FILE = (
    PROJECT_ROOT
    / "data/features/liq_topology_v2_sweep_labels.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/reports/topology_direction_selective_policy_v2"
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

THRESHOLDS = [
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XAUUSDT",
    "XAGUSDT",
]


def safe_balanced_accuracy(y_true, y_pred):
    try:
        return float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        )
    except Exception:
        return None


def train_model(train_df, valid_df):

    cat_features = [
        i
        for i, c in enumerate(FULL)
        if c in CATEGORICAL
    ]

    model = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="BalancedAccuracy",
        task_type="GPU",
        devices="0",
        random_seed=42,
        verbose=100,
        od_type="Iter",
        od_wait=50,
    )

    model.fit(
        train_df[FULL],
        train_df["target"],
        cat_features=cat_features,
        eval_set=(
            valid_df[FULL],
            valid_df["target"],
        ),
        use_best_model=True,
    )

    return model


def build_side_report(
    df,
    thresholds,
):

    rows = []

    total_rows = len(df)

    for threshold in thresholds:

        #
        # LONG SIDE
        #
        long_part = df[
            df["prob_up"] >= threshold
        ].copy()

        if len(long_part):

            long_part["pred"] = 1

            rows.append(
                {
                    "side": "LONG",
                    "threshold": threshold,
                    "rows": len(long_part),
                    "coverage_pct":
                        100.0
                        * len(long_part)
                        / total_rows,
                    "accuracy":
                        accuracy_score(
                            long_part["target"],
                            long_part["pred"],
                        ),
                    "balanced_accuracy":
                        safe_balanced_accuracy(
                            long_part["target"],
                            long_part["pred"],
                        ),
                }
            )

        #
        # SHORT SIDE
        #
        short_part = df[
            df["prob_up"]
            <= (1.0 - threshold)
        ].copy()

        if len(short_part):

            short_part["pred"] = 0

            rows.append(
                {
                    "side": "SHORT",
                    "threshold": threshold,
                    "rows": len(short_part),
                    "coverage_pct":
                        100.0
                        * len(short_part)
                        / total_rows,
                    "accuracy":
                        accuracy_score(
                            short_part["target"],
                            short_part["pred"],
                        ),
                    "balanced_accuracy":
                        safe_balanced_accuracy(
                            short_part["target"],
                            short_part["pred"],
                        ),
                }
            )

    report = pd.DataFrame(rows)

    if len(report):
        report = report.sort_values(
            ["side", "threshold"]
        )

    return report


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading features...")

    feat = pd.read_parquet(
        FEATURES_FILE,
        columns=[
            "id",
            "logged_at",
            *FULL,
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

    df = feat.merge(
        lbl,
        on="id",
        how="inner",
    )

    df = df[
        (df["sweep_valid_1h"] == 1)
        & (df["sweep_code_1h"].isin([-1, 1]))
    ]

    df["target"] = (
        df["sweep_code_1h"] == 1
    ).astype(int)

    df["logged_at"] = pd.to_datetime(
        df["logged_at"],
        utc=True,
    )

    df = df.sort_values("logged_at")

    df = df.dropna(
        subset=[
            c
            for c in FULL
            if c not in CATEGORICAL
        ]
    )

    n = len(df)

    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    valid_df = df.iloc[train_end:valid_end]
    test_df = df.iloc[valid_end:]

    print("Training GPU model...")

    model = train_model(
        train_df,
        valid_df,
    )

    print("Scoring test set...")

    prob_up = model.predict_proba(
        test_df[FULL]
    )[:, 1]

    results = pd.DataFrame(
        {
            "symbol":
                test_df["symbol"].values,
            "target":
                test_df["target"].values,
            "prob_up":
                prob_up,
        }
    )

    #
    # GLOBAL
    #
    global_report = build_side_report(
        results,
        THRESHOLDS,
    )

    global_report.to_csv(
        OUTPUT_DIR
        / "global_selective_policy_v2.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("GLOBAL")
    print("=" * 80)
    print(global_report.to_string(index=False))

    #
    # SYMBOLS
    #
    for symbol in SYMBOLS:

        sym = results[
            results["symbol"] == symbol
        ]

        if len(sym) == 0:
            continue

        sym_report = build_side_report(
            sym,
            THRESHOLDS,
        )

        sym_report.to_csv(
            OUTPUT_DIR
            / f"{symbol.lower()}_selective_policy_v2.csv",
            index=False,
        )

        print()
        print("=" * 80)
        print(symbol)
        print("=" * 80)
        print(sym_report.to_string(index=False))

    print()
    print("Done")


if __name__ == "__main__":
    main()
