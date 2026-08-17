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
    / "data/reports/topology_direction_confidence_v1"
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


def confidence_report(df: pd.DataFrame):

    bins = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        1.01,
    ]

    labels = [
        "0.50-0.55",
        "0.55-0.60",
        "0.60-0.65",
        "0.65-0.70",
        "0.70-0.75",
        "0.75-0.80",
        "0.80-0.85",
        "0.85+",
    ]

    work = df.copy()

    work["confidence"] = np.maximum(
        work["prob"],
        1.0 - work["prob"],
    )

    work["bucket"] = pd.cut(
        work["confidence"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    rows = []

    for bucket, part in work.groupby("bucket"):

        if len(part) == 0:
            continue

        rows.append(
            {
                "bucket": str(bucket),
                "rows": int(len(part)),
                "coverage_pct": float(
                    100 * len(part) / len(work)
                ),
                "accuracy": float(
                    accuracy_score(
                        part["target"],
                        part["pred"],
                    )
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        part["target"],
                        part["pred"],
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


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

    cat_features = [
        i
        for i, col in enumerate(FULL)
        if col in CATEGORICAL
    ]

    print("Training GPU model...")

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

    prob = model.predict_proba(
        test_df[FULL]
    )[:, 1]

    pred = (prob >= 0.5).astype(int)

    results = pd.DataFrame(
        {
            "symbol": test_df["symbol"].values,
            "target": test_df["target"].values,
            "prob": prob,
            "pred": pred,
        }
    )

    print()
    print("GLOBAL CONFIDENCE")

    global_report = confidence_report(
        results
    )

    print(global_report)

    global_report.to_csv(
        OUTPUT_DIR / "global_confidence.csv",
        index=False,
    )

    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "XAUUSDT",
        "XAGUSDT",
    ]

    symbol_reports = {}

    for symbol in symbols:

        part = results[
            results["symbol"] == symbol
        ]

        rep = confidence_report(part)

        rep.to_csv(
            OUTPUT_DIR
            / f"{symbol.lower()}_confidence.csv",
            index=False,
        )

        symbol_reports[symbol] = (
            rep.to_dict("records")
        )

        print()
        print(symbol)
        print(rep)

    with open(
        OUTPUT_DIR / "confidence_report.json",
        "w",
    ) as f:
        json.dump(
            symbol_reports,
            f,
            indent=2,
        )

    print()
    print("Done")


if __name__ == "__main__":
    main()
