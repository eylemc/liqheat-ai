#!/usr/bin/env python3

from pathlib import Path

import pandas as pd

from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
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
    / "data/reports/topology_direction_selective_policy_v3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]

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
].copy()

df["target"] = (
    df["sweep_code_1h"] == 1
).astype(int)

df["logged_at"] = pd.to_datetime(
    df["logged_at"],
    utc=True,
)

df = df.sort_values(
    "logged_at"
)

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
    for i, c in enumerate(FULL)
    if c in CATEGORICAL
]

print("Training...")

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

print("Scoring...")

prob = model.predict_proba(
    test_df[FULL]
)

pred = (
    prob[:, 1] >= 0.5
).astype(int)

confidence = prob.max(axis=1)

eval_df = pd.DataFrame(
    {
        "symbol": test_df["symbol"].values,
        "target": test_df["target"].values,
        "pred": pred,
        "confidence": confidence,
    }
)

rows = []

for threshold in THRESHOLDS:

    part = eval_df[
        eval_df["confidence"] >= threshold
    ]

    if len(part) == 0:
        continue

    rows.append(
        {
            "threshold": threshold,
            "rows": len(part),
            "coverage_pct":
                100.0
                * len(part)
                / len(eval_df),

            "accuracy":
                accuracy_score(
                    part["target"],
                    part["pred"],
                ),

            "balanced_accuracy":
                balanced_accuracy_score(
                    part["target"],
                    part["pred"],
                ),

            "macro_f1":
                f1_score(
                    part["target"],
                    part["pred"],
                    average="macro",
                ),
        }
    )

global_report = pd.DataFrame(rows)

print()
print("=" * 80)
print("GLOBAL")
print("=" * 80)
print(global_report.to_string(index=False))

global_report.to_csv(
    OUTPUT_DIR / "global.csv",
    index=False,
)

for symbol in sorted(
    eval_df["symbol"].unique()
):

    sym = eval_df[
        eval_df["symbol"] == symbol
    ]

    symbol_rows = []

    for threshold in THRESHOLDS:

        part = sym[
            sym["confidence"] >= threshold
        ]

        if len(part) == 0:
            continue

        symbol_rows.append(
            {
                "threshold": threshold,
                "rows": len(part),

                "coverage_pct":
                    100.0
                    * len(part)
                    / len(sym),

                "accuracy":
                    accuracy_score(
                        part["target"],
                        part["pred"],
                    ),

                "balanced_accuracy":
                    balanced_accuracy_score(
                        part["target"],
                        part["pred"],
                    ),

                "macro_f1":
                    f1_score(
                        part["target"],
                        part["pred"],
                        average="macro",
                    ),
            }
        )

    report = pd.DataFrame(
        symbol_rows
    )

    report.to_csv(
        OUTPUT_DIR
        / f"{symbol.lower()}.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print(symbol)
    print("=" * 80)
    print(report.to_string(index=False))

print()
print("DONE")
