from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


FEATURE_PATH = Path(
    "data/features/liq_topology_v2_ml_features.parquet"
)

LABEL_PATH = Path(
    "data/features/liq_topology_v2_sweep_labels.parquet"
)

OUTPUT_DIR = Path(
    "data/models/topology_v2_sweep_distance_ablation_1h"
)

SUMMARY_PATH = OUTPUT_DIR / "ablation_summary.csv"
METRICS_PATH = OUTPUT_DIR / "ablation_metrics.json"

LABEL_COLUMN = "sweep_code_1h"
VALID_LABELS = [-1, 1]

EMBARGO = pd.Timedelta(hours=4)

MAX_TRAIN_ROWS = 1_000_000
MAX_VALIDATION_ROWS = 300_000
MAX_TEST_ROWS = 300_000

RANDOM_STATE = 42


BASE_CATEGORICAL = [
    "symbol",
    "timeframe",
]

DISTANCE_CATEGORICAL = [
    "nearest_side",
]

DISTANCE_NUMERIC = [
    "nearest_side_code",
    "upper_distance_pct",
    "lower_distance_pct",
    "distance_advantage",
    "signed_distance_edge",
    "log1p_upper_distance_pct",
    "log1p_lower_distance_pct",
    "log1p_distance_advantage",
]

VOLUME_TOPOLOGY_NUMERIC = [
    "topology_imbalance",
    "total_volume_imbalance_check",

    "upper_pool_volume",
    "lower_pool_volume",
    "nearest_pool_volume",
    "farther_pool_volume",

    "pool_volume_ratio",
    "distance_pressure_ratio",

    "upper_active_levels",
    "lower_active_levels",
    "active_level_difference",
    "active_level_total",

    "upper_total_volume",
    "lower_total_volume",

    "log1p_upper_pool_volume",
    "log1p_lower_pool_volume",
    "log1p_nearest_pool_volume",
    "log1p_farther_pool_volume",

    "log1p_pool_volume_ratio",
    "log1p_distance_pressure_ratio",

    "log1p_upper_active_levels",
    "log1p_lower_active_levels",
    "log1p_upper_total_volume",
    "log1p_lower_total_volume",
]

CALENDAR_NUMERIC = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend_utc",
]

EXPERIMENTS = {
    "distance_only": {
        "categorical": (
            BASE_CATEGORICAL
            + DISTANCE_CATEGORICAL
        ),
        "numeric": DISTANCE_NUMERIC,
    },
    "volume_only": {
        "categorical": BASE_CATEGORICAL,
        "numeric": VOLUME_TOPOLOGY_NUMERIC,
    },
    "distance_plus_volume": {
        "categorical": (
            BASE_CATEGORICAL
            + DISTANCE_CATEGORICAL
        ),
        "numeric": (
            DISTANCE_NUMERIC
            + VOLUME_TOPOLOGY_NUMERIC
        ),
    },
    "full_combined": {
        "categorical": (
            BASE_CATEGORICAL
            + DISTANCE_CATEGORICAL
        ),
        "numeric": (
            DISTANCE_NUMERIC
            + VOLUME_TOPOLOGY_NUMERIC
            + CALENDAR_NUMERIC
        ),
    },
}


def all_features() -> list[str]:
    columns = []

    for config in EXPERIMENTS.values():
        columns.extend(config["categorical"])
        columns.extend(config["numeric"])

    return list(dict.fromkeys(columns))


def sample_frame(
    df: pd.DataFrame,
    maximum_rows: int,
    seed: int,
) -> pd.DataFrame:
    if len(df) <= maximum_rows:
        return df.copy()

    return (
        df.sample(
            n=maximum_rows,
            random_state=seed,
            replace=False,
        )
        .sort_values(["logged_at", "id"])
        .reset_index(drop=True)
    )


def prepare_features(
    df: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
) -> pd.DataFrame:
    columns = categorical + numeric
    X = df[columns].copy()

    for column in categorical:
        X[column] = (
            X[column]
            .astype("string")
            .fillna("<MISSING>")
            .astype(str)
        )

    for column in numeric:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    return X


def evaluate(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict:
    result = {
        "rows": int(len(y_true)),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }

    if probabilities is not None:
        result["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities[:, 1],
            )
        )
        result["log_loss"] = float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        )

    return result


def train_experiment(
    name: str,
    config: dict,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    categorical = config["categorical"]
    numeric = config["numeric"]
    feature_columns = categorical + numeric

    print()
    print("=" * 80)
    print(f"EXPERIMENT: {name.upper()}")
    print("=" * 80)
    print(f"Features: {len(feature_columns)}")

    X_train = prepare_features(
        train_df,
        categorical,
        numeric,
    )
    y_train = train_df["target_upper_first"]

    X_validation = prepare_features(
        validation_df,
        categorical,
        numeric,
    )
    y_validation = validation_df[
        "target_upper_first"
    ]

    X_test = prepare_features(
        test_df,
        categorical,
        numeric,
    )
    y_test = test_df["target_upper_first"]

    categorical_indices = [
        feature_columns.index(column)
        for column in categorical
    ]

    train_pool = Pool(
        X_train,
        label=y_train,
        cat_features=categorical_indices,
        feature_names=feature_columns,
    )

    validation_pool = Pool(
        X_validation,
        label=y_validation,
        cat_features=categorical_indices,
        feature_names=feature_columns,
    )

    test_pool = Pool(
        X_test,
        label=y_test,
        cat_features=categorical_indices,
        feature_names=feature_columns,
    )

    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        random_strength=1.0,
        l2_leaf_reg=5.0,
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        od_type="Iter",
        od_wait=100,
        use_best_model=True,
        verbose=100,
        allow_writing_files=False,
        thread_count=-1,
    )

    started = time.time()

    model.fit(
        train_pool,
        eval_set=validation_pool,
    )

    training_seconds = time.time() - started

    validation_probabilities = (
        model.predict_proba(validation_pool)
    )
    test_probabilities = (
        model.predict_proba(test_pool)
    )

    validation_predictions = (
        validation_probabilities[:, 1] >= 0.5
    ).astype("int8")

    test_predictions = (
        test_probabilities[:, 1] >= 0.5
    ).astype("int8")

    validation_metrics = evaluate(
        y_validation,
        validation_predictions,
        validation_probabilities,
    )

    test_metrics = evaluate(
        y_test,
        test_predictions,
        test_probabilities,
    )

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": (
                model.get_feature_importance(
                    train_pool,
                    type="FeatureImportance",
                )
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    model.save_model(
        str(OUTPUT_DIR / f"{name}.cbm")
    )

    importance.to_csv(
        OUTPUT_DIR / f"{name}_feature_importance.csv",
        index=False,
    )

    print()
    print(f"Best iteration: {model.get_best_iteration()}")
    print(
        f"Validation balanced: "
        f"{validation_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Test balanced      : "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Test AUC           : "
        f"{test_metrics['roc_auc']:.4f}"
    )
    print(
        f"Test log loss      : "
        f"{test_metrics['log_loss']:.4f}"
    )
    print("Confusion matrix:")
    print(
        np.array(
            test_metrics["confusion_matrix"]
        )
    )

    print()
    print("Top 15 features:")
    print(
        importance.head(15).to_string(
            index=False
        )
    )

    return {
        "experiment": name,
        "feature_count": len(feature_columns),
        "best_iteration": int(
            model.get_best_iteration()
        ),
        "training_seconds": training_seconds,
        "validation": validation_metrics,
        "test": test_metrics,
        "feature_importance": (
            importance.to_dict(
                orient="records"
            )
        ),
    }


def main() -> int:
    started = time.time()

    for path in [FEATURE_PATH, LABEL_PATH]:
        if not path.exists():
            print(
                f"ERROR: Missing file: {path}",
                file=sys.stderr,
            )
            return 1

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("TOPOLOGY V2 — SWEEP DISTANCE ABLATION")
    print("=" * 80)

    selected_columns = list(
        dict.fromkeys(
            [
                "id",
                "logged_at",
                "nearest_side",
            ]
            + all_features()
        )
    )

    features = pd.read_parquet(
        FEATURE_PATH,
        columns=selected_columns,
    )

    labels = pd.read_parquet(
        LABEL_PATH,
        columns=[
            "id",
            LABEL_COLUMN,
        ],
    )

    df = features.merge(
        labels,
        on="id",
        how="inner",
        validate="one_to_one",
    )

    df = df.loc[
        df[LABEL_COLUMN].isin(VALID_LABELS)
    ].copy()

    df["target_upper_first"] = (
        df[LABEL_COLUMN] == 1
    ).astype("int8")

    df = df.sort_values(
        ["logged_at", "id"],
        kind="mergesort",
    ).reset_index(drop=True)

    train_boundary = df[
        "logged_at"
    ].quantile(0.70)

    validation_boundary = df[
        "logged_at"
    ].quantile(0.85)

    train_end = train_boundary - EMBARGO
    validation_start = (
        train_boundary + EMBARGO
    )
    validation_end = (
        validation_boundary - EMBARGO
    )
    test_start = (
        validation_boundary + EMBARGO
    )

    train_df = df.loc[
        df["logged_at"] <= train_end
    ].copy()

    validation_df = df.loc[
        (df["logged_at"] >= validation_start)
        & (df["logged_at"] <= validation_end)
    ].copy()

    test_df = df.loc[
        df["logged_at"] >= test_start
    ].copy()

    train_sample = sample_frame(
        train_df,
        MAX_TRAIN_ROWS,
        RANDOM_STATE,
    )

    validation_sample = sample_frame(
        validation_df,
        MAX_VALIDATION_ROWS,
        RANDOM_STATE + 1,
    )

    test_sample = sample_frame(
        test_df,
        MAX_TEST_ROWS,
        RANDOM_STATE + 2,
    )

    print(f"Usable rows : {len(df):,}")
    print(f"Train       : {len(train_sample):,}")
    print(
        f"Validation  : "
        f"{len(validation_sample):,}"
    )
    print(f"Test        : {len(test_sample):,}")

    y_test = test_sample["target_upper_first"]

    # Simple deterministic baseline:
    # UPPER nearest => predict UPPER_FIRST.
    # LOWER nearest => predict LOWER_FIRST.
    # TIE => use training majority class.
    majority_class = int(
        train_sample["target_upper_first"]
        .value_counts()
        .idxmax()
    )

    nearest_side_prediction = np.where(
        test_sample["nearest_side"].eq("UPPER"),
        1,
        np.where(
            test_sample["nearest_side"].eq("LOWER"),
            0,
            majority_class,
        ),
    ).astype("int8")

    heuristic_metrics = evaluate(
        y_test,
        nearest_side_prediction,
    )

    print()
    print("=" * 80)
    print("NEAREST-SIDE HEURISTIC")
    print("=" * 80)
    print(
        f"Accuracy          : "
        f"{heuristic_metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced accuracy : "
        f"{heuristic_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"F1                : "
        f"{heuristic_metrics['f1']:.4f}"
    )
    print("Confusion matrix:")
    print(
        np.array(
            heuristic_metrics[
                "confusion_matrix"
            ]
        )
    )

    results = []

    for name, config in EXPERIMENTS.items():
        results.append(
            train_experiment(
                name,
                config,
                train_sample,
                validation_sample,
                test_sample,
            )
        )

    summary_rows = [
        {
            "experiment": "nearest_side_heuristic",
            "feature_count": 1,
            "best_iteration": None,
            "validation_balanced_accuracy": None,
            "test_accuracy": (
                heuristic_metrics["accuracy"]
            ),
            "test_balanced_accuracy": (
                heuristic_metrics[
                    "balanced_accuracy"
                ]
            ),
            "test_auc": None,
            "test_f1": heuristic_metrics["f1"],
            "test_log_loss": None,
        }
    ]

    for result in results:
        summary_rows.append(
            {
                "experiment": result["experiment"],
                "feature_count": result["feature_count"],
                "best_iteration": result["best_iteration"],
                "validation_balanced_accuracy": (
                    result["validation"][
                        "balanced_accuracy"
                    ]
                ),
                "test_accuracy": (
                    result["test"]["accuracy"]
                ),
                "test_balanced_accuracy": (
                    result["test"][
                        "balanced_accuracy"
                    ]
                ),
                "test_auc": (
                    result["test"]["roc_auc"]
                ),
                "test_f1": result["test"]["f1"],
                "test_log_loss": (
                    result["test"]["log_loss"]
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    heuristic_balanced = (
        heuristic_metrics[
            "balanced_accuracy"
        ]
    )

    summary[
        "balanced_gain_vs_nearest_side"
    ] = (
        summary["test_balanced_accuracy"]
        - heuristic_balanced
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    report = {
        "target": "UPPER_FIRST vs LOWER_FIRST",
        "usable_rows": int(len(df)),
        "nearest_side_heuristic": heuristic_metrics,
        "experiments": results,
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("DISTANCE ABLATION SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))

    print()
    print(
        f"Elapsed: "
        f"{time.time() - started:.1f}s"
    )
    print(f"Summary: {SUMMARY_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
