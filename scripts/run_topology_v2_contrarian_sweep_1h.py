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
    precision_score,
    recall_score,
    roc_auc_score,
)


FEATURE_PATH = Path(
    "data/features/liq_topology_v2_ml_features.parquet"
)

LABEL_PATH = Path(
    "data/features/liq_topology_v2_sweep_labels.parquet"
)

OUTPUT_DIR = Path(
    "data/models/topology_v2_contrarian_sweep_1h"
)

SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.parquet"

SWEEP_COLUMN = "sweep_code_1h"

EMBARGO = pd.Timedelta(hours=4)

MAX_TRAIN_ROWS = 1_000_000
MAX_VALIDATION_ROWS = 300_000
MAX_TEST_ROWS = 300_000

RANDOM_STATE = 42


BASE_CATEGORICAL = [
    "symbol",
    "timeframe",
    "nearest_side",
]

DISTANCE_FEATURES = [
    "nearest_side_code",
    "upper_distance_pct",
    "lower_distance_pct",
    "distance_advantage",
    "signed_distance_edge",
    "log1p_upper_distance_pct",
    "log1p_lower_distance_pct",
    "log1p_distance_advantage",
]

PURE_VOLUME_FEATURES = [
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
]

STRUCTURE_FEATURES = [
    "pool_volume_ratio",
    "log1p_pool_volume_ratio",

    "topology_imbalance",
    "total_volume_imbalance_check",

    "active_level_difference",
    "active_level_total",
]

# Bu feature mesafe içerdiği için pure-volume grubuna konmadı.
PRESSURE_FEATURES = [
    "distance_pressure_ratio",
    "log1p_distance_pressure_ratio",
]

CALENDAR_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend_utc",
]

EXPERIMENTS = {
    "distance_only": {
        "categorical": BASE_CATEGORICAL,
        "numeric": DISTANCE_FEATURES,
    },
    "pure_volume": {
        "categorical": BASE_CATEGORICAL,
        "numeric": PURE_VOLUME_FEATURES,
    },
    "volume_structure": {
        "categorical": BASE_CATEGORICAL,
        "numeric": (
            PURE_VOLUME_FEATURES
            + STRUCTURE_FEATURES
        ),
    },
    "full_combined": {
        "categorical": BASE_CATEGORICAL,
        "numeric": (
            DISTANCE_FEATURES
            + PURE_VOLUME_FEATURES
            + STRUCTURE_FEATURES
            + PRESSURE_FEATURES
            + CALENDAR_FEATURES
        ),
    },
}


def all_feature_columns() -> list[str]:
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
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
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
    print(f"Feature count: {len(feature_columns)}")

    X_train = prepare_features(
        train_df,
        categorical,
        numeric,
    )
    y_train = train_df["target_contrarian"]

    X_validation = prepare_features(
        validation_df,
        categorical,
        numeric,
    )
    y_validation = validation_df[
        "target_contrarian"
    ]

    X_test = prepare_features(
        test_df,
        categorical,
        numeric,
    )
    y_test = test_df["target_contrarian"]

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
        iterations=1200,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        random_strength=1.0,
        l2_leaf_reg=5.0,
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        od_type="Iter",
        od_wait=120,
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
            "importance": model.get_feature_importance(
                train_pool,
                type="FeatureImportance",
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    model_path = OUTPUT_DIR / f"{name}.cbm"
    importance_path = (
        OUTPUT_DIR / f"{name}_feature_importance.csv"
    )

    model.save_model(str(model_path))

    importance.to_csv(
        importance_path,
        index=False,
    )

    print()
    print(f"Best iteration : {model.get_best_iteration()}")
    print(f"Training time  : {training_seconds:.1f}s")

    print()
    print("Validation:")
    print(
        f"  balanced={validation_metrics['balanced_accuracy']:.4f}  "
        f"auc={validation_metrics['roc_auc']:.4f}  "
        f"f1={validation_metrics['f1']:.4f}"
    )

    print()
    print("Test:")
    print(
        f"  accuracy={test_metrics['accuracy']:.4f}  "
        f"balanced={test_metrics['balanced_accuracy']:.4f}  "
        f"auc={test_metrics['roc_auc']:.4f}  "
        f"precision={test_metrics['precision']:.4f}  "
        f"recall={test_metrics['recall']:.4f}  "
        f"f1={test_metrics['f1']:.4f}  "
        f"log_loss={test_metrics['log_loss']:.4f}"
    )

    print("Confusion matrix [NORMAL, CONTRARIAN]:")
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
        "test_predictions": test_predictions,
        "test_probabilities": test_probabilities,
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
    print("TOPOLOGY V2 — CONTRARIAN SWEEP 1H")
    print("=" * 80)

    selected_columns = list(
        dict.fromkeys(
            [
                "id",
                "logged_at",
                "symbol",
                "timeframe",
                "nearest_side",
            ]
            + all_feature_columns()
        )
    )

    print("Loading features...")

    features = pd.read_parquet(
        FEATURE_PATH,
        columns=selected_columns,
    )

    print("Loading sweep labels...")

    labels = pd.read_parquet(
        LABEL_PATH,
        columns=[
            "id",
            SWEEP_COLUMN,
        ],
    )

    if features["id"].duplicated().any():
        raise ValueError(
            "Duplicate IDs in feature file."
        )

    if labels["id"].duplicated().any():
        raise ValueError(
            "Duplicate IDs in label file."
        )

    df = features.merge(
        labels,
        on="id",
        how="inner",
        validate="one_to_one",
    )

    # Keep only rows with a clear nearest side and clear first sweep.
    df = df.loc[
        df["nearest_side"].isin(
            ["UPPER", "LOWER"]
        )
        & df[SWEEP_COLUMN].isin([-1, 1])
    ].copy()

    # Contrarian means the farther side was swept first.
    df["target_contrarian"] = np.where(
        (
            df["nearest_side"].eq("UPPER")
            & df[SWEEP_COLUMN].eq(-1)
        )
        |
        (
            df["nearest_side"].eq("LOWER")
            & df[SWEEP_COLUMN].eq(1)
        ),
        1,
        0,
    ).astype("int8")

    df = df.sort_values(
        ["logged_at", "id"],
        kind="mergesort",
    ).reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            "No usable contrarian rows remain."
        )

    print()
    print(f"Usable rows: {len(df):,}")
    print("Target distribution:")
    print(
        df["target_contrarian"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Target distribution (%):")
    print(
        (
            df["target_contrarian"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).to_string()
    )

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

    print()
    print("Sample sizes:")
    print(f"  Train      : {len(train_sample):,}")
    print(f"  Validation : {len(validation_sample):,}")
    print(f"  Test       : {len(test_sample):,}")

    y_train = train_sample["target_contrarian"]
    y_test = test_sample["target_contrarian"]

    majority_class = int(
        y_train.value_counts().idxmax()
    )

    majority_predictions = np.full(
        len(y_test),
        majority_class,
        dtype=np.int8,
    )

    majority_metrics = evaluate(
        y_test,
        majority_predictions,
    )

    print()
    print("=" * 80)
    print("MAJORITY BASELINE")
    print("=" * 80)
    print(f"Majority class     : {majority_class}")
    print(
        f"Accuracy           : "
        f"{majority_metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced accuracy  : "
        f"{majority_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"F1                 : "
        f"{majority_metrics['f1']:.4f}"
    )

    prediction_output = test_sample[
        [
            "id",
            "logged_at",
            "symbol",
            "timeframe",
            "nearest_side",
            SWEEP_COLUMN,
            "target_contrarian",
        ]
    ].copy()

    results = []

    for name, config in EXPERIMENTS.items():
        result = train_experiment(
            name,
            config,
            train_sample,
            validation_sample,
            test_sample,
        )

        results.append(result)

        prediction_output[
            f"{name}_prediction"
        ] = result["test_predictions"]

        prediction_output[
            f"{name}_probability_contrarian"
        ] = result["test_probabilities"][
            :, 1
        ].astype("float32")

    prediction_output.to_parquet(
        PREDICTIONS_PATH,
        index=False,
        compression="zstd",
    )

    summary_rows = [
        {
            "experiment": "majority_baseline",
            "feature_count": 0,
            "best_iteration": None,
            "validation_balanced_accuracy": None,
            "validation_auc": None,
            "test_accuracy": (
                majority_metrics["accuracy"]
            ),
            "test_balanced_accuracy": (
                majority_metrics[
                    "balanced_accuracy"
                ]
            ),
            "test_auc": None,
            "test_precision": (
                majority_metrics["precision"]
            ),
            "test_recall": (
                majority_metrics["recall"]
            ),
            "test_f1": majority_metrics["f1"],
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
                "validation_auc": (
                    result["validation"]["roc_auc"]
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
                "test_precision": (
                    result["test"]["precision"]
                ),
                "test_recall": (
                    result["test"]["recall"]
                ),
                "test_f1": result["test"]["f1"],
                "test_log_loss": (
                    result["test"]["log_loss"]
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    pure_volume_balanced = float(
        summary.loc[
            summary["experiment"] == "pure_volume",
            "test_balanced_accuracy",
        ].iloc[0]
    )

    distance_balanced = float(
        summary.loc[
            summary["experiment"] == "distance_only",
            "test_balanced_accuracy",
        ].iloc[0]
    )

    full_balanced = float(
        summary.loc[
            summary["experiment"] == "full_combined",
            "test_balanced_accuracy",
        ].iloc[0]
    )

    summary[
        "balanced_gain_vs_distance"
    ] = (
        summary["test_balanced_accuracy"]
        - distance_balanced
    )

    summary[
        "balanced_gain_vs_pure_volume"
    ] = (
        summary["test_balanced_accuracy"]
        - pure_volume_balanced
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    serializable_results = []

    for result in results:
        serializable_results.append(
            {
                key: value
                for key, value in result.items()
                if key not in {
                    "test_predictions",
                    "test_probabilities",
                }
            }
        )

    report = {
        "target": (
            "farther liquidity side swept first "
            "within 1 hour"
        ),
        "usable_rows": int(len(df)),
        "target_counts": {
            str(key): int(value)
            for key, value in (
                df["target_contrarian"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "majority_baseline": majority_metrics,
        "experiments": serializable_results,
        "comparison": {
            "distance_test_balanced": (
                distance_balanced
            ),
            "pure_volume_test_balanced": (
                pure_volume_balanced
            ),
            "full_test_balanced": (
                full_balanced
            ),
            "full_gain_vs_distance": (
                full_balanced
                - distance_balanced
            ),
            "full_gain_vs_pure_volume": (
                full_balanced
                - pure_volume_balanced
            ),
        },
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
    print("CONTRARIAN SWEEP SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))

    print()
    print("Key comparison:")
    print(
        f"  Distance-only balanced : "
        f"{distance_balanced:.4f}"
    )
    print(
        f"  Pure-volume balanced   : "
        f"{pure_volume_balanced:.4f}"
    )
    print(
        f"  Full combined balanced : "
        f"{full_balanced:.4f}"
    )
    print(
        f"  Full gain vs distance  : "
        f"{full_balanced - distance_balanced:+.4f}"
    )

    print()
    print("=" * 80)
    print("CONTRARIAN SWEEP COMPLETE")
    print("=" * 80)
    print(f"Summary     : {SUMMARY_PATH}")
    print(f"Metrics     : {METRICS_PATH}")
    print(f"Predictions : {PREDICTIONS_PATH}")
    print(
        f"Elapsed     : "
        f"{time.time() - started:.1f}s"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
