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
    "data/models/topology_v2_sweep_ablation_1h"
)

SUMMARY_PATH = OUTPUT_DIR / "ablation_summary.csv"
METRICS_PATH = OUTPUT_DIR / "ablation_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.parquet"

LABEL_CODE_COLUMN = "sweep_code_1h"

# LOWER_FIRST=-1, UPPER_FIRST=1
VALID_LABELS = [-1, 1]

EMBARGO = pd.Timedelta(hours=4)

MAX_TRAIN_ROWS = 1_000_000
MAX_VALIDATION_ROWS = 300_000
MAX_TEST_ROWS = 300_000

RANDOM_STATE = 42

BASE_CATEGORICAL_FEATURES = [
    "symbol",
    "timeframe",
]

CALENDAR_NUMERIC_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend_utc",
]

TOPOLOGY_CATEGORICAL_FEATURES = [
    "nearest_side",
]

TOPOLOGY_NUMERIC_FEATURES = [
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

    "topology_imbalance",
    "total_volume_imbalance_check",

    "active_level_difference",
    "active_level_total",

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

EXPERIMENTS = {
    "calendar_only": {
        "categorical": BASE_CATEGORICAL_FEATURES,
        "numeric": CALENDAR_NUMERIC_FEATURES,
    },
    "topology_only": {
        "categorical": (
            BASE_CATEGORICAL_FEATURES
            + TOPOLOGY_CATEGORICAL_FEATURES
        ),
        "numeric": TOPOLOGY_NUMERIC_FEATURES,
    },
    "combined": {
        "categorical": (
            BASE_CATEGORICAL_FEATURES
            + TOPOLOGY_CATEGORICAL_FEATURES
        ),
        "numeric": (
            CALENDAR_NUMERIC_FEATURES
            + TOPOLOGY_NUMERIC_FEATURES
        ),
    },
}

FORBIDDEN_PATTERNS = [
    "future_",
    "forward_return_",
    "direction_",
    "label_valid_",
    "sweep_code_",
    "sweep_label_",
    "hit_seconds_",
]


def json_value(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.generic):
        return value.item()

    if pd.isna(value):
        return None

    return value


def all_feature_columns() -> list[str]:
    columns = []

    for config in EXPERIMENTS.values():
        columns.extend(config["categorical"])
        columns.extend(config["numeric"])

    return list(dict.fromkeys(columns))


def sample_frame(
    df: pd.DataFrame,
    maximum_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if len(df) <= maximum_rows:
        return df.copy()

    return (
        df.sample(
            n=maximum_rows,
            random_state=random_state,
            replace=False,
        )
        .sort_values(["logged_at", "id"])
        .reset_index(drop=True)
    )


def validate_features(df: pd.DataFrame) -> None:
    selected = all_feature_columns()

    missing = [
        column
        for column in selected
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing feature columns: " + ", ".join(missing)
        )

    leaked = [
        column
        for column in selected
        if any(
            pattern in column
            for pattern in FORBIDDEN_PATTERNS
        )
    ]

    if leaked:
        raise ValueError(
            "Potential leakage columns selected: "
            + ", ".join(leaked)
        )


def prepare_features(
    df: pd.DataFrame,
    categorical_features: list[str],
    numeric_features: list[str],
) -> pd.DataFrame:
    feature_columns = categorical_features + numeric_features
    X = df[feature_columns].copy()

    for column in categorical_features:
        X[column] = (
            X[column]
            .astype("string")
            .fillna("<MISSING>")
            .astype(str)
        )

    for column in numeric_features:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    return X


def evaluate(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    positive_probability = probabilities[:, 1]

    return {
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
        "roc_auc": float(
            roc_auc_score(
                y_true,
                positive_probability,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }


def train_experiment(
    name: str,
    config: dict,
    train_sample: pd.DataFrame,
    validation_sample: pd.DataFrame,
    test_sample: pd.DataFrame,
) -> dict:
    categorical_features = config["categorical"]
    numeric_features = config["numeric"]
    feature_columns = categorical_features + numeric_features

    print()
    print("=" * 80)
    print(f"EXPERIMENT: {name.upper()}")
    print("=" * 80)
    print(f"Features: {len(feature_columns)}")

    X_train = prepare_features(
        train_sample,
        categorical_features,
        numeric_features,
    )
    y_train = train_sample["target_upper_first"]

    X_validation = prepare_features(
        validation_sample,
        categorical_features,
        numeric_features,
    )
    y_validation = validation_sample["target_upper_first"]

    X_test = prepare_features(
        test_sample,
        categorical_features,
        numeric_features,
    )
    y_test = test_sample["target_upper_first"]

    categorical_indices = [
        feature_columns.index(column)
        for column in categorical_features
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

    fit_started = time.time()

    model.fit(
        train_pool,
        eval_set=validation_pool,
    )

    fit_elapsed = time.time() - fit_started

    validation_probabilities = model.predict_proba(
        validation_pool
    )

    test_probabilities = model.predict_proba(
        test_pool
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
        OUTPUT_DIR
        / f"{name}_feature_importance.csv"
    )

    model.save_model(str(model_path))
    importance.to_csv(
        importance_path,
        index=False,
    )

    print()
    print(f"Best iteration: {model.get_best_iteration()}")
    print(f"Training time : {fit_elapsed:.1f}s")

    print()
    print("Validation:")
    print(
        f"  accuracy={validation_metrics['accuracy']:.4f}  "
        f"balanced={validation_metrics['balanced_accuracy']:.4f}  "
        f"auc={validation_metrics['roc_auc']:.4f}  "
        f"f1={validation_metrics['f1']:.4f}"
    )

    print()
    print("Test:")
    print(
        f"  accuracy={test_metrics['accuracy']:.4f}  "
        f"balanced={test_metrics['balanced_accuracy']:.4f}  "
        f"auc={test_metrics['roc_auc']:.4f}  "
        f"f1={test_metrics['f1']:.4f}  "
        f"log_loss={test_metrics['log_loss']:.4f}"
    )

    print("Confusion matrix [LOWER_FIRST, UPPER_FIRST]:")
    print(np.array(test_metrics["confusion_matrix"]))

    print()
    print("Top 15 features:")
    print(importance.head(15).to_string(index=False))

    return {
        "name": name,
        "feature_count": len(feature_columns),
        "best_iteration": int(model.get_best_iteration()),
        "training_seconds": fit_elapsed,
        "validation": validation_metrics,
        "test": test_metrics,
        "feature_importance": importance.to_dict(
            orient="records"
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
    print("TOPOLOGY V2 — 1H SWEEP CATBOOST ABLATION")
    print("=" * 80)

    feature_columns = list(
        dict.fromkeys(
            [
                "id",
                "logged_at",
                "symbol",
                "timeframe",
                "current_price",
            ]
            + all_feature_columns()
        )
    )

    print("Loading features...")

    features = pd.read_parquet(
        FEATURE_PATH,
        columns=feature_columns,
    )

    print("Loading sweep labels...")

    labels = pd.read_parquet(
        LABEL_PATH,
        columns=[
            "id",
            LABEL_CODE_COLUMN,
        ],
    )

    if labels["id"].duplicated().any():
        raise ValueError(
            "Duplicate IDs in sweep label file."
        )

    df = features.merge(
        labels,
        on="id",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(features):
        raise RuntimeError(
            f"Merge lost rows: "
            f"{len(features):,} -> {len(df):,}"
        )

    validate_features(df)

    df = df.loc[
        df[LABEL_CODE_COLUMN].isin(VALID_LABELS)
    ].copy()

    # LOWER_FIRST=0, UPPER_FIRST=1
    df["target_upper_first"] = (
        df[LABEL_CODE_COLUMN] == 1
    ).astype("int8")

    df = df.sort_values(
        ["logged_at", "id"],
        kind="mergesort",
    ).reset_index(drop=True)

    print()
    print(f"Usable rows: {len(df):,}")
    print("Target distribution:")
    print(
        df["target_upper_first"]
        .value_counts(normalize=True)
        .sort_index()
        .to_string()
    )

    train_boundary = df["logged_at"].quantile(0.70)
    validation_boundary = df["logged_at"].quantile(0.85)

    train_end = train_boundary - EMBARGO
    validation_start = train_boundary + EMBARGO
    validation_end = validation_boundary - EMBARGO
    test_start = validation_boundary + EMBARGO

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

    prediction_output = test_sample[
        [
            "id",
            "logged_at",
            "symbol",
            "timeframe",
            "target_upper_first",
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
            f"{name}_probability_upper"
        ] = result["test_probabilities"][
            :, 1
        ].astype("float32")

    prediction_output.to_parquet(
        PREDICTIONS_PATH,
        index=False,
        compression="zstd",
    )

    summary_rows = []

    for result in results:
        summary_rows.append(
            {
                "experiment": result["name"],
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
                "test_f1": result["test"]["f1"],
                "test_log_loss": (
                    result["test"]["log_loss"]
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    calendar_balanced = float(
        summary.loc[
            summary["experiment"] == "calendar_only",
            "test_balanced_accuracy",
        ].iloc[0]
    )

    topology_balanced = float(
        summary.loc[
            summary["experiment"] == "topology_only",
            "test_balanced_accuracy",
        ].iloc[0]
    )

    combined_balanced = float(
        summary.loc[
            summary["experiment"] == "combined",
            "test_balanced_accuracy",
        ].iloc[0]
    )

    calendar_auc = float(
        summary.loc[
            summary["experiment"] == "calendar_only",
            "test_auc",
        ].iloc[0]
    )

    combined_auc = float(
        summary.loc[
            summary["experiment"] == "combined",
            "test_auc",
        ].iloc[0]
    )

    summary[
        "balanced_gain_vs_calendar"
    ] = (
        summary["test_balanced_accuracy"]
        - calendar_balanced
    )

    summary[
        "auc_gain_vs_calendar"
    ] = (
        summary["test_auc"]
        - calendar_auc
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
        "feature_path": str(FEATURE_PATH),
        "label_path": str(LABEL_PATH),
        "target": "UPPER_FIRST vs LOWER_FIRST",
        "usable_rows": int(len(df)),
        "split": {
            "train_end": json_value(train_end),
            "validation_start": json_value(
                validation_start
            ),
            "validation_end": json_value(
                validation_end
            ),
            "test_start": json_value(test_start),
        },
        "experiments": serializable_results,
        "comparison": {
            "calendar_test_balanced": (
                calendar_balanced
            ),
            "topology_test_balanced": (
                topology_balanced
            ),
            "combined_test_balanced": (
                combined_balanced
            ),
            "combined_gain_vs_calendar": (
                combined_balanced
                - calendar_balanced
            ),
            "calendar_test_auc": calendar_auc,
            "combined_test_auc": combined_auc,
            "combined_auc_gain_vs_calendar": (
                combined_auc - calendar_auc
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
    print("SWEEP ABLATION SUMMARY")
    print("=" * 80)

    print(
        summary.to_string(index=False)
    )

    print()
    print("Key comparison:")
    print(
        f"  Calendar-only balanced : "
        f"{calendar_balanced:.4f}"
    )
    print(
        f"  Topology-only balanced : "
        f"{topology_balanced:.4f}"
    )
    print(
        f"  Combined balanced      : "
        f"{combined_balanced:.4f}"
    )
    print(
        f"  Combined gain          : "
        f"{combined_balanced - calendar_balanced:+.4f}"
    )
    print(
        f"  Combined AUC gain      : "
        f"{combined_auc - calendar_auc:+.4f}"
    )

    print()
    print("=" * 80)
    print("SWEEP ABLATION COMPLETE")
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
