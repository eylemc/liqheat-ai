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
)


INPUT_PATH = Path(
    "data/features/liq_topology_v2_ml_labeled.parquet"
)

OUTPUT_DIR = Path(
    "data/models/topology_v2_ablation_1h"
)

SUMMARY_PATH = OUTPUT_DIR / "ablation_summary.csv"
METRICS_PATH = OUTPUT_DIR / "ablation_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.parquet"

TARGET_COLUMN = "direction_1h"
VALID_COLUMN = "label_valid_1h"

EMBARGO = pd.Timedelta(hours=4)

MAX_TRAIN_ROWS = 800_000
MAX_VALIDATION_ROWS = 300_000
MAX_TEST_ROWS = 300_000

RANDOM_STATE = 42

CLASS_VALUES = [-1, 0, 1]

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
    "target_time_",
]


def json_value(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.generic):
        return value.item()

    if pd.isna(value):
        return None

    return value


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


def all_feature_columns() -> list[str]:
    columns = []

    for config in EXPERIMENTS.values():
        columns.extend(config["categorical"])
        columns.extend(config["numeric"])

    return list(dict.fromkeys(columns))


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
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=CLASS_VALUES,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=CLASS_VALUES,
        ).tolist(),
    }


def train_experiment(
    experiment_name: str,
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
    print(f"EXPERIMENT: {experiment_name.upper()}")
    print("=" * 80)
    print(f"Categorical features : {len(categorical_features)}")
    print(f"Numeric features     : {len(numeric_features)}")
    print(f"Total features       : {len(feature_columns)}")
    print()

    X_train = prepare_features(
        train_sample,
        categorical_features,
        numeric_features,
    )
    y_train = train_sample[TARGET_COLUMN]

    X_validation = prepare_features(
        validation_sample,
        categorical_features,
        numeric_features,
    )
    y_validation = validation_sample[TARGET_COLUMN]

    X_test = prepare_features(
        test_sample,
        categorical_features,
        numeric_features,
    )
    y_test = test_sample[TARGET_COLUMN]

    categorical_indices = [
        feature_columns.index(column)
        for column in categorical_features
    ]

    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=categorical_indices,
        feature_names=feature_columns,
    )

    validation_pool = Pool(
        data=X_validation,
        label=y_validation,
        cat_features=categorical_indices,
        feature_names=feature_columns,
    )

    test_pool = Pool(
        data=X_test,
        label=y_test,
        cat_features=categorical_indices,
        feature_names=feature_columns,
    )

    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        random_seed=RANDOM_STATE,
        random_strength=1.0,
        l2_leaf_reg=5.0,
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        auto_class_weights="Balanced",
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

    validation_predictions = (
        model.predict(validation_pool)
        .reshape(-1)
        .astype("int8")
    )

    validation_probabilities = (
        model.predict_proba(validation_pool)
    )

    test_predictions = (
        model.predict(test_pool)
        .reshape(-1)
        .astype("int8")
    )

    test_probabilities = (
        model.predict_proba(test_pool)
    )

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

    feature_importance = pd.DataFrame(
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

    model_path = OUTPUT_DIR / f"{experiment_name}.cbm"
    importance_path = (
        OUTPUT_DIR
        / f"{experiment_name}_feature_importance.csv"
    )

    model.save_model(str(model_path))
    feature_importance.to_csv(
        importance_path,
        index=False,
    )

    print()
    print(f"Best iteration : {model.get_best_iteration()}")
    print(f"Training time  : {fit_elapsed:.1f} seconds")

    print()
    print("Validation:")
    print(
        f"  accuracy          = "
        f"{validation_metrics['accuracy']:.4f}"
    )
    print(
        f"  balanced_accuracy = "
        f"{validation_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"  macro_f1          = "
        f"{validation_metrics['macro_f1']:.4f}"
    )
    print(
        f"  log_loss          = "
        f"{validation_metrics['log_loss']:.4f}"
    )

    print()
    print("Test:")
    print(
        f"  accuracy          = "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        f"  balanced_accuracy = "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"  macro_f1          = "
        f"{test_metrics['macro_f1']:.4f}"
    )
    print(
        f"  log_loss          = "
        f"{test_metrics['log_loss']:.4f}"
    )

    print()
    print("Test confusion matrix [DOWN, NEUTRAL, UP]:")
    print(np.array(test_metrics["confusion_matrix"]))

    print()
    print("Top 15 feature importance:")
    print(
        feature_importance.head(15).to_string(
            index=False
        )
    )

    return {
        "name": experiment_name,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "feature_count": len(feature_columns),
        "best_iteration": int(
            model.get_best_iteration()
        ),
        "training_seconds": fit_elapsed,
        "validation": validation_metrics,
        "test": test_metrics,
        "model_path": str(model_path),
        "importance_path": str(importance_path),
        "feature_importance": (
            feature_importance.to_dict(
                orient="records"
            )
        ),
        "test_predictions": test_predictions,
        "test_probabilities": test_probabilities,
        "classes": [
            int(value)
            for value in model.classes_
        ],
    }


def main() -> int:
    started = time.time()

    if not INPUT_PATH.exists():
        print(
            f"ERROR: Input file not found: {INPUT_PATH}",
            file=sys.stderr,
        )
        return 1

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("TOPOLOGY V2 — 1H CATBOOST ABLATION TEST")
    print("=" * 80)
    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    selected_columns = list(
        dict.fromkeys(
            [
                "id",
                "logged_at",
                "symbol",
                "timeframe",
                "current_price",
                TARGET_COLUMN,
                VALID_COLUMN,
            ]
            + all_feature_columns()
        )
    )

    print("Loading selected columns...")

    df = pd.read_parquet(
        INPUT_PATH,
        columns=selected_columns,
    )

    validate_features(df)

    df = df.loc[
        df[VALID_COLUMN].eq(1)
        & df[TARGET_COLUMN].notna()
    ].copy()

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN].astype("int8")
    )

    df = df.sort_values(
        ["logged_at", "id"],
        kind="mergesort",
    ).reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            "No valid labeled rows remain."
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

    embargoed_rows = (
        len(df)
        - len(train_df)
        - len(validation_df)
        - len(test_df)
    )

    if min(
        len(train_df),
        len(validation_df),
        len(test_df),
    ) == 0:
        raise RuntimeError(
            "At least one temporal split is empty."
        )

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

    print("Temporal split:")
    print(f"  Train end  : {train_end}")
    print(f"  Val start  : {validation_start}")
    print(f"  Val end    : {validation_end}")
    print(f"  Test start : {test_start}")
    print(f"  Embargoed  : {embargoed_rows:,}")
    print()

    print("Sample sizes:")
    print(f"  Train      : {len(train_sample):,}")
    print(f"  Validation : {len(validation_sample):,}")
    print(f"  Test       : {len(test_sample):,}")

    results = []

    prediction_output = test_sample[
        [
            "id",
            "logged_at",
            "symbol",
            "timeframe",
            "current_price",
            TARGET_COLUMN,
        ]
    ].copy()

    for experiment_name, config in EXPERIMENTS.items():
        result = train_experiment(
            experiment_name=experiment_name,
            config=config,
            train_sample=train_sample,
            validation_sample=validation_sample,
            test_sample=test_sample,
        )

        results.append(result)

        prediction_output[
            f"{experiment_name}_prediction"
        ] = result["test_predictions"]

        for index, class_value in enumerate(
            result["classes"]
        ):
            prediction_output[
                f"{experiment_name}_probability_{class_value}"
            ] = result["test_probabilities"][
                :, index
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
                "training_seconds": result[
                    "training_seconds"
                ],
                "validation_accuracy": result[
                    "validation"
                ]["accuracy"],
                "validation_balanced_accuracy": result[
                    "validation"
                ]["balanced_accuracy"],
                "validation_macro_f1": result[
                    "validation"
                ]["macro_f1"],
                "validation_log_loss": result[
                    "validation"
                ]["log_loss"],
                "test_accuracy": result[
                    "test"
                ]["accuracy"],
                "test_balanced_accuracy": result[
                    "test"
                ]["balanced_accuracy"],
                "test_macro_f1": result[
                    "test"
                ]["macro_f1"],
                "test_log_loss": result[
                    "test"
                ]["log_loss"],
            }
        )

    summary = pd.DataFrame(summary_rows)

    calendar_test_balanced = float(
        summary.loc[
            summary["experiment"] == "calendar_only",
            "test_balanced_accuracy",
        ].iloc[0]
    )

    topology_test_balanced = float(
        summary.loc[
            summary["experiment"] == "topology_only",
            "test_balanced_accuracy",
        ].iloc[0]
    )

    combined_test_balanced = float(
        summary.loc[
            summary["experiment"] == "combined",
            "test_balanced_accuracy",
        ].iloc[0]
    )

    summary["test_balanced_gain_vs_calendar"] = (
        summary["test_balanced_accuracy"]
        - calendar_test_balanced
    )

    summary["test_balanced_gain_vs_topology"] = (
        summary["test_balanced_accuracy"]
        - topology_test_balanced
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    serializable_results = []

    for result in results:
        serializable_result = {
            key: value
            for key, value in result.items()
            if key not in {
                "test_predictions",
                "test_probabilities",
            }
        }

        serializable_results.append(
            serializable_result
        )

    report = {
        "input": str(INPUT_PATH),
        "target": TARGET_COLUMN,
        "embargo_hours": (
            EMBARGO.total_seconds() / 3600
        ),
        "split": {
            "train_end": json_value(train_end),
            "validation_start": json_value(
                validation_start
            ),
            "validation_end": json_value(
                validation_end
            ),
            "test_start": json_value(test_start),
            "embargoed_rows": int(
                embargoed_rows
            ),
        },
        "sample_rows": {
            "train": int(len(train_sample)),
            "validation": int(
                len(validation_sample)
            ),
            "test": int(len(test_sample)),
        },
        "experiments": serializable_results,
        "comparison": {
            "calendar_test_balanced_accuracy": (
                calendar_test_balanced
            ),
            "topology_test_balanced_accuracy": (
                topology_test_balanced
            ),
            "combined_test_balanced_accuracy": (
                combined_test_balanced
            ),
            "combined_gain_vs_calendar": (
                combined_test_balanced
                - calendar_test_balanced
            ),
            "combined_gain_vs_topology": (
                combined_test_balanced
                - topology_test_balanced
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

    total_elapsed = time.time() - started

    print()
    print("=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)

    print(
        summary[
            [
                "experiment",
                "feature_count",
                "best_iteration",
                "validation_balanced_accuracy",
                "validation_macro_f1",
                "test_balanced_accuracy",
                "test_macro_f1",
                "test_log_loss",
                "test_balanced_gain_vs_calendar",
            ]
        ].to_string(index=False)
    )

    print()
    print("Key comparison:")
    print(
        f"  Calendar-only test balanced : "
        f"{calendar_test_balanced:.4f}"
    )
    print(
        f"  Topology-only test balanced : "
        f"{topology_test_balanced:.4f}"
    )
    print(
        f"  Combined test balanced      : "
        f"{combined_test_balanced:.4f}"
    )
    print(
        f"  Combined gain vs calendar   : "
        f"{combined_test_balanced - calendar_test_balanced:+.4f}"
    )
    print(
        f"  Combined gain vs topology   : "
        f"{combined_test_balanced - topology_test_balanced:+.4f}"
    )

    print()
    print("=" * 80)
    print("ABLATION COMPLETE")
    print("=" * 80)
    print(f"Summary     : {SUMMARY_PATH}")
    print(f"Metrics     : {METRICS_PATH}")
    print(f"Predictions : {PREDICTIONS_PATH}")
    print(f"Total time  : {total_elapsed:.1f} seconds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
