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
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)


INPUT_PATH = Path(
    "data/features/liq_topology_v2_ml_labeled.parquet"
)

OUTPUT_DIR = Path(
    "data/models/topology_v2_catboost_1h"
)

MODEL_PATH = OUTPUT_DIR / "model.cbm"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.parquet"

TARGET_COLUMN = "direction_1h"
VALID_COLUMN = "label_valid_1h"

EMBARGO = pd.Timedelta(hours=4)

MAX_TRAIN_ROWS = 800_000
MAX_VALIDATION_ROWS = 300_000
MAX_TEST_ROWS = 300_000

RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "symbol",
    "timeframe",
    "nearest_side",
]

NUMERIC_FEATURES = [
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

    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend_utc",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

CLASS_VALUES = [-1, 0, 1]

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


def validate_features(df: pd.DataFrame) -> None:
    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing feature columns: " + ", ".join(missing)
        )

    leaked = [
        column
        for column in FEATURE_COLUMNS
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


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()

    # CatBoost kategorik kolonlarda null kabul etmez.
    for column in CATEGORICAL_FEATURES:
        X[column] = (
            X[column]
            .astype("string")
            .fillna("<MISSING>")
            .astype(str)
        )

    # CatBoost sayısal null değerleri doğal olarak işleyebilir.
    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    return X


def class_counts(series: pd.Series) -> dict:
    counts = series.value_counts(
        dropna=False
    ).sort_index()

    return {
        str(key): int(value)
        for key, value in counts.items()
    }


def evaluate(
    name: str,
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    return {
        "name": name,
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
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=CLASS_VALUES,
            target_names=[
                "DOWN",
                "NEUTRAL",
                "UP",
            ],
            output_dict=True,
            zero_division=0,
        ),
    }


def print_metrics(metrics: dict) -> None:
    print(
        f"{metrics['name']:<22} "
        f"accuracy={metrics['accuracy']:.4f}  "
        f"balanced={metrics['balanced_accuracy']:.4f}  "
        f"macro_f1={metrics['macro_f1']:.4f}  "
        f"log_loss={metrics['log_loss']:.4f}"
    )

    print("Confusion matrix [DOWN, NEUTRAL, UP]:")
    print(np.array(metrics["confusion_matrix"]))


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

    print("=" * 78)
    print("TOPOLOGY V2 — CATBOOST 1H MODEL")
    print("=" * 78)
    print(f"Input : {INPUT_PATH}")
    print(f"Model : {MODEL_PATH}")
    print()

    selected_columns = list(
        dict.fromkeys(
            [
                "id",
                "logged_at",
                "current_price",
                TARGET_COLUMN,
                VALID_COLUMN,
            ]
            + FEATURE_COLUMNS
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
        df[TARGET_COLUMN]
        .astype("int8")
    )

    df = df.sort_values(
        ["logged_at", "id"],
        kind="mergesort",
    ).reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            "No valid labeled rows remain."
        )

    minimum_time = df["logged_at"].min()
    maximum_time = df["logged_at"].max()

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

    print("Temporal range:")
    print(
        f"  Dataset    : "
        f"{minimum_time} -> {maximum_time}"
    )
    print(f"  Train end  : {train_end}")
    print(f"  Val start  : {validation_start}")
    print(f"  Val end    : {validation_end}")
    print(f"  Test start : {test_start}")
    print(f"  Embargo    : {EMBARGO}")
    print()

    print("Full split sizes:")
    print(f"  Train      : {len(train_df):,}")
    print(f"  Validation : {len(validation_df):,}")
    print(f"  Test       : {len(test_df):,}")
    print(f"  Embargoed  : {embargoed_rows:,}")
    print()

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

    print("Model sample sizes:")
    print(f"  Train      : {len(train_sample):,}")
    print(
        f"  Validation : "
        f"{len(validation_sample):,}"
    )
    print(f"  Test       : {len(test_sample):,}")
    print()

    print("Train target distribution:")
    print(
        train_sample[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        .to_string()
    )
    print()

    X_train = prepare_features(train_sample)
    y_train = train_sample[TARGET_COLUMN]

    X_validation = prepare_features(
        validation_sample
    )
    y_validation = validation_sample[
        TARGET_COLUMN
    ]

    X_test = prepare_features(test_sample)
    y_test = test_sample[TARGET_COLUMN]

    categorical_indices = [
        FEATURE_COLUMNS.index(column)
        for column in CATEGORICAL_FEATURES
    ]

    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=categorical_indices,
        feature_names=FEATURE_COLUMNS,
    )

    validation_pool = Pool(
        data=X_validation,
        label=y_validation,
        cat_features=categorical_indices,
        feature_names=FEATURE_COLUMNS,
    )

    test_pool = Pool(
        data=X_test,
        label=y_test,
        cat_features=categorical_indices,
        feature_names=FEATURE_COLUMNS,
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
        verbose=50,
        allow_writing_files=False,
        thread_count=-1,
    )

    print("Training CatBoost model...")
    fit_started = time.time()

    model.fit(
        train_pool,
        eval_set=validation_pool,
    )

    fit_elapsed = time.time() - fit_started

    print()
    print(
        f"Training completed in "
        f"{fit_elapsed:.1f} seconds."
    )
    print(
        f"Best iteration: "
        f"{model.get_best_iteration()}"
    )
    print(
        f"Best score    : "
        f"{model.get_best_score()}"
    )
    print()

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
        "validation_catboost",
        y_validation,
        validation_predictions,
        validation_probabilities,
    )

    test_metrics = evaluate(
        "test_catboost",
        y_test,
        test_predictions,
        test_probabilities,
    )

    print("=" * 78)
    print("VALIDATION RESULTS")
    print("=" * 78)
    print_metrics(validation_metrics)

    print()
    print("=" * 78)
    print("TEST RESULTS")
    print("=" * 78)
    print_metrics(test_metrics)

    feature_importances = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
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

    feature_importances.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    print()
    print("=" * 78)
    print("TOP 30 FEATURE IMPORTANCE")
    print("=" * 78)
    print(
        feature_importances.head(30).to_string(
            index=False
        )
    )

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

    prediction_output[
        "predicted_direction_1h"
    ] = test_predictions

    model_classes = [
        int(value)
        for value in model.classes_
    ]

    for index, class_value in enumerate(
        model_classes
    ):
        prediction_output[
            f"probability_class_{class_value}"
        ] = test_probabilities[
            :, index
        ].astype("float32")

    prediction_output.to_parquet(
        PREDICTIONS_PATH,
        index=False,
        compression="zstd",
    )

    model.save_model(str(MODEL_PATH))

    report = {
        "dataset": {
            "input": str(INPUT_PATH),
            "minimum_time": json_value(
                minimum_time
            ),
            "maximum_time": json_value(
                maximum_time
            ),
            "valid_labeled_rows": int(len(df)),
        },
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "categorical_features": (
            CATEGORICAL_FEATURES
        ),
        "numeric_features": NUMERIC_FEATURES,
        "embargo_hours": (
            EMBARGO.total_seconds() / 3600
        ),
        "boundaries": {
            "train_boundary_raw": json_value(
                train_boundary
            ),
            "validation_boundary_raw": json_value(
                validation_boundary
            ),
            "train_end": json_value(train_end),
            "validation_start": json_value(
                validation_start
            ),
            "validation_end": json_value(
                validation_end
            ),
            "test_start": json_value(test_start),
        },
        "full_split_rows": {
            "train": int(len(train_df)),
            "validation": int(
                len(validation_df)
            ),
            "test": int(len(test_df)),
            "embargoed": int(embargoed_rows),
        },
        "sample_rows": {
            "train": int(len(train_sample)),
            "validation": int(
                len(validation_sample)
            ),
            "test": int(len(test_sample)),
        },
        "class_counts": {
            "train": class_counts(y_train),
            "validation": class_counts(
                y_validation
            ),
            "test": class_counts(y_test),
        },
        "model": {
            "type": "CatBoostClassifier",
            "iterations_requested": 1000,
            "best_iteration": int(
                model.get_best_iteration()
            ),
            "depth": 6,
            "learning_rate": 0.05,
            "auto_class_weights": "Balanced",
            "training_seconds": fit_elapsed,
            "classes": model_classes,
            "best_score": model.get_best_score(),
        },
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "top_30_features": (
            feature_importances
            .head(30)
            .to_dict(orient="records")
        ),
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
    print("=" * 78)
    print("CATBOOST TRAINING COMPLETE")
    print("=" * 78)
    print(f"Model       : {MODEL_PATH}")
    print(f"Metrics     : {METRICS_PATH}")
    print(
        f"Importance  : "
        f"{FEATURE_IMPORTANCE_PATH}"
    )
    print(
        f"Predictions : "
        f"{PREDICTIONS_PATH}"
    )
    print(
        f"Total time  : "
        f"{total_elapsed:.1f} seconds"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
