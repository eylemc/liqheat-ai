#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import (
    CatBoostClassifier,
    CatBoostRegressor,
    Pool,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)


DATASET_PATH = Path(
    "data/forecast_v1/multitimeframe_forecast_dataset.parquet"
)

MODEL_ROOT = Path(
    "models/forecast_v1"
)

REPORT_ROOT = Path(
    "reports/forecast_v1"
)

HORIZONS = [
    15,
    30,
    60,
]

RANDOM_SEED = 42

MAX_TRAIN_ROWS = 700_000

EXCLUDED_PREFIXES = (
    "future_",
    "target_",
)

EXCLUDED_EXACT = {
    "observation_id",
    "observation_time",
    "observation_month",
    "tf4h_source_id",
    "tf4h_source_time",
    "tf24h_source_id",
    "tf24h_source_time",
}

CATEGORICAL_FEATURES = [
    "symbol",
    "tf1h_nearest_side",
    "tf4h_nearest_side",
    "tf24h_nearest_side",
]


def json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        (
            np.integer,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float16,
            np.float32,
            np.float64,
        ),
    ):
        number = float(value)

        if np.isfinite(number):
            return number

        return None

    if isinstance(value, np.ndarray):
        return [
            json_safe(item)
            for item in value.tolist()
        ]

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            json_safe(item)
            for item in value
        ]

    return value


def select_feature_columns(
    frame: pd.DataFrame,
) -> list[str]:
    columns = []

    for column in frame.columns:
        if column in EXCLUDED_EXACT:
            continue

        if column.startswith(
            EXCLUDED_PREFIXES
        ):
            continue

        columns.append(column)

    return columns


def temporal_split(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    ordered = frame.sort_values(
        [
            "observation_time",
            "symbol",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    unique_times = np.array(
        sorted(
            ordered[
                "observation_time"
            ].dropna().unique()
        )
    )

    if len(unique_times) < 100:
        raise RuntimeError(
            "Not enough unique timestamps "
            "for temporal split"
        )

    train_cut = unique_times[
        int(len(unique_times) * 0.70)
    ]

    validation_cut = unique_times[
        int(len(unique_times) * 0.85)
    ]

    train = ordered[
        ordered["observation_time"]
        < train_cut
    ].copy()

    validation = ordered[
        (
            ordered["observation_time"]
            >= train_cut
        )
        & (
            ordered["observation_time"]
            < validation_cut
        )
    ].copy()

    test = ordered[
        ordered["observation_time"]
        >= validation_cut
    ].copy()

    return (
        train,
        validation,
        test,
    )


def deterministic_sample(
    frame: pd.DataFrame,
    maximum_rows: int,
) -> pd.DataFrame:
    if len(frame) <= maximum_rows:
        return frame

    positions = np.linspace(
        0,
        len(frame) - 1,
        maximum_rows,
        dtype=np.int64,
    )

    return frame.iloc[
        positions
    ].copy()


def prepare_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    output = frame[
        feature_columns
    ].copy()

    for column in CATEGORICAL_FEATURES:
        if column not in output.columns:
            continue

        output[column] = (
            output[column]
            .astype("string")
            .fillna("<MISSING>")
            .astype(str)
        )

    for column in feature_columns:
        if column in CATEGORICAL_FEATURES:
            continue

        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        )

    return output


def confidence_bucket_report(
    truth: np.ndarray,
    prediction: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    confidence = probabilities.max(
        axis=1
    )

    rows = []

    buckets = [
        (0.00, 0.50),
        (0.50, 0.60),
        (0.60, 0.70),
        (0.70, 0.80),
        (0.80, 0.90),
        (0.90, 1.01),
    ]

    for lower, upper in buckets:
        mask = (
            (confidence >= lower)
            & (confidence < upper)
        )

        count = int(mask.sum())

        if count == 0:
            continue

        rows.append({
            "confidence_min": lower,
            "confidence_max": min(
                upper,
                1.0,
            ),
            "rows": count,
            "coverage": count / len(truth),
            "accuracy": accuracy_score(
                truth[mask],
                prediction[mask],
            ),
            "mean_confidence": float(
                confidence[mask].mean()
            ),
        })

    return rows


def train_horizon(
    dataset: pd.DataFrame,
    horizon: int,
    feature_columns: list[str],
) -> dict[str, Any]:
    started = time.time()

    target_column = (
        f"target_direction_{horizon}m"
    )

    return_column = (
        f"future_return_bps_{horizon}m"
    )

    valid_column = (
        f"future_valid_{horizon}m"
    )

    frame = dataset[
        (
            dataset[valid_column] == 1
        )
        & dataset[target_column].notna()
        & dataset[return_column].notna()
    ].copy()

    frame[target_column] = (
        frame[target_column]
        .astype(int)
    )

    train, validation, test = (
        temporal_split(frame)
    )

    train = deterministic_sample(
        train,
        MAX_TRAIN_ROWS,
    )

    validation = deterministic_sample(
        validation,
        200_000,
    )

    test = deterministic_sample(
        test,
        250_000,
    )

    X_train = prepare_features(
        train,
        feature_columns,
    )

    X_validation = prepare_features(
        validation,
        feature_columns,
    )

    X_test = prepare_features(
        test,
        feature_columns,
    )

    y_train = train[
        target_column
    ].astype(int)

    y_validation = validation[
        target_column
    ].astype(int)

    y_test = test[
        target_column
    ].astype(int)

    # CatBoost class labels are made explicit and stable.
    class_order = [
        -1,
        0,
        1,
    ]

    classifier = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.045,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=RANDOM_SEED,
        l2_leaf_reg=5.0,
        random_strength=0.4,
        border_count=128,
        auto_class_weights="Balanced",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.90,
        verbose=100,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=100,
    )

    train_pool = Pool(
        X_train,
        y_train,
        cat_features=[
            column
            for column in CATEGORICAL_FEATURES
            if column in feature_columns
        ],
    )

    validation_pool = Pool(
        X_validation,
        y_validation,
        cat_features=[
            column
            for column in CATEGORICAL_FEATURES
            if column in feature_columns
        ],
    )

    classifier.fit(
        train_pool,
        eval_set=validation_pool,
        use_best_model=True,
    )

    raw_prediction = (
        classifier
        .predict(X_test)
        .reshape(-1)
        .astype(int)
    )

    probabilities = (
        classifier
        .predict_proba(X_test)
    )

    learned_classes = [
        int(value)
        for value in classifier.classes_
    ]

    classification_metrics = {
        "rows": int(len(test)),
        "accuracy": accuracy_score(
            y_test,
            raw_prediction,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                y_test,
                raw_prediction,
            )
        ),
        "log_loss": log_loss(
            y_test,
            probabilities,
            labels=learned_classes,
        ),
        "class_order": learned_classes,
        "confusion_matrix": confusion_matrix(
            y_test,
            raw_prediction,
            labels=class_order,
        ).tolist(),
        "classification_report": (
            classification_report(
                y_test,
                raw_prediction,
                labels=class_order,
                output_dict=True,
                zero_division=0,
            )
        ),
        "confidence_buckets": (
            confidence_bucket_report(
                y_test.to_numpy(),
                raw_prediction,
                probabilities,
            )
        ),
    }

    regression_target_train = (
        train[return_column]
        .astype(float)
    )

    regression_target_validation = (
        validation[return_column]
        .astype(float)
    )

    regression_target_test = (
        test[return_column]
        .astype(float)
    )

    regressor = CatBoostRegressor(
        iterations=1000,
        depth=8,
        learning_rate=0.045,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_SEED,
        l2_leaf_reg=6.0,
        random_strength=0.4,
        border_count=128,
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.90,
        verbose=100,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=100,
    )

    regression_train_pool = Pool(
        X_train,
        regression_target_train,
        cat_features=[
            column
            for column in CATEGORICAL_FEATURES
            if column in feature_columns
        ],
    )

    regression_validation_pool = Pool(
        X_validation,
        regression_target_validation,
        cat_features=[
            column
            for column in CATEGORICAL_FEATURES
            if column in feature_columns
        ],
    )

    regressor.fit(
        regression_train_pool,
        eval_set=regression_validation_pool,
        use_best_model=True,
    )

    predicted_return = regressor.predict(
        X_test
    )

    regression_metrics = {
        "rows": int(len(test)),
        "mae_bps": mean_absolute_error(
            regression_target_test,
            predicted_return,
        ),
        "rmse_bps": float(
            np.sqrt(
                mean_squared_error(
                    regression_target_test,
                    predicted_return,
                )
            )
        ),
        "direction_accuracy_from_regression": (
            accuracy_score(
                np.sign(
                    regression_target_test
                ),
                np.sign(
                    predicted_return
                ),
            )
        ),
        "actual_mean_bps": float(
            regression_target_test.mean()
        ),
        "predicted_mean_bps": float(
            np.mean(predicted_return)
        ),
    }

    model_dir = (
        MODEL_ROOT
        / f"{horizon}m"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    classifier.save_model(
        model_dir
        / "direction_classifier.cbm"
    )

    regressor.save_model(
        model_dir
        / "return_regressor.cbm"
    )

    (
        model_dir
        / "features.json"
    ).write_text(
        json.dumps(
            feature_columns,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        model_dir
        / "categorical_features.json"
    ).write_text(
        json.dumps(
            [
                column
                for column in CATEGORICAL_FEATURES
                if column in feature_columns
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    feature_importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": (
            classifier
            .get_feature_importance()
        ),
    }).sort_values(
        "importance",
        ascending=False,
    )

    feature_importance.to_csv(
        model_dir
        / "feature_importance.csv",
        index=False,
    )

    symbol_rows = []

    for symbol, group in test.assign(
        predicted_class=raw_prediction,
        predicted_return_bps=predicted_return,
        confidence=probabilities.max(axis=1),
    ).groupby(
        "symbol",
        observed=True,
    ):
        symbol_rows.append({
            "symbol": str(symbol),
            "rows": int(len(group)),
            "direction_accuracy": (
                accuracy_score(
                    group[target_column],
                    group["predicted_class"],
                )
            ),
            "balanced_accuracy": (
                balanced_accuracy_score(
                    group[target_column],
                    group["predicted_class"],
                )
            ),
            "return_mae_bps": (
                mean_absolute_error(
                    group[return_column],
                    group[
                        "predicted_return_bps"
                    ],
                )
            ),
            "mean_confidence": float(
                group["confidence"].mean()
            ),
        })

    symbol_metrics = pd.DataFrame(
        symbol_rows
    )

    symbol_metrics.to_csv(
        model_dir
        / "symbol_metrics.csv",
        index=False,
    )

    prediction_output = pd.DataFrame({
        "observation_time": (
            test["observation_time"]
            .to_numpy()
        ),
        "symbol": (
            test["symbol"]
            .astype(str)
            .to_numpy()
        ),
        "actual_class": (
            y_test.to_numpy()
        ),
        "predicted_class": (
            raw_prediction
        ),
        "confidence": (
            probabilities.max(axis=1)
        ),
        "actual_return_bps": (
            regression_target_test
            .to_numpy()
        ),
        "predicted_return_bps": (
            predicted_return
        ),
    })

    for index, class_value in enumerate(
        learned_classes
    ):
        prediction_output[
            f"probability_class_{class_value}"
        ] = probabilities[:, index]

    prediction_output.to_parquet(
        model_dir
        / "test_predictions.parquet",
        index=False,
        compression="zstd",
    )

    result = {
        "horizon_minutes": horizon,
        "train_rows": int(len(train)),
        "validation_rows": int(
            len(validation)
        ),
        "test_rows": int(len(test)),
        "train_start": (
            train["observation_time"]
            .min()
            .isoformat()
        ),
        "train_end": (
            train["observation_time"]
            .max()
            .isoformat()
        ),
        "validation_start": (
            validation[
                "observation_time"
            ].min().isoformat()
        ),
        "validation_end": (
            validation[
                "observation_time"
            ].max().isoformat()
        ),
        "test_start": (
            test["observation_time"]
            .min()
            .isoformat()
        ),
        "test_end": (
            test["observation_time"]
            .max()
            .isoformat()
        ),
        "classification": (
            classification_metrics
        ),
        "regression": regression_metrics,
        "elapsed_seconds": (
            time.time() - started
        ),
    }

    (
        model_dir
        / "metrics.json"
    ).write_text(
        json.dumps(
            json_safe(result),
            indent=2,
        ),
        encoding="utf-8",
    )

    return result


def main() -> int:
    started = time.time()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("LIQHEAT FORECAST V1 — WALK-FORWARD BASELINE")
    print("=" * 100)
    print("Dataset:", DATASET_PATH)
    print()

    dataset = pd.read_parquet(
        DATASET_PATH
    )

    dataset["observation_time"] = (
        pd.to_datetime(
            dataset["observation_time"],
            utc=True,
            errors="coerce",
        )
    )

    feature_columns = (
        select_feature_columns(
            dataset
        )
    )

    print(
        "Rows:",
        f"{len(dataset):,}",
    )

    print(
        "Features:",
        len(feature_columns),
    )

    results = []

    for horizon in HORIZONS:
        print()
        print("=" * 100)
        print(
            f"TRAINING {horizon} MINUTE FORECAST"
        )
        print("=" * 100)

        result = train_horizon(
            dataset,
            horizon,
            feature_columns,
        )

        results.append(result)

        print()
        print(
            json.dumps(
                json_safe(result),
                indent=2,
            )
        )

    summary = {
        "status": "complete",
        "dataset": str(DATASET_PATH),
        "rows": int(len(dataset)),
        "feature_count": int(
            len(feature_columns)
        ),
        "horizons": results,
        "elapsed_seconds": (
            time.time() - started
        ),
    }

    (
        REPORT_ROOT
        / "forecast_v1_summary.json"
    ).write_text(
        json.dumps(
            json_safe(summary),
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_rows = []

    for result in results:
        summary_rows.append({
            "horizon_minutes": (
                result["horizon_minutes"]
            ),
            "test_rows": (
                result["test_rows"]
            ),
            "accuracy": (
                result[
                    "classification"
                ]["accuracy"]
            ),
            "balanced_accuracy": (
                result[
                    "classification"
                ][
                    "balanced_accuracy"
                ]
            ),
            "log_loss": (
                result[
                    "classification"
                ]["log_loss"]
            ),
            "return_mae_bps": (
                result[
                    "regression"
                ]["mae_bps"]
            ),
            "return_rmse_bps": (
                result[
                    "regression"
                ]["rmse_bps"]
            ),
            "regression_direction_accuracy": (
                result[
                    "regression"
                ][
                    "direction_accuracy_from_regression"
                ]
            ),
        })

    summary_frame = pd.DataFrame(
        summary_rows
    )

    summary_frame.to_csv(
        REPORT_ROOT
        / "forecast_v1_summary.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("FORECAST V1 COMPLETE")
    print("=" * 100)
    print(
        summary_frame.to_string(
            index=False
        )
    )
    print()
    print(
        "Report:",
        REPORT_ROOT
        / "forecast_v1_summary.json",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
