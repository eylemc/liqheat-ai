#!/usr/bin/env python3
from __future__ import annotations

import json
import math
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
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)


DATASET_PATH = Path(
    "data/forecast_v1/"
    "multitimeframe_forecast_dataset.parquet"
)

MODEL_ROOT = Path("models/forecast_v2")
REPORT_ROOT = Path("reports/forecast_v2")

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

HORIZON_MINUTES = 15
RANDOM_SEED = 42

# Neutral sınıfı, yalnız fold train dönemindeki
# absolute return dağılımından türetilir.
NEUTRAL_ABS_RETURN_QUANTILE = 0.35

# Aşırı küçük veya aşırı geniş neutral band oluşmasını engeller.
MIN_NEUTRAL_BPS = 4.0
MAX_NEUTRAL_BPS = 45.0

# GPU ayarları
GPU_DEVICE = "0"
GPU_RAM_PART = 0.90

CATEGORICAL_FEATURES = [
    "tf1h_nearest_side",
    "tf4h_nearest_side",
    "tf24h_nearest_side",
]

EXCLUDED_EXACT = {
    "symbol",
    "observation_id",
    "observation_time",
    "observation_month",

    "tf4h_source_id",
    "tf4h_source_time",
    "tf24h_source_id",
    "tf24h_source_time",

    # Raw fiyat seviyesi pair-specific modelde zaman/regime
    # ezberine neden olmasın. Return ve delta feature'ları kalır.
    "tf1h_current_price",
    "tf4h_current_price",
    "tf24h_current_price",
}

EXCLUDED_PREFIXES = (
    "future_",
    "target_",
)

CLASS_ORDER = [-1, 0, 1]

CLASS_NAMES = {
    -1: "DOWN",
     0: "NEUTRAL",
     1: "UP",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return [
            json_safe(item)
            for item in value.tolist()
        ]

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def select_feature_columns(
    frame: pd.DataFrame,
) -> list[str]:
    selected: list[str] = []

    for column in frame.columns:
        if column in EXCLUDED_EXACT:
            continue

        if column.startswith(EXCLUDED_PREFIXES):
            continue

        selected.append(column)

    return selected


def prepare_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    output = frame[
        feature_columns
    ].copy()

    for column in feature_columns:
        if column in CATEGORICAL_FEATURES:
            output[column] = (
                output[column]
                .astype("string")
                .fillna("<MISSING>")
                .astype(str)
            )
        else:
            output[column] = pd.to_numeric(
                output[column],
                errors="coerce",
            )

    return output


def build_walk_forward_folds(
    frame: pd.DataFrame,
) -> list[
    tuple[
        str,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]
]:
    """
    Expanding-window, üç temporal fold.

    fold_1:
      train 0-40%, validation 40-50%, test 50-65%

    fold_2:
      train 0-55%, validation 55-65%, test 65-80%

    fold_3:
      train 0-70%, validation 70-80%, test 80-100%
    """
    ordered = (
        frame
        .sort_values(
            "observation_time",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    unique_times = np.array(
        sorted(
            ordered[
                "observation_time"
            ].dropna().unique()
        )
    )

    if len(unique_times) < 1000:
        raise RuntimeError(
            "Not enough unique timestamps "
            "for walk-forward validation"
        )

    specifications = [
        ("fold_1", 0.40, 0.50, 0.65),
        ("fold_2", 0.55, 0.65, 0.80),
        ("fold_3", 0.70, 0.80, 1.00),
    ]

    folds = []

    for (
        fold_name,
        train_end_ratio,
        validation_end_ratio,
        test_end_ratio,
    ) in specifications:
        train_end = unique_times[
            min(
                int(
                    len(unique_times)
                    * train_end_ratio
                ),
                len(unique_times) - 1,
            )
        ]

        validation_end = unique_times[
            min(
                int(
                    len(unique_times)
                    * validation_end_ratio
                ),
                len(unique_times) - 1,
            )
        ]

        if test_end_ratio >= 1.0:
            test_end = None
        else:
            test_end = unique_times[
                min(
                    int(
                        len(unique_times)
                        * test_end_ratio
                    ),
                    len(unique_times) - 1,
                )
            ]

        train = ordered[
            ordered["observation_time"]
            < train_end
        ].copy()

        validation = ordered[
            (
                ordered["observation_time"]
                >= train_end
            )
            & (
                ordered["observation_time"]
                < validation_end
            )
        ].copy()

        if test_end is None:
            test = ordered[
                ordered["observation_time"]
                >= validation_end
            ].copy()
        else:
            test = ordered[
                (
                    ordered["observation_time"]
                    >= validation_end
                )
                & (
                    ordered["observation_time"]
                    < test_end
                )
            ].copy()

        if min(
            len(train),
            len(validation),
            len(test),
        ) == 0:
            raise RuntimeError(
                f"Empty split generated for {fold_name}"
            )

        folds.append(
            (
                fold_name,
                train,
                validation,
                test,
            )
        )

    return folds


def calculate_neutral_threshold(
    train_returns_bps: pd.Series,
) -> float:
    absolute_returns = (
        pd.to_numeric(
            train_returns_bps,
            errors="coerce",
        )
        .abs()
        .dropna()
    )

    threshold = float(
        absolute_returns.quantile(
            NEUTRAL_ABS_RETURN_QUANTILE
        )
    )

    return float(
        np.clip(
            threshold,
            MIN_NEUTRAL_BPS,
            MAX_NEUTRAL_BPS,
        )
    )


def create_direction_target(
    returns_bps: pd.Series,
    neutral_threshold_bps: float,
) -> pd.Series:
    numeric = pd.to_numeric(
        returns_bps,
        errors="coerce",
    )

    values = np.select(
        [
            numeric < -neutral_threshold_bps,
            numeric > neutral_threshold_bps,
        ],
        [
            -1,
            1,
        ],
        default=0,
    )

    target = pd.Series(
        values,
        index=returns_bps.index,
        dtype="int8",
    )

    return target.where(
        numeric.notna()
    )


def fit_direction_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    categorical_features: list[str],
) -> CatBoostClassifier:
    model = CatBoostClassifier(
        iterations=1400,
        depth=8,
        learning_rate=0.035,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",

        random_seed=RANDOM_SEED,
        l2_leaf_reg=6.0,
        random_strength=0.35,
        border_count=128,

        task_type="GPU",
        devices=GPU_DEVICE,
        gpu_ram_part=GPU_RAM_PART,

        verbose=100,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=140,
    )

    train_pool = Pool(
        X_train,
        y_train,
        cat_features=categorical_features,
    )

    validation_pool = Pool(
        X_validation,
        y_validation,
        cat_features=categorical_features,
    )

    model.fit(
        train_pool,
        eval_set=validation_pool,
        use_best_model=True,
    )

    return model


def fit_magnitude_model(
    X_train: pd.DataFrame,
    magnitude_train_bps: pd.Series,
    X_validation: pd.DataFrame,
    magnitude_validation_bps: pd.Series,
    categorical_features: list[str],
) -> CatBoostRegressor:
    """
    Yalnız event satırlarında log1p(abs(return_bps)) öğrenir.
    """
    y_train_log = np.log1p(
        magnitude_train_bps.clip(lower=0)
    )

    y_validation_log = np.log1p(
        magnitude_validation_bps.clip(lower=0)
    )

    model = CatBoostRegressor(
        iterations=1400,
        depth=8,
        learning_rate=0.035,
        loss_function="RMSE",
        eval_metric="RMSE",

        random_seed=RANDOM_SEED,
        l2_leaf_reg=7.0,
        random_strength=0.35,
        border_count=128,

        task_type="GPU",
        devices=GPU_DEVICE,
        gpu_ram_part=GPU_RAM_PART,

        verbose=100,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=140,
    )

    train_pool = Pool(
        X_train,
        y_train_log,
        cat_features=categorical_features,
    )

    validation_pool = Pool(
        X_validation,
        y_validation_log,
        cat_features=categorical_features,
    )

    model.fit(
        train_pool,
        eval_set=validation_pool,
        use_best_model=True,
    )

    return model


def expectancy_bucket_report(
    predictions: pd.DataFrame,
) -> list[dict[str, Any]]:
    valid = predictions[
        predictions["expected_return_bps"]
        .notna()
        & predictions["actual_return_bps"]
        .notna()
    ].copy()

    if len(valid) < 100:
        return []

    try:
        valid["expectancy_bucket"] = pd.qcut(
            valid["expected_return_bps"],
            q=10,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return []

    rows = []

    for bucket, group in valid.groupby(
        "expectancy_bucket",
        observed=True,
        sort=True,
    ):
        actual = group["actual_return_bps"]
        predicted = group["expected_return_bps"]

        rows.append({
            "bucket": int(bucket),
            "rows": int(len(group)),
            "predicted_mean_bps": float(
                predicted.mean()
            ),
            "actual_mean_bps": float(
                actual.mean()
            ),
            "actual_median_bps": float(
                actual.median()
            ),
            "positive_return_rate": float(
                (actual > 0).mean()
            ),
            "negative_return_rate": float(
                (actual < 0).mean()
            ),
            "sign_accuracy": float(
                (
                    np.sign(predicted)
                    == np.sign(actual)
                ).mean()
            ),
        })

    return rows


def confidence_report(
    predictions: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []

    for threshold in [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]:
        selected = predictions[
            predictions["direction_confidence"]
            >= threshold
        ]

        if selected.empty:
            continue

        predicted_event = selected[
            selected["predicted_class"] != 0
        ]

        rows.append({
            "minimum_confidence": threshold,
            "rows": int(len(selected)),
            "coverage": float(
                len(selected)
                / len(predictions)
            ),
            "accuracy": float(
                accuracy_score(
                    selected["actual_class"],
                    selected["predicted_class"],
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    selected["actual_class"],
                    selected["predicted_class"],
                )
            ),
            "predicted_event_rows": int(
                len(predicted_event)
            ),
            "predicted_event_share": float(
                len(predicted_event)
                / len(selected)
            ),
            "event_precision": (
                float(
                    (
                        predicted_event[
                            "actual_class"
                        ] != 0
                    ).mean()
                )
                if len(predicted_event)
                else None
            ),
            "event_direction_accuracy": (
                float(
                    (
                        predicted_event[
                            "actual_class"
                        ]
                        == predicted_event[
                            "predicted_class"
                        ]
                    ).mean()
                )
                if len(predicted_event)
                else None
            ),
            "actual_mean_return_bps": float(
                selected[
                    "actual_return_bps"
                ].mean()
            ),
        })

    return rows


def evaluate_fold(
    symbol: str,
    fold_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    CatBoostClassifier,
    CatBoostRegressor,
]:
    started = time.time()

    return_column = (
        f"future_return_bps_"
        f"{HORIZON_MINUTES}m"
    )

    threshold = calculate_neutral_threshold(
        train[return_column]
    )

    y_train = create_direction_target(
        train[return_column],
        threshold,
    )

    y_validation = create_direction_target(
        validation[return_column],
        threshold,
    )

    y_test = create_direction_target(
        test[return_column],
        threshold,
    )

    train_valid = y_train.notna()
    validation_valid = y_validation.notna()
    test_valid = y_test.notna()

    train = train.loc[train_valid].copy()
    validation = validation.loc[
        validation_valid
    ].copy()
    test = test.loc[test_valid].copy()

    y_train = y_train.loc[
        train_valid
    ].astype(int)

    y_validation = y_validation.loc[
        validation_valid
    ].astype(int)

    y_test = y_test.loc[
        test_valid
    ].astype(int)

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

    categorical = [
        column
        for column in CATEGORICAL_FEATURES
        if column in feature_columns
    ]

    print()
    print("-" * 100)
    print(
        symbol,
        fold_name,
        f"threshold={threshold:.3f} bps",
    )
    print("-" * 100)

    print(
        "Train class distribution:",
        y_train.value_counts().sort_index().to_dict(),
    )

    direction_model = fit_direction_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        categorical,
    )

    event_train_mask = y_train != 0
    event_validation_mask = y_validation != 0

    if event_train_mask.sum() < 1000:
        raise RuntimeError(
            f"Insufficient event rows for magnitude: "
            f"{symbol} {fold_name}"
        )

    magnitude_model = fit_magnitude_model(
        X_train.loc[event_train_mask],
        train.loc[
            event_train_mask,
            return_column,
        ].abs().astype(float),

        X_validation.loc[
            event_validation_mask
        ],
        validation.loc[
            event_validation_mask,
            return_column,
        ].abs().astype(float),

        categorical,
    )

    probabilities = (
        direction_model
        .predict_proba(X_test)
    )

    model_classes = [
        int(value)
        for value in direction_model.classes_
    ]

    class_index = {
        class_value: index
        for index, class_value
        in enumerate(model_classes)
    }

    predicted_class = (
        direction_model
        .predict(X_test)
        .reshape(-1)
        .astype(int)
    )

    predicted_magnitude_log = (
        magnitude_model.predict(X_test)
    )

    predicted_magnitude_bps = np.maximum(
        0.0,
        np.expm1(
            predicted_magnitude_log
        ),
    )

    p_down = probabilities[
        :,
        class_index[-1],
    ]

    p_neutral = probabilities[
        :,
        class_index[0],
    ]

    p_up = probabilities[
        :,
        class_index[1],
    ]

    expected_return_bps = (
        p_up - p_down
    ) * predicted_magnitude_bps

    direction_confidence = probabilities.max(
        axis=1
    )

    actual_returns = (
        test[return_column]
        .astype(float)
        .to_numpy()
    )

    predictions = pd.DataFrame({
        "observation_time": (
            test["observation_time"]
            .to_numpy()
        ),
        "symbol": symbol,
        "fold": fold_name,

        "neutral_threshold_bps": threshold,

        "actual_return_bps": actual_returns,
        "actual_class": y_test.to_numpy(),

        "predicted_class": predicted_class,
        "direction_confidence": (
            direction_confidence
        ),

        "probability_down": p_down,
        "probability_neutral": p_neutral,
        "probability_up": p_up,

        "predicted_magnitude_bps": (
            predicted_magnitude_bps
        ),

        "expected_return_bps": (
            expected_return_bps
        ),
    })

    actual_magnitude_event = (
        predictions.loc[
            predictions["actual_class"] != 0,
            "actual_return_bps",
        ]
        .abs()
    )

    predicted_magnitude_event = (
        predictions.loc[
            predictions["actual_class"] != 0,
            "predicted_magnitude_bps",
        ]
    )

    event_precision, event_recall, event_f1, _ = (
        precision_recall_fscore_support(
            predictions["actual_class"] != 0,
            predictions["predicted_class"] != 0,
            average="binary",
            zero_division=0,
        )
    )

    event_rows = predictions[
        predictions["actual_class"] != 0
    ]

    predicted_event_rows = predictions[
        predictions["predicted_class"] != 0
    ]

    expectancy_pearson = (
        predictions[
            [
                "expected_return_bps",
                "actual_return_bps",
            ]
        ]
        .corr(method="pearson")
        .iloc[0, 1]
    )

    expectancy_spearman = (
        predictions[
            [
                "expected_return_bps",
                "actual_return_bps",
            ]
        ]
        .corr(method="spearman")
        .iloc[0, 1]
    )

    metrics = {
        "symbol": symbol,
        "fold": fold_name,
        "horizon_minutes": HORIZON_MINUTES,

        "neutral_threshold_bps": threshold,

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

        "direction": {
            "accuracy": float(
                accuracy_score(
                    predictions["actual_class"],
                    predictions["predicted_class"],
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    predictions["actual_class"],
                    predictions["predicted_class"],
                )
            ),
            "macro_f1": float(
                f1_score(
                    predictions["actual_class"],
                    predictions["predicted_class"],
                    average="macro",
                    zero_division=0,
                )
            ),
            "confusion_matrix": (
                confusion_matrix(
                    predictions["actual_class"],
                    predictions["predicted_class"],
                    labels=CLASS_ORDER,
                ).tolist()
            ),
            "classification_report": (
                classification_report(
                    predictions["actual_class"],
                    predictions["predicted_class"],
                    labels=CLASS_ORDER,
                    target_names=[
                        "DOWN",
                        "NEUTRAL",
                        "UP",
                    ],
                    output_dict=True,
                    zero_division=0,
                )
            ),
        },

        "event": {
            "precision": float(
                event_precision
            ),
            "recall": float(event_recall),
            "f1": float(event_f1),
            "predicted_event_rows": int(
                len(predicted_event_rows)
            ),
            "actual_event_rows": int(
                len(event_rows)
            ),
            "direction_accuracy_on_actual_events": (
                float(
                    (
                        event_rows[
                            "actual_class"
                        ]
                        == event_rows[
                            "predicted_class"
                        ]
                    ).mean()
                )
            ),
            "direction_accuracy_on_predicted_events": (
                float(
                    (
                        predicted_event_rows[
                            "actual_class"
                        ]
                        == predicted_event_rows[
                            "predicted_class"
                        ]
                    ).mean()
                )
                if len(predicted_event_rows)
                else None
            ),
        },

        "magnitude": {
            "event_mae_bps": (
                float(
                    mean_absolute_error(
                        actual_magnitude_event,
                        predicted_magnitude_event,
                    )
                )
                if len(
                    actual_magnitude_event
                )
                else None
            ),
            "event_rmse_bps": (
                float(
                    np.sqrt(
                        mean_squared_error(
                            actual_magnitude_event,
                            predicted_magnitude_event,
                        )
                    )
                )
                if len(
                    actual_magnitude_event
                )
                else None
            ),
        },

        "expectancy": {
            "pearson_correlation": float(
                expectancy_pearson
            ),
            "spearman_correlation": float(
                expectancy_spearman
            ),
            "sign_accuracy": float(
                (
                    np.sign(
                        predictions[
                            "expected_return_bps"
                        ]
                    )
                    == np.sign(
                        predictions[
                            "actual_return_bps"
                        ]
                    )
                ).mean()
            ),
            "mean_expected_return_bps": (
                float(
                    predictions[
                        "expected_return_bps"
                    ].mean()
                )
            ),
            "mean_actual_return_bps": (
                float(
                    predictions[
                        "actual_return_bps"
                    ].mean()
                )
            ),
            "buckets": (
                expectancy_bucket_report(
                    predictions
                )
            ),
        },

        "confidence_thresholds": (
            confidence_report(
                predictions
            )
        ),

        "direction_trees": int(
            direction_model.tree_count_
        ),
        "magnitude_trees": int(
            magnitude_model.tree_count_
        ),

        "elapsed_seconds": float(
            time.time() - started
        ),
    }

    return (
        metrics,
        predictions,
        direction_model,
        magnitude_model,
    )


def train_production_models(
    symbol_frame: pd.DataFrame,
    symbol: str,
    feature_columns: list[str],
) -> dict[str, Any]:
    """
    Final production model:
    ilk %90 train, son %10 validation.
    Test fold'ları yalnız research içindir.
    """
    ordered = (
        symbol_frame
        .sort_values(
            "observation_time",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    unique_times = np.array(
        sorted(
            ordered[
                "observation_time"
            ].dropna().unique()
        )
    )

    validation_start = unique_times[
        int(len(unique_times) * 0.90)
    ]

    train = ordered[
        ordered["observation_time"]
        < validation_start
    ].copy()

    validation = ordered[
        ordered["observation_time"]
        >= validation_start
    ].copy()

    return_column = (
        f"future_return_bps_"
        f"{HORIZON_MINUTES}m"
    )

    threshold = calculate_neutral_threshold(
        train[return_column]
    )

    y_train = create_direction_target(
        train[return_column],
        threshold,
    )

    y_validation = create_direction_target(
        validation[return_column],
        threshold,
    )

    train_valid = y_train.notna()
    validation_valid = y_validation.notna()

    train = train.loc[train_valid].copy()
    validation = validation.loc[
        validation_valid
    ].copy()

    y_train = y_train.loc[
        train_valid
    ].astype(int)

    y_validation = y_validation.loc[
        validation_valid
    ].astype(int)

    X_train = prepare_features(
        train,
        feature_columns,
    )

    X_validation = prepare_features(
        validation,
        feature_columns,
    )

    categorical = [
        column
        for column in CATEGORICAL_FEATURES
        if column in feature_columns
    ]

    direction_model = fit_direction_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        categorical,
    )

    event_train = y_train != 0
    event_validation = y_validation != 0

    magnitude_model = fit_magnitude_model(
        X_train.loc[event_train],
        train.loc[
            event_train,
            return_column,
        ].abs().astype(float),

        X_validation.loc[
            event_validation
        ],
        validation.loc[
            event_validation,
            return_column,
        ].abs().astype(float),

        categorical,
    )

    model_dir = (
        MODEL_ROOT
        / symbol
        / "15m"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    direction_model.save_model(
        model_dir
        / "direction_classifier.cbm"
    )

    magnitude_model.save_model(
        model_dir
        / "magnitude_regressor.cbm"
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
            categorical,
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = {
        "engine": "liqheat-forecast-v2",
        "symbol": symbol,
        "horizon_minutes": (
            HORIZON_MINUTES
        ),
        "neutral_threshold_bps": (
            threshold
        ),
        "neutral_quantile": (
            NEUTRAL_ABS_RETURN_QUANTILE
        ),
        "train_rows": int(len(train)),
        "validation_rows": int(
            len(validation)
        ),
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
        "feature_count": len(
            feature_columns
        ),
        "direction_trees": int(
            direction_model.tree_count_
        ),
        "magnitude_trees": int(
            magnitude_model.tree_count_
        ),
    }

    (
        model_dir
        / "metadata.json"
    ).write_text(
        json.dumps(
            json_safe(metadata),
            indent=2,
        ),
        encoding="utf-8",
    )

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": (
            direction_model
            .get_feature_importance()
        ),
    }).sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        model_dir
        / "feature_importance.csv",
        index=False,
    )

    return metadata


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
    print(
        "LIQHEAT FORECAST V2 — "
        "PAIR-SPECIFIC RETURN EXPECTANCY"
    )
    print("=" * 100)
    print("Dataset:", DATASET_PATH)
    print("Symbols:", SYMBOLS)
    print("Horizon:", HORIZON_MINUTES, "minutes")
    print()

    dataset = pd.read_parquet(
        DATASET_PATH
    )

    dataset[
        "observation_time"
    ] = pd.to_datetime(
        dataset["observation_time"],
        utc=True,
        errors="coerce",
    )

    valid_column = (
        f"future_valid_"
        f"{HORIZON_MINUTES}m"
    )

    return_column = (
        f"future_return_bps_"
        f"{HORIZON_MINUTES}m"
    )

    dataset = dataset[
        (dataset[valid_column] == 1)
        & dataset[return_column].notna()
        & dataset[
            "observation_time"
        ].notna()
    ].copy()

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

    all_metrics: list[
        dict[str, Any]
    ] = []

    all_predictions: list[
        pd.DataFrame
    ] = []

    production_metadata = []

    for symbol in SYMBOLS:
        print()
        print("=" * 100)
        print(symbol)
        print("=" * 100)

        symbol_frame = dataset[
            dataset["symbol"].astype(str)
            == symbol
        ].copy()

        print(
            "Symbol rows:",
            f"{len(symbol_frame):,}",
        )

        folds = build_walk_forward_folds(
            symbol_frame
        )

        symbol_metrics = []

        for (
            fold_name,
            train,
            validation,
            test,
        ) in folds:
            (
                metrics,
                predictions,
                _direction_model,
                _magnitude_model,
            ) = evaluate_fold(
                symbol,
                fold_name,
                train,
                validation,
                test,
                feature_columns,
            )

            symbol_metrics.append(metrics)
            all_metrics.append(metrics)
            all_predictions.append(
                predictions
            )

            fold_dir = (
                REPORT_ROOT
                / symbol
                / fold_name
            )

            fold_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            (
                fold_dir
                / "metrics.json"
            ).write_text(
                json.dumps(
                    json_safe(metrics),
                    indent=2,
                ),
                encoding="utf-8",
            )

            predictions.to_parquet(
                fold_dir
                / "predictions.parquet",
                index=False,
                compression="zstd",
            )

            print()
            print(
                f"{symbol} {fold_name}:"
            )

            print(
                "  balanced accuracy:",
                round(
                    metrics[
                        "direction"
                    ][
                        "balanced_accuracy"
                    ],
                    4,
                ),
            )

            print(
                "  macro F1:",
                round(
                    metrics[
                        "direction"
                    ]["macro_f1"],
                    4,
                ),
            )

            print(
                "  expectancy Pearson:",
                round(
                    metrics[
                        "expectancy"
                    ][
                        "pearson_correlation"
                    ],
                    4,
                ),
            )

            print(
                "  expectancy Spearman:",
                round(
                    metrics[
                        "expectancy"
                    ][
                        "spearman_correlation"
                    ],
                    4,
                ),
            )

        print()
        print(
            f"Training production model: "
            f"{symbol}"
        )

        production_metadata.append(
            train_production_models(
                symbol_frame,
                symbol,
                feature_columns,
            )
        )

        symbol_report_dir = (
            REPORT_ROOT / symbol
        )

        symbol_report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            symbol_report_dir
            / "walk_forward_summary.json"
        ).write_text(
            json.dumps(
                json_safe(
                    symbol_metrics
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

    metrics_rows = []

    for metrics in all_metrics:
        metrics_rows.append({
            "symbol": metrics["symbol"],
            "fold": metrics["fold"],
            "neutral_threshold_bps": (
                metrics[
                    "neutral_threshold_bps"
                ]
            ),
            "test_rows": (
                metrics["test_rows"]
            ),
            "accuracy": (
                metrics[
                    "direction"
                ]["accuracy"]
            ),
            "balanced_accuracy": (
                metrics[
                    "direction"
                ][
                    "balanced_accuracy"
                ]
            ),
            "macro_f1": (
                metrics[
                    "direction"
                ]["macro_f1"]
            ),
            "event_precision": (
                metrics[
                    "event"
                ]["precision"]
            ),
            "event_recall": (
                metrics[
                    "event"
                ]["recall"]
            ),
            "event_direction_accuracy": (
                metrics[
                    "event"
                ][
                    "direction_accuracy_on_predicted_events"
                ]
            ),
            "magnitude_mae_bps": (
                metrics[
                    "magnitude"
                ]["event_mae_bps"]
            ),
            "expectancy_pearson": (
                metrics[
                    "expectancy"
                ][
                    "pearson_correlation"
                ]
            ),
            "expectancy_spearman": (
                metrics[
                    "expectancy"
                ][
                    "spearman_correlation"
                ]
            ),
            "expectancy_sign_accuracy": (
                metrics[
                    "expectancy"
                ]["sign_accuracy"]
            ),
            "direction_trees": (
                metrics[
                    "direction_trees"
                ]
            ),
            "magnitude_trees": (
                metrics[
                    "magnitude_trees"
                ]
            ),
            "elapsed_seconds": (
                metrics[
                    "elapsed_seconds"
                ]
            ),
        })

    metrics_frame = pd.DataFrame(
        metrics_rows
    )

    metrics_frame.to_csv(
        REPORT_ROOT
        / "walk_forward_metrics.csv",
        index=False,
    )

    combined_predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    combined_predictions.to_parquet(
        REPORT_ROOT
        / "walk_forward_predictions.parquet",
        index=False,
        compression="zstd",
    )

    symbol_summary = (
        metrics_frame
        .groupby(
            "symbol",
            observed=True,
        )
        .agg(
            folds=(
                "fold",
                "count",
            ),
            mean_balanced_accuracy=(
                "balanced_accuracy",
                "mean",
            ),
            minimum_balanced_accuracy=(
                "balanced_accuracy",
                "min",
            ),
            mean_macro_f1=(
                "macro_f1",
                "mean",
            ),
            mean_event_precision=(
                "event_precision",
                "mean",
            ),
            mean_event_direction_accuracy=(
                "event_direction_accuracy",
                "mean",
            ),
            mean_expectancy_pearson=(
                "expectancy_pearson",
                "mean",
            ),
            minimum_expectancy_pearson=(
                "expectancy_pearson",
                "min",
            ),
            mean_expectancy_spearman=(
                "expectancy_spearman",
                "mean",
            ),
            mean_expectancy_sign_accuracy=(
                "expectancy_sign_accuracy",
                "mean",
            ),
            mean_magnitude_mae_bps=(
                "magnitude_mae_bps",
                "mean",
            ),
        )
        .reset_index()
    )

    symbol_summary.to_csv(
        REPORT_ROOT
        / "symbol_summary.csv",
        index=False,
    )

    final_report = {
        "status": "complete",
        "engine": "liqheat-forecast-v2",
        "horizon_minutes": (
            HORIZON_MINUTES
        ),
        "symbols": SYMBOLS,
        "dataset_rows": int(
            len(dataset)
        ),
        "feature_count": int(
            len(feature_columns)
        ),
        "neutral_quantile": (
            NEUTRAL_ABS_RETURN_QUANTILE
        ),
        "walk_forward_folds": 3,
        "production_models": (
            production_metadata
        ),
        "symbol_summary": (
            symbol_summary
            .to_dict(
                orient="records"
            )
        ),
        "elapsed_seconds": float(
            time.time() - started
        ),
    }

    (
        REPORT_ROOT
        / "forecast_v2_report.json"
    ).write_text(
        json.dumps(
            json_safe(final_report),
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("FORECAST V2 COMPLETE")
    print("=" * 120)
    print()
    print(
        symbol_summary.to_string(
            index=False
        )
    )
    print()
    print(
        "Report:",
        REPORT_ROOT
        / "forecast_v2_report.json",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
