#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


DATASET_PATH = Path(
    "data/forecast_v3/"
    "matrix_topology_dataset.parquet"
)

MODEL_ROOT = Path(
    "models/forecast_v3_ablation"
)

REPORT_ROOT = Path(
    "reports/forecast_v3_ablation"
)

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

EXPERIMENTS = [
    "MATRIX_ONLY",
    "TOPOLOGY_ONLY",
    "MATRIX_PLUS_TOPOLOGY",
]

HORIZON_MINUTES = 15
RETURN_COLUMN = "future_return_bps_15m"
VALID_COLUMN = "future_valid_15m"

RANDOM_SEED = 42

NEUTRAL_ABS_RETURN_QUANTILE = 0.35
MIN_NEUTRAL_BPS = 4.0
MAX_NEUTRAL_BPS = 45.0

GPU_DEVICE = "0"
GPU_RAM_PART = 0.90

CLASS_ORDER = [-1, 0, 1]

IDENTIFIER_COLUMNS = {
    "symbol",
    "observation_id",
    "observation_time",
    "observation_month",

    "tf4h_source_id",
    "tf4h_source_time",
    "tf24h_source_id",
    "tf24h_source_time",

    "matrix_1h_source_available_at",
    "matrix_4h_source_available_at",
    "matrix_24h_source_available_at",
}

RAW_PRICE_COLUMNS = {
    "tf1h_current_price",
    "tf4h_current_price",
    "tf24h_current_price",

    "matrix_1h_open",
    "matrix_1h_high",
    "matrix_1h_low",
    "matrix_1h_close",

    "matrix_4h_open",
    "matrix_4h_high",
    "matrix_4h_low",
    "matrix_4h_close",

    "matrix_24h_open",
    "matrix_24h_high",
    "matrix_24h_low",
    "matrix_24h_close",

    "matrix_1h_matrix_source",
    "matrix_1h_matrix_vwma",
    "matrix_1h_matrix_upper",
    "matrix_1h_matrix_lower",

    "matrix_4h_matrix_source",
    "matrix_4h_matrix_vwma",
    "matrix_4h_matrix_upper",
    "matrix_4h_matrix_lower",

    "matrix_24h_matrix_source",
    "matrix_24h_matrix_vwma",
    "matrix_24h_matrix_upper",
    "matrix_24h_matrix_lower",
}

CATEGORICAL_CANDIDATES = {
    "tf1h_nearest_side",
    "tf4h_nearest_side",
    "tf24h_nearest_side",
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
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


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

    return pd.Series(
        values,
        index=returns_bps.index,
        dtype="int8",
    ).where(
        numeric.notna()
    )


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
            "Not enough timestamps "
            "for walk-forward validation"
        )

    specifications = [
        ("fold_1", 0.40, 0.50, 0.65),
        ("fold_2", 0.55, 0.65, 0.80),
        ("fold_3", 0.70, 0.80, 1.00),
    ]

    output = []

    for (
        fold_name,
        train_end_ratio,
        validation_end_ratio,
        test_end_ratio,
    ) in specifications:
        train_end = unique_times[
            int(
                len(unique_times)
                * train_end_ratio
            )
        ]

        validation_end = unique_times[
            int(
                len(unique_times)
                * validation_end_ratio
            )
        ]

        test_end = (
            None
            if test_end_ratio >= 1.0
            else unique_times[
                int(
                    len(unique_times)
                    * test_end_ratio
                )
            ]
        )

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
                f"Empty split: {fold_name}"
            )

        output.append(
            (
                fold_name,
                train,
                validation,
                test,
            )
        )

    return output


def is_matrix_feature(
    column: str,
) -> bool:
    return column.startswith("matrix_")


def is_target_or_future(
    column: str,
) -> bool:
    return (
        column.startswith("future_")
        or column.startswith("target_")
    )


def is_topology_feature(
    column: str,
) -> bool:
    if is_matrix_feature(column):
        return False

    if is_target_or_future(column):
        return False

    if column in IDENTIFIER_COLUMNS:
        return False

    if column in RAW_PRICE_COLUMNS:
        return False

    return True


def select_feature_columns(
    frame: pd.DataFrame,
    experiment: str,
) -> list[str]:
    selected = []

    for column in frame.columns:
        if column in IDENTIFIER_COLUMNS:
            continue

        if column in RAW_PRICE_COLUMNS:
            continue

        if is_target_or_future(column):
            continue

        if experiment == "MATRIX_ONLY":
            if is_matrix_feature(column):
                selected.append(column)

        elif experiment == "TOPOLOGY_ONLY":
            if is_topology_feature(column):
                selected.append(column)

        elif experiment == "MATRIX_PLUS_TOPOLOGY":
            if (
                is_matrix_feature(column)
                or is_topology_feature(column)
            ):
                selected.append(column)

        else:
            raise ValueError(
                f"Unknown experiment: {experiment}"
            )

    selected = sorted(set(selected))

    if not selected:
        raise RuntimeError(
            f"No features selected: {experiment}"
        )

    return selected


def prepare_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    output = frame[
        feature_columns
    ].copy()

    for column in feature_columns:
        if column in CATEGORICAL_CANDIDATES:
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


def get_categorical_features(
    feature_columns: list[str],
) -> list[str]:
    return [
        column
        for column in feature_columns
        if column in CATEGORICAL_CANDIDATES
    ]


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    categorical_features: list[str],
) -> CatBoostClassifier:
    model = CatBoostClassifier(
        iterations=1600,
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
        od_wait=160,
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


def expectancy_from_probabilities(
    probabilities: np.ndarray,
    model_classes: list[int],
    neutral_threshold_bps: float,
) -> np.ndarray:
    """
    Directional expectancy proxy.

    Bu deneyde magnitude modelini bilinçli olarak
    kullanmıyoruz. Üç feature setini yalnız yön
    ayrıştırma gücü bakımından karşılaştırıyoruz.

    Proxy:
        (P(up) - P(down)) × neutral threshold
    """
    class_index = {
        class_value: index
        for index, class_value
        in enumerate(model_classes)
    }

    p_down = probabilities[
        :,
        class_index[-1],
    ]

    p_up = probabilities[
        :,
        class_index[1],
    ]

    return (
        p_up - p_down
    ) * neutral_threshold_bps


def build_expectancy_buckets(
    predictions: pd.DataFrame,
) -> list[dict[str, Any]]:
    working = predictions[
        predictions[
            "expectancy_score"
        ].notna()
        & predictions[
            "actual_return_bps"
        ].notna()
    ].copy()

    if len(working) < 100:
        return []

    try:
        working[
            "expectancy_bucket"
        ] = pd.qcut(
            working[
                "expectancy_score"
            ],
            q=10,
            labels=False,
            duplicates="drop",
        )

    except ValueError:
        return []

    output = []

    for bucket, group in working.groupby(
        "expectancy_bucket",
        observed=True,
        sort=True,
    ):
        output.append({
            "bucket": int(bucket),
            "rows": int(len(group)),
            "predicted_mean": float(
                group[
                    "expectancy_score"
                ].mean()
            ),
            "actual_mean_return_bps": float(
                group[
                    "actual_return_bps"
                ].mean()
            ),
            "actual_median_return_bps": float(
                group[
                    "actual_return_bps"
                ].median()
            ),
            "positive_return_rate": float(
                (
                    group[
                        "actual_return_bps"
                    ] > 0
                ).mean()
            ),
            "negative_return_rate": float(
                (
                    group[
                        "actual_return_bps"
                    ] < 0
                ).mean()
            ),
        })

    return output


def confidence_report(
    predictions: pd.DataFrame,
) -> list[dict[str, Any]]:
    output = []

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
            predictions[
                "direction_confidence"
            ] >= threshold
        ]

        if selected.empty:
            continue

        predicted_events = selected[
            selected[
                "predicted_class"
            ] != 0
        ]

        output.append({
            "minimum_confidence": threshold,
            "rows": int(len(selected)),
            "coverage": float(
                len(selected)
                / len(predictions)
            ),
            "accuracy": float(
                accuracy_score(
                    selected[
                        "actual_class"
                    ],
                    selected[
                        "predicted_class"
                    ],
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    selected[
                        "actual_class"
                    ],
                    selected[
                        "predicted_class"
                    ],
                )
            ),
            "predicted_event_rows": int(
                len(predicted_events)
            ),
            "predicted_event_share": float(
                len(predicted_events)
                / len(selected)
            ),
            "event_precision": (
                float(
                    (
                        predicted_events[
                            "actual_class"
                        ] != 0
                    ).mean()
                )
                if len(predicted_events)
                else None
            ),
            "direction_accuracy": (
                float(
                    (
                        predicted_events[
                            "actual_class"
                        ]
                        == predicted_events[
                            "predicted_class"
                        ]
                    ).mean()
                )
                if len(predicted_events)
                else None
            ),
        })

    return output


def evaluate_fold(
    experiment: str,
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
]:
    started = time.time()

    neutral_threshold = (
        calculate_neutral_threshold(
            train[RETURN_COLUMN]
        )
    )

    y_train = create_direction_target(
        train[RETURN_COLUMN],
        neutral_threshold,
    )

    y_validation = create_direction_target(
        validation[RETURN_COLUMN],
        neutral_threshold,
    )

    y_test = create_direction_target(
        test[RETURN_COLUMN],
        neutral_threshold,
    )

    train_valid = y_train.notna()
    validation_valid = y_validation.notna()
    test_valid = y_test.notna()

    train = train.loc[
        train_valid
    ].copy()

    validation = validation.loc[
        validation_valid
    ].copy()

    test = test.loc[
        test_valid
    ].copy()

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

    categorical_features = (
        get_categorical_features(
            feature_columns
        )
    )

    print()
    print("-" * 110)
    print(
        f"{experiment} | "
        f"{symbol} | "
        f"{fold_name} | "
        f"features={len(feature_columns)} | "
        f"neutral={neutral_threshold:.3f} bps"
    )
    print("-" * 110)

    print(
        "Train classes:",
        y_train
        .value_counts()
        .sort_index()
        .to_dict(),
    )

    model = fit_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        categorical_features,
    )

    predicted_class = (
        model
        .predict(X_test)
        .reshape(-1)
        .astype(int)
    )

    probabilities = (
        model.predict_proba(X_test)
    )

    model_classes = [
        int(value)
        for value in model.classes_
    ]

    class_index = {
        class_value: index
        for index, class_value
        in enumerate(model_classes)
    }

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

    confidence = probabilities.max(
        axis=1
    )

    expectancy_score = (
        expectancy_from_probabilities(
            probabilities,
            model_classes,
            neutral_threshold,
        )
    )

    actual_returns = (
        test[RETURN_COLUMN]
        .astype(float)
        .to_numpy()
    )

    predictions = pd.DataFrame({
        "experiment": experiment,
        "symbol": symbol,
        "fold": fold_name,

        "observation_time": (
            test[
                "observation_time"
            ].to_numpy()
        ),

        "neutral_threshold_bps": (
            neutral_threshold
        ),

        "actual_return_bps": (
            actual_returns
        ),

        "actual_class": (
            y_test.to_numpy()
        ),

        "predicted_class": (
            predicted_class
        ),

        "direction_confidence": (
            confidence
        ),

        "probability_down": p_down,
        "probability_neutral": p_neutral,
        "probability_up": p_up,

        "expectancy_score": (
            expectancy_score
        ),
    })

    event_precision, event_recall, event_f1, _ = (
        precision_recall_fscore_support(
            predictions[
                "actual_class"
            ] != 0,
            predictions[
                "predicted_class"
            ] != 0,
            average="binary",
            zero_division=0,
        )
    )

    predicted_event_rows = predictions[
        predictions[
            "predicted_class"
        ] != 0
    ]

    expectancy_pearson = (
        predictions[
            [
                "expectancy_score",
                "actual_return_bps",
            ]
        ]
        .corr(method="pearson")
        .iloc[0, 1]
    )

    expectancy_spearman = (
        predictions[
            [
                "expectancy_score",
                "actual_return_bps",
            ]
        ]
        .corr(method="spearman")
        .iloc[0, 1]
    )

    report = classification_report(
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

    metrics = {
        "experiment": experiment,
        "symbol": symbol,
        "fold": fold_name,

        "feature_count": int(
            len(feature_columns)
        ),

        "neutral_threshold_bps": (
            neutral_threshold
        ),

        "train_rows": int(len(train)),
        "validation_rows": int(
            len(validation)
        ),
        "test_rows": int(len(test)),

        "train_start": (
            train[
                "observation_time"
            ].min().isoformat()
        ),

        "train_end": (
            train[
                "observation_time"
            ].max().isoformat()
        ),

        "test_start": (
            test[
                "observation_time"
            ].min().isoformat()
        ),

        "test_end": (
            test[
                "observation_time"
            ].max().isoformat()
        ),

        "accuracy": float(
            accuracy_score(
                predictions[
                    "actual_class"
                ],
                predictions[
                    "predicted_class"
                ],
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                predictions[
                    "actual_class"
                ],
                predictions[
                    "predicted_class"
                ],
            )
        ),

        "macro_f1": float(
            f1_score(
                predictions[
                    "actual_class"
                ],
                predictions[
                    "predicted_class"
                ],
                average="macro",
                zero_division=0,
            )
        ),

        "event_precision": float(
            event_precision
        ),

        "event_recall": float(
            event_recall
        ),

        "event_f1": float(
            event_f1
        ),

        "event_direction_accuracy": (
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

        "expectancy_pearson": float(
            expectancy_pearson
        ),

        "expectancy_spearman": float(
            expectancy_spearman
        ),

        "expectancy_sign_accuracy": float(
            (
                np.sign(
                    predictions[
                        "expectancy_score"
                    ]
                )
                == np.sign(
                    predictions[
                        "actual_return_bps"
                    ]
                )
            ).mean()
        ),

        "confusion_matrix": (
            confusion_matrix(
                predictions[
                    "actual_class"
                ],
                predictions[
                    "predicted_class"
                ],
                labels=CLASS_ORDER,
            ).tolist()
        ),

        "classification_report": report,

        "confidence_thresholds": (
            confidence_report(
                predictions
            )
        ),

        "expectancy_buckets": (
            build_expectancy_buckets(
                predictions
            )
        ),

        "tree_count": int(
            model.tree_count_
        ),

        "elapsed_seconds": float(
            time.time() - started
        ),
    }

    return metrics, predictions, model


def save_feature_manifest(
    experiment: str,
    feature_columns: list[str],
) -> None:
    experiment_dir = (
        MODEL_ROOT / experiment
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        experiment_dir
        / "features.json"
    ).write_text(
        json.dumps(
            feature_columns,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    started = time.time()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset missing: "
            f"{DATASET_PATH}"
        )

    MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 120)
    print(
        "LIQHEAT FORECAST V3 — "
        "MATRIX / TOPOLOGY ABLATION"
    )
    print("=" * 120)
    print("Dataset:", DATASET_PATH)
    print("Symbols:", SYMBOLS)
    print("Experiments:", EXPERIMENTS)
    print("Horizon:", HORIZON_MINUTES, "minutes")
    print()

    dataset = pd.read_parquet(
        DATASET_PATH
    )

    dataset[
        "observation_time"
    ] = pd.to_datetime(
        dataset[
            "observation_time"
        ],
        utc=True,
        errors="coerce",
    )

    dataset = dataset[
        (dataset[VALID_COLUMN] == 1)
        & dataset[
            RETURN_COLUMN
        ].notna()
        & dataset[
            "observation_time"
        ].notna()
    ].copy()

    print(
        "Rows:",
        f"{len(dataset):,}",
    )

    feature_sets = {}

    for experiment in EXPERIMENTS:
        feature_sets[experiment] = (
            select_feature_columns(
                dataset,
                experiment,
            )
        )

        save_feature_manifest(
            experiment,
            feature_sets[experiment],
        )

        print(
            f"{experiment}: "
            f"{len(feature_sets[experiment])} "
            f"features"
        )

    all_metrics = []
    all_predictions = []

    for symbol in SYMBOLS:
        symbol_frame = dataset[
            dataset["symbol"].astype(str)
            == symbol
        ].copy()

        print()
        print("=" * 120)
        print(
            symbol,
            f"rows={len(symbol_frame):,}",
        )
        print("=" * 120)

        folds = build_walk_forward_folds(
            symbol_frame
        )

        for experiment in EXPERIMENTS:
            feature_columns = (
                feature_sets[
                    experiment
                ]
            )

            for (
                fold_name,
                train,
                validation,
                test,
            ) in folds:
                (
                    metrics,
                    predictions,
                    model,
                ) = evaluate_fold(
                    experiment,
                    symbol,
                    fold_name,
                    train,
                    validation,
                    test,
                    feature_columns,
                )

                all_metrics.append(
                    metrics
                )

                all_predictions.append(
                    predictions
                )

                fold_dir = (
                    MODEL_ROOT
                    / experiment
                    / symbol
                    / fold_name
                )

                fold_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                model.save_model(
                    fold_dir
                    / "direction_classifier.cbm"
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

                importance = pd.DataFrame({
                    "feature": (
                        feature_columns
                    ),
                    "importance": (
                        model
                        .get_feature_importance()
                    ),
                }).sort_values(
                    "importance",
                    ascending=False,
                )

                importance.to_csv(
                    fold_dir
                    / "feature_importance.csv",
                    index=False,
                )

                print()
                print(
                    f"RESULT "
                    f"{experiment} "
                    f"{symbol} "
                    f"{fold_name}"
                )

                print(
                    "  balanced_accuracy:",
                    round(
                        metrics[
                            "balanced_accuracy"
                        ],
                        4,
                    ),
                )

                print(
                    "  macro_f1:",
                    round(
                        metrics[
                            "macro_f1"
                        ],
                        4,
                    ),
                )

                print(
                    "  event_precision:",
                    round(
                        metrics[
                            "event_precision"
                        ],
                        4,
                    ),
                )

                print(
                    "  event_direction_accuracy:",
                    round(
                        metrics[
                            "event_direction_accuracy"
                        ]
                        or 0,
                        4,
                    ),
                )

                print(
                    "  expectancy_pearson:",
                    round(
                        metrics[
                            "expectancy_pearson"
                        ],
                        4,
                    ),
                )

                print(
                    "  expectancy_spearman:",
                    round(
                        metrics[
                            "expectancy_spearman"
                        ],
                        4,
                    ),
                )

    metrics_frame = pd.DataFrame([
        {
            "experiment": row[
                "experiment"
            ],
            "symbol": row["symbol"],
            "fold": row["fold"],
            "feature_count": row[
                "feature_count"
            ],
            "neutral_threshold_bps": (
                row[
                    "neutral_threshold_bps"
                ]
            ),
            "test_rows": row[
                "test_rows"
            ],
            "accuracy": row[
                "accuracy"
            ],
            "balanced_accuracy": (
                row[
                    "balanced_accuracy"
                ]
            ),
            "macro_f1": row[
                "macro_f1"
            ],
            "event_precision": (
                row[
                    "event_precision"
                ]
            ),
            "event_recall": (
                row[
                    "event_recall"
                ]
            ),
            "event_f1": row[
                "event_f1"
            ],
            "event_direction_accuracy": (
                row[
                    "event_direction_accuracy"
                ]
            ),
            "expectancy_pearson": (
                row[
                    "expectancy_pearson"
                ]
            ),
            "expectancy_spearman": (
                row[
                    "expectancy_spearman"
                ]
            ),
            "expectancy_sign_accuracy": (
                row[
                    "expectancy_sign_accuracy"
                ]
            ),
            "tree_count": row[
                "tree_count"
            ],
            "elapsed_seconds": (
                row[
                    "elapsed_seconds"
                ]
            ),
        }
        for row in all_metrics
    ])

    metrics_frame.to_csv(
        REPORT_ROOT
        / "fold_metrics.csv",
        index=False,
    )

    predictions_frame = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_frame.to_parquet(
        REPORT_ROOT
        / "walk_forward_predictions.parquet",
        index=False,
        compression="zstd",
    )

    experiment_symbol_summary = (
        metrics_frame
        .groupby(
            [
                "experiment",
                "symbol",
            ],
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
        )
        .reset_index()
    )

    experiment_symbol_summary.to_csv(
        REPORT_ROOT
        / "experiment_symbol_summary.csv",
        index=False,
    )

    experiment_summary = (
        metrics_frame
        .groupby(
            "experiment",
            observed=True,
        )
        .agg(
            models=(
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
        )
        .reset_index()
        .sort_values(
            "mean_expectancy_spearman",
            ascending=False,
        )
    )

    experiment_summary.to_csv(
        REPORT_ROOT
        / "experiment_summary.csv",
        index=False,
    )

    matrix_only = (
        experiment_symbol_summary[
            experiment_symbol_summary[
                "experiment"
            ] == "MATRIX_ONLY"
        ]
        .set_index("symbol")
    )

    topology_only = (
        experiment_symbol_summary[
            experiment_symbol_summary[
                "experiment"
            ] == "TOPOLOGY_ONLY"
        ]
        .set_index("symbol")
    )

    combined = (
        experiment_symbol_summary[
            experiment_symbol_summary[
                "experiment"
            ]
            == "MATRIX_PLUS_TOPOLOGY"
        ]
        .set_index("symbol")
    )

    comparison_rows = []

    for symbol in SYMBOLS:
        comparison_rows.append({
            "symbol": symbol,

            "matrix_balanced_accuracy": (
                matrix_only.loc[
                    symbol,
                    "mean_balanced_accuracy",
                ]
            ),

            "topology_balanced_accuracy": (
                topology_only.loc[
                    symbol,
                    "mean_balanced_accuracy",
                ]
            ),

            "combined_balanced_accuracy": (
                combined.loc[
                    symbol,
                    "mean_balanced_accuracy",
                ]
            ),

            "combined_minus_matrix_balanced_accuracy": (
                combined.loc[
                    symbol,
                    "mean_balanced_accuracy",
                ]
                - matrix_only.loc[
                    symbol,
                    "mean_balanced_accuracy",
                ]
            ),

            "combined_minus_topology_balanced_accuracy": (
                combined.loc[
                    symbol,
                    "mean_balanced_accuracy",
                ]
                - topology_only.loc[
                    symbol,
                    "mean_balanced_accuracy",
                ]
            ),

            "matrix_expectancy_spearman": (
                matrix_only.loc[
                    symbol,
                    "mean_expectancy_spearman",
                ]
            ),

            "topology_expectancy_spearman": (
                topology_only.loc[
                    symbol,
                    "mean_expectancy_spearman",
                ]
            ),

            "combined_expectancy_spearman": (
                combined.loc[
                    symbol,
                    "mean_expectancy_spearman",
                ]
            ),

            "combined_minus_matrix_spearman": (
                combined.loc[
                    symbol,
                    "mean_expectancy_spearman",
                ]
                - matrix_only.loc[
                    symbol,
                    "mean_expectancy_spearman",
                ]
            ),

            "combined_minus_topology_spearman": (
                combined.loc[
                    symbol,
                    "mean_expectancy_spearman",
                ]
                - topology_only.loc[
                    symbol,
                    "mean_expectancy_spearman",
                ]
            ),
        })

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison.to_csv(
        REPORT_ROOT
        / "direct_comparison.csv",
        index=False,
    )

    final_report = {
        "status": "complete",
        "engine": (
            "liqheat-forecast-v3-ablation"
        ),
        "dataset": str(
            DATASET_PATH
        ),
        "dataset_rows": int(
            len(dataset)
        ),
        "symbols": SYMBOLS,
        "experiments": (
            EXPERIMENTS
        ),
        "horizon_minutes": (
            HORIZON_MINUTES
        ),
        "feature_counts": {
            experiment: len(
                feature_sets[
                    experiment
                ]
            )
            for experiment
            in EXPERIMENTS
        },
        "experiment_summary": (
            experiment_summary
            .to_dict(
                orient="records"
            )
        ),
        "symbol_comparison": (
            comparison
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
        / "forecast_v3_ablation_report.json"
    ).write_text(
        json.dumps(
            json_safe(
                final_report
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 140)
    print(
        "FORECAST V3 ABLATION COMPLETE"
    )
    print("=" * 140)

    print()
    print("EXPERIMENT SUMMARY")
    print(
        experiment_summary.to_string(
            index=False
        )
    )

    print()
    print("DIRECT SYMBOL COMPARISON")
    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print(
        "Report:",
        REPORT_ROOT
        / "forecast_v3_ablation_report.json",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
