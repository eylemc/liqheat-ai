from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import itertools
import json
import math
import sys
import time

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool


# =============================================================================
# PATHS
# =============================================================================

FEATURE_PATH = Path(
    "data/features/liq_topology_v2_ml_features.parquet"
)

EXPERIMENT_DIR = Path(
    "data/research/topology_v2_squeeze_grid/"
    "tf_1h__future_60m__precursor_5m__q_0p975"
)

EVENT_DATASET_PATH = (
    EXPERIMENT_DIR
    / "squeeze_event_dataset.parquet"
)

DETECTED_EVENTS_PATH = (
    EXPERIMENT_DIR
    / "detected_squeeze_events.parquet"
)

REFERENCE_PREDICTIONS_PATH = (
    EXPERIMENT_DIR
    / "walk_forward_predictions.parquet"
)

OUTPUT_DIR = Path(
    "data/backtests/"
    "topology_v2_squeeze_fixed_50_25_cost_grid"
)

SCORED_SNAPSHOTS_PATH = (
    OUTPUT_DIR
    / "full_stream_oos_scores.parquet"
)

THRESHOLDS_PATH = (
    OUTPUT_DIR
    / "fold_thresholds.csv"
)

ALERTS_PATH = (
    OUTPUT_DIR
    / "all_alerts.parquet"
)

TRADES_PATH = (
    OUTPUT_DIR
    / "all_trades.parquet"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "backtest_summary.csv"
)

TOP_CONFIGS_PATH = (
    OUTPUT_DIR
    / "top_configurations.csv"
)

FOLD_METRICS_PATH = (
    OUTPUT_DIR
    / "fold_metrics.csv"
)

SYMBOL_METRICS_PATH = (
    OUTPUT_DIR
    / "symbol_metrics.csv"
)

BEST_ALERTS_PATH = (
    OUTPUT_DIR
    / "best_configuration_alerts.parquet"
)

BEST_TRADES_PATH = (
    OUTPUT_DIR
    / "best_configuration_trades.parquet"
)

FEATURE_IMPORTANCE_PATH = (
    OUTPUT_DIR
    / "feature_importance.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "report.json"
)


# =============================================================================
# MODEL AND EVENT SETTINGS
# =============================================================================

TOPOLOGY_TIMEFRAME = "1h"

EVENT_WINDOW = pd.Timedelta(
    minutes=60
)

EMBARGO = pd.Timedelta(
    hours=1
)

VALIDATION_FRACTION = 0.20

RANDOM_STATE = 42

LONG_SQUEEZE_CLASS = -1
NO_EVENT_CLASS = 0
SHORT_SQUEEZE_CLASS = 1

CLASS_VALUES = [
    LONG_SQUEEZE_CLASS,
    NO_EVENT_CLASS,
    SHORT_SQUEEZE_CLASS,
]

TRADE_SHORT = -1
TRADE_LONG = 1


# =============================================================================
# EXECUTION COSTS
# =============================================================================

# Total round-trip trading-cost scenarios.
# Includes fees, spread and slippage together.
ROUND_TRIP_COST_BPS_VALUES = [
    0.0,
    4.0,
    8.0,
    14.0,
]


# =============================================================================
# ALERT GRID
# =============================================================================

# Thresholds are selected from validation full-stream percentiles.
# 0.99 means only the highest-scoring 1% of validation snapshots.
ALERT_PERCENTILES = [
    0.990,
    0.995,
    0.9975,
    0.999,
]

MIN_DIRECTION_CONFIDENCE_VALUES = [
    0.55,
    0.60,
    0.65,
    0.70,
]

COOLDOWN_MINUTES_VALUES = [
    60,
    120,
    240,
]

# Alarm re-arms only after the squeeze score has dropped materially.
REARM_SCORE_MULTIPLIER = 0.75


# =============================================================================
# EXECUTION GRID
# =============================================================================

ENTRY_DELAY_SECONDS_VALUES = [
    0,
    73,
]

MAX_HOLD_MINUTES_VALUES = [
    30,
    60,
    120,
]

TARGET_MODES = [
    "nearest_liquidity",
    "fixed_50bp",
    "fixed_100bp",
]

STOP_MODES = [
    "opposite_liquidity",
    "fixed_50bp",
    "fixed_100bp",
]

MIN_TARGET_DISTANCE_BPS_VALUES = [
    0,
    25,
    50,
]

ONE_POSITION_PER_SYMBOL = True

MIN_TRADES_FOR_RANKING = 25

NOTIONAL_FRACTION_PER_TRADE = 0.10


# =============================================================================
# FEATURES
# =============================================================================

CATEGORICAL_FEATURES = [
    "symbol",
    "nearest_side",
]

NUMERIC_FEATURES = [
    "current_price",

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

    "pool_volume_ratio",
    "log1p_pool_volume_ratio",

    "distance_pressure_ratio",
    "log1p_distance_pressure_ratio",

    "topology_imbalance",
    "total_volume_imbalance_check",

    "active_level_difference",
    "active_level_total",

    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend_utc",
]

MODEL_FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)

STREAM_EXTRA_COLUMNS = [
    "id",
    "logged_at",
    "timeframe",

    "nearest_upper_price",
    "nearest_lower_price",
]

STREAM_COLUMNS = list(
    dict.fromkeys(
        STREAM_EXTRA_COLUMNS
        + MODEL_FEATURES
    )
)


# =============================================================================
# CONFIGURATION TYPES
# =============================================================================

@dataclass(frozen=True)
class AlertConfig:
    alert_percentile: float
    minimum_direction_confidence: float
    cooldown_minutes: int


@dataclass(frozen=True)
class ExecutionConfig:
    entry_delay_seconds: int
    max_hold_minutes: int
    target_mode: str
    stop_mode: str
    minimum_target_distance_bps: int
    round_trip_cost_bps: float


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def ensure_files() -> None:
    for path in [
        FEATURE_PATH,
        EVENT_DATASET_PATH,
        DETECTED_EVENTS_PATH,
        REFERENCE_PREDICTIONS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )


def prepare_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame[
        MODEL_FEATURES
    ].copy()

    for column in CATEGORICAL_FEATURES:
        output[column] = (
            output[column]
            .astype("string")
            .fillna("<MISSING>")
            .astype(str)
        )

    for column in NUMERIC_FEATURES:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        )

    return output


def safe_float(
    value,
) -> float | None:
    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def safe_profit_factor(
    returns: np.ndarray,
) -> float | None:
    gains = returns[
        returns > 0
    ].sum()

    losses = -returns[
        returns < 0
    ].sum()

    if losses <= 0:
        return None

    return float(
        gains / losses
    )


def equity_curve(
    returns: np.ndarray,
) -> np.ndarray:
    if len(returns) == 0:
        return np.array(
            [],
            dtype=np.float64,
        )

    allocated_returns = (
        returns
        * NOTIONAL_FRACTION_PER_TRADE
    )

    return np.cumprod(
        1.0 + allocated_returns
    )


def maximum_drawdown(
    equity: np.ndarray,
) -> float:
    if len(equity) == 0:
        return 0.0

    peaks = np.maximum.accumulate(
        equity
    )

    return float(
        np.min(
            equity / peaks - 1.0
        )
    )


# =============================================================================
# FOLD DISCOVERY
# =============================================================================

def discover_folds() -> list[dict]:
    """
    Reuses only the historical test date boundaries from the
    earlier walk-forward run. The old prediction probabilities
    are not used for signal generation or backtesting.
    """

    reference = pd.read_parquet(
        REFERENCE_PREDICTIONS_PATH,
        columns=[
            "fold",
            "logged_at",
        ],
    )

    if reference.empty:
        raise RuntimeError(
            "Reference prediction file is empty."
        )

    folds = []

    for fold_name, group in reference.groupby(
        "fold",
        sort=True,
        observed=True,
    ):
        folds.append(
            {
                "fold": str(
                    fold_name
                ),
                "test_start": (
                    group[
                        "logged_at"
                    ].min()
                ),
                "test_end": (
                    group[
                        "logged_at"
                    ].max()
                ),
            }
        )

    folds.sort(
        key=lambda item: (
            item["test_start"]
        )
    )

    return folds


# =============================================================================
# DATA LOADING
# =============================================================================

def load_event_training_data() -> pd.DataFrame:
    data = pd.read_parquet(
        EVENT_DATASET_PATH
    )

    required = {
        "id",
        "logged_at",
        "symbol",
        "target_event",
        *MODEL_FEATURES,
    }

    missing = required - set(
        data.columns
    )

    if missing:
        raise ValueError(
            "Event dataset missing columns: "
            f"{sorted(missing)}"
        )

    data = data.loc[
        data["target_event"].isin(
            CLASS_VALUES
        )
    ].copy()

    data[
        "target_event"
    ] = data[
        "target_event"
    ].astype("int8")

    return data.sort_values(
        [
            "logged_at",
            "id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def load_full_stream() -> pd.DataFrame:
    stream = pd.read_parquet(
        FEATURE_PATH,
        columns=STREAM_COLUMNS,
        filters=[
            (
                "timeframe",
                "==",
                TOPOLOGY_TIMEFRAME,
            )
        ],
    )

    if stream.empty:
        raise RuntimeError(
            "No full-stream 1h rows found."
        )

    if stream["id"].duplicated().any():
        raise ValueError(
            "Duplicate full-stream IDs found."
        )

    stream = stream.sort_values(
        [
            "symbol",
            "logged_at",
            "id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return stream


def load_detected_events() -> pd.DataFrame:
    events = pd.read_parquet(
        DETECTED_EVENTS_PATH
    )

    required = {
        "symbol",
        "event_time",
        "event_direction",
        "event_name",
        "severity",
    }

    missing = required - set(
        events.columns
    )

    if missing:
        raise ValueError(
            "Detected event file missing columns: "
            f"{sorted(missing)}"
        )

    events[
        "event_direction"
    ] = events[
        "event_direction"
    ].astype("int8")

    return events.sort_values(
        [
            "symbol",
            "event_time",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


# =============================================================================
# MODEL TRAINING AND FULL-STREAM SCORING
# =============================================================================

def chronological_train_validation_split(
    historical: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if len(historical) < 500:
        raise RuntimeError(
            "Not enough historical training rows."
        )

    split_time = historical[
        "logged_at"
    ].quantile(
        1.0 - VALIDATION_FRACTION
    )

    training = historical.loc[
        historical["logged_at"]
        < split_time - EMBARGO
    ].copy()

    validation = historical.loc[
        historical["logged_at"]
        >= split_time
    ].copy()

    if training.empty or validation.empty:
        raise RuntimeError(
            "Chronological train/validation split is empty."
        )

    return training, validation


def train_fold_model(
    fold_number: int,
    training: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[
    CatBoostClassifier,
    pd.DataFrame,
]:
    X_train = prepare_features(
        training
    )

    y_train = training[
        "target_event"
    ]

    X_validation = prepare_features(
        validation
    )

    y_validation = validation[
        "target_event"
    ]

    categorical_indices = [
        MODEL_FEATURES.index(
            column
        )
        for column in CATEGORICAL_FEATURES
    ]

    train_pool = Pool(
        X_train,
        label=y_train,
        cat_features=(
            categorical_indices
        ),
        feature_names=(
            MODEL_FEATURES
        ),
    )

    validation_pool = Pool(
        X_validation,
        label=y_validation,
        cat_features=(
            categorical_indices
        ),
        feature_names=(
            MODEL_FEATURES
        ),
    )

    model = CatBoostClassifier(
        iterations=1800,
        depth=7,
        learning_rate=0.035,

        loss_function="MultiClass",
        eval_metric=(
            "TotalF1:average=Macro"
        ),

        auto_class_weights="Balanced",

        random_seed=(
            RANDOM_STATE
            + fold_number
        ),

        random_strength=1.0,
        l2_leaf_reg=7.0,

        bootstrap_type="Bayesian",
        bagging_temperature=1.0,

        od_type="Iter",
        od_wait=150,

        verbose=100,
        allow_writing_files=False,
        thread_count=-1,
    )

    model.fit(
        train_pool,
        eval_set=validation_pool,
        use_best_model=True,
    )

    importance = pd.DataFrame(
        {
            "feature": (
                MODEL_FEATURES
            ),
            "importance": (
                model.get_feature_importance(
                    train_pool,
                    type="FeatureImportance",
                )
            ),
        }
    )

    return model, importance


def score_frame(
    model: CatBoostClassifier,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    X = prepare_features(
        frame
    )

    categorical_indices = [
        MODEL_FEATURES.index(
            column
        )
        for column in CATEGORICAL_FEATURES
    ]

    pool = Pool(
        X,
        cat_features=(
            categorical_indices
        ),
        feature_names=(
            MODEL_FEATURES
        ),
    )

    probabilities = (
        model.predict_proba(
            pool
        )
    )

    class_order = np.array(
        model.classes_,
        dtype=np.int8,
    )

    probability_columns = {}

    for class_value, name in [
        (
            LONG_SQUEEZE_CLASS,
            "probability_long_squeeze",
        ),
        (
            NO_EVENT_CLASS,
            "probability_no_event",
        ),
        (
            SHORT_SQUEEZE_CLASS,
            "probability_short_squeeze",
        ),
    ]:
        matches = np.where(
            class_order == class_value
        )[0]

        if len(matches) != 1:
            raise RuntimeError(
                "Unexpected CatBoost class ordering: "
                f"{class_order.tolist()}"
            )

        probability_columns[
            name
        ] = probabilities[
            :,
            int(matches[0]),
        ].astype(np.float32)

    output = frame.copy()

    for column, values in (
        probability_columns.items()
    ):
        output[column] = values

    output[
        "squeeze_probability"
    ] = (
        1.0
        - output[
            "probability_no_event"
        ]
    ).astype(np.float32)

    directional_sum = (
        output[
            "probability_long_squeeze"
        ]
        + output[
            "probability_short_squeeze"
        ]
    )

    output[
        "predicted_event_class"
    ] = np.where(
        output[
            "probability_short_squeeze"
        ]
        >= output[
            "probability_long_squeeze"
        ],
        SHORT_SQUEEZE_CLASS,
        LONG_SQUEEZE_CLASS,
    ).astype("int8")

    output[
        "direction_confidence"
    ] = np.where(
        directional_sum > 0,
        np.maximum(
            output[
                "probability_long_squeeze"
            ],
            output[
                "probability_short_squeeze"
            ],
        )
        / directional_sum,
        0.5,
    ).astype(np.float32)

    return output


def train_and_score_full_stream(
    folds: list[dict],
    event_data: pd.DataFrame,
    full_stream: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    scored_test_frames = []
    validation_score_frames = []
    importance_frames = []

    for fold_number, fold in enumerate(
        folds,
        start=1,
    ):
        fold_name = fold["fold"]

        test_start = pd.Timestamp(
            fold["test_start"]
        )

        test_end = pd.Timestamp(
            fold["test_end"]
        )

        historical = event_data.loc[
            event_data[
                "logged_at"
            ]
            < test_start - EMBARGO
        ].copy()

        training, validation = (
            chronological_train_validation_split(
                historical
            )
        )

        validation_start = (
            validation[
                "logged_at"
            ].min()
        )

        validation_end = (
            validation[
                "logged_at"
            ].max()
        )

        validation_stream = (
            full_stream.loc[
                (
                    full_stream[
                        "logged_at"
                    ]
                    >= validation_start
                )
                & (
                    full_stream[
                        "logged_at"
                    ]
                    <= validation_end
                )
            ]
            .copy()
        )

        test_stream = (
            full_stream.loc[
                (
                    full_stream[
                        "logged_at"
                    ]
                    >= test_start
                )
                & (
                    full_stream[
                        "logged_at"
                    ]
                    <= test_end
                )
            ]
            .copy()
        )

        print()
        print("=" * 92)
        print(
            f"FULL-STREAM WALK-FORWARD "
            f"{fold_name.upper()}"
        )
        print("=" * 92)

        print(
            f"Training event rows   : "
            f"{len(training):,}"
        )

        print(
            f"Validation event rows : "
            f"{len(validation):,}"
        )

        print(
            f"Validation stream rows: "
            f"{len(validation_stream):,}"
        )

        print(
            f"Test stream rows      : "
            f"{len(test_stream):,}"
        )

        print(
            f"Test date range       : "
            f"{test_start} -> {test_end}"
        )

        model, importance = (
            train_fold_model(
                fold_number=fold_number,
                training=training,
                validation=validation,
            )
        )

        importance[
            "fold"
        ] = fold_name

        importance_frames.append(
            importance
        )

        scored_validation = (
            score_frame(
                model,
                validation_stream,
            )
        )

        scored_validation[
            "fold"
        ] = fold_name

        scored_test = score_frame(
            model,
            test_stream,
        )

        scored_test[
            "fold"
        ] = fold_name

        validation_score_frames.append(
            scored_validation
        )

        scored_test_frames.append(
            scored_test
        )

    return (
        pd.concat(
            scored_test_frames,
            ignore_index=True,
        ),
        pd.concat(
            validation_score_frames,
            ignore_index=True,
        ),
        pd.concat(
            importance_frames,
            ignore_index=True,
        ),
    )


# =============================================================================
# FOLD THRESHOLDS
# =============================================================================

def build_fold_thresholds(
    validation_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for fold_name, group in (
        validation_scores.groupby(
            "fold",
            observed=True,
            sort=True,
        )
    ):
        values = group[
            "squeeze_probability"
        ].dropna()

        if values.empty:
            raise RuntimeError(
                f"No validation scores for {fold_name}."
            )

        for percentile in (
            ALERT_PERCENTILES
        ):
            threshold = float(
                values.quantile(
                    percentile
                )
            )

            rows.append(
                {
                    "fold": fold_name,
                    "alert_percentile": (
                        percentile
                    ),
                    "score_threshold": (
                        threshold
                    ),
                    "rearm_threshold": (
                        threshold
                        * REARM_SCORE_MULTIPLIER
                    ),
                    "validation_rows": int(
                        len(values)
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# EVENT LOOKUP
# =============================================================================

def prepare_event_lookup(
    events: pd.DataFrame,
) -> dict[str, dict]:
    lookup = {}

    for symbol, group in events.groupby(
        "symbol",
        observed=True,
        sort=False,
    ):
        group = group.sort_values(
            "event_time"
        ).reset_index(drop=True)

        lookup[str(symbol)] = {
            "times_ns": (
                group[
                    "event_time"
                ]
                .to_numpy(
                    dtype="datetime64[ns]"
                )
                .astype(np.int64)
            ),
            "directions": (
                group[
                    "event_direction"
                ].to_numpy(
                    dtype=np.int8
                )
            ),
            "severities": (
                group[
                    "severity"
                ].to_numpy(
                    dtype=np.float64
                )
            ),
            "names": (
                group[
                    "event_name"
                ].astype(str)
                .to_numpy()
            ),
        }

    return lookup


def label_alert_from_events(
    symbol: str,
    alert_time: pd.Timestamp,
    predicted_class: int,
    event_lookup: dict[str, dict],
) -> dict:
    data = event_lookup.get(
        symbol
    )

    if data is None:
        return {
            "event_detected": False,
            "event_direction_correct": False,
            "actual_event_class": 0,
            "actual_event_name": "NO_EVENT",
            "actual_event_time": pd.NaT,
            "event_lead_seconds": np.nan,
            "event_severity": np.nan,
        }

    alert_ns = int(
        alert_time.to_datetime64()
        .astype("datetime64[ns]")
        .astype(np.int64)
    )

    end_ns = (
        alert_ns
        + int(
            EVENT_WINDOW.value
        )
    )

    times_ns = data[
        "times_ns"
    ]

    index = int(
        np.searchsorted(
            times_ns,
            alert_ns,
            side="right",
        )
    )

    if (
        index >= len(times_ns)
        or times_ns[index] > end_ns
    ):
        return {
            "event_detected": False,
            "event_direction_correct": False,
            "actual_event_class": 0,
            "actual_event_name": "NO_EVENT",
            "actual_event_time": pd.NaT,
            "event_lead_seconds": np.nan,
            "event_severity": np.nan,
        }

    actual_direction = int(
        data[
            "directions"
        ][index]
    )

    event_time = pd.Timestamp(
        times_ns[index],
        unit="ns",
        tz="UTC",
    )

    return {
        "event_detected": True,

        "event_direction_correct": bool(
            actual_direction
            == predicted_class
        ),

        "actual_event_class": (
            actual_direction
        ),

        "actual_event_name": str(
            data["names"][index]
        ),

        "actual_event_time": (
            event_time
        ),

        "event_lead_seconds": float(
            (
                times_ns[index]
                - alert_ns
            )
            / 1_000_000_000
        ),

        "event_severity": float(
            data[
                "severities"
            ][index]
        ),
    }


# =============================================================================
# STATEFUL ALERT GENERATION
# =============================================================================

def threshold_map_for_config(
    thresholds: pd.DataFrame,
    percentile: float,
) -> dict[str, tuple[float, float]]:
    selected = thresholds.loc[
        np.isclose(
            thresholds[
                "alert_percentile"
            ],
            percentile,
        )
    ]

    if selected.empty:
        raise RuntimeError(
            f"No thresholds for percentile {percentile}."
        )

    return {
        str(row.fold): (
            float(
                row.score_threshold
            ),
            float(
                row.rearm_threshold
            ),
        )
        for row in selected.itertuples()
    }


def generate_alerts(
    scored_stream: pd.DataFrame,
    thresholds: pd.DataFrame,
    event_lookup: dict[str, dict],
    config: AlertConfig,
) -> pd.DataFrame:
    fold_thresholds = (
        threshold_map_for_config(
            thresholds,
            config.alert_percentile,
        )
    )

    rows = []

    for (
        fold_name,
        symbol,
    ), group in scored_stream.groupby(
        [
            "fold",
            "symbol",
        ],
        observed=True,
        sort=False,
    ):
        group = group.sort_values(
            [
                "logged_at",
                "id",
            ],
            kind="mergesort",
        )

        threshold_pair = (
            fold_thresholds.get(
                str(fold_name)
            )
        )

        if threshold_pair is None:
            continue

        score_threshold, rearm_threshold = (
            threshold_pair
        )

        cooldown = pd.Timedelta(
            minutes=(
                config.cooldown_minutes
            )
        )

        armed = True
        previous_score = 0.0
        last_alert_time = None

        for row in group.itertuples(
            index=False
        ):
            score = float(
                row.squeeze_probability
            )

            direction_confidence = float(
                row.direction_confidence
            )

            timestamp = pd.Timestamp(
                row.logged_at
            )

            if score <= rearm_threshold:
                armed = True

            crossed = (
                previous_score
                < score_threshold
                and score
                >= score_threshold
            )

            cooldown_ready = (
                last_alert_time is None
                or (
                    timestamp
                    - last_alert_time
                    >= cooldown
                )
            )

            if (
                armed
                and crossed
                and cooldown_ready
                and direction_confidence
                >= (
                    config
                    .minimum_direction_confidence
                )
            ):
                predicted_class = int(
                    row.predicted_event_class
                )

                event_result = (
                    label_alert_from_events(
                        symbol=str(symbol),
                        alert_time=timestamp,
                        predicted_class=(
                            predicted_class
                        ),
                        event_lookup=(
                            event_lookup
                        ),
                    )
                )

                rows.append(
                    {
                        "id": row.id,
                        "fold": str(
                            fold_name
                        ),
                        "symbol": str(
                            symbol
                        ),
                        "timeframe": (
                            row.timeframe
                        ),
                        "logged_at": (
                            timestamp
                        ),

                        "current_price": float(
                            row.current_price
                        ),

                        "nearest_upper_price": (
                            safe_float(
                                row.nearest_upper_price
                            )
                        ),

                        "nearest_lower_price": (
                            safe_float(
                                row.nearest_lower_price
                            )
                        ),

                        "squeeze_probability": (
                            score
                        ),

                        "direction_confidence": (
                            direction_confidence
                        ),

                        "predicted_event_class": (
                            predicted_class
                        ),

                        "predicted_event_name": (
                            "SHORT_SQUEEZE"
                            if predicted_class
                            == SHORT_SQUEEZE_CLASS
                            else "LONG_SQUEEZE"
                        ),

                        "score_threshold": (
                            score_threshold
                        ),

                        "rearm_threshold": (
                            rearm_threshold
                        ),

                        "alert_percentile": (
                            config
                            .alert_percentile
                        ),

                        "minimum_direction_confidence": (
                            config
                            .minimum_direction_confidence
                        ),

                        "cooldown_minutes": (
                            config
                            .cooldown_minutes
                        ),

                        **event_result,
                    }
                )

                last_alert_time = (
                    timestamp
                )

                armed = False

            previous_score = score

    return pd.DataFrame(
        rows
    )


# =============================================================================
# PRICE STREAM LOOKUP
# =============================================================================

def prepare_price_lookup(
    full_stream: pd.DataFrame,
) -> dict[str, dict]:
    lookup = {}

    for symbol, group in (
        full_stream.groupby(
            "symbol",
            observed=True,
            sort=False,
        )
    ):
        group = group.sort_values(
            [
                "logged_at",
                "id",
            ],
            kind="mergesort",
        ).reset_index(drop=True)

        lookup[str(symbol)] = {
            "times_ns": (
                group[
                    "logged_at"
                ]
                .to_numpy(
                    dtype="datetime64[ns]"
                )
                .astype(np.int64)
            ),
            "prices": (
                group[
                    "current_price"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            ),
        }

    return lookup


# =============================================================================
# TRADE EXECUTION
# =============================================================================

def resolve_target(
    alert,
    direction: int,
    entry_price: float,
    target_mode: str,
) -> float | None:
    if target_mode == "nearest_liquidity":
        if direction == TRADE_LONG:
            target = safe_float(
                alert.nearest_upper_price
            )
        else:
            target = safe_float(
                alert.nearest_lower_price
            )

        if (
            target is None
            or target <= 0
        ):
            return None

        if (
            direction == TRADE_LONG
            and target <= entry_price
        ):
            return None

        if (
            direction == TRADE_SHORT
            and target >= entry_price
        ):
            return None

        return target

    if target_mode == "fixed_50bp":
        distance = 50 / 10_000

    elif target_mode == "fixed_100bp":
        distance = 100 / 10_000

    else:
        raise ValueError(
            f"Unknown target mode: {target_mode}"
        )

    return (
        entry_price
        * (
            1.0
            + direction * distance
        )
    )


def resolve_stop(
    alert,
    direction: int,
    entry_price: float,
    stop_mode: str,
) -> float | None:
    if stop_mode == "opposite_liquidity":
        if direction == TRADE_LONG:
            stop = safe_float(
                alert.nearest_lower_price
            )
        else:
            stop = safe_float(
                alert.nearest_upper_price
            )

        if (
            stop is None
            or stop <= 0
        ):
            return None

        if (
            direction == TRADE_LONG
            and stop >= entry_price
        ):
            return None

        if (
            direction == TRADE_SHORT
            and stop <= entry_price
        ):
            return None

        return stop

    if stop_mode == "fixed_50bp":
        distance = 50 / 10_000

    elif stop_mode == "fixed_100bp":
        distance = 100 / 10_000

    else:
        raise ValueError(
            f"Unknown stop mode: {stop_mode}"
        )

    return (
        entry_price
        * (
            1.0
            - direction * distance
        )
    )


def simulate_trade(
    alert,
    price_data: dict,
    execution: ExecutionConfig,
) -> dict | None:
    times_ns = price_data[
        "times_ns"
    ]

    prices = price_data[
        "prices"
    ]

    alert_time = pd.Timestamp(
        alert.logged_at
    )

    alert_ns = int(
        alert_time.to_datetime64()
        .astype("datetime64[ns]")
        .astype(np.int64)
    )

    desired_entry_ns = (
        alert_ns
        + execution.entry_delay_seconds
        * 1_000_000_000
    )

    entry_index = int(
        np.searchsorted(
            times_ns,
            desired_entry_ns,
            side="left",
        )
    )

    if entry_index >= len(
        times_ns
    ):
        return None

    entry_price = float(
        prices[entry_index]
    )

    if (
        not math.isfinite(entry_price)
        or entry_price <= 0
    ):
        return None

    predicted_class = int(
        alert.predicted_event_class
    )

    if (
        predicted_class
        == SHORT_SQUEEZE_CLASS
    ):
        direction = TRADE_LONG
        trade_side = "LONG"

    elif (
        predicted_class
        == LONG_SQUEEZE_CLASS
    ):
        direction = TRADE_SHORT
        trade_side = "SHORT"

    else:
        return None

    target_price = resolve_target(
        alert=alert,
        direction=direction,
        entry_price=entry_price,
        target_mode=(
            execution.target_mode
        ),
    )

    stop_price = resolve_stop(
        alert=alert,
        direction=direction,
        entry_price=entry_price,
        stop_mode=execution.stop_mode,
    )

    if (
        target_price is None
        or stop_price is None
    ):
        return None

    target_return = (
        direction
        * (
            target_price
            / entry_price
            - 1.0
        )
    )

    stop_return = (
        -direction
        * (
            stop_price
            / entry_price
            - 1.0
        )
    )

    if (
        target_return <= 0
        or stop_return <= 0
    ):
        return None

    target_distance_bps = (
        target_return
        * 10_000
    )

    stop_distance_bps = (
        stop_return
        * 10_000
    )

    if (
        target_distance_bps
        < execution
        .minimum_target_distance_bps
    ):
        return None

    exit_limit_ns = (
        times_ns[entry_index]
        + execution.max_hold_minutes
        * 60
        * 1_000_000_000
    )

    exit_limit = int(
        np.searchsorted(
            times_ns,
            exit_limit_ns,
            side="right",
        )
    )

    exit_limit = min(
        exit_limit,
        len(times_ns),
    )

    if (
        exit_limit
        <= entry_index + 1
    ):
        return None

    exit_index = (
        exit_limit - 1
    )

    exit_reason = "TIME"

    gross_return = (
        direction
        * (
            float(
                prices[exit_index]
            )
            / entry_price
            - 1.0
        )
    )

    mfe = 0.0
    mae = 0.0

    for index in range(
        entry_index + 1,
        exit_limit,
    ):
        price = float(
            prices[index]
        )

        directional_return = (
            direction
            * (
                price
                / entry_price
                - 1.0
            )
        )

        mfe = max(
            mfe,
            directional_return,
        )

        mae = min(
            mae,
            directional_return,
        )

        target_hit = (
            price >= target_price
            if direction == TRADE_LONG
            else price <= target_price
        )

        stop_hit = (
            price <= stop_price
            if direction == TRADE_LONG
            else price >= stop_price
        )

        # Conservative handling:
        # if one snapshot appears beyond both levels,
        # assume the stop happened first.
        if stop_hit:
            exit_index = index
            exit_reason = "SL"
            gross_return = (
                -stop_return
            )
            break

        if target_hit:
            exit_index = index
            exit_reason = "TP"
            gross_return = (
                target_return
            )
            break

    net_return = (
        gross_return
        - execution.round_trip_cost_bps
        / 10_000
    )

    entry_time = pd.Timestamp(
        times_ns[entry_index],
        unit="ns",
        tz="UTC",
    )

    exit_time = pd.Timestamp(
        times_ns[exit_index],
        unit="ns",
        tz="UTC",
    )

    return {
        "alert_id": alert.id,
        "fold": alert.fold,
        "symbol": alert.symbol,

        "alert_time": alert.logged_at,
        "entry_time": entry_time,
        "exit_time": exit_time,

        "predicted_event_name": (
            alert.predicted_event_name
        ),

        "trade_side": trade_side,

        "squeeze_probability": float(
            alert.squeeze_probability
        ),

        "direction_confidence": float(
            alert.direction_confidence
        ),

        "event_detected": bool(
            alert.event_detected
        ),

        "event_direction_correct": bool(
            alert.event_direction_correct
        ),

        "actual_event_name": (
            alert.actual_event_name
        ),

        "entry_price": (
            entry_price
        ),

        "target_price": (
            target_price
        ),

        "stop_price": (
            stop_price
        ),

        "exit_price": float(
            prices[exit_index]
        ),

        "target_distance_bps": (
            target_distance_bps
        ),

        "stop_distance_bps": (
            stop_distance_bps
        ),

        "exit_reason": (
            exit_reason
        ),

        "gross_return": (
            gross_return
        ),

        "net_return": (
            net_return
        ),

        "gross_bps": (
            gross_return * 10_000
        ),

        "net_bps": (
            net_return * 10_000
        ),

        "mfe_bps": (
            mfe * 10_000
        ),

        "mae_bps": (
            mae * 10_000
        ),

        "holding_seconds": float(
            (
                times_ns[exit_index]
                - times_ns[entry_index]
            )
            / 1_000_000_000
        ),
    }


def simulate_configuration(
    alerts: pd.DataFrame,
    price_lookup: dict[str, dict],
    execution: ExecutionConfig,
) -> pd.DataFrame:
    if alerts.empty:
        return pd.DataFrame()

    rows = []

    last_exit_by_symbol = {}

    for alert in alerts.sort_values(
        [
            "logged_at",
            "symbol",
        ],
        kind="mergesort",
    ).itertuples(
        index=False
    ):
        symbol = str(
            alert.symbol
        )

        previous_exit = (
            last_exit_by_symbol.get(
                symbol
            )
        )

        if (
            ONE_POSITION_PER_SYMBOL
            and previous_exit is not None
            and alert.logged_at
            < previous_exit
        ):
            continue

        price_data = (
            price_lookup.get(
                symbol
            )
        )

        if price_data is None:
            continue

        trade = simulate_trade(
            alert=alert,
            price_data=price_data,
            execution=execution,
        )

        if trade is None:
            continue

        rows.append(
            trade
        )

        last_exit_by_symbol[
            symbol
        ] = trade["exit_time"]

    return pd.DataFrame(
        rows
    )


# =============================================================================
# METRICS
# =============================================================================

def summarize_alerts(
    alerts: pd.DataFrame,
) -> dict:
    if alerts.empty:
        return {
            "alerts": 0,
            "alerts_per_day": None,
            "event_precision": None,
            "direction_accuracy": None,
            "mean_lead_minutes": None,
        }

    span_days = max(
        1.0,
        (
            alerts[
                "logged_at"
            ].max()
            - alerts[
                "logged_at"
            ].min()
        ).total_seconds()
        / 86_400,
    )

    event_alerts = alerts.loc[
        alerts[
            "event_detected"
        ]
    ]

    direction_accuracy = (
        float(
            event_alerts[
                "event_direction_correct"
            ].mean()
        )
        if not event_alerts.empty
        else None
    )

    lead_values = pd.to_numeric(
        event_alerts[
            "event_lead_seconds"
        ],
        errors="coerce",
    ).dropna()

    return {
        "alerts": int(
            len(alerts)
        ),

        "alerts_per_day": float(
            len(alerts) / span_days
        ),

        "event_precision": float(
            alerts[
                "event_detected"
            ].mean()
        ),

        "direction_accuracy": (
            direction_accuracy
        ),

        "mean_lead_minutes": (
            float(
                lead_values.mean()
                / 60
            )
            if not lead_values.empty
            else None
        ),
    }


def summarize_trades(
    trades: pd.DataFrame,
) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "trades_per_day": None,
            "win_rate": None,
            "mean_net_bps": None,
            "median_net_bps": None,
            "total_net_bps": None,
            "profit_factor": None,
            "compounded_return": None,
            "max_drawdown": None,
            "mean_mfe_bps": None,
            "mean_mae_bps": None,
            "mean_holding_minutes": None,
            "target_hit_rate": None,
            "stop_hit_rate": None,
        }

    trades = trades.sort_values(
        [
            "entry_time",
            "symbol",
        ],
        kind="mergesort",
    )

    returns = trades[
        "net_return"
    ].to_numpy(
        dtype=np.float64
    )

    equity = equity_curve(
        returns
    )

    span_days = max(
        1.0,
        (
            trades[
                "entry_time"
            ].max()
            - trades[
                "entry_time"
            ].min()
        ).total_seconds()
        / 86_400,
    )

    return {
        "trades": int(
            len(trades)
        ),

        "trades_per_day": float(
            len(trades) / span_days
        ),

        "win_rate": float(
            (returns > 0).mean()
        ),

        "mean_net_bps": float(
            returns.mean() * 10_000
        ),

        "median_net_bps": float(
            np.median(returns)
            * 10_000
        ),

        "total_net_bps": float(
            returns.sum() * 10_000
        ),

        "profit_factor": (
            safe_profit_factor(
                returns
            )
        ),

        "compounded_return": (
            float(
                equity[-1] - 1.0
            )
            if len(equity)
            else None
        ),

        "max_drawdown": (
            maximum_drawdown(
                equity
            )
        ),

        "mean_mfe_bps": float(
            trades[
                "mfe_bps"
            ].mean()
        ),

        "mean_mae_bps": float(
            trades[
                "mae_bps"
            ].mean()
        ),

        "mean_holding_minutes": float(
            trades[
                "holding_seconds"
            ].mean()
            / 60
        ),

        "target_hit_rate": float(
            trades[
                "exit_reason"
            ].eq("TP").mean()
        ),

        "stop_hit_rate": float(
            trades[
                "exit_reason"
            ].eq("SL").mean()
        ),
    }


def fold_metrics_for_config(
    trades: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    if (
        trades.empty
        or "fold" not in trades.columns
    ):
        return (
            pd.DataFrame(),
            {
                "positive_folds": 0,
                "fold_count": 0,
                "minimum_fold_mean_bps": None,
                "minimum_fold_profit_factor": None,
            },
        )

    rows = []

    for fold_name, group in (
        trades.groupby(
            "fold",
            observed=True,
            sort=True,
        )
    ):
        rows.append(
            {
                "fold": fold_name,
                **summarize_trades(
                    group
                ),
            }
        )

    frame = pd.DataFrame(
        rows
    )

    profit_factors = pd.to_numeric(
        frame[
            "profit_factor"
        ],
        errors="coerce",
    ).dropna()

    return (
        frame,
        {
            "positive_folds": int(
                (
                    frame[
                        "mean_net_bps"
                    ] > 0
                ).sum()
            ),

            "fold_count": int(
                len(frame)
            ),

            "minimum_fold_mean_bps": float(
                frame[
                    "mean_net_bps"
                ].min()
            ),

            "minimum_fold_profit_factor": (
                float(
                    profit_factors.min()
                )
                if not profit_factors.empty
                else None
            ),
        },
    )


def configuration_score(
    alert_metrics: dict,
    trade_metrics: dict,
    consistency: dict,
) -> float:
    trades = int(
        trade_metrics.get(
            "trades",
            0
        )
    )

    if trades < MIN_TRADES_FOR_RANKING:
        return -1_000_000.0

    mean_net = trade_metrics.get(
        "mean_net_bps"
    )

    profit_factor = trade_metrics.get(
        "profit_factor"
    )

    max_drawdown = trade_metrics.get(
        "max_drawdown"
    )

    minimum_fold_mean = (
        consistency.get(
            "minimum_fold_mean_bps"
        )
    )

    event_precision = (
        alert_metrics.get(
            "event_precision"
        )
    )

    direction_accuracy = (
        alert_metrics.get(
            "direction_accuracy"
        )
    )

    required = [
        mean_net,
        profit_factor,
        max_drawdown,
        minimum_fold_mean,
        event_precision,
    ]

    if any(
        value is None
        or not math.isfinite(
            float(value)
        )
        for value in required
    ):
        return -1_000_000.0

    direction_accuracy = (
        float(
            direction_accuracy
        )
        if direction_accuracy is not None
        else 0.0
    )

    positive_folds = int(
        consistency.get(
            "positive_folds",
            0
        )
    )

    fold_count = max(
        1,
        int(
            consistency.get(
                "fold_count",
                1
            )
        ),
    )

    positive_fold_ratio = (
        positive_folds
        / fold_count
    )

    # Economic performance dominates.
    # Alert quality is deliberately secondary.
    return float(
        float(mean_net) * 0.50
        + float(minimum_fold_mean) * 0.35

        + min(
            float(profit_factor),
            4.0,
        ) * 5.0

        + positive_fold_ratio * 8.0

        + float(event_precision) * 3.0
        + direction_accuracy * 2.0

        - abs(
            float(max_drawdown)
        ) * 100 * 0.50
    )


# =============================================================================
# MAIN GRID
# =============================================================================

def main() -> int:
    started = time.time()

    ensure_files()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "TOPOLOGY V2 — TRUE FULL-STREAM "
        "OUT-OF-SAMPLE SQUEEZE BACKTEST"
    )
    print("=" * 100)

    print(
        "This run retrains every fold and scores every "
        "chronological 1h snapshot in each test period."
    )

    print()

    print(
        f"Round-trip costs: "
        f"{ROUND_TRIP_COST_BPS_VALUES} bps"
    )

    print(
        f"Event window    : "
        f"{EVENT_WINDOW}"
    )

    print()

    folds = discover_folds()

    event_data = (
        load_event_training_data()
    )

    full_stream = (
        load_full_stream()
    )

    detected_events = (
        load_detected_events()
    )

    print(
        f"Historical event dataset rows: "
        f"{len(event_data):,}"
    )

    print(
        f"Full 1h stream rows          : "
        f"{len(full_stream):,}"
    )

    print(
        f"Detected events              : "
        f"{len(detected_events):,}"
    )

    (
        scored_stream,
        validation_scores,
        feature_importance,
    ) = train_and_score_full_stream(
        folds=folds,
        event_data=event_data,
        full_stream=full_stream,
    )

    scored_stream.to_parquet(
        SCORED_SNAPSHOTS_PATH,
        index=False,
        compression="zstd",
    )

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    thresholds = build_fold_thresholds(
        validation_scores
    )

    thresholds.to_csv(
        THRESHOLDS_PATH,
        index=False,
    )

    event_lookup = prepare_event_lookup(
        detected_events
    )

    price_lookup = prepare_price_lookup(
        full_stream
    )

    alert_configs = [
        AlertConfig(
            alert_percentile=percentile,
            minimum_direction_confidence=confidence,
            cooldown_minutes=cooldown,
        )
        for (
            percentile,
            confidence,
            cooldown,
        ) in itertools.product(
            ALERT_PERCENTILES,
            MIN_DIRECTION_CONFIDENCE_VALUES,
            COOLDOWN_MINUTES_VALUES,
        )
    ]

    execution_configs = [
        ExecutionConfig(
            entry_delay_seconds=delay,
            max_hold_minutes=hold,
            target_mode=target,
            stop_mode=stop,
            minimum_target_distance_bps=(
                minimum_target
            ),
            round_trip_cost_bps=cost,
        )
        for (
            delay,
            hold,
            target,
            stop,
            minimum_target,
            cost,
        ) in itertools.product(
            ENTRY_DELAY_SECONDS_VALUES,
            MAX_HOLD_MINUTES_VALUES,
            TARGET_MODES,
            STOP_MODES,
            MIN_TARGET_DISTANCE_BPS_VALUES,
            ROUND_TRIP_COST_BPS_VALUES,
        )
    ]

    total_configurations = (
        len(alert_configs)
        * len(execution_configs)
    )

    print()
    print(
        f"Alert configurations    : "
        f"{len(alert_configs):,}"
    )

    print(
        f"Execution configurations: "
        f"{len(execution_configs):,}"
    )

    print(
        f"Total backtests         : "
        f"{total_configurations:,}"
    )

    summary_rows = []
    fold_rows = []
    symbol_rows = []

    alert_frames = []
    trade_frames = []

    config_id = 0
    completed = 0

    for alert_index, alert_config in enumerate(
        alert_configs,
        start=1,
    ):
        alerts = generate_alerts(
            scored_stream=(
                scored_stream
            ),
            thresholds=thresholds,
            event_lookup=event_lookup,
            config=alert_config,
        )

        alert_metrics = summarize_alerts(
            alerts
        )

        if not alerts.empty:
            alert_copy = alerts.copy()

            alert_copy[
                "alert_config_id"
            ] = alert_index

            alert_frames.append(
                alert_copy
            )

        print()
        print("-" * 100)

        print(
            f"ALERT CONFIG "
            f"{alert_index}/"
            f"{len(alert_configs)}"
        )

        print(
            f"percentile="
            f"{alert_config.alert_percentile:.4f} "
            f"direction_conf="
            f"{alert_config.minimum_direction_confidence:.2f} "
            f"cooldown="
            f"{alert_config.cooldown_minutes}m "
            f"alerts="
            f"{len(alerts):,} "
            f"precision="
            f"{alert_metrics['event_precision']}"
        )

        print("-" * 100)

        for execution in execution_configs:
            config_id += 1
            completed += 1

            trades = simulate_configuration(
                alerts=alerts,
                price_lookup=price_lookup,
                execution=execution,
            )

            trade_metrics = (
                summarize_trades(
                    trades
                )
            )

            (
                fold_frame,
                consistency,
            ) = fold_metrics_for_config(
                trades
            )

            score = configuration_score(
                alert_metrics=alert_metrics,
                trade_metrics=trade_metrics,
                consistency=consistency,
            )

            summary_rows.append(
                {
                    "config_id": config_id,

                    **asdict(
                        alert_config
                    ),

                    **asdict(
                        execution
                    ),

                    **alert_metrics,
                    **trade_metrics,
                    **consistency,

                    "research_score": (
                        score
                    ),
                }
            )

            if not fold_frame.empty:
                for row in (
                    fold_frame.to_dict(
                        orient="records"
                    )
                ):
                    fold_rows.append(
                        {
                            "config_id": (
                                config_id
                            ),
                            **row,
                        }
                    )

            if not trades.empty:
                trades = trades.copy()

                trades[
                    "config_id"
                ] = config_id

                trade_frames.append(
                    trades
                )

                for symbol, group in (
                    trades.groupby(
                        "symbol",
                        observed=True,
                        sort=True,
                    )
                ):
                    symbol_rows.append(
                        {
                            "config_id": (
                                config_id
                            ),
                            "symbol": (
                                symbol
                            ),
                            **summarize_trades(
                                group
                            ),
                        }
                    )

            if (
                completed % 100 == 0
                or completed
                == total_configurations
            ):
                elapsed = (
                    time.time()
                    - started
                )

                average = (
                    elapsed
                    / completed
                )

                remaining = (
                    total_configurations
                    - completed
                )

                eta_minutes = (
                    average
                    * remaining
                    / 60
                )

                print(
                    f"Progress "
                    f"{completed:,}/"
                    f"{total_configurations:,} "
                    f"({completed / total_configurations:.1%}) "
                    f"| ETA {eta_minutes:.1f} min",
                    flush=True,
                )

    summary = pd.DataFrame(
        summary_rows
    )

    summary = summary.sort_values(
        [
            "research_score",
            "positive_folds",
            "minimum_fold_mean_bps",
            "mean_net_bps",
            "profit_factor",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    eligible = summary.loc[
        summary[
            "trades"
        ] >= MIN_TRADES_FOR_RANKING
    ].copy()

    top_configs = (
        eligible.head(100)
    )

    top_configs.to_csv(
        TOP_CONFIGS_PATH,
        index=False,
    )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    fold_metrics.to_csv(
        FOLD_METRICS_PATH,
        index=False,
    )

    symbol_metrics = pd.DataFrame(
        symbol_rows
    )

    symbol_metrics.to_csv(
        SYMBOL_METRICS_PATH,
        index=False,
    )

    if alert_frames:
        all_alerts = pd.concat(
            alert_frames,
            ignore_index=True,
        )

        all_alerts.to_parquet(
            ALERTS_PATH,
            index=False,
            compression="zstd",
        )
    else:
        all_alerts = pd.DataFrame()

    if trade_frames:
        all_trades = pd.concat(
            trade_frames,
            ignore_index=True,
        )

        all_trades.to_parquet(
            TRADES_PATH,
            index=False,
            compression="zstd",
        )
    else:
        all_trades = pd.DataFrame()

    best = None

    if not top_configs.empty:
        best = top_configs.iloc[
            0
        ].to_dict()

        best_id = int(
            best["config_id"]
        )

        best_trades = all_trades.loc[
            all_trades[
                "config_id"
            ].eq(best_id)
        ].copy()

        best_trades.to_parquet(
            BEST_TRADES_PATH,
            index=False,
            compression="zstd",
        )

        matching_alert_config = (
            AlertConfig(
                alert_percentile=float(
                    best[
                        "alert_percentile"
                    ]
                ),
                minimum_direction_confidence=float(
                    best[
                        "minimum_direction_confidence"
                    ]
                ),
                cooldown_minutes=int(
                    best[
                        "cooldown_minutes"
                    ]
                ),
            )
        )

        best_alerts = generate_alerts(
            scored_stream=(
                scored_stream
            ),
            thresholds=thresholds,
            event_lookup=event_lookup,
            config=matching_alert_config,
        )

        best_alerts.to_parquet(
            BEST_ALERTS_PATH,
            index=False,
            compression="zstd",
        )

    report = {
        "methodology": {
            "model_training": (
                "Each fold is retrained using only "
                "historical event-dataset rows."
            ),

            "inference": (
                "Every chronological 1h topology "
                "snapshot in each test fold is scored."
            ),

            "threshold_selection": (
                "Alert thresholds are selected from "
                "historical validation full-stream "
                "score percentiles."
            ),

            "event_window_minutes": (
                EVENT_WINDOW
                .total_seconds()
                / 60
            ),

            "round_trip_cost_bps_values": (
                ROUND_TRIP_COST_BPS_VALUES
            ),
        },

        "rows": {
            "event_training_rows": int(
                len(event_data)
            ),

            "full_stream_rows": int(
                len(full_stream)
            ),

            "oos_scored_rows": int(
                len(scored_stream)
            ),

            "detected_events": int(
                len(detected_events)
            ),
        },

        "grid": {
            "alert_configurations": (
                len(alert_configs)
            ),

            "execution_configurations": (
                len(execution_configs)
            ),

            "total_backtests": (
                total_configurations
            ),
        },

        "best_configuration": (
            best
        ),

        "top_20": (
            top_configs.head(20)
            .to_dict(
                orient="records"
            )
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 150)
    print(
        "TOP TRUE FULL-STREAM BACKTESTS"
    )
    print("=" * 150)

    display_columns = [
        "config_id",

        "alert_percentile",
        "minimum_direction_confidence",
        "cooldown_minutes",

        "entry_delay_seconds",
        "max_hold_minutes",

        "target_mode",
        "stop_mode",
        "minimum_target_distance_bps",
        "round_trip_cost_bps",

        "alerts",
        "alerts_per_day",
        "event_precision",
        "direction_accuracy",
        "mean_lead_minutes",

        "trades",
        "trades_per_day",
        "win_rate",

        "mean_net_bps",
        "median_net_bps",
        "profit_factor",

        "compounded_return",
        "max_drawdown",

        "positive_folds",
        "minimum_fold_mean_bps",

        "research_score",
    ]

    if not top_configs.empty:
        print(
            top_configs[
                display_columns
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

        print()
        print("=" * 150)
        print("BEST CONFIGURATION")
        print("=" * 150)

        for column in display_columns:
            print(
                f"{column:38}: "
                f"{best.get(column)}"
            )

        best_id = int(
            best["config_id"]
        )

        print()
        print("Best configuration by fold:")

        print(
            fold_metrics.loc[
                fold_metrics[
                    "config_id"
                ].eq(best_id)
            ].to_string(
                index=False
            )
        )

        print()
        print("Best configuration by symbol:")

        print(
            symbol_metrics.loc[
                symbol_metrics[
                    "config_id"
                ].eq(best_id)
            ].to_string(
                index=False
            )
        )

    else:
        print(
            "No configuration met the "
            "minimum trade requirement."
        )

    elapsed = (
        time.time()
        - started
    )

    print()
    print("=" * 100)
    print(
        "TRUE FULL-STREAM BACKTEST COMPLETE"
    )
    print("=" * 100)

    print(
        f"Scored stream : "
        f"{SCORED_SNAPSHOTS_PATH}"
    )

    print(
        f"Thresholds    : "
        f"{THRESHOLDS_PATH}"
    )

    print(
        f"All alerts    : "
        f"{ALERTS_PATH}"
    )

    print(
        f"All trades    : "
        f"{TRADES_PATH}"
    )

    print(
        f"Summary       : "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Top configs   : "
        f"{TOP_CONFIGS_PATH}"
    )

    print(
        f"Fold metrics  : "
        f"{FOLD_METRICS_PATH}"
    )

    print(
        f"Symbol metrics: "
        f"{SYMBOL_METRICS_PATH}"
    )

    print(
        f"Best alerts   : "
        f"{BEST_ALERTS_PATH}"
    )

    print(
        f"Best trades   : "
        f"{BEST_TRADES_PATH}"
    )

    print(
        f"Report        : "
        f"{REPORT_PATH}"
    )

    print(
        f"Elapsed       : "
        f"{elapsed / 60:.1f} minutes"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except Exception as error:
        print()
        print("=" * 100)
        print(
            "TRUE FULL-STREAM BACKTEST FAILED"
        )
        print("=" * 100)

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        raise
