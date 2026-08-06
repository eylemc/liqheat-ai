from __future__ import annotations

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
    "data/features/liq_topology_v2_ml_features.parquet"
)

OUTPUT_DIR = Path(
    "data/research/topology_v2_squeeze_grid/tf_1w__future_30m__precursor_5m__q_0p9"
)

EVENTS_PATH = OUTPUT_DIR / "detected_squeeze_events.parquet"
DATASET_PATH = OUTPUT_DIR / "squeeze_event_dataset.parquet"
SUMMARY_PATH = OUTPUT_DIR / "walk_forward_summary.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "walk_forward_predictions.parquet"
IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
REPORT_PATH = OUTPUT_DIR / "report.json"

# Bütün sembollerde ortak bulunan topology görünümü.
TOPOLOGY_TIMEFRAME = "1w"

# Olay tanımı.
FUTURE_WINDOW = pd.Timedelta(minutes=30)
PRECURSOR_OFFSET = pd.Timedelta(minutes=5)
EVENT_MERGE_WINDOW = pd.Timedelta(minutes=60)
EVENT_EXCLUSION_WINDOW = pd.Timedelta(minutes=60)

# Negatif örnekleri her 15 dakikada bir al.
NEGATIVE_SAMPLE_INTERVAL = pd.Timedelta(minutes=15)

# Adaptif eşik için geçmiş volatilite penceresi.
VOLATILITY_LOOKBACK = "7D"
VOLATILITY_QUANTILE = 0.9

# Çok sakin dönemlerde eşik aşırı küçülmesin.
MIN_EVENT_THRESHOLD = {
    "BTCUSDT": 0.0040,  # %0.40
    "ETHUSDT": 0.0050,  # %0.50
    "SOLUSDT": 0.0075,  # %0.75
    "XAUUSDT": 0.0020,  # %0.20
    "XAGUSDT": 0.0035,  # %0.35
}

# Bir yön, karşı yön hareketinden en az bu kadar baskın olmalı.
DIRECTION_DOMINANCE = 1.15

# Dataset çok dengesiz olmasın fakat olay nadirliği de korunsun.
MAX_NEGATIVES_PER_EVENT = 8

EMBARGO = pd.Timedelta(hours=1)

RANDOM_STATE = 42

CLASS_VALUES = [-1, 0, 1]

CLASS_NAMES = {
    -1: "LONG_SQUEEZE",
    0: "NO_EVENT",
    1: "SHORT_SQUEEZE",
}

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

FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)

LOAD_COLUMNS = list(
    dict.fromkeys(
        [
            "id",
            "logged_at",
            "symbol",
            "timeframe",
        ]
        + FEATURE_COLUMNS
    )
)


def prepare_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    X = frame[FEATURE_COLUMNS].copy()

    for column in CATEGORICAL_FEATURES:
        X[column] = (
            X[column]
            .astype("string")
            .fillna("<MISSING>")
            .astype(str)
        )

    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    return X


def future_excursions(
    times_ns: np.ndarray,
    prices: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Her snapshot için sonraki 30 dakikadaki:
      - maksimum yukarı hareket
      - maksimum aşağı hareket
      - maksimum yukarı hareket zamanı
      - maksimum aşağı hareket zamanı
    """

    row_count = len(prices)
    window_ns = int(FUTURE_WINDOW.value)

    future_up = np.full(
        row_count,
        np.nan,
        dtype=np.float32,
    )

    future_down = np.full(
        row_count,
        np.nan,
        dtype=np.float32,
    )

    future_up_time_ns = np.full(
        row_count,
        -1,
        dtype=np.int64,
    )

    future_down_time_ns = np.full(
        row_count,
        -1,
        dtype=np.int64,
    )

    chunk_size = 20_000

    for start in range(
        0,
        row_count,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            row_count,
        )

        selected = np.arange(
            start,
            end,
            dtype=np.int64,
        )

        end_times = (
            times_ns[selected]
            + window_ns
        )

        end_indices = np.searchsorted(
            times_ns,
            end_times,
            side="right",
        )

        lengths = (
            end_indices
            - selected
            - 1
        )

        max_steps = int(
            lengths.max(initial=0)
        )

        if max_steps <= 0:
            continue

        offsets = np.arange(
            1,
            max_steps + 1,
            dtype=np.int32,
        )

        future_indices = (
            selected[:, None]
            + offsets[None, :]
        )

        valid = (
            future_indices
            < end_indices[:, None]
        )

        safe_indices = np.minimum(
            future_indices,
            row_count - 1,
        )

        future_prices = prices[
            safe_indices
        ]

        returns = (
            future_prices
            / prices[selected, None]
            - 1.0
        )

        up_matrix = np.where(
            valid,
            returns,
            -np.inf,
        )

        down_matrix = np.where(
            valid,
            -returns,
            -np.inf,
        )

        max_up = np.max(
            up_matrix,
            axis=1,
        )

        max_down = np.max(
            down_matrix,
            axis=1,
        )

        up_offsets = np.argmax(
            up_matrix,
            axis=1,
        )

        down_offsets = np.argmax(
            down_matrix,
            axis=1,
        )

        has_future = lengths > 0

        future_up[selected[has_future]] = (
            max_up[has_future]
        ).astype(np.float32)

        future_down[selected[has_future]] = (
            max_down[has_future]
        ).astype(np.float32)

        up_rows = selected[has_future]
        down_rows = selected[has_future]

        future_up_time_ns[up_rows] = times_ns[
            safe_indices[
                has_future,
                up_offsets[has_future],
            ]
        ]

        future_down_time_ns[down_rows] = times_ns[
            safe_indices[
                has_future,
                down_offsets[has_future],
            ]
        ]

        del (
            future_indices,
            valid,
            safe_indices,
            future_prices,
            returns,
            up_matrix,
            down_matrix,
        )

    return (
        future_up,
        future_down,
        future_up_time_ns,
        future_down_time_ns,
    )


def trailing_move_threshold(
    group: pd.DataFrame,
) -> np.ndarray:
    """
    Yalnız geçmiş fiyat hareketlerinden adaptif squeeze eşiği üretir.

    Yaklaşık 30 dakikalık geriye dönük mutlak hareketin,
    son 7 günlük %95 quantile değeri kullanılır.
    """

    times_ns = (
        group["logged_at"]
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )

    prices = group[
        "current_price"
    ].to_numpy(dtype=np.float64)

    past_target_ns = (
        times_ns
        - int(FUTURE_WINDOW.value)
    )

    past_indices = (
        np.searchsorted(
            times_ns,
            past_target_ns,
            side="right",
        )
        - 1
    )

    valid = past_indices >= 0

    backward_move = np.full(
        len(group),
        np.nan,
        dtype=np.float64,
    )

    backward_move[valid] = np.abs(
        prices[valid]
        / prices[past_indices[valid]]
        - 1.0
    )

    move_series = pd.Series(
        backward_move,
        index=pd.DatetimeIndex(
            group["logged_at"]
        ),
    )

    threshold = (
        move_series
        .rolling(
            VOLATILITY_LOOKBACK,
            min_periods=300,
        )
        .quantile(
            VOLATILITY_QUANTILE
        )
        .to_numpy(dtype=np.float64)
    )

    symbol = str(
        group["symbol"].iloc[0]
    )

    floor = MIN_EVENT_THRESHOLD.get(
        symbol,
        0.0050,
    )

    threshold = np.where(
        np.isfinite(threshold),
        np.maximum(
            threshold,
            floor,
        ),
        floor,
    )

    return threshold.astype(
        np.float32
    )


def merge_event_candidates(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aynı sembol ve yöndeki birbirine yakın adayları tek olay yapar.

    Her kümede severity değeri en yüksek olan aday korunur.
    """

    if candidates.empty:
        return candidates

    merged_rows = []

    for (
        symbol,
        direction,
    ), group in candidates.groupby(
        ["symbol", "event_direction"],
        observed=True,
        sort=False,
    ):
        group = group.sort_values(
            "event_time"
        )

        cluster_rows = []
        last_event_time = None

        for row in group.itertuples(
            index=False
        ):
            if (
                last_event_time is None
                or (
                    row.event_time
                    - last_event_time
                    > EVENT_MERGE_WINDOW
                )
            ):
                if cluster_rows:
                    merged_rows.append(
                        max(
                            cluster_rows,
                            key=lambda item: (
                                item["severity"]
                            ),
                        )
                    )

                cluster_rows = []

            cluster_rows.append(
                row._asdict()
            )

            last_event_time = (
                row.event_time
            )

        if cluster_rows:
            merged_rows.append(
                max(
                    cluster_rows,
                    key=lambda item: (
                        item["severity"]
                    ),
                )
            )

    output = pd.DataFrame(
        merged_rows
    )

    return output.sort_values(
        "event_time"
    ).reset_index(drop=True)


def detect_events(
    data: pd.DataFrame,
) -> pd.DataFrame:
    event_frames = []

    for symbol, group in data.groupby(
        "symbol",
        observed=True,
        sort=False,
    ):
        group = group.sort_values(
            ["logged_at", "id"]
        ).reset_index(drop=True)

        print(
            f"Detecting events: "
            f"{symbol} rows={len(group):,}",
            flush=True,
        )

        times_ns = (
            group["logged_at"]
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )

        prices = group[
            "current_price"
        ].to_numpy(dtype=np.float64)

        (
            future_up,
            future_down,
            future_up_time_ns,
            future_down_time_ns,
        ) = future_excursions(
            times_ns,
            prices,
        )

        threshold = (
            trailing_move_threshold(
                group
            )
        )

        short_candidate = (
            np.isfinite(future_up)
            & (
                future_up >= threshold
            )
            & (
                future_up
                >= (
                    future_down
                    * DIRECTION_DOMINANCE
                )
            )
        )

        long_candidate = (
            np.isfinite(future_down)
            & (
                future_down >= threshold
            )
            & (
                future_down
                >= (
                    future_up
                    * DIRECTION_DOMINANCE
                )
            )
        )

        short_rows = np.flatnonzero(
            short_candidate
        )

        long_rows = np.flatnonzero(
            long_candidate
        )

        rows = []

        for index in short_rows:
            rows.append(
                {
                    "symbol": symbol,
                    "source_id": (
                        group.iloc[index][
                            "id"
                        ]
                    ),
                    "source_time": (
                        group.iloc[index][
                            "logged_at"
                        ]
                    ),
                    "event_time": pd.Timestamp(
                        future_up_time_ns[
                            index
                        ],
                        unit="ns",
                        tz="UTC",
                    ),
                    "event_direction": 1,
                    "event_name": (
                        "SHORT_SQUEEZE"
                    ),
                    "severity": float(
                        future_up[index]
                    ),
                    "threshold": float(
                        threshold[index]
                    ),
                    "opposite_excursion": (
                        float(
                            future_down[
                                index
                            ]
                        )
                    ),
                }
            )

        for index in long_rows:
            rows.append(
                {
                    "symbol": symbol,
                    "source_id": (
                        group.iloc[index][
                            "id"
                        ]
                    ),
                    "source_time": (
                        group.iloc[index][
                            "logged_at"
                        ]
                    ),
                    "event_time": pd.Timestamp(
                        future_down_time_ns[
                            index
                        ],
                        unit="ns",
                        tz="UTC",
                    ),
                    "event_direction": -1,
                    "event_name": (
                        "LONG_SQUEEZE"
                    ),
                    "severity": float(
                        future_down[index]
                    ),
                    "threshold": float(
                        threshold[index]
                    ),
                    "opposite_excursion": (
                        float(
                            future_up[
                                index
                            ]
                        )
                    ),
                }
            )

        symbol_candidates = pd.DataFrame(
            rows
        )

        merged = merge_event_candidates(
            symbol_candidates
        )

        print(
            f"  candidates="
            f"{len(symbol_candidates):,} "
            f"merged_events="
            f"{len(merged):,}",
            flush=True,
        )

        event_frames.append(
            merged
        )

    events = pd.concat(
        event_frames,
        ignore_index=True,
    )

    return events.sort_values(
        "event_time"
    ).reset_index(drop=True)


def build_event_dataset(
    data: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    positive_rows = []
    negative_frames = []

    for symbol, symbol_data in data.groupby(
        "symbol",
        observed=True,
        sort=False,
    ):
        symbol_data = (
            symbol_data
            .sort_values(
                ["logged_at", "id"]
            )
            .reset_index(drop=True)
        )

        times_ns = (
            symbol_data["logged_at"]
            .to_numpy(
                dtype="datetime64[ns]"
            )
            .astype(np.int64)
        )

        symbol_events = events.loc[
            events["symbol"].eq(symbol)
        ].sort_values("event_time")

        event_times_ns = (
            symbol_events["event_time"]
            .to_numpy(
                dtype="datetime64[ns]"
            )
            .astype(np.int64)
        )

        precursor_target_ns = (
            event_times_ns
            - int(PRECURSOR_OFFSET.value)
        )

        precursor_indices = (
            np.searchsorted(
                times_ns,
                precursor_target_ns,
                side="right",
            )
            - 1
        )

        valid = precursor_indices >= 0

        precursor_indices = (
            precursor_indices[valid]
        )

        valid_events = (
            symbol_events.iloc[
                np.flatnonzero(valid)
            ]
            .reset_index(drop=True)
        )

        positives = symbol_data.iloc[
            precursor_indices
        ].copy()

        positives[
            "target_event"
        ] = (
            valid_events[
                "event_direction"
            ]
            .to_numpy(dtype=np.int8)
        )

        positives[
            "event_time"
        ] = (
            valid_events[
                "event_time"
            ].to_numpy()
        )

        positives[
            "event_severity"
        ] = (
            valid_events[
                "severity"
            ].to_numpy(
                dtype=np.float32
            )
        )

        positives[
            "sample_type"
        ] = "EVENT_PRECURSOR"

        positive_rows.append(
            positives
        )

        # Her 15 dakikada bir negatif aday.
        sampled = (
            symbol_data
            .set_index("logged_at")
            .resample(
                NEGATIVE_SAMPLE_INTERVAL
            )
            .last()
            .dropna(
                subset=["id"]
            )
            .reset_index()
        )

        if len(symbol_events):
            candidate_times_ns = (
                sampled["logged_at"]
                .to_numpy(
                    dtype="datetime64[ns]"
                )
                .astype(np.int64)
            )

            nearest_right = np.searchsorted(
                event_times_ns,
                candidate_times_ns,
                side="left",
            )

            safe_right = np.minimum(
                nearest_right,
                len(event_times_ns) - 1,
            )

            right_distance = np.abs(
                event_times_ns[safe_right]
                - candidate_times_ns
            )

            safe_left = np.maximum(
                nearest_right - 1,
                0,
            )

            left_distance = np.abs(
                event_times_ns[safe_left]
                - candidate_times_ns
            )

            nearest_distance = np.minimum(
                left_distance,
                right_distance,
            )

            safe_negative = (
                nearest_distance
                > int(
                    EVENT_EXCLUSION_WINDOW.value
                )
            )

            sampled = sampled.loc[
                safe_negative
            ].copy()

        maximum_negatives = (
            max(
                1,
                len(positives)
                * MAX_NEGATIVES_PER_EVENT,
            )
        )

        if (
            len(sampled)
            > maximum_negatives
        ):
            sampled = sampled.sample(
                n=maximum_negatives,
                random_state=(
                    RANDOM_STATE
                    + sum(
                        ord(char)
                        for char in symbol
                    )
                ),
                replace=False,
            )

        sampled["target_event"] = 0
        sampled["event_time"] = pd.NaT
        sampled["event_severity"] = (
            np.nan
        )
        sampled["sample_type"] = (
            "NO_EVENT"
        )

        negative_frames.append(
            sampled
        )

    dataset = pd.concat(
        positive_rows
        + negative_frames,
        ignore_index=True,
    )

    dataset["target_event"] = (
        dataset["target_event"]
        .astype("int8")
    )

    return dataset.sort_values(
        ["logged_at", "id"],
        kind="mergesort",
    ).reset_index(drop=True)


def top_event_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict]:
    event_probability = (
        1.0 - probabilities[:, 1]
    )

    actual_event = (
        y_true != 0
    ).astype(np.int8)

    base_rate = float(
        actual_event.mean()
    )

    order = np.argsort(
        -event_probability
    )

    rows = []

    for fraction in [
        0.01,
        0.05,
        0.10,
        0.20,
    ]:
        count = max(
            1,
            int(
                len(y_true)
                * fraction
            ),
        )

        selected = order[:count]

        event_rate = float(
            actual_event[
                selected
            ].mean()
        )

        direction_rows = selected[
            actual_event[selected] == 1
        ]

        if len(direction_rows):
            predicted_classes = (
                CLASS_VALUES_ARRAY[
                    np.argmax(
                        probabilities[
                            direction_rows
                        ],
                        axis=1,
                    )
                ]
            )

            direction_accuracy = float(
                (
                    predicted_classes
                    == y_true[
                        direction_rows
                    ]
                ).mean()
            )
        else:
            direction_accuracy = None

        rows.append(
            {
                "fraction": fraction,
                "rows": count,
                "event_rate": event_rate,
                "base_rate": base_rate,
                "lift": (
                    event_rate
                    / base_rate
                    if base_rate > 0
                    else None
                ),
                "direction_accuracy_on_events": (
                    direction_accuracy
                ),
            }
        )

    return rows


CLASS_VALUES_ARRAY = np.array(
    CLASS_VALUES,
    dtype=np.int8,
)


def run_walk_forward(
    dataset: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict],
]:
    q55 = dataset[
        "logged_at"
    ].quantile(0.55)

    q70 = dataset[
        "logged_at"
    ].quantile(0.70)

    q85 = dataset[
        "logged_at"
    ].quantile(0.85)

    folds = [
        {
            "name": "fold_1",
            "train_end": (
                q55 - EMBARGO
            ),
            "test_start": (
                q55 + EMBARGO
            ),
            "test_end": (
                q70 - EMBARGO
            ),
        },
        {
            "name": "fold_2",
            "train_end": (
                q70 - EMBARGO
            ),
            "test_start": (
                q70 + EMBARGO
            ),
            "test_end": (
                q85 - EMBARGO
            ),
        },
        {
            "name": "fold_3",
            "train_end": (
                q85 - EMBARGO
            ),
            "test_start": (
                q85 + EMBARGO
            ),
            "test_end": (
                dataset[
                    "logged_at"
                ].max()
            ),
        },
    ]

    summary_rows = []
    prediction_frames = []
    importance_frames = []
    detailed_results = []

    for fold_index, fold in enumerate(
        folds,
        start=1,
    ):
        train = dataset.loc[
            dataset["logged_at"]
            <= fold["train_end"]
        ].copy()

        test = dataset.loc[
            (
                dataset["logged_at"]
                >= fold["test_start"]
            )
            & (
                dataset["logged_at"]
                <= fold["test_end"]
            )
        ].copy()

        if min(
            len(train),
            len(test),
        ) == 0:
            raise RuntimeError(
                f"{fold['name']}: "
                f"empty split."
            )

        print()
        print("=" * 82)
        print(
            f"EVENT WALK-FORWARD "
            f"{fold['name'].upper()}"
        )
        print("=" * 82)

        print(
            f"Train: {len(train):,} "
            f"{train['logged_at'].min()} "
            f"-> "
            f"{train['logged_at'].max()}"
        )

        print(
            f"Test : {len(test):,} "
            f"{test['logged_at'].min()} "
            f"-> "
            f"{test['logged_at'].max()}"
        )

        print("Train classes:")
        print(
            train["target_event"]
            .value_counts(
                normalize=True
            )
            .sort_index()
            .to_string()
        )

        X_train = prepare_features(
            train
        )

        y_train = train[
            "target_event"
        ]

        X_test = prepare_features(
            test
        )

        y_test = test[
            "target_event"
        ]

        categorical_indices = [
            FEATURE_COLUMNS.index(
                column
            )
            for column
            in CATEGORICAL_FEATURES
        ]

        train_pool = Pool(
            X_train,
            label=y_train,
            cat_features=(
                categorical_indices
            ),
            feature_names=(
                FEATURE_COLUMNS
            ),
        )

        test_pool = Pool(
            X_test,
            label=y_test,
            cat_features=(
                categorical_indices
            ),
            feature_names=(
                FEATURE_COLUMNS
            ),
        )

        model = CatBoostClassifier(
            iterations=1200,
            depth=7,
            learning_rate=0.04,
            loss_function="MultiClass",
            eval_metric=(
                "TotalF1:average=Macro"
            ),
            auto_class_weights=(
                "Balanced"
            ),
            random_seed=(
                RANDOM_STATE
                + fold_index
            ),
            random_strength=1.0,
            l2_leaf_reg=7.0,
            bootstrap_type=(
                "Bayesian"
            ),
            bagging_temperature=1.0,
            verbose=100,
            allow_writing_files=False,
            thread_count=-1,
        )

        fit_started = time.time()

        model.fit(
            train_pool
        )

        training_seconds = (
            time.time()
            - fit_started
        )

        probabilities = (
            model.predict_proba(
                test_pool
            )
        )

        class_order = np.array(
            model.classes_,
            dtype=np.int8,
        )

        prediction_indices = (
            np.argmax(
                probabilities,
                axis=1,
            )
        )

        predictions = class_order[
            prediction_indices
        ]

        accuracy = float(
            accuracy_score(
                y_test,
                predictions,
            )
        )

        balanced = float(
            balanced_accuracy_score(
                y_test,
                predictions,
            )
        )

        macro_f1 = float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        loss = float(
            log_loss(
                y_test,
                probabilities,
                labels=list(
                    class_order
                ),
            )
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=CLASS_VALUES,
        )

        event_metrics = (
            top_event_metrics(
                y_test.to_numpy(
                    dtype=np.int8
                ),
                probabilities[
                    :,
                    [
                        int(
                            np.where(
                                class_order
                                == class_value
                            )[0][0]
                        )
                        for class_value
                        in CLASS_VALUES
                    ],
                ],
            )
        )

        event_map = {
            item["fraction"]: item
            for item in event_metrics
        }

        report = (
            classification_report(
                y_test,
                predictions,
                labels=CLASS_VALUES,
                target_names=[
                    "LONG_SQUEEZE",
                    "NO_EVENT",
                    "SHORT_SQUEEZE",
                ],
                output_dict=True,
                zero_division=0,
            )
        )

        print()
        print(
            f"Accuracy={accuracy:.4f} "
            f"balanced={balanced:.4f} "
            f"macro_f1={macro_f1:.4f} "
            f"log_loss={loss:.4f}"
        )

        print(
            "Confusion matrix "
            "[LONG, NONE, SHORT]:"
        )
        print(matrix)

        print()
        print("Top event buckets:")

        for item in event_metrics:
            print(
                f"  top "
                f"{item['fraction'] * 100:>4.0f}% "
                f"event_rate="
                f"{item['event_rate']:.4f} "
                f"lift="
                f"{item['lift']:.2f}x "
                f"direction_acc="
                f"{item['direction_accuracy_on_events']}"
            )

        importance = pd.DataFrame(
            {
                "feature": (
                    FEATURE_COLUMNS
                ),
                "importance": (
                    model
                    .get_feature_importance(
                        train_pool,
                        type=(
                            "FeatureImportance"
                        ),
                    )
                ),
                "fold": fold["name"],
            }
        )

        importance_frames.append(
            importance
        )

        print()
        print("Top 12 features:")
        print(
            importance
            .sort_values(
                "importance",
                ascending=False,
            )
            .head(12)
            .to_string(index=False)
        )

        prediction_output = test[
            [
                "id",
                "logged_at",
                "symbol",
                "target_event",
                "event_time",
                "event_severity",
            ]
        ].copy()

        prediction_output[
            "fold"
        ] = fold["name"]

        prediction_output[
            "prediction"
        ] = predictions

        for class_value in CLASS_VALUES:
            class_index = int(
                np.where(
                    class_order
                    == class_value
                )[0][0]
            )

            prediction_output[
                f"probability_{CLASS_NAMES[class_value].lower()}"
            ] = probabilities[
                :,
                class_index,
            ].astype(
                np.float32
            )

        prediction_output[
            "squeeze_probability"
        ] = (
            1.0
            - prediction_output[
                "probability_no_event"
            ]
        ).astype(np.float32)

        prediction_frames.append(
            prediction_output
        )

        summary_rows.append(
            {
                "fold": fold["name"],
                "train_rows": len(train),
                "test_rows": len(test),
                "test_event_rate": float(
                    (
                        y_test != 0
                    ).mean()
                ),
                "accuracy": accuracy,
                "balanced_accuracy": (
                    balanced
                ),
                "macro_f1": macro_f1,
                "log_loss": loss,
                "long_recall": (
                    report[
                        "LONG_SQUEEZE"
                    ]["recall"]
                ),
                "short_recall": (
                    report[
                        "SHORT_SQUEEZE"
                    ]["recall"]
                ),
                "top_1pct_event_rate": (
                    event_map[0.01][
                        "event_rate"
                    ]
                ),
                "top_1pct_lift": (
                    event_map[0.01][
                        "lift"
                    ]
                ),
                "top_5pct_event_rate": (
                    event_map[0.05][
                        "event_rate"
                    ]
                ),
                "top_5pct_lift": (
                    event_map[0.05][
                        "lift"
                    ]
                ),
                "top_10pct_event_rate": (
                    event_map[0.10][
                        "event_rate"
                    ]
                ),
                "top_10pct_lift": (
                    event_map[0.10][
                        "lift"
                    ]
                ),
                "training_seconds": (
                    training_seconds
                ),
            }
        )

        detailed_results.append(
            {
                "fold": fold["name"],
                "confusion_matrix": (
                    matrix.tolist()
                ),
                "classification_report": (
                    report
                ),
                "event_buckets": (
                    event_metrics
                ),
            }
        )

    return (
        pd.DataFrame(
            summary_rows
        ),
        pd.concat(
            prediction_frames,
            ignore_index=True,
        ),
        pd.concat(
            importance_frames,
            ignore_index=True,
        ),
        detailed_results,
    )


def main() -> int:
    started = time.time()

    if not INPUT_PATH.exists():
        print(
            f"ERROR: Missing input: "
            f"{INPUT_PATH}",
            file=sys.stderr,
        )
        return 1

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 82)
    print(
        "TOPOLOGY V2 — "
        "SPARSE SQUEEZE EVENT RESEARCH"
    )
    print("=" * 82)
    print(f"Input        : {INPUT_PATH}")
    print(
        f"Topology     : "
        f"{TOPOLOGY_TIMEFRAME}"
    )
    print(
        f"Future window: "
        f"{FUTURE_WINDOW}"
    )
    print(
        f"Precursor    : "
        f"{PRECURSOR_OFFSET} before event"
    )
    print(
        f"Merge window : "
        f"{EVENT_MERGE_WINDOW}"
    )
    print()

    print("Loading 24h topology rows...")

    data = pd.read_parquet(
        INPUT_PATH,
        columns=LOAD_COLUMNS,
        filters=[
            (
                "timeframe",
                "==",
                TOPOLOGY_TIMEFRAME,
            )
        ],
    )

    if data.empty:
        raise RuntimeError(
            "No 24h rows found."
        )

    data = data.sort_values(
        [
            "symbol",
            "logged_at",
            "id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    print(
        f"Rows       : {len(data):,}"
    )
    print(
        f"Date range : "
        f"{data['logged_at'].min()} "
        f"-> {data['logged_at'].max()}"
    )

    print()
    print("Detecting sparse squeeze events...")

    events = detect_events(
        data
    )

    events.to_parquet(
        EVENTS_PATH,
        index=False,
        compression="zstd",
    )

    total_days = (
        (
            data["logged_at"].max()
            - data["logged_at"].min()
        ).total_seconds()
        / 86_400
    )

    print()
    print("=" * 82)
    print("DETECTED EVENTS")
    print("=" * 82)
    print(
        f"Events     : {len(events):,}"
    )
    print(
        f"Events/day : "
        f"{len(events) / total_days:.2f} "
        f"across all symbols"
    )

    print()
    print("By direction:")
    print(
        events[
            "event_name"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("By symbol:")
    print(
        pd.crosstab(
            events["symbol"],
            events["event_name"],
        ).to_string()
    )

    print()
    print("Severity profile:")
    print(
        events[
            "severity"
        ]
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    print()
    print("Building event precursor dataset...")

    dataset = build_event_dataset(
        data,
        events,
    )

    dataset.to_parquet(
        DATASET_PATH,
        index=False,
        compression="zstd",
    )

    print(
        f"Dataset rows: "
        f"{len(dataset):,}"
    )

    print("Dataset classes:")
    print(
        dataset[
            "target_event"
        ]
        .value_counts()
        .sort_index()
        .rename(
            index=CLASS_NAMES
        )
        .to_string()
    )

    (
        summary,
        predictions,
        importance,
        detailed_results,
    ) = run_walk_forward(
        dataset
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    predictions.to_parquet(
        PREDICTIONS_PATH,
        index=False,
        compression="zstd",
    )

    mean_importance = (
        importance
        .groupby(
            "feature",
            observed=True,
        )["importance"]
        .mean()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    mean_importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    aggregate = {
        "mean_balanced_accuracy": float(
            summary[
                "balanced_accuracy"
            ].mean()
        ),
        "min_balanced_accuracy": float(
            summary[
                "balanced_accuracy"
            ].min()
        ),
        "mean_macro_f1": float(
            summary["macro_f1"].mean()
        ),
        "mean_top_1pct_lift": float(
            summary[
                "top_1pct_lift"
            ].mean()
        ),
        "min_top_1pct_lift": float(
            summary[
                "top_1pct_lift"
            ].min()
        ),
        "mean_top_5pct_lift": float(
            summary[
                "top_5pct_lift"
            ].mean()
        ),
        "min_top_5pct_lift": float(
            summary[
                "top_5pct_lift"
            ].min()
        ),
        "mean_top_10pct_lift": float(
            summary[
                "top_10pct_lift"
            ].mean()
        ),
    }

    report = {
        "event_definition": {
            "topology_timeframe": (
                TOPOLOGY_TIMEFRAME
            ),
            "future_window_minutes": (
                FUTURE_WINDOW
                .total_seconds()
                / 60
            ),
            "precursor_minutes": (
                PRECURSOR_OFFSET
                .total_seconds()
                / 60
            ),
            "merge_window_minutes": (
                EVENT_MERGE_WINDOW
                .total_seconds()
                / 60
            ),
            "volatility_quantile": (
                VOLATILITY_QUANTILE
            ),
            "minimum_thresholds": (
                MIN_EVENT_THRESHOLD
            ),
        },
        "events": {
            "count": int(
                len(events)
            ),
            "events_per_day": float(
                len(events)
                / total_days
            ),
            "direction_counts": {
                str(key): int(value)
                for key, value in (
                    events[
                        "event_name"
                    ]
                    .value_counts()
                    .items()
                )
            },
        },
        "dataset": {
            "rows": int(
                len(dataset)
            ),
            "class_counts": {
                CLASS_NAMES[
                    int(key)
                ]: int(value)
                for key, value in (
                    dataset[
                        "target_event"
                    ]
                    .value_counts()
                    .items()
                )
            },
        },
        "walk_forward": (
            detailed_results
        ),
        "aggregate": aggregate,
        "top_features": (
            mean_importance
            .head(20)
            .to_dict(
                orient="records"
            )
        ),
    }

    with REPORT_PATH.open(
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
    print("=" * 82)
    print("SPARSE SQUEEZE WALK-FORWARD SUMMARY")
    print("=" * 82)

    print(
        summary[
            [
                "fold",
                "test_event_rate",
                "balanced_accuracy",
                "macro_f1",
                "long_recall",
                "short_recall",
                "top_1pct_event_rate",
                "top_1pct_lift",
                "top_5pct_event_rate",
                "top_5pct_lift",
                "top_10pct_event_rate",
                "top_10pct_lift",
            ]
        ].to_string(index=False)
    )

    print()
    print("Aggregate:")
    print(
        f"  Mean balanced accuracy : "
        f"{aggregate['mean_balanced_accuracy']:.4f}"
    )
    print(
        f"  Min balanced accuracy  : "
        f"{aggregate['min_balanced_accuracy']:.4f}"
    )
    print(
        f"  Mean macro F1          : "
        f"{aggregate['mean_macro_f1']:.4f}"
    )
    print(
        f"  Mean top 1% lift       : "
        f"{aggregate['mean_top_1pct_lift']:.2f}x"
    )
    print(
        f"  Min top 1% lift        : "
        f"{aggregate['min_top_1pct_lift']:.2f}x"
    )
    print(
        f"  Mean top 5% lift       : "
        f"{aggregate['mean_top_5pct_lift']:.2f}x"
    )
    print(
        f"  Min top 5% lift        : "
        f"{aggregate['min_top_5pct_lift']:.2f}x"
    )

    print()
    print("Mean top features:")
    print(
        mean_importance
        .head(15)
        .to_string(index=False)
    )

    print()
    print("=" * 82)
    print("SQUEEZE EVENT RESEARCH COMPLETE")
    print("=" * 82)
    print(f"Events      : {EVENTS_PATH}")
    print(f"Dataset     : {DATASET_PATH}")
    print(f"Summary     : {SUMMARY_PATH}")
    print(f"Predictions : {PREDICTIONS_PATH}")
    print(f"Importance  : {IMPORTANCE_PATH}")
    print(f"Report      : {REPORT_PATH}")
    print(
        f"Elapsed     : "
        f"{time.time() - started:.1f}s"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
