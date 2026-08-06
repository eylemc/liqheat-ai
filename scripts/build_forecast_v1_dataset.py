#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer


INPUT_PATH = Path(
    "data/features/liq_topology_v2_ml_features.parquet"
)

OUTPUT_PATH = Path(
    "data/forecast_v1/multitimeframe_forecast_dataset.parquet"
)

REPORT_PATH = Path(
    "reports/forecast_v1/dataset_report.json"
)

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

TIMEFRAMES = [
    "1h",
    "4h",
    "24h",
]

HORIZONS = [
    15,
    30,
    60,
]

LAGS_BY_TIMEFRAME = {
    "1h": [5, 15, 30, 60],
    "4h": [15, 30, 60],
    "24h": [30, 60],
}

# Future excursion threshold.
# 25 bps = 0.25%, 35 bps = 0.35%, 50 bps = 0.50%.
EVENT_THRESHOLDS_BPS = {
    15: 25.0,
    30: 35.0,
    60: 50.0,
}

STATIC_COLUMNS = [
    "id",
    "logged_at",
    "symbol",
    "timeframe",
    "current_price",
    "nearest_side",
    "nearest_side_code",
    "has_upper_level",
    "has_lower_level",
    "has_topology",
    "upper_distance_pct",
    "lower_distance_pct",
    "distance_advantage",
    "signed_distance_edge",
    "upper_pool_volume",
    "lower_pool_volume",
    "nearest_pool_volume",
    "farther_pool_volume",
    "upper_total_volume",
    "lower_total_volume",
    "upper_active_levels",
    "lower_active_levels",
    "pool_volume_ratio",
    "distance_pressure_ratio",
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

TEMPORAL_COLUMNS = [
    "current_price",
    "upper_distance_pct",
    "lower_distance_pct",
    "distance_advantage",
    "signed_distance_edge",
    "upper_pool_volume",
    "lower_pool_volume",
    "nearest_pool_volume",
    "farther_pool_volume",
    "upper_total_volume",
    "lower_total_volume",
    "upper_active_levels",
    "lower_active_levels",
    "pool_volume_ratio",
    "distance_pressure_ratio",
    "topology_imbalance",
    "total_volume_imbalance_check",
]


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing input columns: "
            + ", ".join(missing)
        )


def safe_ratio(
    current: pd.Series,
    previous: pd.Series,
) -> pd.Series:
    current_numeric = pd.to_numeric(
        current,
        errors="coerce",
    )

    previous_numeric = pd.to_numeric(
        previous,
        errors="coerce",
    )

    denominator = previous_numeric.abs()

    return np.where(
        denominator > 1e-12,
        (
            current_numeric
            - previous_numeric
        ) / denominator,
        np.nan,
    )


def add_temporal_features(
    frame: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    """
    Her symbol/timeframe stream'inde geçmişe dönük değişim
    feature'ları üretir.

    shift sayısı yaklaşık dakikalık snapshot düzenine dayanır;
    gerçek timestamp farkı ayrıca doğrulanır.
    """
    frame = (
        frame
        .sort_values(
            ["symbol", "logged_at", "id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    groups = frame.groupby(
        "symbol",
        observed=True,
        sort=False,
    )

    for lag_minutes in LAGS_BY_TIMEFRAME[timeframe]:
        previous_time = groups["logged_at"].shift(
            lag_minutes
        )

        actual_lag_minutes = (
            frame["logged_at"]
            - previous_time
        ).dt.total_seconds() / 60.0

        valid_lag = (
            actual_lag_minutes
            .between(
                lag_minutes * 0.65,
                lag_minutes * 1.60,
            )
            .fillna(False)
            .astype(bool)
        )

        frame[
            f"history_valid_{lag_minutes}m"
        ] = valid_lag.astype("int8")

        frame[
            f"history_actual_minutes_{lag_minutes}m"
        ] = actual_lag_minutes.astype("float32")

        for column in TEMPORAL_COLUMNS:
            previous = groups[column].shift(
                lag_minutes
            )

            current_numeric = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            previous_numeric = pd.to_numeric(
                previous,
                errors="coerce",
            )

            delta = (
                current_numeric
                - previous_numeric
            ).where(valid_lag)

            ratio = pd.Series(
                safe_ratio(
                    current_numeric,
                    previous_numeric,
                ),
                index=frame.index,
            ).where(valid_lag)

            frame[
                f"{column}_delta_{lag_minutes}m"
            ] = delta.astype("float32")

            frame[
                f"{column}_change_ratio_{lag_minutes}m"
            ] = ratio.astype("float32")

        current_side = pd.to_numeric(
            frame["nearest_side_code"],
            errors="coerce",
        )

        previous_side = pd.to_numeric(
            groups["nearest_side_code"].shift(
                lag_minutes
            ),
            errors="coerce",
        )

        side_stable = (
            (current_side == previous_side)
            & valid_lag
            & current_side.notna()
            & previous_side.notna()
        )

        frame[
            f"nearest_side_stable_{lag_minutes}m"
        ] = (
            side_stable
            .fillna(False)
            .astype("int8")
        )

    return frame


def prefix_timeframe(
    frame: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    protected = {
        "id",
        "logged_at",
        "symbol",
    }

    rename_map = {
        column: f"{timeframe}_{column}"
        for column in frame.columns
        if column not in protected
    }

    return frame.rename(
        columns=rename_map
    )


def align_timeframes(
    streams: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    1h stream ana gözlem çizgisidir.
    4h ve 24h snapshot'ları symbol bazında backward ASOF ile bağlanır.
    Gelecek bilgi kullanılmaz.
    """
    base = prefix_timeframe(
        streams["1h"],
        "tf1h",
    )

    base = base.rename(
        columns={
            "id": "observation_id",
            "logged_at": "observation_time",
        }
    )

    output_parts = []

    for symbol in SYMBOLS:
        left = (
            base[
                base["symbol"].astype(str)
                == symbol
            ]
            .sort_values("observation_time")
            .reset_index(drop=True)
        )

        if left.empty:
            continue

        merged = left

        for source_timeframe, prefix in [
            ("4h", "tf4h"),
            ("24h", "tf24h"),
        ]:
            right = prefix_timeframe(
                streams[source_timeframe],
                prefix,
            )

            right = (
                right[
                    right["symbol"].astype(str)
                    == symbol
                ]
                .drop(columns=["symbol"])
                .sort_values("logged_at")
                .reset_index(drop=True)
            )

            if right.empty:
                continue

            right = right.rename(
                columns={
                    "id": f"{prefix}_source_id",
                    "logged_at": f"{prefix}_source_time",
                }
            )

            merged = pd.merge_asof(
                merged.sort_values(
                    "observation_time"
                ),
                right.sort_values(
                    f"{prefix}_source_time"
                ),
                left_on="observation_time",
                right_on=f"{prefix}_source_time",
                direction="backward",
                tolerance=pd.Timedelta(
                    minutes=4
                ),
            )

            age_seconds = (
                merged["observation_time"]
                - merged[
                    f"{prefix}_source_time"
                ]
            ).dt.total_seconds()

            merged[
                f"{prefix}_snapshot_age_seconds"
            ] = age_seconds.astype("float32")

        output_parts.append(merged)

    if not output_parts:
        raise RuntimeError(
            "No aligned symbol streams produced"
        )

    return pd.concat(
        output_parts,
        ignore_index=True,
    )


def add_forward_targets(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Her horizon için:
    - endpoint return
    - future maximum upside
    - future maximum downside
    - excursion-based direction class

    Direction classes:
      -1 = bearish expectancy
       0 = no meaningful event
       1 = bullish expectancy
    """
    output_parts = []

    for symbol in SYMBOLS:
        part = (
            frame[
                frame["symbol"].astype(str)
                == symbol
            ]
            .sort_values(
                "observation_time"
            )
            .reset_index(drop=True)
            .copy()
        )

        if part.empty:
            continue

        prices = pd.to_numeric(
            part["tf1h_current_price"],
            errors="coerce",
        ).astype(float)

        timestamps = pd.to_datetime(
            part["observation_time"],
            utc=True,
            errors="coerce",
        )

        for horizon in HORIZONS:
            future_price = prices.shift(
                -horizon
            )

            future_time = timestamps.shift(
                -horizon
            )

            actual_minutes = (
                future_time
                - timestamps
            ).dt.total_seconds() / 60.0

            valid = actual_minutes.between(
                horizon * 0.82,
                horizon * 1.25,
            )

            endpoint_return = (
                future_price / prices - 1.0
            ).where(valid)

            indexer = FixedForwardWindowIndexer(
                window_size=horizon + 1
            )

            future_max_price = (
                prices
                .rolling(
                    window=indexer,
                    min_periods=horizon + 1,
                )
                .max()
            )

            future_min_price = (
                prices
                .rolling(
                    window=indexer,
                    min_periods=horizon + 1,
                )
                .min()
            )

            max_up_bps = (
                (
                    future_max_price
                    / prices
                    - 1.0
                )
                * 10_000.0
            ).where(valid)

            max_down_bps = (
                (
                    future_min_price
                    / prices
                    - 1.0
                )
                * 10_000.0
            ).where(valid)

            threshold = (
                EVENT_THRESHOLDS_BPS[
                    horizon
                ]
            )

            bullish_event = (
                (max_up_bps >= threshold)
                & (
                    max_up_bps
                    > max_down_bps.abs()
                )
            )

            bearish_event = (
                (
                    max_down_bps
                    <= -threshold
                )
                & (
                    max_down_bps.abs()
                    > max_up_bps
                )
            )

            target_class = np.select(
                [
                    bullish_event,
                    bearish_event,
                ],
                [
                    1,
                    -1,
                ],
                default=0,
            )

            target_class = pd.Series(
                target_class,
                index=part.index,
                dtype="Int8",
            ).where(valid)

            part[
                f"future_valid_{horizon}m"
            ] = valid.astype("int8")

            part[
                f"future_actual_minutes_{horizon}m"
            ] = actual_minutes.astype(
                "float32"
            )

            part[
                f"future_return_{horizon}m"
            ] = endpoint_return.astype(
                "float32"
            )

            part[
                f"future_return_bps_{horizon}m"
            ] = (
                endpoint_return
                * 10_000.0
            ).astype("float32")

            part[
                f"future_max_up_bps_{horizon}m"
            ] = max_up_bps.astype(
                "float32"
            )

            part[
                f"future_max_down_bps_{horizon}m"
            ] = max_down_bps.astype(
                "float32"
            )

            part[
                f"target_direction_{horizon}m"
            ] = target_class

            part[
                f"target_event_{horizon}m"
            ] = (
                target_class.fillna(0) != 0
            ).astype("int8").where(valid)

            part[
                f"target_excursion_bps_{horizon}m"
            ] = np.where(
                bullish_event,
                max_up_bps,
                np.where(
                    bearish_event,
                    max_down_bps,
                    endpoint_return
                    * 10_000.0,
                ),
            ).astype("float32")

        output_parts.append(part)

    return pd.concat(
        output_parts,
        ignore_index=True,
    )


def add_cross_timeframe_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Timeframe agreement ve divergence feature'ları.
    """
    side_columns = [
        "tf1h_nearest_side_code",
        "tf4h_nearest_side_code",
        "tf24h_nearest_side_code",
    ]

    side_values = []

    for column in side_columns:
        side_values.append(
            pd.to_numeric(
                frame[column],
                errors="coerce",
            )
        )

    side_matrix = np.column_stack(
        side_values
    )

    valid_count = np.sum(
        np.isfinite(side_matrix),
        axis=1,
    )

    side_sum = np.nansum(
        side_matrix,
        axis=1,
    )

    frame[
        "mtf_side_valid_count"
    ] = valid_count.astype("int8")

    frame[
        "mtf_side_sum"
    ] = side_sum.astype("float32")

    frame[
        "mtf_side_agreement_strength"
    ] = np.where(
        valid_count > 0,
        np.abs(side_sum) / valid_count,
        np.nan,
    ).astype("float32")

    frame[
        "mtf_side_direction"
    ] = np.sign(
        side_sum
    ).astype("int8")

    for feature in [
        "topology_imbalance",
        "signed_distance_edge",
        "pool_volume_ratio",
        "distance_pressure_ratio",
    ]:
        one_hour = pd.to_numeric(
            frame[f"tf1h_{feature}"],
            errors="coerce",
        )

        four_hour = pd.to_numeric(
            frame[f"tf4h_{feature}"],
            errors="coerce",
        )

        daily = pd.to_numeric(
            frame[f"tf24h_{feature}"],
            errors="coerce",
        )

        frame[
            f"mtf_{feature}_1h_minus_4h"
        ] = (
            one_hour
            - four_hour
        ).astype("float32")

        frame[
            f"mtf_{feature}_4h_minus_24h"
        ] = (
            four_hour
            - daily
        ).astype("float32")

        frame[
            f"mtf_{feature}_range"
        ] = (
            pd.concat(
                [
                    one_hour,
                    four_hour,
                    daily,
                ],
                axis=1,
            ).max(axis=1)
            - pd.concat(
                [
                    one_hour,
                    four_hour,
                    daily,
                ],
                axis=1,
            ).min(axis=1)
        ).astype("float32")

    frame[
        "price_return_5m"
    ] = frame[
        "tf1h_current_price_change_ratio_5m"
    ]

    frame[
        "price_return_15m"
    ] = frame[
        "tf1h_current_price_change_ratio_15m"
    ]

    frame[
        "price_return_30m"
    ] = frame[
        "tf1h_current_price_change_ratio_30m"
    ]

    frame[
        "price_acceleration_5m_15m"
    ] = (
        frame["price_return_5m"]
        - frame["price_return_15m"]
    ).astype("float32")

    frame[
        "topology_price_divergence_15m"
    ] = (
        frame[
            "tf1h_topology_imbalance_delta_15m"
        ]
        - frame["price_return_15m"]
    ).astype("float32")

    return frame


def main() -> int:
    started = time.time()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("LIQHEAT FORECAST V1 — MULTI-TIMEFRAME DATASET")
    print("=" * 100)
    print("Input :", INPUT_PATH)
    print("Output:", OUTPUT_PATH)
    print("Symbols:", SYMBOLS)
    print("Timeframes:", TIMEFRAMES)
    print("Horizons:", HORIZONS)
    print()

    frame = pd.read_parquet(
        INPUT_PATH,
        columns=STATIC_COLUMNS,
        filters=[
            (
                "symbol",
                "in",
                SYMBOLS,
            ),
            (
                "timeframe",
                "in",
                TIMEFRAMES,
            ),
        ],
    )

    require_columns(
        frame,
        STATIC_COLUMNS,
    )

    frame["logged_at"] = pd.to_datetime(
        frame["logged_at"],
        utc=True,
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "id",
            "logged_at",
            "symbol",
            "timeframe",
            "current_price",
        ]
    )

    frame = frame.drop_duplicates(
        subset=["id"],
        keep="last",
    )

    print(
        "Loaded rows:",
        f"{len(frame):,}",
    )

    streams: dict[str, pd.DataFrame] = {}

    for timeframe in TIMEFRAMES:
        stream = (
            frame[
                frame["timeframe"].astype(str)
                == timeframe
            ]
            .copy()
        )

        print(
            f"Building temporal features: "
            f"{timeframe} "
            f"({len(stream):,} rows)"
        )

        streams[timeframe] = (
            add_temporal_features(
                stream,
                timeframe,
            )
        )

    print("Aligning 1h + 4h + 24h streams...")

    dataset = align_timeframes(
        streams
    )

    dataset = add_cross_timeframe_features(
        dataset
    )

    print("Building forward targets...")

    dataset = add_forward_targets(
        dataset
    )

    dataset = dataset.sort_values(
        [
            "observation_time",
            "symbol",
            "observation_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    dataset["symbol"] = (
        dataset["symbol"]
        .astype(str)
        .astype("category")
    )

    dataset["observation_month"] = (
        dataset["observation_time"]
        .dt.to_period("M")
        .astype(str)
    )

    temporary_path = OUTPUT_PATH.with_suffix(
        ".parquet.part"
    )

    temporary_path.unlink(
        missing_ok=True
    )

    dataset.to_parquet(
        temporary_path,
        index=False,
        compression="zstd",
    )

    temporary_path.replace(
        OUTPUT_PATH
    )

    target_summary = {}

    for horizon in HORIZONS:
        column = (
            f"target_direction_{horizon}m"
        )

        counts = (
            dataset[column]
            .value_counts(
                dropna=False
            )
            .sort_index()
        )

        target_summary[
            str(horizon)
        ] = {
            str(key): int(value)
            for key, value in counts.items()
        }

    elapsed = time.time() - started

    report = {
        "status": "complete",
        "rows": int(len(dataset)),
        "columns": int(len(dataset.columns)),
        "symbols": SYMBOLS,
        "timeframes": TIMEFRAMES,
        "horizons_minutes": HORIZONS,
        "event_thresholds_bps": (
            EVENT_THRESHOLDS_BPS
        ),
        "minimum_time": (
            dataset["observation_time"]
            .min()
            .isoformat()
        ),
        "maximum_time": (
            dataset["observation_time"]
            .max()
            .isoformat()
        ),
        "target_distribution": (
            target_summary
        ),
        "output": str(OUTPUT_PATH),
        "output_mb": round(
            OUTPUT_PATH.stat().st_size
            / 1024**2,
            2,
        ),
        "elapsed_seconds": round(
            elapsed,
            2,
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("FORECAST DATASET COMPLETE")
    print("=" * 100)
    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
