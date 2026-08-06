#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MARKET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "market"
    / "binance-futures-um"
)

FORECAST_INPUT = (
    PROJECT_ROOT
    / "data"
    / "forecast_v1"
    / "multitimeframe_forecast_dataset.parquet"
)

MATRIX_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "matrix"
    / "matrix_features_v1.parquet"
)

MERGED_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "forecast_v3"
    / "matrix_topology_dataset.parquet"
)

REPORT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "matrix"
    / "matrix_features_v1_report.json"
)

LATEST_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "matrix"
    / "latest_matrix_states.csv"
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

MATRIX_LENGTH = 20

TIMEFRAME_TOLERANCES = {
    "1h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=8),
    "24h": pd.Timedelta(hours=48),
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
        if not np.isfinite(value):
            return None

    return value


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    return numerator / denominator.where(
        denominator.abs() > 1e-12
    )


def compute_bars_since(
    condition: pd.Series,
) -> pd.Series:
    """
    Flip görülen barda 0, sonraki barlarda 1, 2, 3...
    İlk flip öncesinde NaN.
    """
    condition = (
        condition
        .fillna(False)
        .astype(bool)
    )

    positions = np.arange(
        len(condition),
        dtype=np.float64,
    )

    last_flip = pd.Series(
        np.where(
            condition.to_numpy(),
            positions,
            np.nan,
        ),
        index=condition.index,
    ).ffill()

    result = pd.Series(
        positions,
        index=condition.index,
    ) - last_flip

    return result.where(
        last_flip.notna()
    )


def compute_trend_state(
    source: pd.Series,
    previous_upper: pd.Series,
    previous_lower: pd.Series,
) -> pd.Series:
    """
    Pine davranışı:

    trend = 0
    trend :=
        src2 > h[1] ? 1 :
        src2 < l[1] ? -1 :
        trend[1]
    """
    source_values = pd.to_numeric(
        source,
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    upper_values = pd.to_numeric(
        previous_upper,
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    lower_values = pd.to_numeric(
        previous_lower,
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    trend = np.zeros(
        len(source_values),
        dtype=np.int8,
    )

    previous_state = 0

    for index in range(len(source_values)):
        source_value = source_values[index]
        upper_value = upper_values[index]
        lower_value = lower_values[index]

        if (
            np.isfinite(source_value)
            and np.isfinite(upper_value)
            and source_value > upper_value
        ):
            previous_state = 1

        elif (
            np.isfinite(source_value)
            and np.isfinite(lower_value)
            and source_value < lower_value
        ):
            previous_state = -1

        trend[index] = previous_state

    return pd.Series(
        trend,
        index=source.index,
        dtype="int8",
    )


def compute_matrix(
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    frame = frame.copy()

    frame["open_time"] = pd.to_datetime(
        frame["open_time"],
        utc=True,
        errors="coerce",
    )

    frame["close_time"] = pd.to_datetime(
        frame["close_time"],
        utc=True,
        errors="coerce",
    )

    frame = (
        frame
        .dropna(
            subset=[
                "open_time",
                "close_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )
        .sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if "is_complete" in frame.columns:
        frame = frame[
            frame["is_complete"]
            .fillna(False)
            .astype(bool)
        ].copy()

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).astype("float64")

    # Pine defaults:
    # src1 = ohlc4
    # src2 = ohlc4
    frame["matrix_source"] = (
        frame["open"]
        + frame["high"]
        + frame["low"]
        + frame["close"]
    ) / 4.0

    weighted_value = (
        frame["matrix_source"]
        * frame["volume"]
    )

    rolling_weighted = weighted_value.rolling(
        MATRIX_LENGTH,
        min_periods=MATRIX_LENGTH,
    ).sum()

    rolling_volume = frame["volume"].rolling(
        MATRIX_LENGTH,
        min_periods=MATRIX_LENGTH,
    ).sum()

    # Pine vwma(src1, len)
    frame["matrix_vwma"] = safe_divide(
        rolling_weighted,
        rolling_volume,
    )

    # Pine:
    # h = highest(ma, len)
    # l = lowest(ma, len)
    frame["matrix_upper"] = (
        frame["matrix_vwma"]
        .rolling(
            MATRIX_LENGTH,
            min_periods=MATRIX_LENGTH,
        )
        .max()
    )

    frame["matrix_lower"] = (
        frame["matrix_vwma"]
        .rolling(
            MATRIX_LENGTH,
            min_periods=MATRIX_LENGTH,
        )
        .min()
    )

    previous_upper = (
        frame["matrix_upper"]
        .shift(1)
    )

    previous_lower = (
        frame["matrix_lower"]
        .shift(1)
    )

    frame["matrix_trend"] = (
        compute_trend_state(
            frame["matrix_source"],
            previous_upper,
            previous_lower,
        )
    )

    previous_trend = (
        frame["matrix_trend"]
        .shift(1)
        .fillna(0)
        .astype("int8")
    )

    # Pine longC / shortC
    frame["matrix_long_flip"] = (
        (frame["matrix_trend"] == 1)
        & (previous_trend == -1)
    ).astype("int8")

    frame["matrix_short_flip"] = (
        (frame["matrix_trend"] == -1)
        & (previous_trend == 1)
    ).astype("int8")

    frame["matrix_flip"] = (
        frame["matrix_long_flip"]
        - frame["matrix_short_flip"]
    ).astype("int8")

    frame["matrix_any_flip"] = (
        frame["matrix_flip"] != 0
    )

    frame["matrix_bars_since_flip"] = (
        compute_bars_since(
            frame["matrix_any_flip"]
        )
        .astype("float32")
    )

    trend_change_group = (
        frame["matrix_trend"]
        .ne(
            frame["matrix_trend"].shift(1)
        )
        .cumsum()
    )

    frame["matrix_trend_age_bars"] = (
        frame.groupby(
            trend_change_group,
            observed=True,
        )
        .cumcount()
        .astype("int32")
    )

    frame["matrix_distance_to_vwma_pct"] = (
        safe_divide(
            frame["matrix_source"]
            - frame["matrix_vwma"],
            frame["matrix_vwma"],
        )
    )

    frame["matrix_distance_to_upper_pct"] = (
        safe_divide(
            frame["matrix_upper"]
            - frame["matrix_source"],
            frame["matrix_source"],
        )
    )

    frame["matrix_distance_to_lower_pct"] = (
        safe_divide(
            frame["matrix_source"]
            - frame["matrix_lower"],
            frame["matrix_source"],
        )
    )

    frame["matrix_channel_width_pct"] = (
        safe_divide(
            frame["matrix_upper"]
            - frame["matrix_lower"],
            frame["matrix_vwma"],
        )
    )

    frame["matrix_above_upper"] = (
        frame["matrix_source"]
        > frame["matrix_upper"]
    ).fillna(False).astype("int8")

    frame["matrix_below_lower"] = (
        frame["matrix_source"]
        < frame["matrix_lower"]
    ).fillna(False).astype("int8")

    frame["matrix_inside_channel"] = (
        (
            frame["matrix_source"]
            <= frame["matrix_upper"]
        )
        & (
            frame["matrix_source"]
            >= frame["matrix_lower"]
        )
    ).fillna(False).astype("int8")

    for lag in [
        1,
        3,
        5,
    ]:
        frame[
            f"matrix_vwma_slope_{lag}bar_pct"
        ] = safe_divide(
            frame["matrix_vwma"]
            - frame["matrix_vwma"].shift(lag),
            frame["matrix_vwma"].shift(lag),
        )

        frame[
            f"matrix_channel_width_change_{lag}bar"
        ] = (
            frame["matrix_channel_width_pct"]
            - frame[
                "matrix_channel_width_pct"
            ].shift(lag)
        )

        frame[
            f"matrix_source_return_{lag}bar"
        ] = frame[
            "matrix_source"
        ].pct_change(
            periods=lag,
            fill_method=None,
        )

    frame["matrix_volume_ratio_20"] = (
        safe_divide(
            frame["volume"],
            frame["volume"].rolling(
                MATRIX_LENGTH,
                min_periods=MATRIX_LENGTH,
            ).mean(),
        )
    )

    # Feature ancak mum kapandıktan sonra erişilebilir.
    frame["matrix_available_at"] = (
        frame["close_time"]
    )

    frame["symbol"] = symbol
    frame["matrix_timeframe"] = timeframe

    keep_columns = [
        "symbol",
        "matrix_timeframe",
        "open_time",
        "close_time",
        "matrix_available_at",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "matrix_source",
        "matrix_vwma",
        "matrix_upper",
        "matrix_lower",
        "matrix_trend",
        "matrix_long_flip",
        "matrix_short_flip",
        "matrix_flip",
        "matrix_bars_since_flip",
        "matrix_trend_age_bars",
        "matrix_distance_to_vwma_pct",
        "matrix_distance_to_upper_pct",
        "matrix_distance_to_lower_pct",
        "matrix_channel_width_pct",
        "matrix_above_upper",
        "matrix_below_lower",
        "matrix_inside_channel",
        "matrix_vwma_slope_1bar_pct",
        "matrix_vwma_slope_3bar_pct",
        "matrix_vwma_slope_5bar_pct",
        "matrix_channel_width_change_1bar",
        "matrix_channel_width_change_3bar",
        "matrix_channel_width_change_5bar",
        "matrix_source_return_1bar",
        "matrix_source_return_3bar",
        "matrix_source_return_5bar",
        "matrix_volume_ratio_20",
    ]

    return frame[keep_columns].copy()


def prefix_matrix_columns(
    frame: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    prefix = f"matrix_{timeframe}"

    protected = {
        "symbol",
        "matrix_available_at",
    }

    rename_map = {
        column: f"{prefix}_{column}"
        for column in frame.columns
        if column not in protected
    }

    return frame.rename(
        columns=rename_map
    )


def merge_matrix_into_forecast(
    forecast: pd.DataFrame,
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    forecast = forecast.copy()

    forecast["observation_time"] = (
        pd.to_datetime(
            forecast["observation_time"],
            utc=True,
            errors="coerce",
        )
    )

    merged_parts = []

    for symbol in SYMBOLS:
        symbol_forecast = (
            forecast[
                forecast["symbol"].astype(str)
                == symbol
            ]
            .sort_values(
                "observation_time"
            )
            .reset_index(drop=True)
        )

        if symbol_forecast.empty:
            continue

        current = symbol_forecast

        for timeframe in TIMEFRAMES:
            matrix_part = (
                matrix[
                    (
                        matrix["symbol"].astype(str)
                        == symbol
                    )
                    & (
                        matrix[
                            "matrix_timeframe"
                        ].astype(str)
                        == timeframe
                    )
                ]
                .sort_values(
                    "matrix_available_at"
                )
                .reset_index(drop=True)
            )

            if matrix_part.empty:
                raise RuntimeError(
                    f"No Matrix rows: "
                    f"{symbol} {timeframe}"
                )

            matrix_part = prefix_matrix_columns(
                matrix_part,
                timeframe,
            )

            matrix_part = matrix_part.drop(
                columns=["symbol"]
            )

            source_time_column = (
                f"matrix_{timeframe}_source_available_at"
            )

            matrix_part = matrix_part.rename(
                columns={
                    "matrix_available_at": (
                        source_time_column
                    )
                }
            )

            current = pd.merge_asof(
                current.sort_values(
                    "observation_time"
                ),
                matrix_part.sort_values(
                    source_time_column
                ),
                left_on="observation_time",
                right_on=source_time_column,
                direction="backward",
                tolerance=(
                    TIMEFRAME_TOLERANCES[
                        timeframe
                    ]
                ),
            )

            current[
                f"matrix_{timeframe}_age_seconds"
            ] = (
                current["observation_time"]
                - current[source_time_column]
            ).dt.total_seconds().astype(
                "float32"
            )

        merged_parts.append(current)

    if not merged_parts:
        raise RuntimeError(
            "No forecast/matrix rows merged"
        )

    merged = pd.concat(
        merged_parts,
        ignore_index=True,
    )

    trend_columns = [
        "matrix_1h_matrix_trend",
        "matrix_4h_matrix_trend",
        "matrix_24h_matrix_trend",
    ]

    trend_matrix = np.column_stack([
        pd.to_numeric(
            merged[column],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        for column in trend_columns
    ])

    valid_count = np.sum(
        np.isfinite(trend_matrix),
        axis=1,
    )

    trend_sum = np.nansum(
        trend_matrix,
        axis=1,
    )

    merged[
        "matrix_mtf_valid_count"
    ] = valid_count.astype("int8")

    merged[
        "matrix_mtf_trend_sum"
    ] = trend_sum.astype("float32")

    merged[
        "matrix_mtf_direction"
    ] = np.sign(
        trend_sum
    ).astype("int8")

    merged[
        "matrix_mtf_agreement_strength"
    ] = np.where(
        valid_count > 0,
        np.abs(trend_sum) / valid_count,
        np.nan,
    ).astype("float32")

    merged[
        "matrix_mtf_full_bullish"
    ] = (
        trend_sum == 3
    ).astype("int8")

    merged[
        "matrix_mtf_full_bearish"
    ] = (
        trend_sum == -3
    ).astype("int8")

    merged[
        "matrix_mtf_mixed"
    ] = (
        (valid_count == 3)
        & (np.abs(trend_sum) < 3)
    ).astype("int8")

    # Matrix ile topology arasındaki çapraz feature'lar.
    if "mtf_side_direction" in merged.columns:
        topology_direction = pd.to_numeric(
            merged["mtf_side_direction"],
            errors="coerce",
        )

        merged[
            "matrix_topology_direction_agreement"
        ] = (
            topology_direction
            == merged["matrix_mtf_direction"]
        ).fillna(False).astype("int8")

        merged[
            "matrix_topology_direction_product"
        ] = (
            topology_direction
            * merged["matrix_mtf_direction"]
        ).astype("float32")

    for timeframe, topology_prefix in [
        ("1h", "tf1h"),
        ("4h", "tf4h"),
        ("24h", "tf24h"),
    ]:
        matrix_trend_column = (
            f"matrix_{timeframe}_matrix_trend"
        )

        topology_column = (
            f"{topology_prefix}_topology_imbalance"
        )

        if (
            matrix_trend_column
            in merged.columns
            and topology_column
            in merged.columns
        ):
            merged[
                f"matrix_{timeframe}_"
                f"trend_x_topology_imbalance"
            ] = (
                pd.to_numeric(
                    merged[matrix_trend_column],
                    errors="coerce",
                )
                * pd.to_numeric(
                    merged[topology_column],
                    errors="coerce",
                )
            ).astype("float32")

    return merged


def main() -> int:
    started = time.time()

    MATRIX_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MERGED_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("KOINVIZYON MATRIX FEATURE ENGINE V1")
    print("=" * 100)
    print("MA type : VWMA")
    print("Length  :", MATRIX_LENGTH)
    print("Symbols :", SYMBOLS)
    print("Frames  :", TIMEFRAMES)
    print()

    matrix_parts = []

    source_report = []

    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            source_path = (
                MARKET_ROOT
                / symbol
                / timeframe
                / f"{symbol}-{timeframe}.parquet"
            )

            if not source_path.exists():
                raise FileNotFoundError(
                    f"Missing OHLCV: {source_path}"
                )

            candles = pd.read_parquet(
                source_path
            )

            print(
                f"Computing Matrix: "
                f"{symbol} {timeframe} "
                f"({len(candles):,} candles)"
            )

            matrix_frame = compute_matrix(
                candles,
                symbol,
                timeframe,
            )

            matrix_parts.append(
                matrix_frame
            )

            valid_matrix = (
                matrix_frame[
                    "matrix_vwma"
                ].notna()
                & matrix_frame[
                    "matrix_upper"
                ].notna()
                & matrix_frame[
                    "matrix_lower"
                ].notna()
            )

            source_report.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "source_rows": int(
                    len(candles)
                ),
                "matrix_rows": int(
                    len(matrix_frame)
                ),
                "valid_matrix_rows": int(
                    valid_matrix.sum()
                ),
                "long_flips": int(
                    matrix_frame[
                        "matrix_long_flip"
                    ].sum()
                ),
                "short_flips": int(
                    matrix_frame[
                        "matrix_short_flip"
                    ].sum()
                ),
                "latest_time": (
                    matrix_frame[
                        "matrix_available_at"
                    ].max().isoformat()
                ),
                "latest_trend": int(
                    matrix_frame[
                        "matrix_trend"
                    ].iloc[-1]
                ),
            })

    matrix = pd.concat(
        matrix_parts,
        ignore_index=True,
    )

    temporary_matrix = (
        MATRIX_OUTPUT.with_suffix(
            ".parquet.part"
        )
    )

    temporary_matrix.unlink(
        missing_ok=True
    )

    matrix.to_parquet(
        temporary_matrix,
        index=False,
        compression="zstd",
    )

    temporary_matrix.replace(
        MATRIX_OUTPUT
    )

    print()
    print(
        "Matrix feature rows:",
        f"{len(matrix):,}",
    )

    print(
        "Matrix feature output:",
        MATRIX_OUTPUT,
    )

    if not FORECAST_INPUT.exists():
        raise FileNotFoundError(
            f"Forecast dataset missing: "
            f"{FORECAST_INPUT}"
        )

    print()
    print("Loading forecast dataset...")

    forecast = pd.read_parquet(
        FORECAST_INPUT
    )

    print(
        "Forecast rows:",
        f"{len(forecast):,}",
    )

    print(
        "Forecast columns:",
        len(forecast.columns),
    )

    print()
    print(
        "Merging Matrix into "
        "topology forecast dataset..."
    )

    merged = merge_matrix_into_forecast(
        forecast,
        matrix,
    )

    temporary_merged = (
        MERGED_OUTPUT.with_suffix(
            ".parquet.part"
        )
    )

    temporary_merged.unlink(
        missing_ok=True
    )

    merged.to_parquet(
        temporary_merged,
        index=False,
        compression="zstd",
    )

    temporary_merged.replace(
        MERGED_OUTPUT
    )

    matrix_feature_columns = [
        column
        for column in merged.columns
        if column.startswith("matrix_")
    ]

    coverage = {}

    for timeframe in TIMEFRAMES:
        trend_column = (
            f"matrix_{timeframe}_matrix_trend"
        )

        coverage[timeframe] = {
            "non_null_rows": int(
                merged[
                    trend_column
                ].notna().sum()
            ),
            "coverage_pct": round(
                100.0
                * merged[
                    trend_column
                ].notna().mean(),
                4,
            ),
        }

    latest_states = (
        matrix
        .sort_values(
            "matrix_available_at"
        )
        .groupby(
            [
                "symbol",
                "matrix_timeframe",
            ],
            observed=True,
            as_index=False,
        )
        .tail(1)
        [
            [
                "symbol",
                "matrix_timeframe",
                "matrix_available_at",
                "close",
                "matrix_vwma",
                "matrix_upper",
                "matrix_lower",
                "matrix_trend",
                "matrix_flip",
                "matrix_bars_since_flip",
                "matrix_channel_width_pct",
            ]
        ]
        .sort_values(
            [
                "symbol",
                "matrix_timeframe",
            ]
        )
    )

    latest_states.to_csv(
        LATEST_OUTPUT,
        index=False,
    )

    elapsed = time.time() - started

    report = {
        "status": "complete",
        "engine": (
            "koinvizyon-matrix-features-v1"
        ),
        "matrix_type": "VWMA",
        "matrix_length": MATRIX_LENGTH,
        "symbols": SYMBOLS,
        "timeframes": TIMEFRAMES,
        "matrix_rows": int(
            len(matrix)
        ),
        "matrix_columns": int(
            len(matrix.columns)
        ),
        "forecast_rows": int(
            len(forecast)
        ),
        "merged_rows": int(
            len(merged)
        ),
        "merged_columns": int(
            len(merged.columns)
        ),
        "matrix_feature_count": int(
            len(matrix_feature_columns)
        ),
        "matrix_coverage": coverage,
        "sources": source_report,
        "matrix_output": str(
            MATRIX_OUTPUT
        ),
        "matrix_output_mb": round(
            MATRIX_OUTPUT.stat().st_size
            / 1024**2,
            3,
        ),
        "merged_output": str(
            MERGED_OUTPUT
        ),
        "merged_output_mb": round(
            MERGED_OUTPUT.stat().st_size
            / 1024**2,
            3,
        ),
        "latest_states_output": str(
            LATEST_OUTPUT
        ),
        "elapsed_seconds": round(
            elapsed,
            3,
        ),
    }

    REPORT_OUTPUT.write_text(
        json.dumps(
            json_safe(report),
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("MATRIX FEATURE ENGINE COMPLETE")
    print("=" * 100)
    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("=" * 100)
    print("LATEST MATRIX STATES")
    print("=" * 100)
    print(
        latest_states.to_string(
            index=False
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
