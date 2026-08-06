#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


BASE_URL = "https://fapi.binance.com"
KLINE_ENDPOINT = "/fapi/v1/klines"

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

START_TIME = pd.Timestamp(
    "2025-08-06T00:00:00Z"
)

ONE_MINUTE_MS = 60_000
REQUEST_LIMIT = 1000

MATRIX_LENGTH = 20

TIMEFRAMES = {
    "5m": {
        "rule": "5min",
        "minutes": 5,
    },
    "15m": {
        "rule": "15min",
        "minutes": 15,
    },
    "1h": {
        "rule": "1h",
        "minutes": 60,
    },
    "4h": {
        "rule": "4h",
        "minutes": 240,
    },
    "1d": {
        "rule": "1D",
        "minutes": 1440,
    },
}

HORIZONS = [
    15,
    30,
    60,
]

DATA_ROOT = Path(
    "data/market/binance-futures-um"
)

REPORT_ROOT = Path(
    "reports/matrix_true_backtest"
)


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


def last_closed_minute() -> pd.Timestamp:
    return (
        pd.Timestamp.now(tz="UTC")
        .floor("min")
        - pd.Timedelta(minutes=1)
    )


def timestamp_ms(
    value: pd.Timestamp,
) -> int:
    return int(
        value.timestamp() * 1000
    )


def empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "open_time",
            "close_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ]
    )


def request_klines(
    session: requests.Session,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> list[list[Any]]:
    parameters = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": REQUEST_LIMIT,
    }

    for attempt in range(1, 9):
        try:
            response = session.get(
                BASE_URL + KLINE_ENDPOINT,
                params=parameters,
                timeout=30,
            )

            if response.status_code == 429:
                delay = int(
                    response.headers.get(
                        "Retry-After",
                        attempt * 3,
                    )
                )

                print(
                    f"{symbol}: rate limit, "
                    f"sleeping {delay}s",
                    flush=True,
                )

                time.sleep(delay)
                continue

            response.raise_for_status()

            payload = response.json()

            if not isinstance(
                payload,
                list,
            ):
                raise RuntimeError(
                    f"Unexpected response: "
                    f"{payload}"
                )

            return payload

        except (
            requests.RequestException,
            RuntimeError,
            ValueError,
        ) as exc:
            if attempt == 8:
                raise

            delay = min(
                30,
                2 ** attempt,
            )

            print(
                f"{symbol}: request error "
                f"{type(exc).__name__}: {exc}; "
                f"retry={delay}s",
                flush=True,
            )

            time.sleep(delay)

    raise RuntimeError(
        "Request retry exhausted"
    )


def payload_to_frame(
    payload: list[list[Any]],
    symbol: str,
) -> pd.DataFrame:
    if not payload:
        return empty_ohlcv()

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    frame = pd.DataFrame(
        payload,
        columns=columns,
    )

    frame["symbol"] = symbol

    frame["open_time"] = pd.to_datetime(
        frame["open_time"],
        unit="ms",
        utc=True,
    )

    frame["close_time"] = pd.to_datetime(
        frame["close_time"],
        unit="ms",
        utc=True,
    )

    float_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    for column in float_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).astype("float64")

    frame["trade_count"] = pd.to_numeric(
        frame["trade_count"],
        errors="coerce",
    ).astype("Int64")

    return frame.drop(
        columns=["ignore"]
    )


def load_existing(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        return empty_ohlcv()

    frame = pd.read_parquet(path)

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

    return frame


def atomic_parquet(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".part"
    )

    temporary.unlink(
        missing_ok=True
    )

    frame.to_parquet(
        temporary,
        index=False,
        compression="zstd",
    )

    temporary.replace(path)


def download_symbol(
    session: requests.Session,
    symbol: str,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    output_path = (
        DATA_ROOT
        / symbol
        / "1m"
        / f"{symbol}-1m.parquet"
    )

    existing = load_existing(
        output_path
    )

    existing = (
        existing
        .sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
    )

    if existing.empty:
        cursor = START_TIME
    else:
        cursor = max(
            START_TIME,
            existing["open_time"].max()
            + pd.Timedelta(minutes=1),
        )

    print()
    print("=" * 100)
    print(symbol)
    print("=" * 100)
    print(
        "Existing:",
        f"{len(existing):,}",
    )
    print(
        "Download from:",
        cursor.isoformat(),
    )
    print(
        "Download to:",
        end_time.isoformat(),
    )

    parts: list[pd.DataFrame] = []

    cursor_ms = timestamp_ms(cursor)
    end_ms = timestamp_ms(end_time)

    request_count = 0
    row_count = 0

    while cursor_ms <= end_ms:
        payload = request_klines(
            session,
            symbol,
            cursor_ms,
            end_ms,
        )

        if not payload:
            break

        part = payload_to_frame(
            payload,
            symbol,
        )

        part = part[
            part["open_time"]
            <= end_time
        ]

        if part.empty:
            break

        parts.append(part)

        request_count += 1
        row_count += len(part)

        latest = part[
            "open_time"
        ].max()

        cursor_ms = (
            timestamp_ms(latest)
            + ONE_MINUTE_MS
        )

        if (
            request_count % 25 == 0
            or len(payload) < REQUEST_LIMIT
        ):
            print(
                f"{symbol}: "
                f"requests={request_count:,} "
                f"downloaded={row_count:,} "
                f"latest={latest.isoformat()}",
                flush=True,
            )

        if len(payload) < REQUEST_LIMIT:
            break

        time.sleep(0.08)

    frames = []

    if not existing.empty:
        frames.append(existing)

    frames.extend(parts)

    if not frames:
        raise RuntimeError(
            f"No data for {symbol}"
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = (
        combined
        .sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
    )

    combined = combined[
        combined["open_time"].between(
            START_TIME,
            end_time,
        )
    ].reset_index(drop=True)

    atomic_parquet(
        combined,
        output_path,
    )

    time_gap = (
        combined["open_time"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    gap_count = int(
        (time_gap > 1).sum()
    )

    print(
        f"{symbol}: final={len(combined):,} "
        f"gaps={gap_count} "
        f"output={output_path}"
    )

    if gap_count:
        print(
            f"WARNING: {symbol} contains "
            f"{gap_count} time gaps"
        )

    return combined


def aggregate_candles(
    one_minute: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    definition = TIMEFRAMES[
        timeframe
    ]

    rule = definition["rule"]
    expected_count = definition[
        "minutes"
    ]

    indexed = (
        one_minute
        .sort_values("open_time")
        .set_index("open_time")
    )

    aggregation = indexed.resample(
        rule,
        origin="epoch",
        label="left",
        closed="left",
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "trade_count": "sum",
        "taker_buy_base_volume": "sum",
        "taker_buy_quote_volume": "sum",
    })

    counts = (
        indexed["close"]
        .resample(
            rule,
            origin="epoch",
            label="left",
            closed="left",
        )
        .count()
        .rename("source_count")
    )

    aggregation = aggregation.join(
        counts
    )

    aggregation = aggregation[
        aggregation["source_count"]
        == expected_count
    ].copy()

    aggregation = aggregation.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    aggregation[
        "open_time"
    ] = aggregation.index

    aggregation[
        "close_time"
    ] = (
        aggregation.index
        + pd.Timedelta(
            minutes=expected_count
        )
        - pd.Timedelta(
            milliseconds=1
        )
    )

    aggregation[
        "timeframe"
    ] = timeframe

    aggregation[
        "symbol"
    ] = str(
        one_minute[
            "symbol"
        ].iloc[0]
    )

    return aggregation.reset_index(
        drop=True
    )


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    return numerator / denominator.where(
        denominator.abs() > 1e-12
    )


def persistent_matrix_trend(
    source: pd.Series,
    previous_upper: pd.Series,
    previous_lower: pd.Series,
) -> pd.Series:
    source_values = source.to_numpy(
        dtype=np.float64
    )

    upper_values = (
        previous_upper.to_numpy(
            dtype=np.float64
        )
    )

    lower_values = (
        previous_lower.to_numpy(
            dtype=np.float64
        )
    )

    output = np.zeros(
        len(source_values),
        dtype=np.int8,
    )

    state = 0

    for index in range(
        len(source_values)
    ):
        source_value = (
            source_values[index]
        )

        upper_value = (
            upper_values[index]
        )

        lower_value = (
            lower_values[index]
        )

        if (
            np.isfinite(source_value)
            and np.isfinite(upper_value)
            and source_value > upper_value
        ):
            state = 1

        elif (
            np.isfinite(source_value)
            and np.isfinite(lower_value)
            and source_value < lower_value
        ):
            state = -1

        output[index] = state

    return pd.Series(
        output,
        index=source.index,
        dtype="int8",
    )


def add_matrix(
    candles: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    frame = candles.copy()

    # Pine defaults:
    # src1 = ohlc4
    # src2 = ohlc4
    frame["source"] = (
        frame["open"]
        + frame["high"]
        + frame["low"]
        + frame["close"]
    ) / 4.0

    weighted_source = (
        frame["source"]
        * frame["volume"]
    )

    weighted_sum = (
        weighted_source
        .rolling(
            MATRIX_LENGTH,
            min_periods=MATRIX_LENGTH,
        )
        .sum()
    )

    volume_sum = (
        frame["volume"]
        .rolling(
            MATRIX_LENGTH,
            min_periods=MATRIX_LENGTH,
        )
        .sum()
    )

    # Pine: vwma(ohlc4, 20)
    frame["vwma"] = safe_divide(
        weighted_sum,
        volume_sum,
    )

    # Pine:
    # h = highest(ma, 20)
    # l = lowest(ma, 20)
    frame["upper"] = (
        frame["vwma"]
        .rolling(
            MATRIX_LENGTH,
            min_periods=MATRIX_LENGTH,
        )
        .max()
    )

    frame["lower"] = (
        frame["vwma"]
        .rolling(
            MATRIX_LENGTH,
            min_periods=MATRIX_LENGTH,
        )
        .min()
    )

    frame["trend"] = (
        persistent_matrix_trend(
            frame["source"],
            frame["upper"].shift(1),
            frame["lower"].shift(1),
        )
    )

    previous_trend = (
        frame["trend"]
        .shift(1)
        .fillna(0)
        .astype("int8")
    )

    frame["long_flip"] = (
        (frame["trend"] == 1)
        & (previous_trend == -1)
    )

    frame["short_flip"] = (
        (frame["trend"] == -1)
        & (previous_trend == 1)
    )

    frame["flip"] = np.select(
        [
            frame["long_flip"],
            frame["short_flip"],
        ],
        [
            1,
            -1,
        ],
        default=0,
    ).astype("int8")

    frame[
        "distance_to_vwma_pct"
    ] = safe_divide(
        frame["source"]
        - frame["vwma"],
        frame["vwma"],
    )

    frame[
        "channel_width_pct"
    ] = safe_divide(
        frame["upper"]
        - frame["lower"],
        frame["vwma"],
    )

    frame["available_at"] = (
        frame["close_time"]
    )

    keep = [
        "symbol",
        "timeframe",
        "open_time",
        "close_time",
        "available_at",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "vwma",
        "upper",
        "lower",
        "trend",
        "flip",
        "long_flip",
        "short_flip",
        "distance_to_vwma_pct",
        "channel_width_pct",
    ]

    return frame[keep].copy()


def merge_higher_timeframes(
    one_minute_matrix: pd.DataFrame,
    matrix_frames: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    timeline = (
        one_minute_matrix
        .sort_values("available_at")
        .copy()
    )

    timeline = timeline.rename(
        columns={
            "trend": "trend_1m",
            "flip": "flip_1m",
            "long_flip": "long_flip_1m",
            "short_flip": "short_flip_1m",
            "vwma": "vwma_1m",
            "upper": "upper_1m",
            "lower": "lower_1m",
            "source": "source_1m",
            "distance_to_vwma_pct": (
                "distance_to_vwma_pct_1m"
            ),
            "channel_width_pct": (
                "channel_width_pct_1m"
            ),
        }
    )

    protected = {
        "symbol",
        "timeframe",
        "open_time",
        "close_time",
        "available_at",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    for timeframe in [
        "5m",
        "15m",
        "1h",
        "4h",
        "1d",
    ]:
        higher = matrix_frames[
            timeframe
        ].copy()

        rename = {}

        for column in higher.columns:
            if column in {
                "symbol",
                "available_at",
            }:
                continue

            rename[column] = (
                f"{column}_{timeframe}"
            )

        higher = higher.rename(
            columns=rename
        )

        higher = higher.drop(
            columns=["symbol"]
        )

        higher = higher.sort_values(
            "available_at"
        )

        timeline = pd.merge_asof(
            timeline.sort_values(
                "available_at"
            ),
            higher,
            on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )

    return timeline


def add_alignment_states(
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    frame = timeline.copy()

    trend_columns = {
        "1d": "trend_1d",
        "4h": "trend_4h",
        "1h": "trend_1h",
        "15m": "trend_15m",
        "5m": "trend_5m",
    }

    for column in trend_columns.values():
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).fillna(0).astype("int8")

    daily = frame["trend_1d"]

    frame["daily_regime"] = (
        daily.where(
            daily.isin([-1, 1]),
            0,
        )
    ).astype("int8")

    alignment_definitions = {
        "DAILY_ONLY": [
            "1d",
        ],
        "DAILY_4H": [
            "1d",
            "4h",
        ],
        "DAILY_4H_1H": [
            "1d",
            "4h",
            "1h",
        ],
        "DAILY_4H_1H_15M": [
            "1d",
            "4h",
            "1h",
            "15m",
        ],
        "FULL_ALIGNMENT": [
            "1d",
            "4h",
            "1h",
            "15m",
            "5m",
        ],
    }

    for name, timeframes in (
        alignment_definitions.items()
    ):
        columns = [
            trend_columns[timeframe]
            for timeframe in timeframes
        ]

        same_as_daily = pd.Series(
            True,
            index=frame.index,
        )

        for column in columns:
            same_as_daily &= (
                frame[column]
                == daily
            )

        same_as_daily &= daily.isin(
            [-1, 1]
        )

        frame[
            f"aligned_{name}"
        ] = same_as_daily

        frame[
            f"signal_{name}"
        ] = (
            same_as_daily
            & (
                frame["flip_1m"]
                == daily
            )
        )

    return frame


def forward_trade_metrics(
    timeline: pd.DataFrame,
    alignment: str,
    horizon: int,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:
    signal_column = (
        f"signal_{alignment}"
    )

    signal_positions = np.flatnonzero(
        timeline[
            signal_column
        ].fillna(False).to_numpy()
    )

    opens = timeline[
        "open"
    ].to_numpy(dtype=np.float64)

    highs = timeline[
        "high"
    ].to_numpy(dtype=np.float64)

    lows = timeline[
        "low"
    ].to_numpy(dtype=np.float64)

    closes = timeline[
        "close"
    ].to_numpy(dtype=np.float64)

    directions = timeline[
        "daily_regime"
    ].to_numpy(dtype=np.int8)

    times = timeline[
        "available_at"
    ].to_numpy()

    rows = []

    for signal_position in signal_positions:
        entry_position = (
            signal_position + 1
        )

        exit_position = (
            signal_position + horizon
        )

        if (
            entry_position >= len(timeline)
            or exit_position >= len(timeline)
        ):
            continue

        entry_price = opens[
            entry_position
        ]

        exit_price = closes[
            exit_position
        ]

        direction = int(
            directions[
                signal_position
            ]
        )

        if (
            direction not in (-1, 1)
            or not np.isfinite(entry_price)
            or not np.isfinite(exit_price)
            or entry_price <= 0
        ):
            continue

        raw_return_bps = (
            exit_price / entry_price - 1.0
        ) * 10_000.0

        directional_return_bps = (
            raw_return_bps
            * direction
        )

        window_high = np.nanmax(
            highs[
                entry_position:
                exit_position + 1
            ]
        )

        window_low = np.nanmin(
            lows[
                entry_position:
                exit_position + 1
            ]
        )

        if direction == 1:
            mfe_bps = (
                window_high / entry_price
                - 1.0
            ) * 10_000.0

            mae_bps = (
                window_low / entry_price
                - 1.0
            ) * 10_000.0

        else:
            mfe_bps = (
                entry_price / window_low
                - 1.0
            ) * 10_000.0

            mae_bps = (
                entry_price / window_high
                - 1.0
            ) * 10_000.0

        rows.append({
            "symbol": (
                timeline[
                    "symbol"
                ].iloc[0]
            ),
            "alignment": alignment,
            "horizon_minutes": horizon,
            "signal_time": (
                pd.Timestamp(
                    times[signal_position]
                )
            ),
            "entry_time": (
                pd.Timestamp(
                    times[entry_position]
                )
            ),
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "raw_return_bps": (
                raw_return_bps
            ),
            "directional_return_bps": (
                directional_return_bps
            ),
            "mfe_bps": mfe_bps,
            "mae_bps": mae_bps,
        })

    trades = pd.DataFrame(rows)

    if trades.empty:
        return {
            "symbol": (
                timeline[
                    "symbol"
                ].iloc[0]
            ),
            "alignment": alignment,
            "horizon_minutes": horizon,
            "trades": 0,
        }, trades

    returns = trades[
        "directional_return_bps"
    ]

    wins = returns[
        returns > 0
    ]

    losses = returns[
        returns < 0
    ]

    gross_profit = float(
        wins.sum()
    )

    gross_loss = float(
        losses.sum()
    )

    profit_factor = (
        gross_profit / abs(gross_loss)
        if gross_loss < 0
        else np.inf
    )

    mean_return = float(
        returns.mean()
    )

    standard_deviation = float(
        returns.std(ddof=1)
    ) if len(returns) > 1 else np.nan

    t_stat = (
        mean_return
        / (
            standard_deviation
            / math.sqrt(len(returns))
        )
        if (
            len(returns) > 1
            and np.isfinite(
                standard_deviation
            )
            and standard_deviation > 0
        )
        else np.nan
    )

    metrics = {
        "symbol": trades[
            "symbol"
        ].iloc[0],
        "alignment": alignment,
        "horizon_minutes": horizon,
        "trades": int(
            len(trades)
        ),
        "long_trades": int(
            (trades["direction"] == 1)
            .sum()
        ),
        "short_trades": int(
            (trades["direction"] == -1)
            .sum()
        ),
        "win_rate": float(
            (returns > 0).mean()
        ),
        "mean_return_bps": (
            mean_return
        ),
        "median_return_bps": float(
            returns.median()
        ),
        "profit_factor": float(
            profit_factor
        ),
        "mean_mfe_bps": float(
            trades["mfe_bps"].mean()
        ),
        "mean_mae_bps": float(
            trades["mae_bps"].mean()
        ),
        "q10_return_bps": float(
            returns.quantile(0.10)
        ),
        "q90_return_bps": float(
            returns.quantile(0.90)
        ),
        "t_stat": float(
            t_stat
        ),
        "first_trade": (
            trades[
                "signal_time"
            ].min().isoformat()
        ),
        "last_trade": (
            trades[
                "signal_time"
            ].max().isoformat()
        ),
    }

    return metrics, trades


def analyze_symbol(
    one_minute: pd.DataFrame,
    symbol: str,
) -> tuple[
    list[dict[str, Any]],
    list[pd.DataFrame],
    pd.DataFrame,
]:
    print()
    print("=" * 100)
    print(
        f"BUILDING MATRIX: {symbol}"
    )
    print("=" * 100)

    matrix_frames: dict[
        str,
        pd.DataFrame,
    ] = {}

    one_minute_candles = (
        one_minute.copy()
    )

    one_minute_candles[
        "timeframe"
    ] = "1m"

    matrix_frames["1m"] = add_matrix(
        one_minute_candles,
        "1m",
    )

    for timeframe in TIMEFRAMES:
        candles = aggregate_candles(
            one_minute,
            timeframe,
        )

        matrix = add_matrix(
            candles,
            timeframe,
        )

        matrix_frames[
            timeframe
        ] = matrix

        output_path = (
            DATA_ROOT
            / symbol
            / timeframe
            / (
                f"{symbol}-{timeframe}-"
                f"matrix.parquet"
            )
        )

        atomic_parquet(
            matrix,
            output_path,
        )

        print(
            f"{symbol} {timeframe}: "
            f"candles={len(candles):,} "
            f"long_flips="
            f"{int(matrix['long_flip'].sum()):,} "
            f"short_flips="
            f"{int(matrix['short_flip'].sum()):,}"
        )

    timeline = (
        merge_higher_timeframes(
            matrix_frames["1m"],
            matrix_frames,
        )
    )

    timeline = add_alignment_states(
        timeline
    )

    timeline_output = (
        REPORT_ROOT
        / f"{symbol}_matrix_timeline.parquet"
    )

    atomic_parquet(
        timeline,
        timeline_output,
    )

    alignments = [
        "DAILY_ONLY",
        "DAILY_4H",
        "DAILY_4H_1H",
        "DAILY_4H_1H_15M",
        "FULL_ALIGNMENT",
    ]

    metrics = []
    trade_frames = []

    for alignment in alignments:
        for horizon in HORIZONS:
            (
                result,
                trades,
            ) = forward_trade_metrics(
                timeline,
                alignment,
                horizon,
            )

            metrics.append(result)

            if not trades.empty:
                trade_frames.append(
                    trades
                )

            print(
                f"{symbol:<8} "
                f"{alignment:<24} "
                f"{horizon:>2}m "
                f"trades={result.get('trades', 0):>4} "
                f"win={result.get('win_rate', np.nan):>7.2%} "
                f"mean={result.get('mean_return_bps', np.nan):>8.2f} "
                f"PF={result.get('profit_factor', np.nan):>7.3f}"
            )

    return (
        metrics,
        trade_frames,
        timeline,
    )


def main() -> int:
    started = time.time()

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "LiqHeat-Koinvizyon-"
            "Matrix-Research/1.0"
        ),
        "Accept": "application/json",
    })

    end_time = last_closed_minute()

    print("=" * 110)
    print(
        "KOINVIZYON MATRIX — "
        "TRUE MULTI-TIMEFRAME BACKTEST"
    )
    print("=" * 110)
    print(
        "Source:",
        BASE_URL + KLINE_ENDPOINT,
    )
    print(
        "Period:",
        START_TIME.isoformat(),
        "→",
        end_time.isoformat(),
    )
    print(
        "Matrix:",
        "VWMA(20), OHLC4 source",
    )
    print(
        "Entry:",
        "1m flip, next 1m open",
    )
    print()

    all_metrics = []
    all_trade_frames = []

    data_reports = []

    for symbol in SYMBOLS:
        one_minute = download_symbol(
            session,
            symbol,
            end_time,
        )

        gap_count = int(
            (
                one_minute[
                    "open_time"
                ]
                .diff()
                .dt.total_seconds()
                .div(60)
                > 1
            ).sum()
        )

        data_reports.append({
            "symbol": symbol,
            "rows": int(
                len(one_minute)
            ),
            "minimum_time": (
                one_minute[
                    "open_time"
                ].min().isoformat()
            ),
            "maximum_time": (
                one_minute[
                    "open_time"
                ].max().isoformat()
            ),
            "gap_count": gap_count,
        })

        (
            metrics,
            trade_frames,
            _timeline,
        ) = analyze_symbol(
            one_minute,
            symbol,
        )

        all_metrics.extend(
            metrics
        )

        all_trade_frames.extend(
            trade_frames
        )

    metrics_frame = pd.DataFrame(
        all_metrics
    )

    metrics_frame.to_csv(
        REPORT_ROOT
        / "matrix_backtest_metrics.csv",
        index=False,
    )

    if all_trade_frames:
        trades_frame = pd.concat(
            all_trade_frames,
            ignore_index=True,
        )

        trades_frame.to_parquet(
            REPORT_ROOT
            / "matrix_backtest_trades.parquet",
            index=False,
            compression="zstd",
        )
    else:
        trades_frame = pd.DataFrame()

    aggregate = (
        metrics_frame
        .groupby(
            [
                "alignment",
                "horizon_minutes",
            ],
            observed=True,
        )
        .agg(
            symbols=(
                "symbol",
                "nunique",
            ),
            trades=(
                "trades",
                "sum",
            ),
            mean_win_rate=(
                "win_rate",
                "mean",
            ),
            mean_return_bps=(
                "mean_return_bps",
                "mean",
            ),
            median_return_bps=(
                "median_return_bps",
                "mean",
            ),
            mean_profit_factor=(
                "profit_factor",
                "mean",
            ),
            mean_mfe_bps=(
                "mean_mfe_bps",
                "mean",
            ),
            mean_mae_bps=(
                "mean_mae_bps",
                "mean",
            ),
            mean_t_stat=(
                "t_stat",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "horizon_minutes",
                "alignment",
            ]
        )
    )

    aggregate.to_csv(
        REPORT_ROOT
        / "matrix_backtest_summary.csv",
        index=False,
    )

    report = {
        "status": "complete",
        "engine": (
            "koinvizyon-matrix-"
            "true-backtest-v1"
        ),
        "source": (
            BASE_URL + KLINE_ENDPOINT
        ),
        "market": (
            "Binance USD-M Futures"
        ),
        "symbols": SYMBOLS,
        "start_time": (
            START_TIME.isoformat()
        ),
        "end_time": (
            end_time.isoformat()
        ),
        "matrix": {
            "ma_type": "VWMA",
            "length": MATRIX_LENGTH,
            "source": "OHLC4",
            "signal_source": "OHLC4",
        },
        "entry_rule": (
            "Signal on closed 1m Matrix "
            "flip; entry at next 1m open"
        ),
        "alignment_levels": [
            "DAILY_ONLY",
            "DAILY_4H",
            "DAILY_4H_1H",
            "DAILY_4H_1H_15M",
            "FULL_ALIGNMENT",
        ],
        "horizons_minutes": (
            HORIZONS
        ),
        "data": data_reports,
        "metric_rows": int(
            len(metrics_frame)
        ),
        "trade_rows": int(
            len(trades_frame)
        ),
        "elapsed_seconds": round(
            time.time() - started,
            3,
        ),
        "outputs": {
            "metrics": str(
                REPORT_ROOT
                / "matrix_backtest_metrics.csv"
            ),
            "summary": str(
                REPORT_ROOT
                / "matrix_backtest_summary.csv"
            ),
            "trades": str(
                REPORT_ROOT
                / "matrix_backtest_trades.parquet"
            ),
        },
    }

    (
        REPORT_ROOT
        / "matrix_true_backtest_report.json"
    ).write_text(
        json.dumps(
            json_safe(report),
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 140)
    print(
        "TRUE MATRIX BACKTEST COMPLETE"
    )
    print("=" * 140)
    print()
    print(
        aggregate.to_string(
            index=False
        )
    )
    print()
    print(
        "Report:",
        REPORT_ROOT
        / "matrix_true_backtest_report.json",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
