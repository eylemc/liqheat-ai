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

import matrix_true_backtest as base


REPORT_ROOT = Path(
    "reports/matrix_event_study_v3"
)

ENTRY_TIMEFRAMES = [
    "1m",
    "5m",
    "15m",
    "1h",
]

ALIGNMENTS = {
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

WINDOWS_MINUTES = [
    15,
    30,
    60,
    120,
    240,
]

TARGET_LEVELS_BPS = [
    10,
    25,
    50,
    75,
    100,
    150,
    200,
]

STOP_LEVELS_BPS = [
    25,
    50,
    100,
]

FIRST_HIT_PAIRS = [
    (25, 25),
    (50, 25),
    (50, 50),
    (100, 50),
    (100, 100),
]

# Günlük candle UTC 00:00 bazlı 24 saat.
base.TIMEFRAMES["1d"]["rule"] = "24h"


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


def force_ns_utc(
    values: pd.Series,
) -> pd.Series:
    return (
        pd.to_datetime(
            values,
            utc=True,
            errors="coerce",
        )
        .astype("datetime64[ns, UTC]")
    )


def prepare_matrix_frames(
    one_minute: pd.DataFrame,
    symbol: str,
) -> dict[str, pd.DataFrame]:
    matrix_frames: dict[
        str,
        pd.DataFrame
    ] = {}

    one_minute = one_minute.copy()

    one_minute["open_time"] = force_ns_utc(
        one_minute["open_time"]
    )

    one_minute["close_time"] = force_ns_utc(
        one_minute["close_time"]
    )

    minute_candles = one_minute.copy()
    minute_candles["timeframe"] = "1m"

    matrix_1m = base.add_matrix(
        minute_candles,
        "1m",
    )

    matrix_1m["available_at"] = force_ns_utc(
        matrix_1m["available_at"]
    )

    matrix_frames["1m"] = matrix_1m

    for timeframe in [
        "5m",
        "15m",
        "1h",
        "4h",
        "1d",
    ]:
        candles = base.aggregate_candles(
            one_minute,
            timeframe,
        )

        candles["open_time"] = force_ns_utc(
            candles["open_time"]
        )

        candles["close_time"] = force_ns_utc(
            candles["close_time"]
        )

        matrix = base.add_matrix(
            candles,
            timeframe,
        )

        matrix["available_at"] = force_ns_utc(
            matrix["available_at"]
        )

        matrix_frames[timeframe] = matrix

        print(
            f"{symbol:<8} "
            f"{timeframe:<3} "
            f"candles={len(candles):>7,} "
            f"long_flips="
            f"{int(matrix['long_flip'].sum()):>5,} "
            f"short_flips="
            f"{int(matrix['short_flip'].sum()):>5,}",
            flush=True,
        )

    return matrix_frames


def merge_matrix_timeline(
    matrix_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    minute = matrix_frames["1m"].copy()

    minute["available_at"] = force_ns_utc(
        minute["available_at"]
    )

    minute = (
        minute
        .dropna(
            subset=["available_at"]
        )
        .sort_values("available_at")
        .reset_index(drop=True)
    )

    timeline = minute.rename(
        columns={
            "trend": "trend_1m",
            "flip": "flip_1m",
            "long_flip": "long_flip_1m",
            "short_flip": "short_flip_1m",
            "source": "source_1m",
            "vwma": "vwma_1m",
            "upper": "upper_1m",
            "lower": "lower_1m",
            "distance_to_vwma_pct":
                "distance_to_vwma_pct_1m",
            "channel_width_pct":
                "channel_width_pct_1m",
        }
    )

    timeline["new_candle_1m"] = True

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

        higher["available_at"] = force_ns_utc(
            higher["available_at"]
        )

        higher = (
            higher
            .dropna(
                subset=["available_at"]
            )
            .sort_values("available_at")
            .drop_duplicates(
                subset=["available_at"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        source_time_column = (
            f"matrix_close_time_{timeframe}"
        )

        higher[source_time_column] = (
            higher["available_at"]
        )

        rename_map = {}

        for column in higher.columns:
            if column in {
                "symbol",
                "available_at",
                source_time_column,
            }:
                continue

            rename_map[column] = (
                f"{column}_{timeframe}"
            )

        higher = higher.rename(
            columns=rename_map
        )

        higher = higher.drop(
            columns=["symbol"]
        )

        timeline = pd.merge_asof(
            timeline.sort_values(
                "available_at"
            ),
            higher.sort_values(
                "available_at"
            ),
            on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )

        timeline[
            source_time_column
        ] = force_ns_utc(
            timeline[source_time_column]
        )

        timeline[
            f"new_candle_{timeframe}"
        ] = (
            timeline[
                source_time_column
            ].notna()
            & timeline[
                source_time_column
            ].ne(
                timeline[
                    source_time_column
                ].shift(1)
            )
        )

    return (
        timeline
        .sort_values("available_at")
        .reset_index(drop=True)
    )


def add_alignment_signals(
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

    flip_columns = {
        "1m": "flip_1m",
        "5m": "flip_5m",
        "15m": "flip_15m",
        "1h": "flip_1h",
    }

    for column in [
        *trend_columns.values(),
        *flip_columns.values(),
    ]:
        frame[column] = (
            pd.to_numeric(
                frame[column],
                errors="coerce",
            )
            .fillna(0)
            .astype("int8")
        )

    daily_direction = frame["trend_1d"]

    frame["daily_direction"] = (
        daily_direction
        .where(
            daily_direction.isin([-1, 1]),
            0,
        )
        .astype("int8")
    )

    for alignment_name, timeframes in (
        ALIGNMENTS.items()
    ):
        aligned = frame[
            "daily_direction"
        ].isin([-1, 1])

        for timeframe in timeframes:
            aligned &= (
                frame[
                    trend_columns[timeframe]
                ]
                == frame["daily_direction"]
            )

        frame[
            f"aligned_{alignment_name}"
        ] = aligned

    for entry_timeframe in (
        ENTRY_TIMEFRAMES
    ):
        flip_column = (
            flip_columns[entry_timeframe]
        )

        new_candle_column = (
            f"new_candle_{entry_timeframe}"
        )

        flip_event = (
            frame[
                new_candle_column
            ].fillna(False)
            & frame[
                flip_column
            ].isin([-1, 1])
        )

        frame[
            f"flip_event_{entry_timeframe}"
        ] = flip_event

        for alignment_name in (
            ALIGNMENTS
        ):
            frame[
                f"event_"
                f"{entry_timeframe}_"
                f"{alignment_name}"
            ] = (
                flip_event
                & frame[
                    f"aligned_{alignment_name}"
                ]
                & (
                    frame[flip_column]
                    == frame["daily_direction"]
                )
            )

    return frame


def first_hit_minutes(
    favorable_path_bps: np.ndarray,
    adverse_path_bps: np.ndarray,
    target_bps: float,
    stop_bps: float,
) -> tuple[
    str,
    float | None,
    float | None,
]:
    target_hits = np.flatnonzero(
        favorable_path_bps
        >= target_bps
    )

    stop_hits = np.flatnonzero(
        adverse_path_bps
        <= -stop_bps
    )

    target_minute = (
        int(target_hits[0] + 1)
        if len(target_hits)
        else None
    )

    stop_minute = (
        int(stop_hits[0] + 1)
        if len(stop_hits)
        else None
    )

    if (
        target_minute is None
        and stop_minute is None
    ):
        result = "NONE"

    elif stop_minute is None:
        result = "TARGET_FIRST"

    elif target_minute is None:
        result = "STOP_FIRST"

    elif target_minute < stop_minute:
        result = "TARGET_FIRST"

    elif stop_minute < target_minute:
        result = "STOP_FIRST"

    else:
        # Aynı 1m mumda hem hedef hem stop görülürse
        # mum içi sıralama bilinmediği için belirsiz.
        result = "SAME_BAR_AMBIGUOUS"

    return (
        result,
        target_minute,
        stop_minute,
    )


def build_event_rows(
    timeline: pd.DataFrame,
    entry_timeframe: str,
    alignment: str,
) -> pd.DataFrame:
    event_column = (
        f"event_"
        f"{entry_timeframe}_"
        f"{alignment}"
    )

    positions = np.flatnonzero(
        timeline[
            event_column
        ]
        .fillna(False)
        .to_numpy(dtype=bool)
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
        "daily_direction"
    ].to_numpy(dtype=np.int8)

    times = pd.to_datetime(
        timeline["available_at"],
        utc=True,
    ).to_numpy()

    symbol = str(
        timeline["symbol"].iloc[0]
    )

    event_rows = []

    maximum_window = max(
        WINDOWS_MINUTES
    )

    for signal_position in positions:
        entry_position = (
            signal_position + 1
        )

        maximum_exit_position = (
            entry_position
            + maximum_window
            - 1
        )

        if (
            entry_position >= len(timeline)
            or maximum_exit_position
            >= len(timeline)
        ):
            continue

        direction = int(
            directions[signal_position]
        )

        entry_price = float(
            opens[entry_position]
        )

        if (
            direction not in (-1, 1)
            or not np.isfinite(entry_price)
            or entry_price <= 0
        ):
            continue

        row: dict[str, Any] = {
            "symbol": symbol,
            "entry_timeframe": (
                entry_timeframe
            ),
            "alignment": alignment,
            "signal_time": pd.Timestamp(
                times[signal_position]
            ),
            "entry_time": pd.Timestamp(
                times[entry_position]
            ),
            "direction": direction,
            "entry_price": entry_price,
        }

        for window in WINDOWS_MINUTES:
            exit_position = (
                entry_position + window - 1
            )

            window_highs = highs[
                entry_position:
                exit_position + 1
            ]

            window_lows = lows[
                entry_position:
                exit_position + 1
            ]

            window_closes = closes[
                entry_position:
                exit_position + 1
            ]

            if (
                not len(window_highs)
                or not len(window_lows)
            ):
                continue

            if direction == 1:
                favorable_path_bps = (
                    window_highs
                    / entry_price
                    - 1.0
                ) * 10_000.0

                adverse_path_bps = (
                    window_lows
                    / entry_price
                    - 1.0
                ) * 10_000.0

                terminal_return_bps = (
                    window_closes[-1]
                    / entry_price
                    - 1.0
                ) * 10_000.0

            else:
                favorable_path_bps = (
                    entry_price
                    / window_lows
                    - 1.0
                ) * 10_000.0

                adverse_path_bps = (
                    entry_price
                    / window_highs
                    - 1.0
                ) * 10_000.0

                terminal_return_bps = (
                    entry_price
                    / window_closes[-1]
                    - 1.0
                ) * 10_000.0

            maximum_favorable = float(
                np.nanmax(
                    favorable_path_bps
                )
            )

            maximum_adverse = float(
                np.nanmin(
                    adverse_path_bps
                )
            )

            time_to_mfe = int(
                np.nanargmax(
                    favorable_path_bps
                ) + 1
            )

            time_to_mae = int(
                np.nanargmin(
                    adverse_path_bps
                ) + 1
            )

            row[
                f"mfe_bps_{window}m"
            ] = maximum_favorable

            row[
                f"mae_bps_{window}m"
            ] = maximum_adverse

            row[
                f"terminal_return_bps_"
                f"{window}m"
            ] = float(
                terminal_return_bps
            )

            row[
                f"time_to_mfe_minutes_"
                f"{window}m"
            ] = time_to_mfe

            row[
                f"time_to_mae_minutes_"
                f"{window}m"
            ] = time_to_mae

            row[
                f"mfe_mae_ratio_{window}m"
            ] = (
                maximum_favorable
                / abs(maximum_adverse)
                if maximum_adverse < 0
                else np.inf
            )

            for target_bps in (
                TARGET_LEVELS_BPS
            ):
                hit_positions = np.flatnonzero(
                    favorable_path_bps
                    >= target_bps
                )

                row[
                    f"hit_target_"
                    f"{target_bps}bps_"
                    f"{window}m"
                ] = bool(
                    len(hit_positions)
                )

                row[
                    f"time_to_target_"
                    f"{target_bps}bps_"
                    f"{window}m"
                ] = (
                    int(
                        hit_positions[0] + 1
                    )
                    if len(hit_positions)
                    else np.nan
                )

            for (
                target_bps,
                stop_bps,
            ) in FIRST_HIT_PAIRS:
                (
                    result,
                    target_minute,
                    stop_minute,
                ) = first_hit_minutes(
                    favorable_path_bps,
                    adverse_path_bps,
                    target_bps,
                    stop_bps,
                )

                prefix = (
                    f"first_hit_"
                    f"t{target_bps}_"
                    f"s{stop_bps}_"
                    f"{window}m"
                )

                row[prefix] = result

                row[
                    f"{prefix}_target_minute"
                ] = (
                    target_minute
                    if target_minute
                    is not None
                    else np.nan
                )

                row[
                    f"{prefix}_stop_minute"
                ] = (
                    stop_minute
                    if stop_minute
                    is not None
                    else np.nan
                )

        event_rows.append(row)

    return pd.DataFrame(
        event_rows
    )


def summarize_event_group(
    group: pd.DataFrame,
    window: int,
) -> dict[str, Any]:
    mfe = pd.to_numeric(
        group[
            f"mfe_bps_{window}m"
        ],
        errors="coerce",
    )

    mae = pd.to_numeric(
        group[
            f"mae_bps_{window}m"
        ],
        errors="coerce",
    )

    terminal = pd.to_numeric(
        group[
            f"terminal_return_bps_"
            f"{window}m"
        ],
        errors="coerce",
    )

    output: dict[str, Any] = {
        "events": int(len(group)),
        "mean_mfe_bps": float(
            mfe.mean()
        ),
        "median_mfe_bps": float(
            mfe.median()
        ),
        "q25_mfe_bps": float(
            mfe.quantile(0.25)
        ),
        "q75_mfe_bps": float(
            mfe.quantile(0.75)
        ),
        "q90_mfe_bps": float(
            mfe.quantile(0.90)
        ),
        "mean_mae_bps": float(
            mae.mean()
        ),
        "median_mae_bps": float(
            mae.median()
        ),
        "q10_mae_bps": float(
            mae.quantile(0.10)
        ),
        "mean_terminal_return_bps": (
            float(terminal.mean())
        ),
        "median_terminal_return_bps": (
            float(terminal.median())
        ),
        "terminal_win_rate": float(
            (terminal > 0).mean()
        ),
        "mean_time_to_mfe_minutes": (
            float(
                group[
                    f"time_to_mfe_minutes_"
                    f"{window}m"
                ].mean()
            )
        ),
        "median_time_to_mfe_minutes": (
            float(
                group[
                    f"time_to_mfe_minutes_"
                    f"{window}m"
                ].median()
            )
        ),
        "mean_mfe_mae_ratio": float(
            group[
                f"mfe_mae_ratio_"
                f"{window}m"
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .mean()
        ),
    }

    for target_bps in TARGET_LEVELS_BPS:
        hit_column = (
            f"hit_target_"
            f"{target_bps}bps_"
            f"{window}m"
        )

        time_column = (
            f"time_to_target_"
            f"{target_bps}bps_"
            f"{window}m"
        )

        output[
            f"target_{target_bps}bps_"
            f"hit_rate"
        ] = float(
            group[hit_column]
            .fillna(False)
            .mean()
        )

        hit_times = pd.to_numeric(
            group.loc[
                group[hit_column]
                .fillna(False),
                time_column,
            ],
            errors="coerce",
        )

        output[
            f"target_{target_bps}bps_"
            f"median_hit_minutes"
        ] = (
            float(hit_times.median())
            if len(hit_times)
            else np.nan
        )

    for (
        target_bps,
        stop_bps,
    ) in FIRST_HIT_PAIRS:
        column = (
            f"first_hit_"
            f"t{target_bps}_"
            f"s{stop_bps}_"
            f"{window}m"
        )

        values = group[column].astype(
            "string"
        )

        output[
            f"t{target_bps}_"
            f"s{stop_bps}_"
            f"target_first_rate"
        ] = float(
            (
                values
                == "TARGET_FIRST"
            ).mean()
        )

        output[
            f"t{target_bps}_"
            f"s{stop_bps}_"
            f"stop_first_rate"
        ] = float(
            (
                values
                == "STOP_FIRST"
            ).mean()
        )

        resolved = values.isin([
            "TARGET_FIRST",
            "STOP_FIRST",
        ])

        output[
            f"t{target_bps}_"
            f"s{stop_bps}_"
            f"resolved_target_win_rate"
        ] = (
            float(
                (
                    values.loc[resolved]
                    == "TARGET_FIRST"
                ).mean()
            )
            if resolved.any()
            else np.nan
        )

    return output


def build_summary(
    events: pd.DataFrame,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    rows = []

    grouping_columns = [
        "symbol",
        "entry_timeframe",
        "alignment",
    ]

    for keys, group in events.groupby(
        grouping_columns,
        observed=True,
        sort=True,
    ):
        (
            symbol,
            entry_timeframe,
            alignment,
        ) = keys

        for window in WINDOWS_MINUTES:
            metrics = summarize_event_group(
                group,
                window,
            )

            rows.append({
                "symbol": symbol,
                "entry_timeframe": (
                    entry_timeframe
                ),
                "alignment": alignment,
                "window_minutes": window,
                **metrics,
            })

    # Pooled ALL sembol özeti.
    pooled_columns = [
        "entry_timeframe",
        "alignment",
    ]

    for keys, group in events.groupby(
        pooled_columns,
        observed=True,
        sort=True,
    ):
        (
            entry_timeframe,
            alignment,
        ) = keys

        for window in WINDOWS_MINUTES:
            metrics = summarize_event_group(
                group,
                window,
            )

            rows.append({
                "symbol": "ALL",
                "entry_timeframe": (
                    entry_timeframe
                ),
                "alignment": alignment,
                "window_minutes": window,
                **metrics,
            })

    return (
        pd.DataFrame(rows)
        .sort_values([
            "symbol",
            "entry_timeframe",
            "window_minutes",
            "alignment",
        ])
        .reset_index(drop=True)
    )


def build_best_results(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    eligible = summary[
        (summary["symbol"] == "ALL")
        & (summary["events"] >= 10)
    ].copy()

    best_parts = []

    for metric in [
        "mean_mfe_bps",
        "terminal_win_rate",
        "target_50bps_hit_rate",
        "target_100bps_hit_rate",
        "t50_s25_resolved_target_win_rate",
        "t100_s50_resolved_target_win_rate",
    ]:
        if metric not in eligible.columns:
            continue

        ranking = (
            eligible
            .sort_values(
                metric,
                ascending=False,
            )
            .groupby(
                [
                    "entry_timeframe",
                    "window_minutes",
                ],
                observed=True,
                as_index=False,
            )
            .head(3)
            .copy()
        )

        ranking[
            "ranking_metric"
        ] = metric

        ranking[
            "ranking_value"
        ] = ranking[metric]

        best_parts.append(ranking)

    if not best_parts:
        return pd.DataFrame()

    return pd.concat(
        best_parts,
        ignore_index=True,
    )


def analyze_symbol(
    session: requests.Session,
    symbol: str,
    end_time: pd.Timestamp,
) -> list[pd.DataFrame]:
    one_minute = base.download_symbol(
        session,
        symbol,
        end_time,
    )

    one_minute["open_time"] = force_ns_utc(
        one_minute["open_time"]
    )

    one_minute["close_time"] = force_ns_utc(
        one_minute["close_time"]
    )

    print()
    print("=" * 110)
    print(
        f"MATRIX EVENT STUDY: {symbol}"
    )
    print("=" * 110)

    matrix_frames = prepare_matrix_frames(
        one_minute,
        symbol,
    )

    timeline = merge_matrix_timeline(
        matrix_frames
    )

    timeline = add_alignment_signals(
        timeline
    )

    timeline_path = (
        REPORT_ROOT
        / f"{symbol}_event_timeline.parquet"
    )

    base.atomic_parquet(
        timeline,
        timeline_path,
    )

    symbol_event_frames = []

    for entry_timeframe in (
        ENTRY_TIMEFRAMES
    ):
        for alignment in ALIGNMENTS:
            events = build_event_rows(
                timeline,
                entry_timeframe,
                alignment,
            )

            if events.empty:
                print(
                    f"{symbol:<8} "
                    f"entry={entry_timeframe:<3} "
                    f"{alignment:<23} "
                    f"events=0"
                )
                continue

            symbol_event_frames.append(
                events
            )

            print(
                f"{symbol:<8} "
                f"entry={entry_timeframe:<3} "
                f"{alignment:<23} "
                f"events={len(events):>5} "
                f"60m MFE="
                f"{events['mfe_bps_60m'].mean():>8.2f} "
                f"60m hit50="
                f"{events['hit_target_50bps_60m'].mean():>7.2%} "
                f"60m terminal win="
                f"{(events['terminal_return_bps_60m'] > 0).mean():>7.2%}",
                flush=True,
            )

    return symbol_event_frames


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
            "Matrix-Event-Study-V3"
        ),
        "Accept": "application/json",
    })

    end_time = (
        base.last_closed_minute()
    )

    print("=" * 120)
    print(
        "KOINVIZYON MATRIX "
        "DELAYED-OPPORTUNITY EVENT STUDY V3"
    )
    print("=" * 120)

    print(
        "Source:",
        base.BASE_URL
        + base.KLINE_ENDPOINT,
    )

    print(
        "Period:",
        base.START_TIME.isoformat(),
        "→",
        end_time.isoformat(),
    )

    print(
        "Matrix:",
        "VWMA(20), OHLC4",
    )

    print(
        "Entry events:",
        ENTRY_TIMEFRAMES,
    )

    print(
        "Opportunity windows:",
        WINDOWS_MINUTES,
    )

    print(
        "Targets bps:",
        TARGET_LEVELS_BPS,
    )

    print()

    all_event_frames = []

    for symbol in base.SYMBOLS:
        all_event_frames.extend(
            analyze_symbol(
                session,
                symbol,
                end_time,
            )
        )

    if not all_event_frames:
        raise RuntimeError(
            "No Matrix events were produced."
        )

    events = pd.concat(
        all_event_frames,
        ignore_index=True,
    )

    events.to_parquet(
        REPORT_ROOT
        / "matrix_event_observations.parquet",
        index=False,
        compression="zstd",
    )

    summary = build_summary(events)

    summary.to_csv(
        REPORT_ROOT
        / "matrix_event_summary.csv",
        index=False,
    )

    best = build_best_results(
        summary
    )

    best.to_csv(
        REPORT_ROOT
        / "matrix_event_best_results.csv",
        index=False,
    )

    compact_columns = [
        "symbol",
        "entry_timeframe",
        "alignment",
        "window_minutes",
        "events",
        "mean_mfe_bps",
        "median_mfe_bps",
        "mean_mae_bps",
        "mean_terminal_return_bps",
        "terminal_win_rate",
        "mean_time_to_mfe_minutes",
        "target_25bps_hit_rate",
        "target_50bps_hit_rate",
        "target_100bps_hit_rate",
        "t50_s25_resolved_target_win_rate",
        "t100_s50_resolved_target_win_rate",
    ]

    available_columns = [
        column
        for column in compact_columns
        if column in summary.columns
    ]

    compact = summary[
        available_columns
    ].copy()

    compact.to_csv(
        REPORT_ROOT
        / "matrix_event_compact_summary.csv",
        index=False,
    )

    report = {
        "status": "complete",
        "engine": (
            "koinvizyon-matrix-"
            "delayed-opportunity-event-study-v3"
        ),
        "symbols": base.SYMBOLS,
        "market": (
            "Binance USD-M Futures"
        ),
        "start_time": (
            base.START_TIME.isoformat()
        ),
        "end_time": (
            end_time.isoformat()
        ),
        "matrix": {
            "ma_type": "VWMA",
            "length": 20,
            "source": "OHLC4",
            "signal_source": "OHLC4",
        },
        "entry_timeframes": (
            ENTRY_TIMEFRAMES
        ),
        "alignment_levels": list(
            ALIGNMENTS.keys()
        ),
        "opportunity_windows_minutes": (
            WINDOWS_MINUTES
        ),
        "target_levels_bps": (
            TARGET_LEVELS_BPS
        ),
        "first_hit_pairs": (
            FIRST_HIT_PAIRS
        ),
        "event_rows": int(
            len(events)
        ),
        "summary_rows": int(
            len(summary)
        ),
        "elapsed_seconds": round(
            time.time() - started,
            3,
        ),
        "generated_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "outputs": {
            "events": str(
                REPORT_ROOT
                / "matrix_event_observations.parquet"
            ),
            "summary": str(
                REPORT_ROOT
                / "matrix_event_summary.csv"
            ),
            "compact_summary": str(
                REPORT_ROOT
                / "matrix_event_compact_summary.csv"
            ),
            "best_results": str(
                REPORT_ROOT
                / "matrix_event_best_results.csv"
            ),
        },
    }

    (
        REPORT_ROOT
        / "matrix_event_study_report.json"
    ).write_text(
        json.dumps(
            json_safe(report),
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 150)
    print(
        "MATRIX EVENT STUDY V3 COMPLETE"
    )
    print("=" * 150)
    print()

    pooled = compact[
        compact["symbol"] == "ALL"
    ]

    print(
        pooled.to_string(
            index=False
        )
    )

    print()
    print(
        "Report:",
        REPORT_ROOT
        / "matrix_event_study_report.json",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
